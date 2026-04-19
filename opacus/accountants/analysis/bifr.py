"""
BIFR analysis helpers for exact finite-horizon runtime/accountant objects.

This module owns the BIFR-specific coefficient algebra used by both the runtime
surface and the accountant bridge:
- inverse-side generalized-binomial coefficient generation
- exact finite-horizon factor-column resolution
- structural-stability checks for the exact finite-horizon slice
- amplified accountant-column construction on top of the exact factor column

Source: BIFR (Kalinin et al., 2026) for the `γ`-indexed inverse family and its
low-bandwidth interpolation between DP-`λ`CGD-style and BISR-style behavior.

Claim-type notes:
- inverse/factor coefficient builders are implementation-contract surfaces
- structural-stability checks are runtime/accountant safety checks, not paper
  theorems
- amplified BNB helpers build accountant-side bridge objects rather than full
  amplified privacy guarantees
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from opacus.accountants.analysis.bnb import build_bnb_toeplitz_c_matrix_and_contract
from opacus.accountants.analysis.toeplitz_family import (
    InverseSideToeplitzFamily,
    ToeplitzMechanismFamily,
)

_BIFR_EXACT_FACTOR_STRUCTURAL_RADIUS_TOL = 1e-9
_BIFR_EXACT_FACTOR_MAX_ABS_COEFF_CAP = 1e6


def validate_bifr_frac(frac: float) -> float:
    """Validate the BIFR interpolation parameter `γ` on the closed interval `[0, 1]`."""
    resolved = float(frac)
    if not math.isfinite(resolved):
        raise ValueError("BIFR frac must be finite")

    if not (0.0 <= resolved <= 1.0):
        raise ValueError("BIFR frac must satisfy 0 <= frac <= 1")

    return resolved


def _effective_sgd_alpha_beta(*, momentum: float, weight_decay: float) -> tuple[float, float]:
    """
    Resolve the effective `(alpha, beta)` workload pair used by BIFR generation.

    Here `alpha` is the effective decay term and `beta` is the momentum term.
    The current implementation uses `alpha = 1` when weight decay is disabled.
    """
    beta = float(momentum)
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)
    if not (0.0 <= beta < 1.0):
        raise ValueError("momentum must satisfy 0 <= momentum < 1")

    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            "weight_decay must satisfy 0 < weight_decay <= 1 (or be exactly 0 to represent no weight decay)"
        )

    if beta > alpha:
        raise ValueError("BIFR inverse-side generation requires momentum <= effective weight decay")

    return alpha, beta


def generate_bifr_series_coeffs(*, bands: int, frac: float) -> list[float]:
    """
    Generate the inverse-side generalized-binomial BIFR series coefficients.

    This matches the theorem-side inverse-series recurrence

        a_0 = 1,
        a_{k+1} = ((k - theta) / (k + 1)) a_k.
    """
    resolved_frac = validate_bifr_frac(frac)
    if int(bands) < 1:
        raise ValueError("bands must be >= 1")

    coeffs = [0.0] * int(bands)
    coeffs[0] = 1.0
    for idx in range(int(bands) - 1):
        # Generalized-binomial recurrence for the inverse-side `γ` series.
        coeffs[idx + 1] = coeffs[idx] * ((float(idx) - resolved_frac) / float(idx + 1))

    return coeffs


def generate_bifr_inverse_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
    frac: float = 0.5,
) -> list[float]:
    """
    Generate BIFR inverse-side Toeplitz coefficients from the SGD workload.

    The returned list is the band-truncated first column of `C^{-1}`.
    """
    alpha, beta = _effective_sgd_alpha_beta(
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    series_coeffs = generate_bifr_series_coeffs(
        bands=int(bands),
        frac=float(frac),
    )
    coeffs = [0.0] * int(bands)
    for j in range(int(bands)):
        total = 0.0
        for i in range(j + 1):
            # Convolve the two generalized-binomial sides induced by the
            # effective SGD workload `(alpha, beta)`.
            total += (
                series_coeffs[j - i]
                * (alpha ** (j - i))
                * series_coeffs[i]
                * (beta ** i)
            )

        coeffs[j] = total

    if not all(math.isfinite(c) for c in coeffs):
        raise ValueError("derived BIFR inverse coefficients must be finite")

    return coeffs


def bifr_exact_factor_recurrence_spectral_radius_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
) -> float:
    """
    Spectral radius of the exact first-column recurrence induced by `C^{-1}`.

    If `C^{-1}` has first column `[c_0, c_1, ..., c_m]`, then the factor-side
    first column satisfies

        c_0 x_t + c_1 x_{t-1} + ... + c_m x_{t-m} = 0.

    The companion-matrix spectral radius controls asymptotic growth of the
    exact finite-horizon factor column.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    leading = float(coeff_list[0])
    if abs(leading) <= 0.0:
        raise ValueError("leading inverse coefficient must be nonzero")

    order = len(coeff_list) - 1
    if order <= 0:
        return 0.0

    normalized_tail = [float(c) / leading for c in coeff_list[1:]]
    # Companion-matrix reduction for the exact factor-side recurrence implied
    # by the inverse-side first column.
    companion = torch.zeros((order, order), dtype=torch.float64)
    companion[0, :] = -torch.tensor(normalized_tail, dtype=torch.float64)
    if order > 1:
        companion[1:, :-1] = torch.eye(order - 1, dtype=torch.float64)

    eigvals = torch.linalg.eigvals(companion)
    radius = float(torch.max(torch.abs(eigvals)).item())
    if not math.isfinite(radius):
        raise ValueError("exact BIFR recurrence spectral radius must be finite")

    return radius


def validate_bifr_exact_factor_structural_stability(
    *,
    inverse_coeffs: Iterable[float],
    factor_coeffs: Iterable[float],
    steps: int,
    radius_tol: float = _BIFR_EXACT_FACTOR_STRUCTURAL_RADIUS_TOL,
    max_abs_coeff_cap: float = _BIFR_EXACT_FACTOR_MAX_ABS_COEFF_CAP,
) -> float:
    """
    Reject exact finite-horizon BIFR slices that are structurally unstable for
    the requested horizon.

    We do not reject every unstable recurrence unconditionally: low-horizon
    exact slices can still be useful and theorem-facing tests rely on that.
    The maintained runtime/accountant concern is the long-horizon regime where
    the exact factor-side first column explodes to enormous magnitude.
    """
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")

    factor_list = [float(c) for c in factor_coeffs]
    if len(factor_list) == 0:
        raise ValueError("factor_coeffs must be non-empty")

    if not all(math.isfinite(c) for c in factor_list):
        raise ValueError("factor_coeffs must be finite")

    radius = bifr_exact_factor_recurrence_spectral_radius_from_inverse_coeffs(
        coeffs=inverse_coeffs
    )
    max_abs_coeff = max(abs(float(c)) for c in factor_list)
    if radius > 1.0 + float(radius_tol) and max_abs_coeff > float(max_abs_coeff_cap):
        raise ValueError(
            "unstable_exact_bifr_slice: "
            f"recurrence spectral radius {radius:.6g} exceeds 1 and "
            f"max |C[:,0]| coefficient {max_abs_coeff:.6g} exceeds the "
            f"maintained cap {float(max_abs_coeff_cap):.6g} at horizon {int(steps)}"
        )

    return radius


def derive_bifr_factor_coeffs_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Derive exact finite-horizon factor-side BIFR coefficients from inverse-side coefficients.
    """
    inverse_coeffs = [float(c) for c in coeffs]
    factor_coeffs = InverseSideToeplitzFamily(
        inv_coeffs=inverse_coeffs,
        steps=int(steps),
        source="bifr",
    ).factor_coeffs()
    validate_bifr_exact_factor_structural_stability(
        inverse_coeffs=inverse_coeffs,
        factor_coeffs=factor_coeffs,
        steps=int(steps),
    )

    return factor_coeffs


def build_bifr_exact_factor_family_from_inverse_coeffs(
    *,
    coeffs: Sequence[float],
    steps: int,
) -> ToeplitzMechanismFamily:
    """Build the exact finite-horizon factor-side Toeplitz family from inverse coeffs."""
    return ToeplitzMechanismFamily(
        coeffs=derive_bifr_factor_coeffs_from_inverse_coeffs(
            coeffs=coeffs,
            steps=steps,
        ),
        steps=int(steps),
        source="bifr",
    )


def build_bifr_exact_factor_family_from_sgd_workload(
    *,
    bands: int,
    steps: int,
    momentum: float,
    weight_decay: float,
    frac: float = 0.5,
) -> ToeplitzMechanismFamily:
    """Build the exact finite-horizon factor-side Toeplitz family from workload inputs."""
    inv_coeffs = generate_bifr_inverse_coeffs_from_sgd_workload(
        bands=int(bands),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        frac=float(frac),
    )
    return build_bifr_exact_factor_family_from_inverse_coeffs(
        coeffs=inv_coeffs,
        steps=int(steps),
    )


def compute_bifr_fixed_batch_sensitivity_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
    allow_disjoint_fallback: bool = False,
) -> float:
    """Compute fixed-batch BIFR sensitivity from explicit inverse-side coefficients."""
    return build_bifr_exact_factor_family_from_inverse_coeffs(
        coeffs=[float(c) for c in coeffs],
        steps=int(steps),
    ).fixed_batch_sensitivity(
        max_participations=max_participations,
        min_separation=min_separation,
        allow_disjoint_fallback=allow_disjoint_fallback,
    )


def compute_bifr_fixed_batch_sensitivity_from_sgd_workload(
    *,
    bands: int,
    steps: int,
    max_participations: int,
    min_separation: int,
    momentum: float,
    weight_decay: float,
    frac: float = 0.5,
    allow_disjoint_fallback: bool = False,
) -> float:
    """Compute fixed-batch BIFR sensitivity directly from workload-shaped inputs."""
    return build_bifr_exact_factor_family_from_sgd_workload(
        bands=bands,
        steps=steps,
        momentum=momentum,
        weight_decay=weight_decay,
        frac=frac,
    ).fixed_batch_sensitivity(
        max_participations=max_participations,
        min_separation=min_separation,
        allow_disjoint_fallback=allow_disjoint_fallback,
    )


def resolve_bifr_exact_factor_coeffs_for_accounting(
    *,
    coeffs: Sequence[float] | None = None,
    inverse_coeffs: Sequence[float] | None = None,
    steps: int | None = None,
    bands: int | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    frac: float = 0.5,
) -> tuple[list[float], str]:
    """
    Resolve exact finite-horizon factor-side BIFR coefficients for accountant use.

    Accepted sources:
    - explicit exact factor coefficients
    - explicit inverse-side coefficients plus horizon
    - exact workload parameters `(bands, steps, momentum, weight_decay, frac)`
    """
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        resolved = [float(c) for c in coeffs]
        if not all(math.isfinite(c) for c in resolved):
            raise ValueError("explicit BIFR factor coefficients must be finite")

        return resolved, "explicit_exact_factor_c_col"

    if isinstance(inverse_coeffs, (list, tuple)) and len(inverse_coeffs) > 0:
        if steps is None:
            raise ValueError("explicit BIFR inverse coefficients require an exact finite horizon")

        return (
            derive_bifr_factor_coeffs_from_inverse_coeffs(
                coeffs=[float(c) for c in inverse_coeffs],
                steps=int(steps),
            ),
            "exact_factor_c_col_from_explicit_inverse",
        )

    if bands is None or steps is None or momentum is None or weight_decay is None:
        raise ValueError(
            "BIFR accountant coefficient resolution requires explicit exact factor coefficients, "
            "explicit inverse coefficients with `steps`, or exact workload parameters "
            "`(bands, steps, momentum, weight_decay)`"
        )

    inv_coeffs = generate_bifr_inverse_coeffs_from_sgd_workload(
        bands=int(bands),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        frac=float(validate_bifr_frac(frac)),
    )
    return (
        derive_bifr_factor_coeffs_from_inverse_coeffs(
            coeffs=inv_coeffs,
            steps=int(steps),
        ),
        "exact_factor_c_col_from_workload_inverse",
    )


def derive_bifr_amplified_accountant_coeffs_from_factor_coeffs(
    *,
    coeffs: Iterable[float],
) -> list[float]:
    """
    Derive the non-negative accountant-side first column for amplified BIFR.
    """
    # The current amplified BNB bridge consumes a non-negative first column, so
    # BIFR exports the absolute factor-side visible-horizon column here.
    accountant_coeffs = [abs(float(c)) for c in coeffs]
    if not accountant_coeffs:
        raise ValueError("derived amplified BIFR accountant coefficients must be non-empty")

    if not all(math.isfinite(c) for c in accountant_coeffs):
        raise ValueError("derived amplified BIFR accountant coefficients must be finite")

    if accountant_coeffs[0] <= 0.0:
        raise ValueError("derived amplified BIFR accountant coefficients must be positive")

    return accountant_coeffs


def build_bifr_amplified_bnb_inputs_from_factor_coeffs(
    *,
    coeffs: Sequence[float],
    bands: int,
    horizon: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """
    Build the amplified BIFR BNB bridge inputs from exact factor coefficients.

    Returns:
        A dictionary containing the non-negative accountant column, the explicit
        Toeplitz `C` matrix, its contract payload, and the resolved visible
        horizon metadata.
    """
    accountant_coeffs = derive_bifr_amplified_accountant_coeffs_from_factor_coeffs(
        coeffs=coeffs,
    )
    c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=accountant_coeffs,
        bands=int(bands),
        horizon=int(horizon),
        dtype=dtype,
        device=device,
        atol=float(atol),
    )

    return {
        "bnb_accountant_coeffs": [float(c) for c in accountant_coeffs],
        "bnb_accountant_coeffs_source": "abs_exact_factor_c_col",
        "bnb_c_matrix": c_matrix,
        "bnb_c_matrix_contract": c_matrix_contract,
        "bnb_bands": int(bands),
        "bnb_horizon": int(horizon),
    }
