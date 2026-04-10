from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from opacus.accountants.analysis.bnb import build_bnb_toeplitz_c_matrix_and_contract
from opacus.accountants.analysis.toeplitz_family import (
    InverseSideToeplitzFamily,
    ToeplitzMechanismFamily,
)


def validate_bifr_frac(frac: float) -> float:
    resolved = float(frac)
    if not math.isfinite(resolved):
        raise ValueError("BIFR frac must be finite")

    if not (0.0 <= resolved <= 1.0):
        raise ValueError("BIFR frac must satisfy 0 <= frac <= 1")

    return resolved


def _effective_sgd_alpha_beta(*, momentum: float, weight_decay: float) -> tuple[float, float]:
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


def derive_bifr_factor_coeffs_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Derive exact finite-horizon factor-side BIFR coefficients from inverse-side coefficients.
    """
    return InverseSideToeplitzFamily(
        inv_coeffs=[float(c) for c in coeffs],
        steps=int(steps),
        source="bifr",
    ).factor_coeffs()


def build_bifr_exact_factor_family_from_inverse_coeffs(
    *,
    coeffs: Sequence[float],
    steps: int,
) -> ToeplitzMechanismFamily:
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
