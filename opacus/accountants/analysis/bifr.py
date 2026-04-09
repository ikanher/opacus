from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from opacus.accountants.analysis.bnb import build_bnb_toeplitz_c_matrix_and_contract
from opacus.accountants.analysis.toeplitz_family import ToeplitzMechanismFamily


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
        raise ValueError("BIFR analytic coefficient generation requires momentum <= effective weight decay")

    return alpha, beta


def generate_bifr_series_coeffs(*, bands: int, frac: float) -> list[float]:
    resolved_frac = validate_bifr_frac(frac)
    if int(bands) < 1:
        raise ValueError("bands must be >= 1")

    coeffs = [0.0] * int(bands)
    coeffs[0] = 1.0
    for idx in range(int(bands) - 1):
        coeffs[idx + 1] = coeffs[idx] * ((float(idx) + resolved_frac) / float(idx + 1))

    return coeffs


def generate_bifr_factor_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
    frac: float = 0.5,
) -> list[float]:
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

    return coeffs


def build_bifr_analytic_factor_family(
    *,
    bands: int,
    steps: int,
    momentum: float,
    weight_decay: float,
    frac: float = 0.5,
) -> ToeplitzMechanismFamily:
    return ToeplitzMechanismFamily(
        coeffs=generate_bifr_factor_coeffs_from_sgd_workload(
            bands=bands,
            momentum=momentum,
            weight_decay=weight_decay,
            frac=frac,
        ),
        steps=int(steps),
        source="bifr",
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
    return build_bifr_analytic_factor_family(
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


def resolve_bifr_factor_coeffs_for_accounting(
    *,
    coeffs: Sequence[float] | None = None,
    bands: int | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    frac: float = 0.5,
) -> tuple[list[float], str]:
    """
    Resolve factor-side BIFR coefficients for accountant use.

    Accepted sources:
    - explicit factor coefficients already carried by runtime/public state
    - analytic BIFR workload parameters `(bands, momentum, weight_decay, frac)`

    The returned source string is machine-readable accountant metadata.
    """
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        resolved = [float(c) for c in coeffs]
        if not all(math.isfinite(c) for c in resolved):
            raise ValueError("explicit BIFR factor coefficients must be finite")
        return resolved, "explicit_factor_c_col"

    if bands is None or momentum is None or weight_decay is None:
        raise ValueError(
            "BIFR accountant coefficient resolution requires either explicit factor coefficients "
            "or analytic workload parameters `(bands, momentum, weight_decay)`"
        )

    return (
        generate_bifr_factor_coeffs_from_sgd_workload(
            bands=int(bands),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            frac=float(validate_bifr_frac(frac)),
        ),
        "analytic_factor_c_col",
    )


def derive_bifr_amplified_accountant_coeffs_from_factor_coeffs(
    *,
    coeffs: Iterable[float],
) -> list[float]:
    """
    Derive the non-negative accountant-side first column for amplified BIFR.

    BIFR is already parameterized by factor-side coefficients for `C`, so the
    amplified accountant object is the finite-horizon first column of `|C|`.
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
    """
    Build accountant-side BNB inputs from factor-side BIFR coefficients.

    This packages:
    - non-negative accountant coefficients
    - Toeplitz `C` matrix materialization
    - BNB matrix contract metadata
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
        "bnb_accountant_coeffs_source": "abs_factor_c_col",
        "bnb_c_matrix": c_matrix,
        "bnb_c_matrix_contract": c_matrix_contract,
        "bnb_bands": int(bands),
        "bnb_horizon": int(horizon),
    }
