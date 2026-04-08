from __future__ import annotations

import math

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
