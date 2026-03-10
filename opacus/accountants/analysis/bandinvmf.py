from __future__ import annotations

"""
BandInvMF analysis helpers.

BandInvMF is the optimized inverse-band path from "Back to Square Roots: An
Optimal Bound on The Matrix Factorization Eerror for Multiepoch Differentially
Private SGD" (Kalinin et al., 2025)


It keeps the same inverse-band lower triangular Toeplitz parameterization as
BISR, initializes those inverse coefficients from the analytic BISR
coefficients, and then numerically optimizes the separated-participation upper
bound for the corresponding SGD workload.

The first delivery here is analysis-only. It is intentionally separate from the
runtime `PrivacyEngine` mechanism surface.

Experiment-surface note:
- BISR is the analytic inverse-band path with closed-form coefficients.
- BandInvMF is the optimized inverse-band path initialized from those analytic
  coefficients.
- Future run scripts should treat the paper text and runtime implementation
  semantics as authoritative for parameter values; legacy scripts are not the
  source of truth.
"""

import math
from typing import Iterable

import numpy as np
from scipy import optimize

from opacus.accountants.analysis.bisr import (
    generate_bisr_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bsr import compute_bsr_mf_sensitivity_from_coeffs


def _validate_workload_params(*, bands: int, momentum: float, weight_decay: float) -> None:
    if bands < 1:
        raise ValueError("bands must be >= 1")

    if not (0.0 <= float(momentum) < 1.0):
        raise ValueError("momentum must satisfy 0 <= momentum < 1")

    if not (0.0 < float(weight_decay) <= 1.0):
        raise ValueError(
            "weight_decay must satisfy 0 < weight_decay <= 1 for BandInvMF "
            "coefficient generation (or be exactly 0 to represent no weight decay)"
        )


def _validate_optimization_contract(
    *,
    steps: int,
    max_participations: int,
    min_separation: int,
    optimizer_steps: int,
) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    if max_participations < 1:
        raise ValueError("max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")

    if optimizer_steps < 1:
        raise ValueError("optimizer_steps must be >= 1")


def _workload_coeffs(*, steps: int, momentum: float, weight_decay: float) -> np.ndarray:
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)
    beta = float(momentum)
    a = np.array([alpha**k for k in range(steps)], dtype=np.float64)
    b = np.array([beta**k for k in range(steps)], dtype=np.float64)

    return np.convolve(a, b)[:steps]


def _pad_coeffs(coeffs: Iterable[float], *, steps: int) -> np.ndarray:
    coeff_array = np.array([float(c) for c in coeffs], dtype=np.float64)
    if coeff_array.size == 0:
        raise ValueError("coeffs must be non-empty")

    if not np.all(np.isfinite(coeff_array)):
        raise ValueError("coeffs must be finite")

    out = np.zeros(steps, dtype=np.float64)
    out[: min(steps, coeff_array.size)] = coeff_array[:steps]

    return out


def _toeplitz_inverse_coeffs(inv_coeffs: np.ndarray) -> np.ndarray:
    n = int(inv_coeffs.size)
    coef = np.zeros(n, dtype=np.float64)
    coef[0] = 1.0 / inv_coeffs[0]
    for i in range(1, n):
        coef[i] = -np.dot(coef[:i], inv_coeffs[i:0:-1]) / inv_coeffs[0]

    return coef


def derive_bandinvmf_runtime_coeffs_from_inv_coeffs(
    *,
    inv_coeffs: Iterable[float],
) -> list[float]:
    """
    Derive runtime Toeplitz noising coefficients from inverse-band coefficients.

    BandInvMF is optimized in the inverse-band parameterization, while the
    Opacus correlated runtime consumes the Toeplitz noising coefficients of the
    realized strategy. This helper makes the production mapping explicit.
    """
    coeff_array = np.array([float(c) for c in inv_coeffs], dtype=np.float64)
    if coeff_array.size == 0:
        raise ValueError("inv_coeffs must be non-empty")
    if not np.all(np.isfinite(coeff_array)):
        raise ValueError("inv_coeffs must be finite")
    if float(coeff_array[0]) <= 0.0:
        raise ValueError("inv_coeffs[0] must be > 0")

    return [float(x) for x in _toeplitz_inverse_coeffs(coeff_array)]


def derive_bandinvmf_inv_coeffs_from_runtime_coeffs(
    *,
    coeffs: Iterable[float],
) -> list[float]:
    """
    Recover inverse-band coefficients from runtime Toeplitz noising coefficients.

    The lower-triangular Toeplitz inverse is again Toeplitz, so the same
    inversion routine gives the inverse-band view needed for deterministic
    checkpoint resume and runtime-state canonicalization.
    """
    coeff_array = np.array([float(c) for c in coeffs], dtype=np.float64)
    if coeff_array.size == 0:
        raise ValueError("coeffs must be non-empty")
    if not np.all(np.isfinite(coeff_array)):
        raise ValueError("coeffs must be finite")
    if float(coeff_array[0]) <= 0.0:
        raise ValueError("coeffs[0] must be > 0")

    return [float(x) for x in _toeplitz_inverse_coeffs(coeff_array)]


def _toeplitz_per_query_error_from_noising_coeffs(
    *,
    noising_coeffs: np.ndarray,
    workload_coeffs: np.ndarray,
) -> np.ndarray:
    b_coeffs = np.convolve(workload_coeffs, noising_coeffs)[: noising_coeffs.size]
    return np.cumsum(b_coeffs * b_coeffs)


def _toeplitz_mean_error_from_noising_coeffs(
    *,
    noising_coeffs: np.ndarray,
    workload_coeffs: np.ndarray,
) -> float:
    return float(
        np.mean(
            _toeplitz_per_query_error_from_noising_coeffs(
                noising_coeffs=noising_coeffs,
                workload_coeffs=workload_coeffs,
            )
        )
    )


def _decreasing_envelope(coeffs: np.ndarray) -> np.ndarray:
    out = np.array(coeffs, dtype=np.float64, copy=True)
    running = -math.inf
    for idx in range(out.size - 1, -1, -1):
        running = max(running, float(out[idx]))
        out[idx] = running

    return out


def generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
) -> list[float]:
    """
    Initialize BandInvMF inverse-band coefficients from analytic BISR.

    The paper initializes the optimized inverse-band path from the analytic BISR
    coefficients. This helper makes that initialization explicit.
    """
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)
    beta = float(momentum)

    _validate_workload_params(bands=bands, momentum=beta, weight_decay=alpha)
    return generate_bisr_coeffs_from_sgd_workload(
        bands=bands,
        momentum=beta,
        weight_decay=alpha,
    )


def compute_bandinvmf_objective_from_inv_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
    momentum: float,
    weight_decay: float,
) -> float:
    """
    Compute the BandInvMF optimization objective for inverse-band coefficients.

    The objective is the product of:
    - the mean squared workload error induced by the inverse-band noising path,
    - and the separated-participation sensitivity upper bound of the implied
      strategy Toeplitz factor.

    For the sensitivity term, the implied strategy coefficients are mapped to
    their decreasing nonnegative envelope before applying the closed-form
    separated-participation upper bound. This matches the paper's monotone
    upper-bound justification for the optimized inverse-band path.
    """
    inv_coeff_list = [float(c) for c in inv_coeffs]
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)
    beta = float(momentum)

    _validate_workload_params(bands=len(inv_coeff_list), momentum=beta, weight_decay=alpha)
    _validate_optimization_contract(
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
        optimizer_steps=1,
    )

    inv = _pad_coeffs(inv_coeff_list, steps=steps)
    if inv[0] <= 0.0:
        raise ValueError("inv_coeffs[0] must be > 0")

    workload = _workload_coeffs(
        steps=steps,
        momentum=beta,
        weight_decay=alpha,
    )
    mean_error = _toeplitz_mean_error_from_noising_coeffs(
        noising_coeffs=inv,
        workload_coeffs=workload,
    )

    strategy = _toeplitz_inverse_coeffs(inv)
    strategy_envelope = _decreasing_envelope(np.maximum(strategy, 0.0))
    sensitivity = compute_bsr_mf_sensitivity_from_coeffs(
        coeffs=strategy_envelope.tolist(),
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
    )

    return float(mean_error * (sensitivity**2))


def optimize_bandinvmf_inv_coeffs_for_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
    steps: int,
    max_participations: int,
    min_separation: int,
    optimizer_steps: int = 20,
) -> list[float]:
    """
    Optimize BandInvMF inverse-band coefficients for an SGD workload.

    The optimization is deterministic under fixed inputs:
    - initialize from analytic BISR,
    - keep the main diagonal coefficient fixed to 1,
    - optimize the remaining inverse-band coefficients for at most
      `optimizer_steps` Powell iterations.
    """
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)
    beta = float(momentum)

    _validate_workload_params(bands=bands, momentum=beta, weight_decay=alpha)
    _validate_optimization_contract(
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
        optimizer_steps=optimizer_steps,
    )

    init = np.array(
        generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
            bands=bands,
            momentum=beta,
            weight_decay=alpha,
        ),
        dtype=np.float64,
    )

    if bands == 1:
        return [1.0]

    x0 = init[1:].copy()

    def loss(x: np.ndarray) -> float:
        inv = np.concatenate(([1.0], np.asarray(x, dtype=np.float64)))
        return compute_bandinvmf_objective_from_inv_coeffs(
            inv_coeffs=inv.tolist(),
            steps=steps,
            max_participations=max_participations,
            min_separation=min_separation,
            momentum=beta,
            weight_decay=alpha,
        )

    result = optimize.minimize(
        loss,
        x0=x0,
        method="Powell",
        options={"maxiter": int(optimizer_steps), "disp": False},
    )
    opt = np.concatenate(([1.0], np.asarray(result.x, dtype=np.float64)))

    return [float(x) for x in opt]
