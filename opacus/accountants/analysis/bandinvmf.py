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
import warnings
from typing import Iterable

import numpy as np
import torch

from opacus.accountants.analysis.bisr import generate_bisr_coeffs_from_sgd_workload
from opacus.accountants.analysis.bsr import compute_bsr_mf_sensitivity_from_coeffs
from opacus.accountants.analysis.toeplitz_family import (
    InverseSideToeplitzFamily,
    ToeplitzMechanismFamily,
)


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


def derive_bandinvmf_factor_coeffs_from_inv_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Derive finite-horizon factor-side Toeplitz coefficients from BandInvMF inverse coefficients.

    BandInvMF exposes inverse-band coefficients of the paper-facing lower
    triangular Toeplitz object ``C^{-1}``, while the fixed-batch paper
    sensitivity is written in terms of the factor-side object ``C``. This
    helper keeps the BandInvMF namespace explicit while reusing the same
    finite-horizon inverse-to-factor derivation pattern as BISR.
    """
    return InverseSideToeplitzFamily(
        inv_coeffs=[float(c) for c in inv_coeffs],
        steps=int(steps),
        source="bandinvmf",
    ).factor_coeffs()


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


def compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Evaluate the fixed-batch BandInvMF paper sensitivity from inverse-band coefficients.

    BandInvMF optimization and runtime state are expressed using inverse-band
    coefficients, but the paper-facing non-amplified calibration uses
    ``sens_{k,b}(C)`` on the implied factor-side object ``C``. This helper
    derives the finite-horizon factor-side coefficients and evaluates the
    fixed-batch separated-participation sensitivity on that factor-side view.
    """
    factor_coeffs = derive_bandinvmf_factor_coeffs_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=steps,
    )
    factor_envelope = _decreasing_envelope(np.maximum(np.asarray(factor_coeffs, dtype=np.float64), 0.0))
    return ToeplitzMechanismFamily(
        coeffs=factor_envelope.tolist(),
        steps=int(steps),
        source="bandinvmf",
    ).fixed_batch_sensitivity(
        max_participations=max_participations,
        min_separation=min_separation,
    )


def derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Derive a non-negative accountant-side first column for amplified BandInvMF.

    Runtime BandInvMF is parameterized by inverse-band coefficients and executed
    via the finite Toeplitz noising operator, but the amplified balls-in-bins
    accountant consumes a non-negative first column ``c_col`` over the full
    finite horizon. We therefore derive the factor-side finite-horizon
    coefficients for ``C`` and return the first column of ``|C|``.
    """
    factor_coeffs = derive_bandinvmf_factor_coeffs_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=steps,
    )
    accountant_coeffs = [abs(float(c)) for c in factor_coeffs]
    if not all(math.isfinite(c) for c in accountant_coeffs):
        raise ValueError("derived amplified BandInvMF accountant coefficients must be finite")

    if accountant_coeffs[0] <= 0.0:
        raise ValueError("derived amplified BandInvMF accountant coefficients must be positive")

    return accountant_coeffs


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
      `optimizer_steps` L-BFGS iterations.
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
    init_obj = compute_bandinvmf_objective_from_inv_coeffs(
        inv_coeffs=init.tolist(),
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
        momentum=beta,
        weight_decay=alpha,
    )

    dtype = torch.float64
    params = torch.nn.Parameter(torch.tensor(x0, dtype=dtype))
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=int(optimizer_steps),
        line_search_fn="strong_wolfe",
    )
    workload = torch.tensor(
        _workload_coeffs(steps=steps, momentum=beta, weight_decay=alpha),
        dtype=dtype,
    )
    k_eff = min(max_participations, (steps - 1) // min_separation + 1)
    best_opt = init.copy()
    best_obj = init_obj

    def _loss(v: torch.Tensor) -> torch.Tensor:
        inv = torch.cat(
            (
                torch.ones(1, dtype=v.dtype, device=v.device),
                v,
            )
        )

        b_vals: list[torch.Tensor] = []
        for t in range(int(steps)):
            acc = torch.zeros((), dtype=v.dtype, device=v.device)
            max_lag = min(t, inv.numel() - 1)
            for lag in range(max_lag + 1):
                acc = acc + inv[lag] * workload[t - lag]
            b_vals.append(acc)

        b_vec = torch.stack(b_vals)
        mean_error = torch.cumsum(b_vec * b_vec, dim=0).mean()

        strategy_vals: list[torch.Tensor] = [
            torch.ones((), dtype=v.dtype, device=v.device)
        ]
        for i in range(1, int(steps)):
            acc = torch.zeros((), dtype=v.dtype, device=v.device)
            max_lag = min(i, inv.numel() - 1)
            for lag in range(1, max_lag + 1):
                acc = acc + inv[lag] * strategy_vals[i - lag]
            strategy_vals.append(-acc / inv[0])

        strategy = torch.stack(strategy_vals)
        strategy_nonnegative = torch.relu(strategy)
        strategy_envelope = torch.flip(
            torch.cummax(torch.flip(strategy_nonnegative, dims=[0]), dim=0).values,
            dims=[0],
        )

        total_sq = torch.zeros((), dtype=v.dtype, device=v.device)
        for i in range(int(steps)):
            j_max = min(k_eff - 1, i // min_separation)
            row_sum = torch.zeros((), dtype=v.dtype, device=v.device)
            for j in range(j_max + 1):
                row_sum = row_sum + strategy_envelope[i - j * min_separation]

            total_sq = total_sq + row_sum * row_sum

        loss = mean_error * total_sq
        if not torch.isfinite(loss):
            # Keep the current objective semantics but push LBFGS back toward
            # the finite region when the inverse-to-factor mapping explodes.
            return torch.tensor(1e100, dtype=v.dtype, device=v.device) + (
                torch.sum(v * v) * 0.0
            )

        return loss

    def closure() -> torch.Tensor:
        nonlocal best_obj, best_opt
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(params)
        with torch.no_grad():
            candidate = np.concatenate(
                ([1.0], np.asarray(params.detach().cpu().numpy(), dtype=np.float64))
            )
        if np.all(np.isfinite(candidate)):
            try:
                candidate_obj = compute_bandinvmf_objective_from_inv_coeffs(
                    inv_coeffs=candidate.tolist(),
                    steps=steps,
                    max_participations=max_participations,
                    min_separation=min_separation,
                    momentum=beta,
                    weight_decay=alpha,
                )
            except (FloatingPointError, OverflowError, ValueError):
                candidate_obj = math.inf
            if math.isfinite(candidate_obj) and candidate_obj < best_obj - 1e-12:
                # CIFAR-scale line search can leave the finite region after an
                # earlier valid improvement; keep that best finite candidate.
                best_obj = candidate_obj
                best_opt = candidate
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        opt = np.concatenate(
            ([1.0], np.asarray(params.detach().cpu().numpy(), dtype=np.float64))
        )
    try:
        opt_obj = compute_bandinvmf_objective_from_inv_coeffs(
            inv_coeffs=opt.tolist(),
            steps=steps,
            max_participations=max_participations,
            min_separation=min_separation,
            momentum=beta,
            weight_decay=alpha,
        )

    except (FloatingPointError, OverflowError, ValueError):
        opt_obj = math.inf

    if not np.all(np.isfinite(opt)) or not math.isfinite(opt_obj):
        if np.all(np.isfinite(best_opt)) and math.isfinite(best_obj):
            if best_obj < init_obj - 1e-12:
                warnings.warn(
                    "BandInvMF optimization produced a non-finite final candidate; "
                    "returning the best earlier finite improving candidate",
                    UserWarning,
                )
            else:
                warnings.warn(
                    "BandInvMF optimization produced a non-finite final candidate; "
                    "returning the finite initialization because no improving finite candidate was found",
                    UserWarning,
                )
            return [float(x) for x in best_opt]
        raise RuntimeError(
            "BandInvMF optimization produced a non-finite final candidate; "
            "and no finite candidate was found"
        )
    if opt_obj >= init_obj - 1e-12:
        if np.all(np.isfinite(best_opt)) and math.isfinite(best_obj):
            if best_obj < init_obj - 1e-12:
                warnings.warn(
                    "BandInvMF optimization did not finish with an improved final candidate; "
                    "returning the best earlier finite improving candidate",
                    UserWarning,
                )
            else:
                warnings.warn(
                    "BandInvMF optimization did not improve over initialization; "
                    "returning the finite initialization",
                    UserWarning,
                )
            return [float(x) for x in best_opt]
        raise RuntimeError(
            "BandInvMF optimization did not improve over initialization; "
            "and no finite candidate was found"
        )

    return [float(x) for x in opt]
