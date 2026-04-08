from __future__ import annotations

"""
BandMF analysis helpers.

The production BandMF contract follows the paper/JAX fixed-batch object:
- optimize Toeplitz-style coefficients for the prefix workload;
- materialize the finite-horizon banded lower-triangular matrix;
- normalize each nonzero column independently.

This matches `jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded`
and is the canonical fixed-batch BandMF sensitivity surface.

NB: This factorization is called "dense" and apparently the standard one is
    unnormalized. We might need to change to unnormalized.
"""

import math
from typing import Iterable

import numpy as np
import torch

from opacus.accountants.analysis.toeplitz_family import ToeplitzMechanismFamily


def _validate_bands(*, bands: int) -> None:
    if bands < 1:
        raise ValueError("bands must be >= 1")


def _validate_steps(*, steps: int, bands: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps < bands:
        raise ValueError(f"steps must be >= bands; got steps={steps}, bands={bands}")


def _as_float_array(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")

    return array


def normalize_bandmf_strategy_coeffs(
    *,
    coeffs: Iterable[float],
) -> list[float]:
    """
    Normalize Toeplitz coefficients to unit single-participation sensitivity.

    For lower-triangular Toeplitz strategies, the single-participation
    sensitivity is the L2 norm of the first column, so normalization is just
    `theta / ||theta||_2`.
    """
    coeff_array = _as_float_array(coeffs, name="coeffs")
    norm = float(np.linalg.norm(coeff_array))
    if norm <= 0.0:
        raise ValueError("coeffs must have positive L2 norm")

    normalized = coeff_array / norm
    if normalized[0] < 0.0:
        normalized = -normalized

    return [float(x) for x in normalized]


def materialize_bandmf_toeplitz_matrix(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> np.ndarray:
    """
    Materialize the globally normalized lower-triangular Toeplitz BandMF matrix.
    """
    coeff_array = _as_float_array(coeffs, name="coeffs")
    _validate_steps(steps=int(steps), bands=int(min(len(coeff_array), steps)))
    normalized = np.asarray(
        normalize_bandmf_strategy_coeffs(coeffs=coeff_array.tolist()), dtype=np.float64
    )

    n = int(steps)
    matrix = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        max_lag = min(normalized.size, n - j)
        matrix[j : j + max_lag, j] = normalized[:max_lag]

    return matrix


def build_bandmf_toeplitz_family_from_runtime_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> ToeplitzMechanismFamily:
    """
    Build the shared Toeplitz-family view of BandMF's globally normalized strategy.

    This does not replace BandMF's paper/JAX fixed-batch object, which remains
    column-normalized and therefore family-local. It only exposes the shared
    first-column family where the contracts truly coincide.
    """
    normalized = normalize_bandmf_strategy_coeffs(coeffs=coeffs)
    return ToeplitzMechanismFamily(
        coeffs=normalized,
        steps=int(steps),
        source="bandmf",
    )


def materialize_column_normalized_banded_bandmf_matrix(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> np.ndarray:
    """
    Materialize the paper/JAX BandMF matrix for a finite horizon.

    The input Toeplitz-style coefficients are first globally normalized, then
    the lower-triangular banded matrix is materialized, and finally each
    nonzero column is normalized independently.
    """
    matrix = materialize_bandmf_toeplitz_matrix(coeffs=coeffs, steps=steps)
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix.copy()
    nonzero = column_norms > 0.0
    normalized[:, nonzero] /= column_norms[nonzero]

    return normalized


def derive_bandmf_amplified_accountant_coeffs_from_runtime_coeffs(
    *,
    coeffs: Iterable[float],
) -> list[float]:
    """
    Derive the current amplified BandMF accountant-side first column.

    The current Opacus balls-in-bins ownership model for BandMF uses the
    runtime Toeplitz first column directly as the accountant-side `c_col`.
    This is an ownership/provenance helper, not a paper-parity claim.
    """
    coeff_array = _as_float_array(coeffs, name="coeffs")
    if float(coeff_array[0]) <= 0.0:
        raise ValueError("coeffs[0] must be > 0")

    return [float(x) for x in coeff_array]


def compute_bandmf_max_column_norm_from_column_normalized_matrix(
    *,
    matrix: np.ndarray,
) -> float:
    """Return the maximum column norm of a materialized BandMF matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    return float(np.max(np.linalg.norm(matrix, axis=0)))


def _max_participation_for_linear_fn(
    values: np.ndarray,
    *,
    min_separation: int,
    max_participations: int,
) -> float:
    """Dynamic program for max min-sep participation on a 1-D objective."""
    n = int(values.size)
    if n == 0:
        return 0.0

    dp = np.zeros((int(max_participations) + 1, n + int(min_separation) + 1), dtype=np.float64)
    for k in range(1, int(max_participations) + 1):
        for i in range(n - 1, -1, -1):
            take = float(values[i]) + dp[k - 1, i + int(min_separation)]
            skip = dp[k, i + 1]
            dp[k, i] = max(take, skip)

    return float(dp[int(max_participations), 0])


def compute_bandmf_fixed_batch_sensitivity_from_column_normalized_matrix(
    *,
    matrix: np.ndarray,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compute a fixed-batch BandMF sensitivity bound on the column-normalized matrix.

    When `min_separation >= bands`, column-normalized banded BandMF columns are
    orthogonal under the participation contract, so sensitivity is exactly
    `sqrt(k_eff)`. Otherwise compute the generic absolute-Gram upper bound on
    the same column-normalized matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    if max_participations < 1:
        raise ValueError("max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")

    n = int(matrix.shape[0])
    if n < 1:
        raise ValueError("matrix must be non-empty")

    support = np.abs(matrix) > 0.0
    if not np.any(support):
        raise ValueError("matrix must have at least one nonzero entry")

    row_idx, col_idx = np.nonzero(support)
    bands = int(np.max(row_idx - col_idx)) + 1
    k_eff = min(int(max_participations), (n - 1) // int(min_separation) + 1)
    if int(min_separation) >= bands:
        return math.sqrt(float(k_eff))

    gram_abs = np.abs(matrix.T @ matrix)
    row_max = np.asarray(
        [
            _max_participation_for_linear_fn(
                gram_abs[i],
                min_separation=int(min_separation),
                max_participations=int(max_participations),
            )
            for i in range(n)
        ],
        dtype=np.float64,
    )
    sens_sq = _max_participation_for_linear_fn(
        row_max,
        min_separation=int(min_separation),
        max_participations=int(max_participations),
    )

    return math.sqrt(float(sens_sq))


def generate_bandmf_initial_strategy_coeffs(
    *,
    bands: int,
) -> list[float]:
    """
    Deterministic BandMF initialization.

    This uses the lower-triangular Toeplitz coefficients from the optimal
    max-error prefix-workload factorization and truncates them to the requested
    band width before unit-sensitivity normalization.
    """
    _validate_bands(bands=int(bands))
    k = np.arange(int(bands), dtype=np.float64)
    coeffs = np.ones(int(bands), dtype=np.float64)
    for idx in range(1, int(bands)):
        coeffs[idx] = coeffs[idx - 1] * ((2.0 * idx - 1.0) / (2.0 * idx))

    return normalize_bandmf_strategy_coeffs(coeffs=coeffs.tolist())


def compute_bandmf_prefix_workload_per_query_error_from_strategy(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Compute prefix-workload per-query squared error for a Toeplitz strategy.

    For a lower-triangular Toeplitz strategy with coefficients `theta`, let
    `w = C(theta)^(-1) 1`. The prefix-workload per-query squared error is
    `cumsum(w_i^2)`.
    """
    coeff_array = _as_float_array(coeffs, name="coeffs")
    _validate_steps(steps=int(steps), bands=int(min(len(coeff_array), steps)))

    n = int(steps)
    reconciled = np.zeros(n, dtype=np.float64)
    reconciled[: min(n, coeff_array.size)] = coeff_array[:n]
    if reconciled[0] <= 0.0:
        raise ValueError("coeffs[0] must be > 0")

    w = np.zeros(n, dtype=np.float64)
    for t in range(n):
        acc = 1.0
        max_lag = min(t, reconciled.size - 1)
        for lag in range(1, max_lag + 1):
            acc -= reconciled[lag] * w[t - lag]

        w[t] = acc / reconciled[0]

    return [float(x) for x in np.cumsum(w * w)]


def compute_bandmf_prefix_workload_mean_error_from_strategy(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    Mean prefix-workload squared error for the Toeplitz strategy coefficients.
    """
    per_query = compute_bandmf_prefix_workload_per_query_error_from_strategy(
        coeffs=coeffs,
        steps=steps,
    )
    return float(np.mean(np.asarray(per_query, dtype=np.float64)))


def compute_bandmf_objective_from_strategy(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    BandMF optimization objective in paper terms.

    This is `mean_error(theta) * ||theta||_2^2`, matching the prefix-workload
    objective with Toeplitz sensitivity folded in.
    """
    coeff_array = _as_float_array(coeffs, name="coeffs")
    mean_error = compute_bandmf_prefix_workload_mean_error_from_strategy(
        coeffs=coeff_array.tolist(),
        steps=int(steps),
    )
    return float(mean_error * float(np.dot(coeff_array, coeff_array)))


def optimize_bandmf_strategy_coeffs(
    *,
    steps: int,
    bands: int,
    init_coeffs: Iterable[float] | None = None,
    max_optimizer_steps: int = 250,
) -> list[float]:
    """
    Optimize the forward Toeplitz BandMF strategy for the prefix workload.

    The returned coefficients are normalized to unit L2 norm, which makes them
    the runtime/accounting-facing strategy coefficients.
    """
    _validate_bands(bands=int(bands))
    _validate_steps(steps=int(steps), bands=int(bands))
    if max_optimizer_steps < 1:
        raise ValueError("max_optimizer_steps must be >= 1")

    if init_coeffs is None:
        start = np.asarray(
            generate_bandmf_initial_strategy_coeffs(bands=int(bands)),
            dtype=np.float64,
        )
    else:
        start = _as_float_array(init_coeffs, name="init_coeffs")
        if start.size != int(bands):
            raise ValueError(
                f"init_coeffs must have length bands; got len={start.size}, bands={bands}"
            )

        start = np.asarray(
            normalize_bandmf_strategy_coeffs(coeffs=start.tolist()), dtype=np.float64
        )

    dtype = torch.float64
    params = torch.nn.Parameter(torch.tensor(start, dtype=dtype))
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=int(max_optimizer_steps),
        line_search_fn="strong_wolfe",
    )

    def _loss(v: torch.Tensor) -> torch.Tensor:
        coeffs = v / (torch.linalg.norm(v) + torch.tensor(1e-24, dtype=v.dtype))
        if coeffs[0] < 0:
            coeffs = -coeffs

        penalty = (
            torch.relu(torch.tensor(1e-9, dtype=v.dtype) - coeffs[0])
            * torch.tensor(1e9, dtype=v.dtype)
        )

        rhs_val = torch.ones((), dtype=v.dtype)
        w_vals: list[torch.Tensor] = []
        for t in range(int(steps)):
            acc = rhs_val
            max_lag = min(t, coeffs.numel() - 1)

            for lag in range(1, max_lag + 1):
                acc = acc - coeffs[lag] * w_vals[t - lag]

            w_vals.append(acc / coeffs[0])

        w = torch.stack(w_vals)
        mean_error = torch.cumsum(w * w, dim=0).mean()

        return mean_error * torch.sum(coeffs * coeffs) + penalty

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(params)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        coeffs = params.detach() / (torch.linalg.norm(params.detach()) + 1e-24)
        if coeffs[0] < 0:
            coeffs = -coeffs

    return [float(x) for x in coeffs.tolist()]


def generate_bandmf_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
    steps: int,
    max_optimizer_steps: int = 250,
) -> list[float]:
    """
    Derive paper-faithful BandMF Toeplitz coefficients for an SGD training run.

    BandMF is optimized for the prefix workload. The current production BandMF
    contract depends on the training horizon `steps`; `momentum` and
    `weight_decay` are accepted for API compatibility with other MF helpers but
    do not change the optimized forward Toeplitz strategy.
    """
    del momentum, weight_decay
    return optimize_bandmf_strategy_coeffs(
        steps=int(steps),
        bands=int(bands),
        max_optimizer_steps=int(max_optimizer_steps),
    )

def compute_bandmf_mf_sensitivity_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compute fixed-batch BandMF sensitivity on the paper/JAX column-normalized object.
    """
    matrix = materialize_column_normalized_banded_bandmf_matrix(
        coeffs=coeffs,
        steps=steps,
    )
    return compute_bandmf_fixed_batch_sensitivity_from_column_normalized_matrix(
        matrix=matrix,
        max_participations=max_participations,
        min_separation=min_separation,
    )
