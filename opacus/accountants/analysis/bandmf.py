from __future__ import annotations

"""
BandMF analysis helpers.

BandMF is the optimized forward banded Toeplitz method from
"Scaling up the Banded Matrix Factorization Mechanism for Differentially
Private ML" (McKenna, 2024).

The production BandMF contract is:
- optimize a banded lower-triangular Toeplitz strategy for the prefix workload;
- use the prefix-workload mean squared error objective from Proposition 3.1;
- normalize the resulting Toeplitz coefficients to unit single-participation
  sensitivity before using them in runtime/accounting.
"""

import math
from typing import Iterable

import numpy as np
import torch

from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload


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


def generate_legacy_bandmf_placeholder_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
) -> list[float]:
    """
    Legacy migration oracle for the historical `BandMF == BSR coeffs` contract.

    This is not the paper-faithful BandMF method. It exists only so tests can
    prove the refactor materially changed the old placeholder behavior.
    """
    return generate_bsr_coeffs_from_sgd_workload(
        bands=int(bands),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )


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
    Compute fixed-batch BandMF sensitivity from Toeplitz coefficients.

    Paper-faithful BandMF uses a column-normalized lower-triangular banded
    strategy. Under a `(k, b)` fixed-batch contract with `b >= bands`, the
    worst-case participations are orthogonal, so the sensitivity squared is
    just the number of actual participations.

    The legacy Toeplitz recurrence path is retained only as a fallback for
    non-paper configurations with `min_separation < bands`.
    """
    coeff_list = [float(c) for c in coeffs]
    if not coeff_list:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    if any(c < 0.0 for c in coeff_list):
        raise ValueError("coeffs must be nonnegative")

    if steps < 1:
        raise ValueError("steps must be >= 1")

    if max_participations < 1:
        raise ValueError("max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")

    k_eff = min(int(max_participations), (int(steps) - 1) // int(min_separation) + 1)

    coeff_norm = math.sqrt(sum(c * c for c in coeff_list))
    if not math.isfinite(coeff_norm) or coeff_norm <= 0.0:
        raise ValueError("coeffs must have positive L2 norm")

    # Column-normalized banded strategies have unit single-participation
    # sensitivity. When participations are separated by at least the band
    # width, cross terms vanish and sensitivity^2 equals the number of
    # participations.
    if int(min_separation) >= len(coeff_list):
        return math.sqrt(float(k_eff))

    for prev, cur in zip(coeff_list, coeff_list[1:]):
        if cur > prev + 1e-12:
            raise ValueError("coeffs must be non-increasing")

    padding = (int(min_separation) - int(steps)) % int(min_separation)
    padded = coeff_list + [0.0] * max(0, int(steps) - len(coeff_list) + padding)

    vector: list[float] = [0.0] * len(padded)
    for block_start in range(0, len(padded), int(min_separation)):
        running = 0.0
        block_end = min(block_start + int(min_separation), len(padded))

        for idx in range(block_start, block_end):
            running += padded[idx]
            vector[idx] = running

    stride = int(min_separation) * int(k_eff)
    for idx in range(stride, len(vector)):
        vector[idx] -= vector[idx - stride]

    total_sq = sum(v * v for v in vector[: int(steps)])

    return math.sqrt(total_sq)
