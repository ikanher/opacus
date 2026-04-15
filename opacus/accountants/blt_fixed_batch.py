from __future__ import annotations

"""
BLT fixed-batch calibration search.

This module owns BLT candidate generation and accountant-backed search for the
fixed-batch training contract. It does not own the runtime mechanism or the BLT
coefficient algebra; those live in `opacus.noise_mechanisms.blt` and
`opacus.accountants.analysis.blt`.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from opacus.accountants.analysis.blt import blt_pair_from_theta_pair
from opacus.accountants.utils import get_noise_multiplier
from opacus.mechanism_contracts import SamplingSemantics


def _as_descending_positive_vector(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if not np.all(arr > 0.0):
        raise ValueError(f"{name} must be > 0")
    arr = np.sort(arr)[::-1]
    if np.any(arr > 1.0):
        raise ValueError(f"{name} values must be <= 1")
    return arr


def _bsr_calibration_denominator(*, loss_reduction: str, logical_batch_size: int) -> float:
    if loss_reduction == "sum":
        return 1.0
    if logical_batch_size <= 0:
        raise ValueError(
            "BLT optimization requires logical_batch_size > 0 under loss_reduction='mean'"
        )
    return float(logical_batch_size)


def _resolve_steps_per_epoch(*, dataset_size: int, logical_batch_size: int) -> int:
    if dataset_size < 1:
        raise ValueError("dataset_size must be >= 1")
    if logical_batch_size < 1:
        raise ValueError("logical_batch_size must be >= 1")
    return int(math.ceil(float(dataset_size) / float(logical_batch_size)))


def _default_theta(buffers: int, *, theta_min: float, theta_max: float) -> np.ndarray:
    if buffers < 1:
        raise ValueError("buffers must be >= 1")
    if not (0.0 < theta_min <= theta_max <= 1.0):
        raise ValueError("require 0 < theta_min <= theta_max <= 1")
    if buffers == 1:
        return np.array([float(theta_max)], dtype=np.float64)
    return np.geomspace(float(theta_min), float(theta_max), int(buffers), dtype=np.float64)[
        ::-1
    ]


def _resolve_base_theta(
    *,
    buffers: int | None,
    theta: Sequence[float] | None,
    theta_min: float,
    theta_max: float,
) -> np.ndarray:
    if theta is None:
        if buffers is None:
            raise ValueError("either buffers or theta must be provided")
        return _default_theta(int(buffers), theta_min=theta_min, theta_max=theta_max)

    base_theta = _as_descending_positive_vector("theta", theta)
    if buffers is not None and int(buffers) != int(base_theta.size):
        raise ValueError("buffers must match len(theta) when both are provided")
    return base_theta


def _validated_scale_grid(name: str, values: Sequence[float]) -> list[float]:
    valid: list[float] = []
    for value in values:
        scale = float(value)
        if math.isfinite(scale) and scale > 0.0:
            valid.append(scale)
    if not valid:
        raise ValueError(f"{name} must contain at least one finite positive value")
    return valid


def _scaled_descending_candidate(values: np.ndarray, *, scale: float) -> np.ndarray:
    return np.sort(np.clip(float(scale) * values, 1e-6, 1.0))[::-1]


def generate_blt_theta_pair_candidates(
    *,
    buffers: int | None = None,
    theta: Sequence[float] | None = None,
    theta_min: float = 0.2,
    theta_max: float = 0.8,
    theta_scale_grid: Sequence[float] = (1.0, 0.92),
    theta_hat_scale_grid: Sequence[float] = (0.75, 0.6, 0.45),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate a deterministic family of candidate `(theta, theta_hat)` BLT decay pairs.

    The first implementation keeps the search bounded and reproducible: it varies a
    base forward decay family together with a smaller inverse-side scale family.

    These are real BLT parameter candidates. Any report-facing `lambda` sweep is
    only a local way to choose a canonical `theta` slice before entering this
    search surface.
    """

    base_theta = _resolve_base_theta(
        buffers=buffers,
        theta=theta,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    theta_scales = _validated_scale_grid("theta_scale_grid", theta_scale_grid)
    theta_hat_scales = _validated_scale_grid(
        "theta_hat_scale_grid",
        theta_hat_scale_grid,
    )

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for theta_scale in theta_scales:
        theta_candidate = _scaled_descending_candidate(base_theta, scale=theta_scale)
        for theta_hat_scale in theta_hat_scales:
            theta_hat_candidate = _scaled_descending_candidate(
                theta_candidate,
                scale=theta_hat_scale,
            )
            signature = (
                tuple(float(x) for x in theta_candidate),
                tuple(float(x) for x in theta_hat_candidate),
            )
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((theta_candidate.copy(), theta_hat_candidate.copy()))
    if not candidates:
        raise ValueError("no valid BLT theta-pair candidates were generated")
    return candidates


@dataclass(frozen=True)
class BLTFixedBatchOptimizationResult:
    """Canonical result package for the fixed-batch BLT calibration search."""
    mechanism_state: Mapping[str, Any]
    selected_theta: tuple[float, ...]
    selected_theta_hat: tuple[float, ...]
    selected_candidate_index: int
    noise_multiplier_ref: float
    score: float
    candidate_count: int


def _canonical_blt_mechanism_state(
    *,
    pair,
    total_steps: int,
    steps_per_epoch: int,
    max_participations: int,
) -> dict[str, Any]:
    forward = pair.forward.canonicalized()
    inverse = pair.inverse.canonicalized()
    return {
        "forward": {
            "theta": [float(x) for x in forward.theta_array()],
            "omega": [float(x) for x in forward.omega_array()],
        },
        "inverse": {
            "theta": [float(x) for x in inverse.theta_array()],
            "omega": [float(x) for x in inverse.omega_array()],
        },
        "z_std": 1.0,
        "blt_horizon": int(total_steps),
        "blt_min_separation": int(steps_per_epoch),
        "blt_max_participations": int(max_participations),
    }


def _score_blt_candidate(
    *,
    idx: int,
    theta_candidate: np.ndarray,
    theta_hat_candidate: np.ndarray,
    target_epsilon: float,
    target_delta: float,
    total_steps: int,
    max_grad_norm: float,
    denominator: float,
    steps_per_epoch: int,
    max_participations: int,
    epsilon_tolerance: float,
    semantics: SamplingSemantics,
    candidate_count: int,
) -> BLTFixedBatchOptimizationResult:
    pair = blt_pair_from_theta_pair(
        theta=theta_candidate,
        theta_hat=theta_hat_candidate,
    )
    mechanism_state = _canonical_blt_mechanism_state(
        pair=pair,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        max_participations=max_participations,
    )
    noise_multiplier_ref = float(
        get_noise_multiplier(
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            sample_rate=1.0,
            steps=int(total_steps),
            accountant="blt",
            epsilon_tolerance=float(epsilon_tolerance),
            mechanism_state=mechanism_state,
            sampling_semantics=semantics,
        )
    )
    mechanism_state["noise_multiplier_ref"] = noise_multiplier_ref
    mechanism_state["z_std"] = float(noise_multiplier_ref) * float(max_grad_norm) / float(
        denominator
    )
    return BLTFixedBatchOptimizationResult(
        mechanism_state=mechanism_state,
        selected_theta=tuple(float(x) for x in theta_candidate),
        selected_theta_hat=tuple(float(x) for x in theta_hat_candidate),
        selected_candidate_index=int(idx),
        noise_multiplier_ref=float(noise_multiplier_ref),
        score=float(noise_multiplier_ref),
        candidate_count=int(candidate_count),
    )


def optimize_blt_fixed_batch(
    *,
    target_epsilon: float,
    target_delta: float,
    total_steps: int,
    dataset_size: int,
    logical_batch_size: int,
    max_grad_norm: float,
    loss_reduction: str = "mean",
    sampling_semantics: SamplingSemantics | None = None,
    theta: Sequence[float] | None = None,
    buffers: int | None = None,
    theta_min: float = 0.2,
    theta_max: float = 0.8,
    theta_scale_grid: Sequence[float] = (1.0, 0.92),
    theta_hat_scale_grid: Sequence[float] = (0.75, 0.6, 0.45),
    epsilon_tolerance: float = 0.05,
) -> BLTFixedBatchOptimizationResult:
    """
    Search a small deterministic BLT decay-pair family and return canonical state
    for the best fixed-batch BLT candidate under the current accountant-backed
    calibration contract.

    Claim type: implementation contract for the current fixed-batch BLT
    calibration path. This does not search the amplified BNB route directly.
    """

    semantics = sampling_semantics or SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={},
    )
    if semantics.sampling_mode != "torch_sampler":
        raise ValueError(
            "BLT optimization currently supports the fixed-batch torch_sampler contract only"
        )
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if max_grad_norm <= 0.0 or not math.isfinite(max_grad_norm):
        raise ValueError("max_grad_norm must be finite and > 0")

    steps_per_epoch = _resolve_steps_per_epoch(
        dataset_size=dataset_size,
        logical_batch_size=logical_batch_size,
    )
    max_participations = int(math.ceil(float(total_steps) / float(steps_per_epoch)))
    denominator = _bsr_calibration_denominator(
        loss_reduction=loss_reduction,
        logical_batch_size=logical_batch_size,
    )

    candidates = generate_blt_theta_pair_candidates(
        buffers=buffers,
        theta=theta,
        theta_min=theta_min,
        theta_max=theta_max,
        theta_scale_grid=theta_scale_grid,
        theta_hat_scale_grid=theta_hat_scale_grid,
    )

    best_result: BLTFixedBatchOptimizationResult | None = None
    last_error: Exception | None = None

    for idx, (theta_candidate, theta_hat_candidate) in enumerate(candidates):
        try:
            result = _score_blt_candidate(
                idx=idx,
                theta_candidate=theta_candidate,
                theta_hat_candidate=theta_hat_candidate,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                total_steps=total_steps,
                max_grad_norm=max_grad_norm,
                denominator=denominator,
                steps_per_epoch=steps_per_epoch,
                max_participations=max_participations,
                epsilon_tolerance=epsilon_tolerance,
                semantics=semantics,
                candidate_count=len(candidates),
            )
            if best_result is None or result.score < best_result.score:
                best_result = result
        except Exception as exc:  # keep deterministic search bounded and explicit
            last_error = exc

    if best_result is None:
        if last_error is not None:
            raise ValueError(
                "BLT optimization did not find a supported accountant-backed candidate"
            ) from last_error
        raise ValueError("BLT optimization did not generate any supported candidates")

    return best_result


__all__ = [
    "BLTFixedBatchOptimizationResult",
    "generate_blt_theta_pair_candidates",
    "optimize_blt_fixed_batch",
]
