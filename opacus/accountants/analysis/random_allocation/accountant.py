"""Repeated random-allocation accounting on discretized PLDs.

This module owns the production repeated-accounting path. It resolves
mechanism-facing repeated `k`-out-of-`t` inputs, builds one-step Gaussian PLD
realizations, composes the repeated law on discretized grids, and answers
`δ(ε)` / `ε(δ)` queries on the resulting remove/add PLDs.

The later fixed-bin balls-in-bins bridge is intentionally out of scope here and
lives in `fixed_bin.py`.

See `random_allocation.__init__` for the package-level citation registry.

Traceability:
- `paper/random-allocation/main_RA.tex`
- `paper/bnb-accounting-wo-mc/PLD_components/body.tex`
- `Mf/DP/PLDRandomAllocation.lean`
- `Mf/DP/PLDRandomAllocationKOutOfTReduction.lean`
- `Mf/DP/PLDRandomAllocationNumerics.lean`
- `randomAllocationTransforms`
- `exactKOutOfTReductionTheoremTarget`
- `BNBDeterministicRandomAllocationBridge.lean`
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Literal, Sequence, TypeAlias, overload

import numpy as np
from scipy import stats
from scipy.signal import fftconvolve


PMF_MASS_TOL = 10 * np.finfo(float).eps
SPACING_ATOL = 1e-12
SPACING_RTOL = 1e-6
MIN_GRID_SIZE = 100
FLOAT64_LOG_MAX = math.log(np.finfo(np.float64).max)

__all__ = [
    "RandomAllocationAccountantInputs",
    "RandomAllocationGaussianRuntimeConfig",
    "resolve_random_allocation_accountant_inputs",
    "resolve_random_allocation_gaussian_runtime_config",
    "build_gaussian_random_allocation_realization",
    "estimate_epsilon_random_allocation",
    "estimate_epsilon_range_random_allocation",
]


def _timing_enabled() -> bool:
    return os.getenv("DEBUG_TIMING", "").strip().lower() not in ("", "0", "false", "no")


def _debug_timing(message: str) -> None:
    if _timing_enabled():
        print(f"[opacus.random_allocation] [timing] {message}", flush=True)


@contextmanager
def _timed(label: str):
    if not _timing_enabled():
        yield
        return

    start = time.perf_counter()
    _debug_timing(f"start {label}")

    try:
        yield
    finally:
        _debug_timing(f"done {label} elapsed={time.perf_counter() - start:.3f}s")


def _log_exp_neg_loss_moment(
    pmf: np.ndarray,
    x_array: np.ndarray,
) -> float:
    probs = np.asarray(pmf, dtype=np.float64)
    losses = np.asarray(x_array, dtype=np.float64)
    if probs.shape != losses.shape:
        raise ValueError("pmf and x_array must have the same shape")

    positive = probs > 0.0
    if not np.any(positive):
        return -math.inf

    log_terms = np.log(probs[positive]) - losses[positive]
    max_log = float(np.max(log_terms))
    if not math.isfinite(max_log):
        return max_log

    return max_log + math.log(float(np.sum(np.exp(log_terms - max_log), dtype=np.float64)))


def _rescale_pmf_to_exp_neg_loss_moment_at_most_one(
    pmf: np.ndarray,
    x_array: np.ndarray,
    *,
    atol: float,
) -> np.ndarray:
    probs = np.asarray(pmf, dtype=np.float64)
    log_moment = _log_exp_neg_loss_moment(probs, np.asarray(x_array, dtype=np.float64))
    if not math.isfinite(log_moment):
        if log_moment > 0.0:
            return np.zeros_like(probs, dtype=np.float64)
        return probs

    if log_moment <= math.log1p(float(atol)):
        return probs

    return probs * math.exp(-log_moment)


class _BoundType(Enum):
    DOMINATES = "DOMINATES"
    IS_DOMINATED = "IS_DOMINATED"


class _SpacingType(Enum):
    LINEAR = "linear"
    GEOMETRIC = "geometric"


@dataclass(frozen=True)
class RandomAllocationAccountantInputs:
    """Resolved repeated random-allocation contract for the local runtime.

    This dataclass is the stable input object consumed by the repeated
    random-allocation accountant. It captures the reduced repeated shape, the
    coefficient vector, and the effective Gaussian scale derived from the
    mechanism-facing inputs.

    Source: `RA`, `PLD`.

    Attributes:
        contract_kind: Stable identifier for this repeated-accounting contract.
        package_alignment_kind: Name of the resolved repeated-accounting family.
        package_alignment_notes: Human-readable explanation of the resolved
            route and theorem layer.
        route: Reduced repeated route selected by the resolver.
        mechanism: Mechanism name used for diagnostics and routing.
        accountant_coeffs: Coefficient vector used by the one-step realization.
        accountant_coeff_l2: `ℓ₂` norm of `accountant_coeffs`.
        noise_multiplier: User-facing Gaussian noise multiplier `σ`.
        effective_sigma: Effective one-step Gaussian scale `σ / ||c||₂`.
        horizon: Total number of mechanism applications covered by the route.
        cycle_length: Number of positions in one repeated round.
        pld_num_steps: Repeated-accounting `t` parameter on the PLD side.
        pld_num_selected: Repeated-accounting `k` parameter on the PLD side.
        pld_num_epochs: Number of repeated rounds after reduction.
        reduced_num_steps_per_round: Reduced per-round length passed to the
            repeated recurrence.
        reduced_num_rounds: Reduced number of repeated rounds.
        exact_law_route: Exact-law route used by the supported public repeated
            path, if one was resolved.
        initial_package_route: Deterministic initial-package route used by the
            supported public repeated path, if one was resolved.
        pair_driven_inputs: Internal package-backed repeated-accounting object
            used by the supported public exact-law route.
    """

    contract_kind: str
    package_alignment_kind: str
    package_alignment_notes: str
    route: str
    mechanism: str
    accountant_coeffs: tuple[float, ...]
    accountant_coeff_l2: float
    noise_multiplier: float
    effective_sigma: float
    horizon: int
    cycle_length: int
    pld_num_steps: int
    pld_num_selected: int
    pld_num_epochs: int
    reduced_num_steps_per_round: int
    reduced_num_rounds: int
    exact_law_route: str | None = None
    initial_package_route: str | None = None
    pair_driven_inputs: Any | None = None


_SUPPORTED_PUBLIC_EXACT_MECHANISMS = frozenset(
    {"gaussian", "bsr", "bisr", "bandmf", "bandinvmf"}
)


@dataclass(frozen=True)
class RandomAllocationGaussianRuntimeConfig:
    """Runtime policy for discretizing and composing repeated Gaussian PLDs.

    Source: `PLD`.

    Attributes:
        policy_name: Stable name for the runtime policy.
        runtime_policy: Repeated-accounting runtime policy. Public repeated
            routes currently distinguish `strict_exact_package` from
            `efficient_staged_grid`.
        loss_discretization: Linear grid spacing used by the one-step PLD
            realization before optional geometric conversion. In efficient mode
            this is the requested output-grid spacing.
        tail_truncation: Total tail budget spent on truncation and
            rediscretization during composition. In efficient mode this is the
            output-level tail budget before staged splitting.
        max_grid_fft: Maximum grid size permitted for FFT-based convolution.
        max_grid_mult: Maximum grid size permitted for iterative multiplication.
        convolution_method: Selected composition backend, e.g. `fft` or
            `geometric`.
        matches_package_defaults: Whether the policy matches the older package
            defaults rather than the repository-fast defaults.
        clamp_to_package_grid: Whether package-backed evaluation should force
            runtime loss spacing down to the package realization spacing.
        remove_convolution_method: Preferred efficient remove-side convolution
            policy.
        add_convolution_method: Preferred efficient add-side convolution
            policy.
        refinement_rounds: Maximum bounded-refinement steps for repeated
            calibration under the efficient policy.
    """

    policy_name: str
    runtime_policy: str
    loss_discretization: float
    tail_truncation: float
    max_grid_fft: int
    max_grid_mult: int
    convolution_method: str
    matches_package_defaults: bool
    clamp_to_package_grid: bool = True
    remove_convolution_method: str = "fft"
    add_convolution_method: str = "geometric"
    refinement_rounds: int = 0


@dataclass(frozen=True)
class _RepeatedRuntimeStages:
    output_loss_discretization: float
    pre_composition_loss_discretization: float
    inner_loss_discretization: float
    output_tail_truncation: float
    pre_composition_tail_truncation: float
    inner_tail_truncation: float
    remove_convolution_method: str
    add_convolution_method: str
    refinement_rounds: int


class _LinearDiscreteDist:
    def __init__(
        self,
        x_min: float,
        x_gap: float,
        pmf: np.ndarray,
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ):
        self.x_min = float(x_min)
        self.x_gap = float(x_gap)
        self.PMF_array = np.asarray(pmf, dtype=np.float64)
        self.p_neg_inf = float(p_neg_inf)
        self.p_pos_inf = float(p_pos_inf)
        self._validate_basic()

    @classmethod
    def from_x_array(
        cls,
        x_array: np.ndarray,
        pmf: np.ndarray,
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ):
        x = np.asarray(x_array, dtype=np.float64)
        gap = _compute_bin_width(x)
        return cls(
            float(x[0]), gap, np.asarray(pmf, dtype=np.float64), p_neg_inf, p_pos_inf
        )

    @property
    def x_array(self) -> np.ndarray:
        return self.x_min + self.x_gap * np.arange(
            self.PMF_array.size, dtype=np.float64
        )

    def copy(self) -> _LinearDiscreteDist:
        return _LinearDiscreteDist(
            self.x_min,
            self.x_gap,
            self.PMF_array.copy(),
            self.p_neg_inf,
            self.p_pos_inf,
        )

    def truncate_edges(
        self, tail_truncation: float, bound_type: _BoundType
    ) -> _LinearDiscreteDist:
        return _truncate_linear(self, tail_truncation, bound_type)

    def _validate_basic(self) -> None:
        if self.PMF_array.ndim != 1 or self.PMF_array.size < 2:
            raise ValueError("PMF must be a 1-D array with at least 2 points")

        if self.x_gap <= 0.0 or not math.isfinite(self.x_gap):
            raise ValueError("x_gap must be positive and finite")

        if np.any(self.PMF_array < -PMF_MASS_TOL):
            raise ValueError("PMF must be nonnegative")

        if self.p_neg_inf < 0.0 or self.p_pos_inf < 0.0:
            raise ValueError("Infinity masses must be nonnegative")


class _GeometricDiscreteDist:
    def __init__(
        self,
        x_min: float,
        ratio: float,
        pmf: np.ndarray,
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ):
        self.x_min = float(x_min)
        self.ratio = float(ratio)
        self.PMF_array = np.asarray(pmf, dtype=np.float64)
        self.p_neg_inf = float(p_neg_inf)
        self.p_pos_inf = float(p_pos_inf)
        self._validate_basic()

    @property
    def x_array(self) -> np.ndarray:
        log_x_min = math.log(self.x_min)
        log_ratio = math.log(self.ratio)
        logs = log_x_min + log_ratio * np.arange(self.PMF_array.size, dtype=np.float64)
        out = np.empty(self.PMF_array.size, dtype=np.float64)
        finite = logs <= FLOAT64_LOG_MAX
        out[finite] = np.exp(logs[finite])
        out[~finite] = np.inf
        return out

    def truncate_edges(
        self, tail_truncation: float, bound_type: _BoundType
    ) -> _GeometricDiscreteDist:
        nonzero = np.nonzero(self.PMF_array)[0]
        if nonzero.size == 0:
            if self.p_neg_inf > 0.0 or self.p_pos_inf > 0.0:
                return self

            raise ValueError("Cannot truncate distribution with zero finite mass")

        if tail_truncation == 0.0:
            min_ind = int(nonzero[0])
            max_ind = int(nonzero[-1])
        else:
            cumsum_left = np.cumsum(self.PMF_array, dtype=np.float64)
            cumsum_right = np.cumsum(self.PMF_array[::-1], dtype=np.float64)
            min_ind = int(np.searchsorted(cumsum_left, tail_truncation, side="right"))
            right_cnt = int(
                np.searchsorted(cumsum_right, tail_truncation, side="right")
            )
            max_ind = self.PMF_array.size - 1 - right_cnt

        if min_ind > max_ind:
            min_ind = int(nonzero[0])
            max_ind = int(nonzero[-1])

        if max_ind - min_ind < 1:
            if min_ind > 0:
                min_ind -= 1
            elif max_ind < self.PMF_array.size - 1:
                max_ind += 1

        left_mass = float(np.sum(self.PMF_array[:min_ind], dtype=np.float64))
        right_mass = float(np.sum(self.PMF_array[max_ind + 1 :], dtype=np.float64))
        new_pmf = self.PMF_array[min_ind : max_ind + 1].copy()
        if bound_type == _BoundType.DOMINATES:
            new_pmf[0] += left_mass
            expected_neg_inf = self.p_neg_inf
            expected_pos_inf = self.p_pos_inf + right_mass
        else:
            new_pmf[-1] += right_mass
            expected_neg_inf = self.p_neg_inf + left_mass
            expected_pos_inf = self.p_pos_inf

        new_pmf, new_p_neg_inf, new_p_pos_inf = _enforce_mass_conservation(
            new_pmf, expected_neg_inf, expected_pos_inf, bound_type
        )
        return _GeometricDiscreteDist(
            self.x_array[min_ind], self.ratio, new_pmf, new_p_neg_inf, new_p_pos_inf
        )

    def _validate_basic(self) -> None:
        if self.PMF_array.ndim != 1 or self.PMF_array.size < 2:
            raise ValueError("PMF must be a 1-D array with at least 2 points")

        if self.x_min <= 0.0 or self.ratio <= 1.0:
            raise ValueError("Geometric grid requires x_min > 0 and ratio > 1")

        if np.any(self.PMF_array < -PMF_MASS_TOL):
            raise ValueError("PMF must be nonnegative")

        if self.p_neg_inf < 0.0 or self.p_pos_inf < 0.0:
            raise ValueError("Infinity masses must be nonnegative")


class _PLDRealization(_LinearDiscreteDist):
    def __init__(
        self,
        x_min: float,
        x_gap: float,
        pmf: np.ndarray,
        p_loss_inf: float = 0.0,
        p_loss_neg_inf: float = 0.0,
    ):
        super().__init__(x_min, x_gap, pmf, p_loss_neg_inf, p_loss_inf)
        self.validate_pld_realization()

    @classmethod
    def from_linear_dist(cls, dist: _LinearDiscreteDist):
        return cls(
            dist.x_min,
            dist.x_gap,
            dist.PMF_array.copy(),
            dist.p_pos_inf,
            dist.p_neg_inf,
        )

    @property
    def p_loss_inf(self) -> float:
        return self.p_pos_inf

    @property
    def p_loss_neg_inf(self) -> float:
        return self.p_neg_inf

    @property
    def loss_values(self) -> np.ndarray:
        return self.x_array

    @property
    def probabilities(self) -> np.ndarray:
        return self.PMF_array

    def validate_pld_realization(self) -> _PLDRealization:
        self._validate_basic()
        pmf_sum = float(np.sum(self.PMF_array, dtype=np.float64))
        total_mass = pmf_sum + self.p_neg_inf + self.p_pos_inf
        if abs(total_mass - 1.0) > PMF_MASS_TOL:
            raise ValueError("MASS CONSERVATION ERROR")

        if self.p_neg_inf > 0.0:
            raise ValueError("DOMINATES bound_type requires p_neg_inf=0")

        log_moment = _log_exp_neg_loss_moment(self.PMF_array, self.x_array)
        if log_moment > math.log1p(1e-9):
            raise ValueError("E[exp(-L)] must be <= 1 for a PLD realization")

        return self


_DiscreteDist: TypeAlias = _LinearDiscreteDist | _GeometricDiscreteDist


def _dist_debug_summary(dist: _DiscreteDist) -> str:
    spacing = (
        f"x_gap={dist.x_gap:.12g}"
        if isinstance(dist, _LinearDiscreteDist)
        else f"ratio={dist.ratio:.12g}"
    )
    return (
        f"len={dist.PMF_array.size} x_min={dist.x_min:.12g} {spacing} "
        f"p_neg_inf={dist.p_neg_inf:.12g} p_pos_inf={dist.p_pos_inf:.12g}"
    )


def _stable_isclose(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL))


def _compute_bin_width(x_array: np.ndarray) -> float:
    if x_array.size < 2:
        raise ValueError("Cannot compute width with less than 2 bins")

    diffs = np.diff(x_array)
    median = float(np.median(diffs))
    if not np.allclose(diffs, median, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        raise ValueError("Distribution has non-uniform bin widths")

    return median


def _compute_bin_ratio(x_array: np.ndarray) -> float:
    if np.any(x_array <= 0):
        raise ValueError("Cannot compute geometric bin ratio for non-positive values")

    finite = np.isfinite(x_array)
    if np.count_nonzero(finite) < 2:
        raise ValueError("Cannot compute geometric bin ratio with less than 2 finite bins")

    finite_x = np.asarray(x_array[finite], dtype=np.float64)
    logs = np.diff(np.log(finite_x))
    median = float(np.median(logs))
    if not np.allclose(logs, median, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        raise ValueError("Distribution has non-uniform bin widths")

    return float(np.exp(median))


def _compute_bin_ratio_two_arrays(
    x_array_1: np.ndarray, x_array_2: np.ndarray
) -> float:
    r1 = _compute_bin_ratio(x_array_1)
    r2 = _compute_bin_ratio(x_array_2)
    if not _stable_isclose(r1, r2):
        raise ValueError("Grid ratios must match")

    return 0.5 * (r1 + r2)


def _expected_infinity_mass(
    *,
    expected_neg_inf: float,
    expected_pos_inf: float,
    bound_type: _BoundType,
) -> float:
    return expected_pos_inf if bound_type == _BoundType.DOMINATES else expected_neg_inf


def _all_infinity_mass_result(
    *,
    pmf: np.ndarray,
    expected_inf: float,
    bound_type: _BoundType,
) -> tuple[np.ndarray, float, float]:
    if expected_inf >= 1.0 - PMF_MASS_TOL:
        zero = np.zeros_like(pmf, dtype=np.float64)
        if bound_type == _BoundType.DOMINATES:
            return zero, 0.0, 1.0

        return zero, 1.0, 0.0

    raise ValueError("Cannot enforce mass conservation with zero finite mass")


def _trim_linear_mass_to_target(
    *,
    pmf: np.ndarray,
    finite_target: float,
    bound_type: _BoundType,
) -> np.ndarray:
    pmf_sum = float(np.sum(pmf, dtype=np.float64))
    if pmf_sum <= finite_target:
        return pmf

    excess = min(pmf_sum - finite_target, pmf_sum)
    if bound_type == _BoundType.DOMINATES:
        cumsum = np.cumsum(pmf, dtype=np.float64)
        pivot = int(np.searchsorted(cumsum, excess, side="left"))
        pivot = min(max(pivot, 0), pmf.size - 1)
        removed_before = float(cumsum[pivot - 1]) if pivot > 0 else 0.0
        if pivot > 0:
            pmf[:pivot] = 0.0

        pmf[pivot] = max(0.0, pmf[pivot] - (excess - removed_before))

        return pmf

    cumsum_rev = np.cumsum(pmf[::-1], dtype=np.float64)
    rev_pivot = int(np.searchsorted(cumsum_rev, excess, side="left"))
    rev_pivot = min(max(rev_pivot, 0), pmf.size - 1)
    removed_before = float(cumsum_rev[rev_pivot - 1]) if rev_pivot > 0 else 0.0
    pivot = pmf.size - 1 - rev_pivot

    if rev_pivot > 0:
        pmf[pivot + 1 :] = 0.0

    pmf[pivot] = max(0.0, pmf[pivot] - (excess - removed_before))

    return pmf


def _project_infinity_mass(
    *,
    pmf: np.ndarray,
    expected_inf: float,
    bound_type: _BoundType,
) -> tuple[np.ndarray, float, float]:
    """Project any remaining mass mismatch onto the active infinite atom.

    Dominating routes keep their infinite mass on `+∞`; dominated routes keep
    it on `-∞`. If the expected infinite mass exceeds the current residual mass,
    the finite PMF is rescaled first so the final law still sums to one.
    """
    pmf_sum = float(np.sum(pmf, dtype=np.float64))
    remaining_mass = max(0.0, 1.0 - pmf_sum)
    output_inf = max(expected_inf, remaining_mass)

    if output_inf > remaining_mass and pmf_sum > 0.0:
        pmf = pmf * ((1.0 - expected_inf) / pmf_sum)

    if bound_type == _BoundType.DOMINATES:
        return pmf, 0.0, output_inf

    return pmf, output_inf, 0.0


def _enforce_mass_conservation(
    pmf: np.ndarray,
    expected_neg_inf: float,
    expected_pos_inf: float,
    bound_type: _BoundType,
):
    """Repair finite and infinite masses so the discrete law sums to one.

    The repair respects the one-sided semantics encoded by `bound_type`.
    Finite-grid mass is trimmed toward the appropriate edge, then any remaining
    mismatch is projected onto the active infinite atom.
    """
    expected_inf = _expected_infinity_mass(
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=bound_type,
    )
    pmf_sum = float(np.sum(pmf, dtype=np.float64))
    if pmf_sum <= 0.0:
        # A zero finite slice is still valid when the entire distribution mass is
        # already accounted for at +inf / -inf. Several direct ambient bridge
        # routes rely on this exact case.
        return _all_infinity_mass_result(
            pmf=pmf,
            expected_inf=expected_inf,
            bound_type=bound_type,
        )

    finite_target = 1.0 - expected_inf
    if finite_target < 0.0:
        raise ValueError("Expected infinity mass cannot exceed 1")

    pmf = _trim_linear_mass_to_target(
        pmf=pmf,
        finite_target=finite_target,
        bound_type=bound_type,
    )

    return _project_infinity_mass(
        pmf=pmf,
        expected_inf=expected_inf,
        bound_type=bound_type,
    )


def _truncate_linear(
    dist: _LinearDiscreteDist, tail_truncation: float, bound_type: _BoundType
) -> _LinearDiscreteDist:
    """Trim a linear-grid PLD under a one-sided finite tail budget.

    The left and right finite tails are removed according to `tail_truncation`,
    and the surviving edge bins plus `±∞` masses are adjusted in the domination
    direction requested by `bound_type`.
    """
    nonzero = np.nonzero(dist.PMF_array)[0]
    if nonzero.size == 0:
        if dist.p_neg_inf > 0.0 or dist.p_pos_inf > 0.0:
            return dist
        raise ValueError("Cannot truncate distribution with zero finite mass")

    if tail_truncation == 0.0:
        min_ind = int(nonzero[0])
        max_ind = int(nonzero[-1])
    else:
        cumsum_left = np.cumsum(dist.PMF_array, dtype=np.float64)
        cumsum_right = np.cumsum(dist.PMF_array[::-1], dtype=np.float64)
        min_ind = int(np.searchsorted(cumsum_left, tail_truncation, side="right"))
        right_cnt = int(np.searchsorted(cumsum_right, tail_truncation, side="right"))
        max_ind = dist.PMF_array.size - 1 - right_cnt

    if min_ind > max_ind:
        min_ind = int(nonzero[0])
        max_ind = int(nonzero[-1])

    if max_ind - min_ind < 1:
        if min_ind > 0:
            min_ind -= 1
        elif max_ind < dist.PMF_array.size - 1:
            max_ind += 1

    left_mass = float(np.sum(dist.PMF_array[:min_ind], dtype=np.float64))
    right_mass = float(np.sum(dist.PMF_array[max_ind + 1 :], dtype=np.float64))
    new_pmf = dist.PMF_array[min_ind : max_ind + 1].copy()
    if bound_type == _BoundType.DOMINATES:
        new_pmf[0] += left_mass
        expected_neg_inf = dist.p_neg_inf
        expected_pos_inf = dist.p_pos_inf + right_mass
    else:
        new_pmf[-1] += right_mass
        expected_neg_inf = dist.p_neg_inf + left_mass
        expected_pos_inf = dist.p_pos_inf

    new_pmf, new_p_neg_inf, new_p_pos_inf = _enforce_mass_conservation(
        new_pmf, expected_neg_inf, expected_pos_inf, bound_type
    )
    return _LinearDiscreteDist(
        dist.x_min + min_ind * dist.x_gap,
        dist.x_gap,
        new_pmf,
        new_p_neg_inf,
        new_p_pos_inf,
    )


def _discretize_aligned_range(
    x_min: float, x_max: float, spacing_type: _SpacingType, discretization: float
) -> np.ndarray:
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")

    if discretization <= 0.0:
        raise ValueError("discretization must be positive")

    if spacing_type == _SpacingType.GEOMETRIC:
        if x_min <= 0.0:
            raise ValueError("Geometric spacing requires positive values")

        n_grid = max(
            int(np.ceil(np.log(x_max / x_min) / discretization)) + 1, MIN_GRID_SIZE
        )
        x_min = float(np.exp(np.floor(np.log(x_min) / discretization) * discretization))
        x_max = float(np.exp(np.ceil(np.log(x_max) / discretization) * discretization))
        n_grid = int(np.ceil(np.log(x_max / x_min) / discretization)) + 1
        return np.geomspace(x_min, x_max, n_grid)

    n_grid = max(int(np.ceil((x_max - x_min) / discretization)) + 1, MIN_GRID_SIZE)
    x_min = float(np.floor(x_min / discretization) * discretization)
    x_max = float(np.ceil(x_max / discretization) * discretization)
    n_grid = int(np.ceil((x_max - x_min) / discretization)) + 1

    return x_min + discretization * np.arange(n_grid, dtype=np.float64)


def _rediscretize_pmf(
    x_array: np.ndarray, pmf_array: np.ndarray, x_array_out: np.ndarray, dominates: bool
) -> np.ndarray:
    """Move a PMF onto a new aligned grid with one-sided rounding.

    When `dominates` is true, each source mass is rounded to the first output
    bin at or to the right of the source location. Otherwise it is rounded to
    the last output bin at or to the left of the source location.
    """
    n_out = x_array_out.size
    pmf_out = np.zeros(n_out, dtype=np.float64)
    comp = np.zeros(n_out, dtype=np.float64)
    j = 0

    if dominates:
        for i in range(x_array.size):
            z = float(x_array[i])
            mass = float(pmf_array[i])
            if mass <= 0.0:
                continue

            while j < n_out and x_array_out[j] < z:
                j += 1

            if j >= n_out:
                continue

            y = mass - comp[j]
            t = pmf_out[j] + y
            comp[j] = (t - pmf_out[j]) - y
            pmf_out[j] = t
    else:
        for i in range(x_array.size):
            z = float(x_array[i])
            mass = float(pmf_array[i])
            if mass <= 0.0:
                continue

            while j + 1 < n_out and x_array_out[j + 1] <= z:
                j += 1

            if z < x_array_out[0]:
                continue

            y = mass - comp[j]
            t = pmf_out[j] + y
            comp[j] = (t - pmf_out[j]) - y
            pmf_out[j] = t

    return pmf_out


@overload
def _change_spacing_type(
    dist: _DiscreteDist,
    tail_truncation: float,
    loss_discretization: float,
    spacing_type: Literal[_SpacingType.LINEAR],
    bound_type: _BoundType,
) -> _LinearDiscreteDist: ...


@overload
def _change_spacing_type(
    dist: _DiscreteDist,
    tail_truncation: float,
    loss_discretization: float,
    spacing_type: Literal[_SpacingType.GEOMETRIC],
    bound_type: _BoundType,
) -> _GeometricDiscreteDist: ...


def _change_spacing_type(
    dist: _DiscreteDist,
    tail_truncation: float,
    loss_discretization: float,
    spacing_type: _SpacingType,
    bound_type: _BoundType,
) -> _DiscreteDist:
    trunc_dist = dist.truncate_edges(tail_truncation / 2, bound_type)
    x_array = trunc_dist.x_array
    x_array_out = _discretize_aligned_range(
        float(x_array[0]), float(x_array[-1]), spacing_type, loss_discretization
    )
    pmf_out = _rediscretize_pmf(
        x_array, trunc_dist.PMF_array, x_array_out, bound_type == _BoundType.DOMINATES
    )
    pmf_out, p_neg_inf, p_pos_inf = _enforce_mass_conservation(
        pmf_out, trunc_dist.p_neg_inf, trunc_dist.p_pos_inf, bound_type
    )

    if spacing_type == _SpacingType.LINEAR:
        return _LinearDiscreteDist.from_x_array(
            x_array_out, pmf_out, p_neg_inf, p_pos_inf
        )

    return _GeometricDiscreteDist(
        float(x_array_out[0]),
        _compute_bin_ratio(x_array_out),
        pmf_out,
        p_neg_inf,
        p_pos_inf,
    )


def _combine_infinite_mass_probabilities(p1: float, p2: float) -> float:
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    if p1 == 1.0 or p2 == 1.0:
        return 1.0
    return float(np.clip(p1 + p2 - p1 * p2, 0.0, 1.0))


def _convolve_infinite_masses(
    p_neg_inf_1: float, p_pos_inf_1: float, p_neg_inf_2: float, p_pos_inf_2: float
):
    p_neg_inf = _combine_infinite_mass_probabilities(p_neg_inf_1, p_neg_inf_2)
    p_pos_inf = _combine_infinite_mass_probabilities(p_pos_inf_1, p_pos_inf_2)

    return p_neg_inf, p_pos_inf


def _binary_self_convolve(
    dist: _GeometricDiscreteDist, t: int, tail_truncation: float, bound_type: _BoundType
) -> _GeometricDiscreteDist:
    """Exponentiate a geometric-grid PMF by repeated self-convolution.

    This uses binary powering so `t` copies are composed with `O(log t)`
    convolutions instead of `O(t)` direct multiplications. The tail budget is
    spent across the intermediate convolutions in the same direction as
    `bound_type`.
    """
    if t < 1:
        raise ValueError("T must be >= 1")

    if t == 1:
        return dist

    base = dist
    acc = None
    tail = tail_truncation / 4.0
    remaining = int(t)
    while remaining > 0:
        if remaining & 1:
            acc = (
                base
                if acc is None
                else _geometric_convolve(
                    acc, base, tail / max(remaining, 1), bound_type
                )
            )

        remaining >>= 1
        if remaining > 0:
            base = _geometric_convolve(base, base, tail / max(remaining, 1), bound_type)

    return acc if acc is not None else base


def _linear_convolve_fft(
    dist_1: _LinearDiscreteDist,
    dist_2: _LinearDiscreteDist,
    tail_truncation: float,
    bound_type: _BoundType,
) -> _LinearDiscreteDist:
    if not math.isclose(
        float(dist_1.x_gap),
        float(dist_2.x_gap),
        rel_tol=SPACING_RTOL,
        abs_tol=SPACING_ATOL,
    ):
        raise ValueError("Linear convolution requires matching x_gap")

    pmf = fftconvolve(dist_1.PMF_array, dist_2.PMF_array, mode="full")
    pmf = np.maximum(np.asarray(pmf, dtype=np.float64), 0.0)
    p_neg_inf, p_pos_inf = _convolve_infinite_masses(
        dist_1.p_neg_inf,
        dist_1.p_pos_inf,
        dist_2.p_neg_inf,
        dist_2.p_pos_inf,
    )
    pmf, p_neg_inf, p_pos_inf = _enforce_mass_conservation(
        pmf, p_neg_inf, p_pos_inf, bound_type
    )
    return _LinearDiscreteDist(
        dist_1.x_min + dist_2.x_min,
        dist_1.x_gap,
        pmf,
        p_neg_inf,
        p_pos_inf,
    ).truncate_edges(tail_truncation, bound_type)


def _binary_self_convolve_linear(
    dist: _LinearDiscreteDist,
    t: int,
    tail_truncation: float,
    bound_type: _BoundType,
) -> _LinearDiscreteDist:
    if t < 1:
        raise ValueError("T must be >= 1")

    if t == 1:
        return dist

    base = dist
    acc: _LinearDiscreteDist | None = None
    tail = tail_truncation / 4.0
    remaining = int(t)
    while remaining > 0:
        if remaining & 1:
            acc = (
                base
                if acc is None
                else _linear_convolve_fft(
                    acc, base, tail / max(remaining, 1), bound_type
                )
            )

        remaining >>= 1
        if remaining > 0:
            base = _linear_convolve_fft(
                base, base, tail / max(remaining, 1), bound_type
            )

    return acc if acc is not None else base


def _iter_constant_int_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.ndim != 1:
        raise ValueError("values must be a 1-D array")

    if values.size == 0:
        return []

    change_indices = np.nonzero(np.diff(values))[0]
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), change_indices + 1),
        dtype=np.int64,
    )
    ends = np.concatenate(
        (change_indices, np.array([values.size - 1], dtype=np.int64)),
        dtype=np.int64,
    )
    return [
        (int(start), int(end), int(values[start]))
        for start, end in zip(starts, ends, strict=False)
    ]


def _compensated_slice_add_inplace(
    pmf_out: np.ndarray,
    comp: np.ndarray,
    start: int,
    values: np.ndarray,
) -> None:
    if values.size == 0:
        return

    if start < 0:
        offset = -start
        if offset >= values.size:
            return
        start = 0
        values = values[offset:]

    stop = min(start + values.size, pmf_out.size)
    if stop <= start:
        return

    values = values[: stop - start]
    sl = slice(start, stop)
    y = values - comp[sl]
    t = pmf_out[sl] + y
    comp[sl] = (t - pmf_out[sl]) - y
    pmf_out[sl] = t


def _geometric_convolve(
    dist_1: _GeometricDiscreteDist,
    dist_2: _GeometricDiscreteDist,
    tail_truncation: float,
    bound_type: _BoundType,
) -> _GeometricDiscreteDist:
    """Convolve two geometric-grid PMFs and preserve one-sided PLD semantics.

    The finite-grid part is composed on a shared geometric grid, while the
    infinite masses are combined analytically. The result is then truncated in
    the domination direction requested by `bound_type`.
    """
    if not _stable_isclose(dist_1.ratio, dist_2.ratio):
        raise ValueError("Grid ratios must match")

    ratio = 0.5 * (dist_1.ratio + dist_2.ratio)
    x1_min = dist_1.x_min
    x2_min = dist_2.x_min
    p1 = dist_1.PMF_array
    p2 = dist_2.PMF_array
    if x1_min > x2_min:
        x1_min, p1, x2_min, p2 = x2_min, p2, x1_min, p1

    scale = float(x2_min / x1_min)
    n = max(p1.size, p2.size)
    if p1.size < n:
        p1 = np.pad(p1, (0, n - p1.size), mode="constant")

    if p2.size < n:
        p2 = np.pad(p2, (0, n - p2.size), mode="constant")

    if n == 1:
        mass = p1[0] * p2[0]
        x_out_min = x1_min + x2_min
        pmf_out = np.array([mass], dtype=np.float64)
    else:
        log_r = math.log(ratio)
        log_scale = math.log(scale)
        log_ap1 = math.log(scale + 1.0)
        d_vec = np.arange(n, dtype=np.float64)
        log_r_d = d_vec * log_r
        log_lohi = np.logaddexp(0.0, log_scale + log_r_d)
        tau_lohi = (log_lohi - log_ap1) / log_r
        log_hilo = np.logaddexp(log_scale, log_r_d)
        tau_hilo = (log_hilo - log_ap1) / log_r
        rounding_eps = 1e-16

        if bound_type == _BoundType.DOMINATES:
            delta_lohi = np.ceil(tau_lohi - rounding_eps).astype(np.int64)
            delta_hilo = np.ceil(tau_hilo - rounding_eps).astype(np.int64)
        else:
            delta_lohi = np.floor(tau_lohi + rounding_eps).astype(np.int64)
            delta_hilo = np.floor(tau_hilo + rounding_eps).astype(np.int64)

        pmf_out = np.zeros(n, dtype=np.float64)
        comp = np.zeros(n, dtype=np.float64)
        _compensated_slice_add_inplace(pmf_out, comp, 0, p1 * p2)

        if n > 1:
            diag_offsets = np.arange(n, dtype=np.int64)
            diff_lohi = delta_lohi - diag_offsets
            diff_hilo = delta_hilo - diag_offsets
            prefix_1 = np.concatenate(
                (np.array([0.0], dtype=np.float64), np.cumsum(p1, dtype=np.float64))
            )
            prefix_2 = np.concatenate(
                (np.array([0.0], dtype=np.float64), np.cumsum(p2, dtype=np.float64))
            )
            target_indices = np.arange(n, dtype=np.int64)

            for start_idx, end_idx, offset in _iter_constant_int_runs(diff_lohi[1:]):
                start_d = start_idx + 1
                end_d = end_idx + 1
                j = target_indices[start_d:]
                left = np.maximum(j - end_d, 0)
                right = j - start_d
                window_mass = prefix_1[right + 1] - prefix_1[left]
                values = p2[start_d:] * window_mass
                _compensated_slice_add_inplace(
                    pmf_out,
                    comp,
                    start_d + offset,
                    values,
                )

            for start_idx, end_idx, offset in _iter_constant_int_runs(diff_hilo[1:]):
                start_d = start_idx + 1
                end_d = end_idx + 1
                j = target_indices[start_d:]
                left = np.maximum(j - end_d, 0)
                right = j - start_d
                window_mass = prefix_2[right + 1] - prefix_2[left]
                values = p1[start_d:] * window_mass
                _compensated_slice_add_inplace(
                    pmf_out,
                    comp,
                    start_d + offset,
                    values,
                )

        x_out_min = x1_min + x2_min

    expected_neg_inf, expected_pos_inf = _convolve_infinite_masses(
        dist_1.p_neg_inf, dist_1.p_pos_inf, dist_2.p_neg_inf, dist_2.p_pos_inf
    )
    pmf_out, p_neg_inf, p_pos_inf = _enforce_mass_conservation(
        pmf_out, expected_neg_inf, expected_pos_inf, bound_type
    )

    return _GeometricDiscreteDist(
        float(x_out_min), ratio, pmf_out, p_neg_inf, p_pos_inf
    ).truncate_edges(tail_truncation, bound_type)


def _exp_linear_to_geometric(dist: _LinearDiscreteDist) -> _GeometricDiscreteDist:
    return _GeometricDiscreteDist(
        float(np.exp(dist.x_min)),
        float(np.exp(dist.x_gap)),
        dist.PMF_array.copy(),
        dist.p_neg_inf,
        dist.p_pos_inf,
    )


def _log_geometric_to_linear(dist: _GeometricDiscreteDist) -> _LinearDiscreteDist:
    return _LinearDiscreteDist(
        float(np.log(dist.x_min)),
        float(np.log(dist.ratio)),
        dist.PMF_array.copy(),
        dist.p_neg_inf,
        dist.p_pos_inf,
    )


def _negate_reverse_linear_distribution(
    dist: _LinearDiscreteDist,
) -> _LinearDiscreteDist:
    n = dist.PMF_array.size

    return _LinearDiscreteDist(
        -(dist.x_min + dist.x_gap * (n - 1)),
        dist.x_gap,
        np.flip(dist.PMF_array),
        dist.p_pos_inf,
        dist.p_neg_inf,
    )


def _calc_pld_dual(realization: _PLDRealization) -> _PLDRealization:
    dual_probs_aligned = np.zeros_like(realization.PMF_array)
    mask = realization.PMF_array > 0
    dual_probs_aligned[mask] = np.exp(
        np.log(realization.PMF_array[mask]) - realization.x_array[mask]
    )
    dual_probs = np.flip(dual_probs_aligned)
    sum_prob = float(np.sum(dual_probs, dtype=np.float64))
    if sum_prob > 1.0:
        dual_probs *= 1.0 / sum_prob
        sum_prob = 1.0

    return _PLDRealization(
        -(realization.x_min + realization.x_gap * (realization.PMF_array.size - 1)),
        realization.x_gap,
        dual_probs,
        max(0.0, 1.0 - sum_prob),
        0.0,
    )


def _decompose_allocation_compositions(
    num_steps: int, num_selected: int, num_epochs: int
):
    if num_steps < 1 or num_selected < 1 or num_epochs < 1:
        raise ValueError("num_steps, num_selected, num_epochs must all be >= 1")

    num_steps_per_round = int(num_steps // num_selected)
    if num_steps_per_round < 1:
        raise ValueError("num_steps must be >= num_selected")

    return num_steps_per_round, int(num_selected * num_epochs)


def _log_mean_exp_remove(
    lower_loss_factor: _LinearDiscreteDist,
    upper_loss_factor: _LinearDiscreteDist,
    num_steps: int,
    tail_truncation: float,
    bound_type: _BoundType,
):
    """Build the remove-side per-round PLD from the `RA` log-mean-exp rule.

    It computes

        log_remove(t) = log((1 / T) · ∑_{j=1}^{T} exp(log_loss_j(t)))

    with `T = num_steps`, where one distinguished term uses
    `upper_loss_factor` and the other `T - 1` terms use `lower_loss_factor`.
    The `-log(T)` shift absorbs the averaging factor into the grid before
    exponentiation.
    """
    log_num_steps = float(np.log(num_steps))
    scaled_lower = _LinearDiscreteDist(
        lower_loss_factor.x_min - log_num_steps,
        lower_loss_factor.x_gap,
        lower_loss_factor.PMF_array.copy(),
        lower_loss_factor.p_neg_inf,
        lower_loss_factor.p_pos_inf,
    )
    scaled_upper = _LinearDiscreteDist(
        upper_loss_factor.x_min - log_num_steps,
        upper_loss_factor.x_gap,
        upper_loss_factor.PMF_array.copy(),
        upper_loss_factor.p_neg_inf,
        upper_loss_factor.p_pos_inf,
    )
    lower_exp = _exp_linear_to_geometric(scaled_lower)
    upper_exp = _exp_linear_to_geometric(scaled_upper)

    if num_steps == 1:
        composed = upper_exp
    else:
        convolved_lower = _binary_self_convolve(
            lower_exp, num_steps - 1, tail_truncation / 3.0, bound_type
        )
        composed = _geometric_convolve(
            convolved_lower, upper_exp, tail_truncation / 3.0, bound_type
        )

    return _log_geometric_to_linear(composed)


def _log_mean_exp_add(
    add_loss_factor: _LinearDiscreteDist,
    num_steps: int,
    tail_truncation: float,
    bound_type: _BoundType,
):
    """Build the add-side per-round PLD from the same `RA` mixture rule.

    Every selected step contributes the same add-side one-step factor, so this
    is the add-direction analog of `_log_mean_exp_remove`.
    """
    log_num_steps = float(np.log(num_steps))
    scaled_add = _LinearDiscreteDist(
        add_loss_factor.x_min - log_num_steps,
        add_loss_factor.x_gap,
        add_loss_factor.PMF_array.copy(),
        add_loss_factor.p_neg_inf,
        add_loss_factor.p_pos_inf,
    )
    exp_factor = _exp_linear_to_geometric(scaled_add)
    exp_bound_type = (
        _BoundType.IS_DOMINATED
        if bound_type == _BoundType.DOMINATES
        else _BoundType.DOMINATES
    )

    if num_steps == 1:
        conv = exp_factor
    else:
        conv = _binary_self_convolve(
            exp_factor, num_steps, tail_truncation / 2.0, exp_bound_type
        )

    return _negate_reverse_linear_distribution(_log_geometric_to_linear(conv))


def _allocation_pmf_remove_from_realization(
    realization: _PLDRealization,
    num_steps_per_round: int,
    config: RandomAllocationGaussianRuntimeConfig,
    bound_type: _BoundType,
) -> _LinearDiscreteDist:
    with _timed(
        f"allocation_remove_pmf num_steps_per_round={num_steps_per_round} bound={bound_type.value}"
    ):
        dual_realization = _calc_pld_dual(realization)

        return _allocation_pmf_remove_from_realization_with_dual(
            realization=realization,
            dual_realization=dual_realization,
            num_steps_per_round=num_steps_per_round,
            config=config,
            bound_type=bound_type,
        )


def _allocation_pmf_remove_from_realization_with_dual(
    *,
    realization: _PLDRealization,
    dual_realization: _LinearDiscreteDist,
    num_steps_per_round: int,
    config: RandomAllocationGaussianRuntimeConfig,
    bound_type: _BoundType,
) -> _LinearDiscreteDist:
    with _timed(
        f"allocation_remove_pmf_with_dual num_steps_per_round={num_steps_per_round} bound={bound_type.value}"
    ):
        neg_dual = _negate_reverse_linear_distribution(dual_realization)
        upper_linear: _LinearDiscreteDist = realization
        if upper_linear.x_gap < config.loss_discretization:
            upper_linear = _change_spacing_type(
                upper_linear,
                config.tail_truncation,
                config.loss_discretization,
                _SpacingType.LINEAR,
                bound_type,
            )

        dual_linear: _LinearDiscreteDist = neg_dual
        if dual_linear.x_gap < config.loss_discretization:
            dual_linear = _change_spacing_type(
                dual_linear,
                config.tail_truncation,
                config.loss_discretization,
                _SpacingType.LINEAR,
                bound_type,
            )

        composed = _log_mean_exp_remove(
            dual_linear,
            upper_linear,
            num_steps_per_round,
            config.tail_truncation,
            bound_type,
        )
        result = _change_spacing_type(
            composed,
            config.tail_truncation,
            config.loss_discretization,
            _SpacingType.LINEAR,
            bound_type,
        )

        _debug_timing(
            "allocation_remove_pmf result "
            f"bound={bound_type.value} {_dist_debug_summary(result)}"
        )

        return result


def _allocation_pmf_add_from_realization(
    realization: _PLDRealization,
    num_steps_per_round: int,
    config: RandomAllocationGaussianRuntimeConfig,
    bound_type: _BoundType,
) -> _LinearDiscreteDist:
    with _timed(
        f"allocation_add_pmf num_steps_per_round={num_steps_per_round} bound={bound_type.value}"
    ):
        exp_bound_type = (
            _BoundType.IS_DOMINATED
            if bound_type == _BoundType.DOMINATES
            else _BoundType.DOMINATES
        )
        add_realization = _negate_reverse_linear_distribution(realization)
        add_linear: _LinearDiscreteDist = add_realization

        if add_linear.x_gap < config.loss_discretization:
            add_linear = _change_spacing_type(
                add_linear,
                config.tail_truncation,
                config.loss_discretization,
                _SpacingType.LINEAR,
                exp_bound_type,
            )

        composed = _log_mean_exp_add(
            add_linear, num_steps_per_round, config.tail_truncation, bound_type
        )
        result = _change_spacing_type(
            composed,
            config.tail_truncation,
            config.loss_discretization,
            _SpacingType.LINEAR,
            bound_type,
        )

        _debug_timing(
            "allocation_add_pmf result "
            f"bound={bound_type.value} {_dist_debug_summary(result)}"
        )

        return result


def _compose_linear_pmfs(
    round_dist: _LinearDiscreteDist,
    num_rounds: int,
    tail_truncation: float,
    bound_type: _BoundType,
    convolution_method: str = "direct",
) -> _LinearDiscreteDist:
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    if num_rounds == 1:
        return round_dist

    if str(convolution_method) == "fft":
        with _timed(
            f"compose_linear_pmfs_fft num_rounds={num_rounds} bound={bound_type.value}"
        ):
            result = _binary_self_convolve_linear(
                round_dist,
                num_rounds,
                tail_truncation,
                bound_type,
            )
            _debug_timing(
                "compose_linear_pmfs_fft result "
                f"bound={bound_type.value} {_dist_debug_summary(result)}"
            )
            return result

    with _timed(
        f"compose_linear_pmfs num_rounds={num_rounds} bound={bound_type.value}"
    ):
        current = round_dist
        _debug_timing(
            "compose_linear_pmfs start "
            f"bound={bound_type.value} num_rounds={num_rounds} {_dist_debug_summary(current)}"
        )

        for idx in range(1, num_rounds):
            len_before = int(current.PMF_array.size)
            pmf = np.convolve(current.PMF_array, round_dist.PMF_array)
            len_after_conv = int(pmf.size)
            p_neg_inf, p_pos_inf = _convolve_infinite_masses(
                current.p_neg_inf,
                current.p_pos_inf,
                round_dist.p_neg_inf,
                round_dist.p_pos_inf,
            )
            pmf, p_neg_inf, p_pos_inf = _enforce_mass_conservation(
                pmf.astype(np.float64), p_neg_inf, p_pos_inf, bound_type
            )
            current = _LinearDiscreteDist(
                current.x_min + round_dist.x_min,
                round_dist.x_gap,
                pmf,
                p_neg_inf,
                p_pos_inf,
            )
            current = current.truncate_edges(tail_truncation, bound_type)

            _debug_timing(
                "compose_linear_pmfs iter "
                f"{idx}/{num_rounds - 1} bound={bound_type.value} "
                f"len_before={len_before} len_round={round_dist.PMF_array.size} "
                f"len_after_conv={len_after_conv} len_after_trunc={current.PMF_array.size} "
                f"p_neg_inf={current.p_neg_inf:.12g} p_pos_inf={current.p_pos_inf:.12g}"
            )
        _debug_timing(
            "compose_linear_pmfs result "
            f"bound={bound_type.value} {_dist_debug_summary(current)}"
        )

        return current


def _delta_from_linear_dist_for_epsilon(
    dist: _LinearDiscreteDist,
    epsilon: float,
) -> float:
    """Evaluate `δ(ε)` on a discrete privacy-loss grid.

    For a finite grid this computes

        δ(ε) = p(+∞) + ∑_{t > ε} PMF(t) · (1 - exp(ε - t)).
    """
    losses = dist.x_array
    probs = dist.PMF_array
    indices = losses > float(epsilon)
    return float(
        dist.p_pos_inf
        + np.dot(-np.expm1(float(epsilon) - losses[indices]), probs[indices])
    )


def _epsilon_from_linear_dist_for_delta(
    dist: _LinearDiscreteDist,
    delta: float,
) -> float:
    """Invert a discrete privacy-loss grid from `δ` back to `ε`.

    The helper first removes the `p(+∞)` atom by using
    `δ_residual = δ - p(+∞)`. It then performs the standard cumulative-tail
    inversion with

        d₁[i] = ∑_{j ≥ i} PMF[j],
        d₂[i] = ∑_{j ≥ i} PMF[j] · exp(-loss[j]).
    """
    if float(delta) <= dist.p_pos_inf:
        return math.inf

    dtype = np.longdouble
    losses = dist.x_array.astype(dtype)
    probs = dist.PMF_array.astype(dtype)
    delta_residual = dtype(float(delta) - float(dist.p_pos_inf))

    d1 = np.flip(np.cumsum(np.flip(probs), dtype=dtype))
    d2 = np.flip(np.cumsum(np.flip(probs * np.exp(-losses)), dtype=dtype))
    ndelta = np.exp(losses) * d2 - d1
    i = int(np.searchsorted(ndelta, -delta_residual, side="left"))

    if i >= len(losses):
        return math.inf

    while i > 0:
        d1_i = d1[i]
        d2_i = d2[i]
        if d1_i <= delta_residual:
            return 0.0

        if d2_i == 0.0:
            return max(0.0, float(losses[i]))

        epsilon = float(np.log((d1_i - delta_residual) / d2_i))
        if epsilon >= float(losses[i - 1]):
            break

        i -= 1

    d1_i = d1[i]
    d2_i = d2[i]
    if d1_i <= delta_residual:
        return 0.0

    if d2_i == 0.0:
        return max(0.0, float(losses[i]))

    return max(0.0, float(np.log((d1_i - delta_residual) / d2_i)))


def _epsilon_from_remove_add_pmfs_for_delta(
    remove_dist: _LinearDiscreteDist,
    add_dist: _LinearDiscreteDist,
    delta: float,
) -> float:
    """Return the two-sided `ε(δ)` query on final remove/add discrete PLDs.

    The production answer is `max(ε_remove(δ), ε_add(δ))` across the final
    remove-side and add-side PLDs.
    """
    epsilon_remove = _epsilon_from_linear_dist_for_delta(remove_dist, delta)
    epsilon_add = _epsilon_from_linear_dist_for_delta(add_dist, delta)
    return float(max(epsilon_remove, epsilon_add))


def _general_allocation_pmfs(
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: _PLDRealization,
    add_realization: _PLDRealization,
    config: RandomAllocationGaussianRuntimeConfig,
    bound_type: _BoundType,
    remove_dual_realization: _PLDRealization | None = None,
):
    with _timed(
        f"general_allocation_pmfs num_steps={num_steps} num_selected={num_selected} "
        f"num_epochs={num_epochs} bound={bound_type.value}"
    ):
        num_steps_per_round, num_rounds = _decompose_allocation_compositions(
            num_steps, num_selected, num_epochs
        )
        _debug_timing(
            "general_allocation_pmfs contract "
            f"bound={bound_type.value} num_steps_per_round={num_steps_per_round} num_rounds={num_rounds}"
        )
        if remove_dual_realization is None:
            remove_dist = _allocation_pmf_remove_from_realization(
                remove_realization,
                num_steps_per_round,
                config,
                bound_type,
            )
        else:
            remove_dist = _allocation_pmf_remove_from_realization_with_dual(
                realization=remove_realization,
                dual_realization=remove_dual_realization,
                num_steps_per_round=num_steps_per_round,
                config=config,
                bound_type=bound_type,
            )

        add_dist = _allocation_pmf_add_from_realization(
            add_realization, num_steps_per_round, config, bound_type
        )
        remove_final = _compose_linear_pmfs(
            remove_dist, num_rounds, config.tail_truncation, bound_type
        )
        add_final = _compose_linear_pmfs(
            add_dist, num_rounds, config.tail_truncation, bound_type
        )

        _debug_timing(
            "general_allocation_pmfs final "
            f"bound={bound_type.value} remove=({_dist_debug_summary(remove_final)}) "
            f"add=({_dist_debug_summary(add_final)})"
        )
        return remove_final, add_final


def _resolve_accountant_coeffs(
    *,
    accountant_coeffs: Sequence[float] | None = None,
    mechanism_state: dict[str, Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    state = mechanism_state if isinstance(mechanism_state, dict) else {}
    local_kwargs = kwargs if isinstance(kwargs, dict) else {}
    coeffs = local_kwargs.get(
        "random_allocation_accountant_coeffs",
        local_kwargs.get(
            "bnb_accountant_coeffs",
            (
                accountant_coeffs
                if accountant_coeffs is not None
                else state.get(
                    "random_allocation_accountant_coeffs",
                    state.get("bnb_accountant_coeffs", state.get("coeffs")),
                )
            ),
        ),
    )
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
        raise ValueError(
            "random_allocation accounting requires accountant-side coefficients via `bnb_accountant_coeffs` or `coeffs`"
        )

    resolved = tuple(float(c) for c in coeffs)
    l2 = math.sqrt(sum(c * c for c in resolved))
    if not math.isfinite(l2) or l2 <= 0.0:
        raise ValueError(
            "random_allocation accounting requires positive finite accountant coefficient norm"
        )

    return resolved


@dataclass(frozen=True)
class _ResolvedRandomAllocationShape:
    cycle_length: int
    num_selected: int
    horizon: int
    num_epochs: int
    reduced_num_steps_per_round: int
    reduced_num_rounds: int
    route: str


def _privacy_metadata_dict(sampling_semantics: Any | None) -> dict[str, Any]:
    if sampling_semantics is None or not hasattr(
        sampling_semantics, "privacy_metadata"
    ):
        return {}

    metadata = sampling_semantics.privacy_metadata

    return metadata if isinstance(metadata, dict) else {}


def _resolve_cycle_length(
    *,
    cycle_length: int | None,
    metadata: dict[str, Any],
    state: dict[str, Any],
    local_kwargs: dict[str, Any],
) -> int:
    resolved = local_kwargs.get(
        "bnb_cycle_length",
        (
            cycle_length
            if cycle_length is not None
            else metadata.get(
                "num_steps",
                metadata.get(
                    "bins", state.get("bnb_cycle_length", state.get("bnb_bins"))
                ),
            )
        ),
    )
    if resolved is None:
        raise ValueError(
            "random_allocation accounting requires `cycle_length`/`num_steps`/`bins` metadata"
        )

    resolved = int(resolved)
    if resolved < 1:
        raise ValueError("random_allocation accounting requires cycle_length >= 1")

    return resolved


def _resolve_num_selected(
    *,
    cycle_length: int,
    metadata: dict[str, Any],
    state: dict[str, Any],
    local_kwargs: dict[str, Any],
) -> int:
    resolved = local_kwargs.get(
        "pld_num_selected",
        local_kwargs.get(
            "num_selected",
            metadata.get("num_selected", state.get("pld_num_selected", 1)),
        ),
    )
    resolved = int(resolved)
    if resolved < 1:
        raise ValueError("random_allocation accounting requires num_selected >= 1")

    if resolved > cycle_length:
        raise ValueError(
            "random_allocation accounting requires num_steps >= num_selected"
        )

    return resolved


def _resolve_horizon(
    *,
    cycle_length: int,
    horizon: int | None,
    state: dict[str, Any],
    local_kwargs: dict[str, Any],
) -> int:
    c_matrix = local_kwargs.get("bnb_c_matrix", state.get("bnb_c_matrix"))
    resolved = local_kwargs.get("bnb_horizon", horizon)
    if resolved is None and c_matrix is not None and hasattr(c_matrix, "shape"):
        resolved = int(c_matrix.shape[1])

    if resolved is None:
        raise ValueError(
            "random_allocation accounting requires `horizon` or `bnb_c_matrix`"
        )

    resolved = int(resolved)
    if resolved < 1:
        raise ValueError("random_allocation accounting requires horizon >= 1")

    if resolved % cycle_length != 0:
        raise ValueError(
            "random_allocation accounting currently requires horizon divisible by cycle_length "
            f"(got horizon={resolved}, cycle_length={cycle_length})"
        )

    return resolved


def _resolve_noise_multiplier(noise_multiplier: float) -> float:
    sigma = float(noise_multiplier)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "random_allocation accounting requires positive finite noise_multiplier"
        )

    return sigma


def _resolve_random_allocation_shape(
    *,
    cycle_length: int,
    num_selected: int,
    horizon: int,
) -> _ResolvedRandomAllocationShape:
    num_epochs = horizon // cycle_length
    reduced_num_steps_per_round, reduced_num_rounds = (
        _decompose_allocation_compositions(
            cycle_length,
            num_selected,
            num_epochs,
        )
    )
    route = (
        "single_step_realization"
        if cycle_length == 1 and horizon == 1 and num_selected == 1
        else "multi_step_gaussian"
    )

    return _ResolvedRandomAllocationShape(
        cycle_length=cycle_length,
        num_selected=num_selected,
        horizon=horizon,
        num_epochs=num_epochs,
        reduced_num_steps_per_round=reduced_num_steps_per_round,
        reduced_num_rounds=reduced_num_rounds,
        route=route,
    )


def _resolve_public_exact_pair_driven_inputs(
    *,
    mechanism: str,
    accountant_coeffs: tuple[float, ...],
    noise_multiplier: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
):
    mechanism_name = str(mechanism)
    if mechanism_name not in _SUPPORTED_PUBLIC_EXACT_MECHANISMS:
        raise ValueError(
            "random_allocation public exact-law route does not support "
            f"mechanism={mechanism_name!r}; supported mechanisms are "
            f"{sorted(_SUPPORTED_PUBLIC_EXACT_MECHANISMS)!r}"
        )

    # Import lazily to avoid a module cycle with `initial_package.py`.
    from .exact_laws import build_realizable_gaussian_one_step_neighboring_pair
    from .initial_package import resolve_pair_driven_random_allocation_inputs

    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism=mechanism_name,
        forward_mean=np.asarray(accountant_coeffs, dtype=np.float64),
        reverse_mean=np.zeros(len(accountant_coeffs), dtype=np.float64),
        noise_multiplier=float(noise_multiplier),
    )
    return resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=int(num_steps),
        num_selected=int(num_selected),
        num_epochs=int(num_epochs),
    )


def resolve_random_allocation_accountant_inputs(
    *,
    mechanism: str,
    accountant_coeffs: Sequence[float] | None = None,
    cycle_length: int | None = None,
    horizon: int | None = None,
    noise_multiplier: float,
    mechanism_state: dict[str, Any] | None = None,
    sampling_semantics: Any | None = None,
    kwargs: dict[str, Any] | None = None,
) -> RandomAllocationAccountantInputs:
    """Resolve mechanism-facing state into repeated random-allocation inputs.

    This resolver validates the repeated `k`-out-of-`t` shape, computes the
    reduced repeated-accounting decomposition, and derives the effective
    Gaussian scale `σ / ||c||₂` from the provided coefficient vector.

    It consumes explicit arguments first and then falls back to
    `mechanism_state`, `sampling_semantics.privacy_metadata`, and `kwargs` for
    repeated-shape metadata such as cycle length, number selected, and horizon.

    Source: `RA`.

    Raises:
        ValueError: If the repeated shape or coefficient data is inconsistent.
    """

    state = mechanism_state if isinstance(mechanism_state, dict) else {}
    metadata = _privacy_metadata_dict(sampling_semantics)
    local_kwargs = kwargs if isinstance(kwargs, dict) else {}

    resolved_cycle_length = _resolve_cycle_length(
        cycle_length=cycle_length,
        metadata=metadata,
        state=state,
        local_kwargs=local_kwargs,
    )
    resolved_num_selected = _resolve_num_selected(
        cycle_length=resolved_cycle_length,
        metadata=metadata,
        state=state,
        local_kwargs=local_kwargs,
    )
    resolved_horizon = _resolve_horizon(
        cycle_length=resolved_cycle_length,
        horizon=horizon,
        state=state,
        local_kwargs=local_kwargs,
    )
    resolved_shape = _resolve_random_allocation_shape(
        cycle_length=resolved_cycle_length,
        num_selected=resolved_num_selected,
        horizon=resolved_horizon,
    )
    mechanism_name = str(mechanism)
    resolved_coeffs = _resolve_accountant_coeffs(
        accountant_coeffs=accountant_coeffs, mechanism_state=state, kwargs=local_kwargs
    )
    l2 = math.sqrt(sum(c * c for c in resolved_coeffs))

    sigma = _resolve_noise_multiplier(noise_multiplier)
    effective_sigma = sigma / l2
    pair_driven_inputs = _resolve_public_exact_pair_driven_inputs(
        mechanism=mechanism_name,
        accountant_coeffs=resolved_coeffs,
        noise_multiplier=sigma,
        num_steps=resolved_shape.cycle_length,
        num_selected=resolved_shape.num_selected,
        num_epochs=resolved_shape.num_epochs,
    )

    return RandomAllocationAccountantInputs(
        contract_kind="pld_accounting_random_allocation",
        package_alignment_kind="repeated_k_out_of_t",
        package_alignment_notes=(
            "Repeated k-out-of-t deterministic random-allocation route from `RA`; "
            "resolved through an exact common-covariance one-step neighboring law and the "
            "deterministic initial-package evaluator; "
            f"uses the {'direct k = 1 transform' if resolved_num_selected == 1 else 'reduced general k-out-of-t transform'} "
            "rather than the later fixed-bin balls-in-bins bridge."
        ),
        route="pair_driven_public_exact_initial_package",
        mechanism=mechanism_name,
        accountant_coeffs=resolved_coeffs,
        accountant_coeff_l2=l2,
        noise_multiplier=sigma,
        effective_sigma=effective_sigma,
        horizon=resolved_shape.horizon,
        cycle_length=resolved_shape.cycle_length,
        pld_num_steps=resolved_shape.cycle_length,
        pld_num_selected=resolved_shape.num_selected,
        pld_num_epochs=resolved_shape.num_epochs,
        reduced_num_steps_per_round=resolved_shape.reduced_num_steps_per_round,
        reduced_num_rounds=resolved_shape.reduced_num_rounds,
        exact_law_route=pair_driven_inputs.initial_package.exact_law_route,
        initial_package_route=pair_driven_inputs.initial_package.route,
        pair_driven_inputs=pair_driven_inputs,
    )


def resolve_random_allocation_gaussian_runtime_config(
    *,
    target_delta: float,
    runtime_policy: str | None = None,
    loss_discretization: float | None = None,
    tail_truncation: float | None = None,
    max_grid_fft: int | None = None,
    max_grid_mult: int | None = None,
    convolution_method: str | None = None,
) -> RandomAllocationGaussianRuntimeConfig:
    """Resolve the Gaussian PLD runtime policy for a target `δ`.

    The defaults favor the fast local repeated-accounting route, while allowing
    callers to override grid spacing, tail budget, grid ceilings, or the
    convolution backend.

    Source: `PLD`.

    The default values are:

    - `loss_discretization = 5e-2`
    - `tail_truncation = min(δ / 1000, 1e-8)`
    - `max_grid_fft = 1_000_000`
    - `max_grid_mult = 100_000`
    - `convolution_method = "fft"`
    """

    resolved_policy = (
        str(runtime_policy) if runtime_policy is not None else "repository_fast_random_allocation"
    )
    resolved_loss = (
        float(loss_discretization) if loss_discretization is not None else 5e-2
    )
    resolved_tail = (
        float(tail_truncation)
        if tail_truncation is not None
        else min(float(target_delta) / 1000.0, 1e-8)
    )
    resolved_fft = int(max_grid_fft) if max_grid_fft is not None else 1_000_000
    resolved_mult = int(max_grid_mult) if max_grid_mult is not None else 100_000
    resolved_conv = str(convolution_method) if convolution_method is not None else "fft"

    if resolved_policy == "strict_exact_package":
        policy_name = "strict_exact_package"
        clamp_to_package_grid = True
        refinement_rounds = 0
    elif resolved_policy == "efficient_staged_grid":
        policy_name = "efficient_staged_grid"
        clamp_to_package_grid = False
        refinement_rounds = 2
    elif resolved_policy == "repository_fast_random_allocation":
        policy_name = "repository_fast_random_allocation"
        clamp_to_package_grid = False
        refinement_rounds = 0
    else:
        raise ValueError(
            "random_allocation runtime_policy must be one of "
            "['strict_exact_package', 'efficient_staged_grid', 'repository_fast_random_allocation']"
        )

    return RandomAllocationGaussianRuntimeConfig(
        policy_name=policy_name,
        runtime_policy=resolved_policy,
        loss_discretization=resolved_loss,
        tail_truncation=resolved_tail,
        max_grid_fft=resolved_fft,
        max_grid_mult=resolved_mult,
        convolution_method=resolved_conv,
        matches_package_defaults=(
            math.isclose(resolved_loss, 1e-2)
            and math.isclose(resolved_tail, 1e-12)
            and resolved_conv == "geometric"
            and resolved_fft == 1_000_000
            and resolved_mult == -1
        ),
        clamp_to_package_grid=clamp_to_package_grid,
        remove_convolution_method="fft",
        add_convolution_method="geometric",
        refinement_rounds=refinement_rounds,
    )


@lru_cache(maxsize=None)
def build_gaussian_random_allocation_realization(
    sigma: float,
    *,
    x_gap: float = 5e-3,
    num_std: float = 8.0,
) -> _PLDRealization:
    """Build the one-step Gaussian PLD realization used by repeated accounting.

    The result is the analytic common-covariance Gaussian privacy-loss law on a
    linear grid, truncated to a finite window determined by `num_std` and
    `x_gap`.

    Source: `PLD`.
    """

    with _timed(f"build_gaussian_realization sigma={sigma:.12g} x_gap={x_gap:.12g}"):
        sigma_loss = 1.0 / float(sigma)
        mean = 1.0 / (2.0 * float(sigma) * float(sigma))
        raw_min = mean - num_std * sigma_loss
        raw_max = mean + num_std * sigma_loss
        lower_index = int(math.floor(raw_min / x_gap))
        upper_index = int(math.ceil(raw_max / x_gap))
        x_array = x_gap * np.arange(lower_index, upper_index + 1, dtype=np.float64)
        probabilities = stats.norm.pdf(x_array, loc=mean, scale=sigma_loss) * x_gap
        probabilities = probabilities / float(np.sum(probabilities, dtype=np.float64))
        realization = _PLDRealization(
            float(x_array[0]), float(x_gap), probabilities, 0.0, 0.0
        )
        _debug_timing(
            "build_gaussian_realization result "
            f"sigma={sigma:.12g} len={realization.PMF_array.size} "
            f"x_min={realization.x_min:.12g} x_gap={realization.x_gap:.12g}"
        )

        return realization


def _derive_repeated_runtime_stages(
    *,
    inputs: RandomAllocationAccountantInputs,
    runtime: RandomAllocationGaussianRuntimeConfig,
) -> _RepeatedRuntimeStages:
    if runtime.runtime_policy != "efficient_staged_grid":
        loss = float(runtime.loss_discretization)
        tail = float(runtime.tail_truncation)
        return _RepeatedRuntimeStages(
            output_loss_discretization=loss,
            pre_composition_loss_discretization=loss,
            inner_loss_discretization=loss,
            output_tail_truncation=tail,
            pre_composition_tail_truncation=tail,
            inner_tail_truncation=tail,
            remove_convolution_method=str(runtime.remove_convolution_method),
            add_convolution_method=str(runtime.add_convolution_method),
            refinement_rounds=int(runtime.refinement_rounds),
        )

    num_rounds = max(1, int(inputs.reduced_num_rounds))
    num_steps_per_round = max(1, int(inputs.reduced_num_steps_per_round))
    output_loss = float(runtime.loss_discretization)
    output_tail = float(runtime.tail_truncation)
    pre_loss = output_loss / math.sqrt(float(num_rounds))
    inner_divisor = 2.0 * math.ceil(math.log2(float(num_steps_per_round))) + 1.0
    inner_loss = pre_loss / inner_divisor
    pre_tail = output_tail / float(num_rounds)
    inner_tail = max(
        pre_tail / float(num_steps_per_round),
        float(np.finfo(float).eps * 1e-10),
    )
    return _RepeatedRuntimeStages(
        output_loss_discretization=output_loss,
        pre_composition_loss_discretization=pre_loss,
        inner_loss_discretization=inner_loss,
        output_tail_truncation=output_tail,
        pre_composition_tail_truncation=pre_tail,
        inner_tail_truncation=inner_tail,
        remove_convolution_method=str(runtime.remove_convolution_method),
        add_convolution_method=str(runtime.add_convolution_method),
        refinement_rounds=int(runtime.refinement_rounds),
    )


def estimate_epsilon_random_allocation(
    *,
    inputs: RandomAllocationAccountantInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> float:
    """Compute the dominating repeated random-allocation `ε(δ)` value.

    This is the production upper-bound query for the repeated accountant in
    this module.

    Source: `PLD`.
    """

    if inputs.pair_driven_inputs is not None:
        from .initial_package import estimate_epsilon_random_allocation_from_initial_package

        return estimate_epsilon_random_allocation_from_initial_package(
            inputs=inputs.pair_driven_inputs,
            target_delta=target_delta,
            runtime_config=runtime_config,
        )

    return _estimate_epsilon_random_allocation_bound(
        inputs=inputs,
        target_delta=target_delta,
        runtime_config=runtime_config,
        bound_type=_BoundType.DOMINATES,
    )


def _estimate_epsilon_random_allocation_bound(
    *,
    inputs: RandomAllocationAccountantInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None,
    bound_type: _BoundType,
    package_remove_realization: _PLDRealization | None = None,
    package_add_realization: _PLDRealization | None = None,
    package_remove_dual_realization: _PLDRealization | None = None,
) -> float:
    with _timed(
        f"estimate_epsilon_random_allocation_bound mechanism={inputs.mechanism} "
        f"route={inputs.route} bound={bound_type.value}"
    ):
        runtime = runtime_config or resolve_random_allocation_gaussian_runtime_config(
            target_delta=target_delta
        )
        _debug_timing(
            "random_allocation bound context "
            f"mechanism={inputs.mechanism} route={inputs.route} "
            f"effective_sigma={inputs.effective_sigma:.12g} "
            f"num_steps={inputs.pld_num_steps} num_selected={inputs.pld_num_selected} "
            f"num_epochs={inputs.pld_num_epochs} reduced_num_steps_per_round={inputs.reduced_num_steps_per_round} "
            f"reduced_num_rounds={inputs.reduced_num_rounds} "
            f"loss_discretization={runtime.loss_discretization:.12g} "
            f"tail_truncation={runtime.tail_truncation:.12g} "
            f"convolution_method={runtime.convolution_method}"
        )
        remove_realization = (
            package_remove_realization
            if package_remove_realization is not None
            else build_gaussian_random_allocation_realization(
                inputs.effective_sigma, x_gap=runtime.loss_discretization
            )
        )
        add_realization = (
            package_add_realization
            if package_add_realization is not None
            else remove_realization
        )
        remove_final, add_final = _general_allocation_pmfs(
            num_steps=int(inputs.pld_num_steps),
            num_selected=int(inputs.pld_num_selected),
            num_epochs=int(inputs.pld_num_epochs),
            remove_realization=remove_realization,
            add_realization=add_realization,
            config=runtime,
            bound_type=bound_type,
            remove_dual_realization=package_remove_dual_realization,
        )

        with _timed(
            f"epsilon_query mechanism={inputs.mechanism} route={inputs.route} "
            f"bound={bound_type.value} target_delta={target_delta:.12g}"
        ):
            epsilon = _epsilon_from_remove_add_pmfs_for_delta(
                remove_final,
                add_final,
                float(target_delta),
            )

        _debug_timing(
            "epsilon_query result "
            f"mechanism={inputs.mechanism} route={inputs.route} bound={bound_type.value} "
            f"epsilon={epsilon!r}"
        )

        return epsilon


def estimate_epsilon_range_random_allocation(
    *,
    inputs: RandomAllocationAccountantInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> tuple[float, float]:
    """Compute `(ε_upper, ε_lower)` for repeated random allocation.

    The first component comes from the dominating PLD route and the second from
    the dominated route, giving a conservative interval at the requested `δ`.

    Source: `PLD`.
    """

    runtime = runtime_config or resolve_random_allocation_gaussian_runtime_config(
        target_delta=target_delta
    )
    if inputs.pair_driven_inputs is not None:
        from .initial_package import (
            estimate_epsilon_range_random_allocation_from_initial_package,
        )

        return estimate_epsilon_range_random_allocation_from_initial_package(
            inputs=inputs.pair_driven_inputs,
            target_delta=target_delta,
            runtime_config=runtime,
        )

    upper = _estimate_epsilon_random_allocation_bound(
        inputs=inputs,
        target_delta=target_delta,
        runtime_config=runtime,
        bound_type=_BoundType.DOMINATES,
    )
    lower = _estimate_epsilon_random_allocation_bound(
        inputs=inputs,
        target_delta=target_delta,
        runtime_config=runtime,
        bound_type=_BoundType.IS_DOMINATED,
    )
    if lower > upper and not math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"random_allocation interval endpoints are inverted: lower={lower}, upper={upper}"
        )

    if lower > upper:
        lower = upper

    return float(upper), float(lower)
