"""Deterministic initial-package assembly above exact random-allocation laws.

This module turns exact neighboring-pair inputs into deterministic package
objects that the local random-allocation runtime can evaluate. It also builds
ambient quantitative-window metadata for finite Gaussian mixtures and materializes
those windows into conservative remove/add PLD realizations when possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .accountant import (
    RandomAllocationGaussianRuntimeConfig,
    _BoundType,
    _change_spacing_type,
    _compose_linear_pmfs,
    _derive_repeated_runtime_stages,
    _calc_pld_dual,
    _estimate_epsilon_random_allocation_bound,
    _epsilon_from_remove_add_pmfs_for_delta,
    _allocation_pmf_add_from_realization,
    _allocation_pmf_remove_from_realization_with_dual,
    _rescale_pmf_to_exp_neg_loss_moment_at_most_one,
    _PLDRealization,
    _SpacingType,
    build_gaussian_random_allocation_realization,
    resolve_random_allocation_gaussian_runtime_config,
)
from .exact_laws import (
    ExactLawMetadata,
    FiniteGaussianMixtureNeighboringPair,
    RealizableGaussianOneStepNeighboringPair,
)


__all__ = [
    "build_ambient_quantitative_window_package_from_exact_law",
    "build_deterministic_initial_package_from_ambient_quantitative_window",
    "build_deterministic_initial_package_from_exact_law",
    "resolve_pair_driven_random_allocation_inputs",
    "resolve_pair_driven_ambient_quantitative_window_inputs",
    "estimate_epsilon_random_allocation_from_initial_package",
    "estimate_epsilon_range_random_allocation_from_initial_package",
]


@dataclass(frozen=True)
class _DeterministicInitialDirection:
    """One direction of a deterministic initial package.

    This is the minimal pairing of direction label, source-law label, and the
    PLD realization used downstream by the package-backed evaluator.
    """

    direction: str
    source_law_kind: str
    realization: _PLDRealization


@dataclass(frozen=True)
class _DeterministicInitialBounds:
    """Initial `(α, β)` metadata carried with a deterministic package."""

    alpha: float
    beta: float
    has_initial_bounds: bool
    justification: str


@dataclass(frozen=True)
class _RandomAllocationDeterministicInitialPackage:
    """Deterministic package consumed by the pair-driven evaluator.

    It bundles the remove, remove-dual, add, and optional lower-envelope
    realizations together with exact-law routing and initial bound metadata.
    """

    layer_kind: str
    route: str
    exact_law_route: str
    mechanism: str
    theorem_alignment: tuple[str, ...]
    remove: _DeterministicInitialDirection
    remove_dual: _DeterministicInitialDirection
    add: _DeterministicInitialDirection
    bounds: _DeterministicInitialBounds
    remove_lower: _DeterministicInitialDirection | None = None
    add_lower: _DeterministicInitialDirection | None = None


@dataclass(frozen=True)
class _DeterministicWitnessFamilyMember:
    """One weighted source-law member in the exact witness-family detour."""

    source_index: int
    weight: float
    source_law_kind: str
    forward_mode: tuple[float, ...]
    remove: _DeterministicInitialDirection
    add: _DeterministicInitialDirection


@dataclass(frozen=True)
class _RandomAllocationDeterministicWitnessFamilyPackage:
    """Weighted family of exact Gaussian source realizations for one mixture."""

    layer_kind: str
    route: str
    exact_law_route: str
    mechanism: str
    theorem_alignment: tuple[str, ...]
    centered_reference_mean: tuple[float, ...]
    family: tuple[_DeterministicWitnessFamilyMember, ...]
    bounds: _DeterministicInitialBounds


@dataclass(frozen=True)
class _ExactFamilySourceLawMember:
    """One exact Gaussian source-law member in the family-level contract."""

    source_index: int
    weight: float
    source_law_kind: str
    forward_mean: tuple[float, ...]
    reference_mean: tuple[float, ...]
    covariance_kind: str
    effective_sigma: float


@dataclass(frozen=True)
class _RandomAllocationExactFamilyAccountantContract:
    """Family-level exact accountant contract below the round-pair lift."""

    layer_kind: str
    route: str
    exact_law_route: str
    mechanism: str
    theorem_alignment: tuple[str, ...]
    centered_reference_mean: tuple[float, ...]
    family: tuple[_ExactFamilySourceLawMember, ...]
    witness_family_route: str
    bounds: _DeterministicInitialBounds


@dataclass(frozen=True)
class _PairDrivenRandomAllocationInputs:
    """Resolved pair-driven evaluator inputs above a deterministic package."""

    route: str
    mechanism: str
    num_steps: int
    num_selected: int
    num_epochs: int
    initial_package: _RandomAllocationDeterministicInitialPackage


@dataclass(frozen=True)
class _PairDrivenExactFamilyRandomAllocationInputs:
    """Resolved family-contract inputs before round-pair lowering."""

    route: str
    mechanism: str
    num_steps: int
    num_selected: int
    num_epochs: int
    exact_family_contract: _RandomAllocationExactFamilyAccountantContract


@dataclass(frozen=True)
class _ExactFamilyRoundPairInputs:
    """Exact family contract lifted to one round-level finite-mixture pair."""

    route: str
    mechanism: str
    num_steps_per_round: int
    num_rounds: int
    exact_family_contract: _RandomAllocationExactFamilyAccountantContract
    round_pair: FiniteGaussianMixtureNeighboringPair


@dataclass(frozen=True)
class _ExactFamilyRoundPairPackageInputs:
    """Round-pair package plus the pair-driven route derived from it."""

    route: str
    mechanism: str
    num_steps_per_round: int
    num_rounds: int
    exact_family_round_pair: _ExactFamilyRoundPairInputs
    pair_driven_inputs: _PairDrivenRandomAllocationInputs


@dataclass(frozen=True)
class _AmbientQuantitativeWindowPolicy:
    """Shared-grid quantitative window policy for an ambient finite mixture."""

    lower: float
    upper: float
    alpha: float
    beta: float
    per_event_tail: float
    grid_size: int
    forward_lower_budget: float
    forward_upper_budget: float
    reverse_lower_budget: float
    reverse_upper_budget: float


@dataclass(frozen=True)
class _RandomAllocationAmbientQuantitativeWindowPackage:
    """Ambient quantitative-window metadata before realization materialization."""

    layer_kind: str
    route: str
    exact_law_route: str
    mechanism: str
    theorem_alignment: tuple[str, ...]
    num_modes: int
    policy: _AmbientQuantitativeWindowPolicy
    bounds: _DeterministicInitialBounds


@dataclass(frozen=True)
class _PairDrivenAmbientQuantitativeWindowInputs:
    """Resolved ambient quantitative-window route metadata for one workload."""

    route: str
    mechanism: str
    num_steps: int
    num_selected: int
    num_epochs: int
    quantitative_window_package: _RandomAllocationAmbientQuantitativeWindowPackage


@dataclass(frozen=True)
class _AmbientQuantitativeWindowValidatedInputs:
    """Validated numeric inputs used to derive ambient window geometry."""

    reference_mean: np.ndarray
    forward_modes: np.ndarray
    weights: np.ndarray
    sigma: float
    runtime: RandomAllocationGaussianRuntimeConfig


@dataclass(frozen=True)
class _AmbientQuantitativeWindowGeometry:
    """Derived geometric terms behind the ambient quantitative window."""

    num_modes: int
    weights: np.ndarray
    log_weights: np.ndarray
    loss_std: np.ndarray
    forward_means: np.ndarray
    reverse_means: np.ndarray
    cross_means: np.ndarray
    cross_scales: np.ndarray
    per_event_tail: float
    lower: float
    upper: float
    alpha: float


@dataclass(frozen=True)
class _AmbientQuantitativeWindowBudgets:
    """Tail-budget summary used to define the ambient window's `β`."""

    forward_lower_budget: float
    forward_upper_budget: float
    reverse_lower_budget: float
    reverse_upper_budget: float
    beta: float


def _gaussian_effective_sigma_from_pair(
    pair: RealizableGaussianOneStepNeighboringPair,
) -> float:
    delta = np.asarray(pair.forward_mean, dtype=np.float64) - np.asarray(
        pair.reverse_mean, dtype=np.float64
    )
    l2 = float(np.linalg.norm(delta))
    if l2 <= 0.0:
        raise ValueError(
            "deterministic initial package requires distinct forward and reverse means"
        )

    return float(pair.noise_multiplier) / l2


def _gaussian_effective_sigma_from_mode(
    *,
    forward_mode: tuple[float, ...],
    centered_reference_mean: tuple[float, ...],
    noise_multiplier: float,
) -> float:
    delta = np.asarray(forward_mode, dtype=np.float64) - np.asarray(
        centered_reference_mean, dtype=np.float64
    )
    l2 = float(np.linalg.norm(delta))
    if l2 <= 0.0:
        raise NotImplementedError(
            "exact-family witness packages require non-degenerate forward modes against the centered reference law"
        )

    return float(noise_multiplier) / l2


def _degenerate_gaussian_effective_sigma_from_finite_mixture(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> float | None:
    forward_modes = np.asarray(pair.forward_modes, dtype=np.float64)
    if forward_modes.ndim != 2 or forward_modes.shape[0] < 1:
        return None

    reference_mean = np.asarray(pair.centered_reference_mean, dtype=np.float64)
    if reference_mean.shape != forward_modes.shape[1:]:
        return None

    if len(pair.reverse_modes) != 1:
        return None

    reverse_mode = np.asarray(pair.reverse_modes[0], dtype=np.float64)
    if reverse_mode.shape != reference_mean.shape or not np.allclose(
        reverse_mode, reference_mean
    ):
        return None

    if any(not np.allclose(mode, forward_modes[0]) for mode in forward_modes[1:]):
        return None

    delta = forward_modes[0] - reference_mean
    l2 = float(np.linalg.norm(delta))
    if l2 <= 0.0:
        return None

    return float(pair.noise_multiplier) / l2


def _orthogonal_equal_norm_gaussian_effective_sigma_from_finite_mixture(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> float | None:
    forward_modes = np.asarray(pair.forward_modes, dtype=np.float64)
    if forward_modes.ndim != 2 or forward_modes.shape[0] < 1:
        return None

    reference_mean = np.asarray(pair.centered_reference_mean, dtype=np.float64)
    if reference_mean.shape != forward_modes.shape[1:]:
        return None

    if len(pair.reverse_modes) != 1:
        return None

    reverse_mode = np.asarray(pair.reverse_modes[0], dtype=np.float64)
    if reverse_mode.shape != reference_mean.shape or not np.allclose(
        reverse_mode, reference_mean
    ):
        return None

    weights = np.asarray(pair.forward_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size != forward_modes.shape[0]:
        return None

    uniform = np.full(weights.shape, 1.0 / float(weights.size), dtype=np.float64)
    if not np.allclose(weights, uniform):
        return None

    centered_modes = forward_modes - reference_mean[None, :]
    gram = centered_modes @ centered_modes.T
    diag = np.diag(gram)
    if not np.all(diag > 0.0):
        return None

    if not np.allclose(diag, diag[0]):
        return None

    off_diag = gram - np.diag(diag)
    if not np.allclose(off_diag, 0.0):
        return None

    return float(pair.noise_multiplier) / math.sqrt(float(diag[0]))


def _resolve_ambient_quantitative_window_inputs(
    pair: FiniteGaussianMixtureNeighboringPair,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None,
) -> _AmbientQuantitativeWindowValidatedInputs:
    reference_mean = _resolve_ambient_quantitative_window_reference_mean(pair)
    sigma = _resolve_ambient_quantitative_window_sigma(pair)
    forward_modes = _resolve_ambient_quantitative_window_forward_modes(
        pair, reference_mean
    )
    weights = _resolve_ambient_quantitative_window_forward_weights(pair)
    runtime = runtime_config or resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5
    )

    return _AmbientQuantitativeWindowValidatedInputs(
        reference_mean=reference_mean,
        forward_modes=forward_modes,
        weights=weights,
        sigma=sigma,
        runtime=runtime,
    )


def _resolve_ambient_quantitative_window_reference_mean(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> np.ndarray:
    if pair.covariance_kind != "common_covariance_sigma_squared_identity":
        raise NotImplementedError(
            "ambient quantitative-window packages require common-covariance sigma-squared identity pairs"
        )

    if len(pair.forward_modes) != len(pair.forward_weights):
        raise ValueError(
            "finite Gaussian-mixture pair has mismatched forward mode and weight counts"
        )

    if len(pair.forward_modes) < 1:
        raise ValueError(
            "ambient quantitative-window packages require at least one forward mode"
        )

    if len(pair.reverse_modes) != 1:
        raise NotImplementedError(
            "ambient quantitative-window packages currently require a singleton reverse reference law"
        )

    reference_mean = np.asarray(pair.centered_reference_mean, dtype=np.float64)
    reverse_mode = np.asarray(pair.reverse_modes[0], dtype=np.float64)
    if reverse_mode.shape != reference_mean.shape or not np.allclose(
        reverse_mode, reference_mean
    ):
        raise NotImplementedError(
            "ambient quantitative-window packages currently require the reverse singleton to equal the centered reference mean"
        )

    return reference_mean


def _resolve_ambient_quantitative_window_sigma(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> float:
    sigma = float(pair.noise_multiplier)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "ambient quantitative-window packages require positive finite noise_multiplier"
        )

    return sigma


def _resolve_ambient_quantitative_window_forward_modes(
    pair: FiniteGaussianMixtureNeighboringPair,
    reference_mean: np.ndarray,
) -> np.ndarray:
    forward_modes = np.asarray(pair.forward_modes, dtype=np.float64)
    if forward_modes.ndim != 2 or forward_modes.shape[1] != reference_mean.size:
        raise ValueError("finite Gaussian-mixture pair has invalid forward mode shape")

    return forward_modes


def _resolve_ambient_quantitative_window_forward_weights(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> np.ndarray:
    weights = np.asarray(pair.forward_weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(
            "finite Gaussian-mixture pair requires finite nonnegative forward weights"
        )

    total_weight = float(np.sum(weights, dtype=np.float64))
    if total_weight <= 0.0:
        raise ValueError(
            "finite Gaussian-mixture pair requires positive total forward weight"
        )

    weights = weights / total_weight
    if np.any(weights <= 0.0):
        raise NotImplementedError(
            "ambient quantitative-window packages currently require strictly positive forward weights"
        )

    return weights


def _build_ambient_quantitative_window_geometry(
    inputs: _AmbientQuantitativeWindowValidatedInputs,
) -> _AmbientQuantitativeWindowGeometry:
    """Derive the shared-grid geometry for an ambient quantitative window.

    The output fixes one common linear grid for forward and reverse PLDs:
    lower endpoint, upper endpoint, step size `α`, and the mode-dependent
    Gaussian terms needed by the tail-budget builder.
    """
    deltas = inputs.forward_modes - inputs.reference_mean[None, :]
    norms = np.linalg.norm(deltas, axis=1)
    if np.any(norms <= 0.0):
        raise NotImplementedError(
            "ambient quantitative-window packages currently require non-degenerate forward modes"
        )

    num_modes = int(deltas.shape[0])
    per_event_tail = min(
        max(float(inputs.runtime.tail_truncation) / float(max(num_modes, 1)), 1e-300),
        0.25,
    )
    loss_std = norms / inputs.sigma
    forward_means = 0.5 * loss_std * loss_std
    reverse_means = -forward_means
    log_weights = np.log(inputs.weights)
    cross_means = (deltas @ deltas.T) / (inputs.sigma * inputs.sigma) - 0.5 * np.square(
        loss_std
    )[None, :]
    cross_scales = loss_std[None, :]

    lower_forward = (
        stats.norm.ppf(per_event_tail, loc=forward_means, scale=loss_std) + log_weights
    )
    lower_reverse = stats.norm.ppf(per_event_tail, loc=reverse_means, scale=loss_std)
    upper_forward = stats.norm.ppf(
        1.0 - per_event_tail, loc=cross_means, scale=cross_scales
    )
    upper_reverse = stats.norm.ppf(
        1.0 - per_event_tail, loc=reverse_means, scale=loss_std
    )

    lower = float(min(np.min(lower_forward), np.min(lower_reverse)))
    upper = float(max(np.max(upper_forward), np.max(upper_reverse)))
    alpha = float(inputs.runtime.loss_discretization)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            "ambient quantitative-window packages require positive finite loss_discretization"
        )
    if upper < lower:
        raise ValueError("ambient quantitative-window policy produced upper < lower")

    return _AmbientQuantitativeWindowGeometry(
        num_modes=num_modes,
        weights=inputs.weights,
        log_weights=log_weights,
        loss_std=loss_std,
        forward_means=forward_means,
        reverse_means=reverse_means,
        cross_means=cross_means,
        cross_scales=cross_scales,
        per_event_tail=per_event_tail,
        lower=lower,
        upper=upper,
        alpha=alpha,
    )


def _build_ambient_quantitative_window_budgets(
    geometry: _AmbientQuantitativeWindowGeometry,
) -> _AmbientQuantitativeWindowBudgets:
    """Compute the four tail budgets that define the ambient window.

    The returned budgets are the forward lower tail, forward upper cross-source
    tail, reverse lower tail, and reverse upper tail. The final `β` is the
    maximum of those four conservative contributions.
    """
    forward_lower_budget = float(
        np.sum(
            geometry.weights
            * stats.norm.cdf(
                geometry.lower - geometry.log_weights,
                loc=geometry.forward_means,
                scale=geometry.loss_std,
            ),
            dtype=np.float64,
        )
    )
    reverse_lower_budget = float(
        np.sum(
            stats.norm.cdf(
                geometry.lower, loc=geometry.reverse_means, scale=geometry.loss_std
            ),
            dtype=np.float64,
        )
    )
    reverse_upper_budget = float(
        np.sum(
            stats.norm.sf(
                geometry.upper, loc=geometry.reverse_means, scale=geometry.loss_std
            ),
            dtype=np.float64,
        )
    )
    forward_upper_budget = float(
        np.sum(
            geometry.weights[:, None]
            * stats.norm.sf(
                geometry.upper,
                loc=geometry.cross_means,
                scale=geometry.cross_scales,
            ),
            dtype=np.float64,
        )
    )
    beta = float(
        max(
            forward_lower_budget,
            forward_upper_budget,
            reverse_lower_budget,
            reverse_upper_budget,
        )
    )

    return _AmbientQuantitativeWindowBudgets(
        forward_lower_budget=forward_lower_budget,
        forward_upper_budget=forward_upper_budget,
        reverse_lower_budget=reverse_lower_budget,
        reverse_upper_budget=reverse_upper_budget,
        beta=beta,
    )


def _assemble_ambient_quantitative_window_package(
    *,
    pair: FiniteGaussianMixtureNeighboringPair,
    geometry: _AmbientQuantitativeWindowGeometry,
    budgets: _AmbientQuantitativeWindowBudgets,
) -> _RandomAllocationAmbientQuantitativeWindowPackage:
    grid_size = int(math.ceil((geometry.upper - geometry.lower) / geometry.alpha)) + 1
    policy = _AmbientQuantitativeWindowPolicy(
        lower=geometry.lower,
        upper=geometry.upper,
        alpha=geometry.alpha,
        beta=budgets.beta,
        per_event_tail=geometry.per_event_tail,
        grid_size=grid_size,
        forward_lower_budget=budgets.forward_lower_budget,
        forward_upper_budget=budgets.forward_upper_budget,
        reverse_lower_budget=budgets.reverse_lower_budget,
        reverse_upper_budget=budgets.reverse_upper_budget,
    )

    return _RandomAllocationAmbientQuantitativeWindowPackage(
        layer_kind="ambient_quantitative_window_policy_package",
        route="pair_driven_ambient_quantitative_window_policy",
        exact_law_route=pair.metadata.route,
        mechanism=pair.mechanism,
        theorem_alignment=(
            "AmbientFiniteGaussianMixtureQuantitativeTailBounds.lean",
            "AmbientFiniteGaussianMixtureComplementaryTailBounds.lean",
            "AmbientFiniteGaussianMixtureQuantitativeWindows.lean",
            "PLDRandomAllocationNumerics.lean",
        ),
        num_modes=geometry.num_modes,
        policy=policy,
        bounds=_DeterministicInitialBounds(
            alpha=policy.alpha,
            beta=policy.beta,
            has_initial_bounds=True,
            justification=(
                "ambient exact finite Gaussian-mixture pair exposes a shared quantitative truncation "
                "window policy and grid-size/runtime metadata, but this route does not yet construct "
                "an evaluable PLD realization package"
            ),
        ),
    )


def build_ambient_quantitative_window_package_from_exact_law(
    pair: FiniteGaussianMixtureNeighboringPair,
    *,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> _RandomAllocationAmbientQuantitativeWindowPackage:
    """Build a quantitative window policy from an exact finite-mixture law.

    The result is metadata only: shared endpoints, step size, grid size, and
    `(α, β)` bounds derived from the exact mixture pair. It is not yet an
    evaluable PLD realization package.

    Source: `PLD`.
    """
    validated = _resolve_ambient_quantitative_window_inputs(pair, runtime_config)
    geometry = _build_ambient_quantitative_window_geometry(validated)
    budgets = _build_ambient_quantitative_window_budgets(geometry)

    return _assemble_ambient_quantitative_window_package(
        pair=pair,
        geometry=geometry,
        budgets=budgets,
    )


def _ambient_quantitative_window_terms(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    reference_mean = np.asarray(pair.centered_reference_mean, dtype=np.float64)
    forward_modes = np.asarray(pair.forward_modes, dtype=np.float64)
    weights = np.asarray(pair.forward_weights, dtype=np.float64)
    total_weight = float(np.sum(weights, dtype=np.float64))
    weights = weights / total_weight
    deltas = forward_modes - reference_mean[None, :]
    loss_std = np.linalg.norm(deltas, axis=1) / float(pair.noise_multiplier)
    forward_means = 0.5 * loss_std * loss_std
    reverse_means = -forward_means
    log_weights = np.log(weights)
    cross_means = (deltas @ deltas.T) / (
        float(pair.noise_multiplier) ** 2
    ) - 0.5 * np.square(loss_std)[None, :]

    return (
        weights,
        log_weights,
        loss_std,
        forward_means,
        reverse_means,
        cross_means,
        deltas,
    )


def _ambient_quantitative_window_grid(
    policy: _AmbientQuantitativeWindowPolicy,
) -> np.ndarray:
    return policy.lower + policy.alpha * np.arange(
        int(policy.grid_size), dtype=np.float64
    )


def _build_dominating_realization_from_cdf_lower_bounds(
    *,
    x_array: np.ndarray,
    cdf_lower: np.ndarray,
) -> _PLDRealization:
    """Build a dominating realization from CDF lower bounds.

    A lower bound on the CDF yields conservative PMF increments on the shared
    grid, so the returned realization dominates the target law.
    """
    seed = np.concatenate(([0.0], np.maximum.accumulate(np.clip(cdf_lower, 0.0, 1.0))))
    pmf = np.diff(seed)
    pmf = _rescale_pmf_to_exp_neg_loss_moment_at_most_one(
        pmf,
        x_array,
        atol=1e-12,
    )

    p_loss_inf = float(max(0.0, 1.0 - float(np.sum(pmf, dtype=np.float64))))
    x_gap = float(x_array[1] - x_array[0]) if x_array.size > 1 else 1.0

    return _PLDRealization(
        x_min=float(x_array[0]),
        x_gap=x_gap,
        pmf=pmf,
        p_loss_inf=p_loss_inf,
        p_loss_neg_inf=0.0,
    )


def _build_dominated_realization_from_cdf_upper_bounds(
    *,
    x_array: np.ndarray,
    cdf_upper: np.ndarray,
) -> _PLDRealization:
    """Build a dominated realization from CDF upper bounds.

    An upper bound on the CDF yields dominated PMF increments on the shared
    grid, so the returned realization is conservative in the lower direction.
    """
    clipped = np.maximum.accumulate(np.clip(cdf_upper, 0.0, 1.0))
    seed = np.concatenate(([0.0], clipped))
    pmf = np.diff(seed)
    pmf = _rescale_pmf_to_exp_neg_loss_moment_at_most_one(
        pmf,
        x_array,
        atol=1e-12,
    )

    p_loss_inf = float(max(0.0, 1.0 - float(np.sum(pmf, dtype=np.float64))))
    x_gap = float(x_array[1] - x_array[0]) if x_array.size > 1 else 1.0

    return _PLDRealization(
        x_min=float(x_array[0]),
        x_gap=x_gap,
        pmf=pmf,
        p_loss_inf=p_loss_inf,
        p_loss_neg_inf=0.0,
    )


def build_deterministic_initial_package_from_ambient_quantitative_window(
    pair: FiniteGaussianMixtureNeighboringPair,
    *,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> _RandomAllocationDeterministicInitialPackage:
    """Materialize a deterministic package from ambient window bounds.

    This upgrades the ambient quantitative-window policy into explicit remove,
    remove-dual, add, and lower-envelope PLD realizations on one shared grid.
    The output is conservative and directly consumable by the pair-driven
    random-allocation evaluator.

    Source: `PLD`.
    """

    policy_package = build_ambient_quantitative_window_package_from_exact_law(
        pair,
        runtime_config=runtime_config,
    )
    policy = policy_package.policy
    x_array = _ambient_quantitative_window_grid(policy)
    (
        weights,
        log_weights,
        loss_std,
        forward_means,
        reverse_means,
        cross_means,
        _deltas,
    ) = _ambient_quantitative_window_terms(pair)

    forward_cdf_upper = np.sum(
        weights[None, :]
        * stats.norm.cdf(
            x_array[:, None] - log_weights[None, :],
            loc=forward_means[None, :],
            scale=loss_std[None, :],
        ),
        axis=1,
        dtype=np.float64,
    )
    reverse_cdf_upper = np.sum(
        stats.norm.cdf(
            x_array[:, None], loc=reverse_means[None, :], scale=loss_std[None, :]
        ),
        axis=1,
        dtype=np.float64,
    )
    reverse_tail_upper = np.sum(
        stats.norm.sf(
            x_array[:, None], loc=reverse_means[None, :], scale=loss_std[None, :]
        ),
        axis=1,
        dtype=np.float64,
    )
    forward_tail_upper = np.zeros_like(x_array)
    for source_index, weight in enumerate(weights):
        row_tail = np.sum(
            stats.norm.sf(
                x_array[:, None],
                loc=cross_means[source_index][None, :],
                scale=loss_std[None, :],
            ),
            axis=1,
            dtype=np.float64,
        )
        forward_tail_upper += float(weight) * row_tail

    forward_cdf_lower = np.maximum.accumulate(
        np.clip(1.0 - forward_tail_upper, 0.0, 1.0)
    )
    reverse_cdf_lower = np.maximum.accumulate(
        np.clip(1.0 - reverse_tail_upper, 0.0, 1.0)
    )
    forward_cdf_upper = np.maximum(
        forward_cdf_lower, np.maximum.accumulate(np.clip(forward_cdf_upper, 0.0, 1.0))
    )
    reverse_cdf_upper = np.maximum(
        reverse_cdf_lower, np.maximum.accumulate(np.clip(reverse_cdf_upper, 0.0, 1.0))
    )

    remove = _build_dominating_realization_from_cdf_lower_bounds(
        x_array=x_array,
        cdf_lower=forward_cdf_lower,
    )
    add = _build_dominating_realization_from_cdf_lower_bounds(
        x_array=x_array,
        cdf_lower=reverse_cdf_lower,
    )
    remove_lower = _build_dominated_realization_from_cdf_upper_bounds(
        x_array=x_array,
        cdf_upper=forward_cdf_upper,
    )
    add_lower = _build_dominated_realization_from_cdf_upper_bounds(
        x_array=x_array,
        cdf_upper=reverse_cdf_upper,
    )

    return _RandomAllocationDeterministicInitialPackage(
        layer_kind="deterministic_initial_accountant_package",
        route="pair_driven_ambient_quantitative_window_realization_package",
        exact_law_route=pair.metadata.route,
        mechanism=pair.mechanism,
        theorem_alignment=(
            "AmbientFiniteGaussianMixtureQuantitativeWindows.lean",
            "PLDRandomAllocationNumerics.lean",
            "ConcreteRandomAllocationInitialApprox",
        ),
        remove=_DeterministicInitialDirection(
            direction="remove",
            source_law_kind="forward",
            realization=remove,
        ),
        remove_dual=_DeterministicInitialDirection(
            direction="remove_dual",
            source_law_kind="reverse",
            realization=add,
        ),
        add=_DeterministicInitialDirection(
            direction="add",
            source_law_kind="reverse",
            realization=add,
        ),
        bounds=_DeterministicInitialBounds(
            alpha=policy.alpha,
            beta=policy.beta,
            has_initial_bounds=True,
            justification=(
                "ambient exact finite Gaussian-mixture pair lowered through the quantitative-window "
                "policy to a shared-grid deterministic realization package with explicit overflow masses"
            ),
        ),
        remove_lower=_DeterministicInitialDirection(
            direction="remove",
            source_law_kind="forward",
            realization=remove_lower,
        ),
        add_lower=_DeterministicInitialDirection(
            direction="add",
            source_law_kind="reverse",
            realization=add_lower,
        ),
    )


def build_deterministic_initial_package_from_exact_law(
    pair: (
        RealizableGaussianOneStepNeighboringPair | FiniteGaussianMixtureNeighboringPair
    ),
) -> _RandomAllocationDeterministicInitialPackage:
    """Resolve the strongest supported deterministic package for `pair`.

    Exact one-step Gaussian pairs lower directly to exact deterministic
    realizations. Exact finite Gaussian-mixture pairs first try special closed
    forms and then fall back to the ambient quantitative-window route.

    Source: `PLD`.
    """

    if isinstance(pair, FiniteGaussianMixtureNeighboringPair):
        degenerate_sigma = _degenerate_gaussian_effective_sigma_from_finite_mixture(
            pair
        )
        if degenerate_sigma is not None:
            remove = build_gaussian_random_allocation_realization(degenerate_sigma)
            justification = (
                "exact finite Gaussian-mixture pair collapses to an exact common-covariance Gaussian "
                "one-step pair with zero slack"
            )
        else:
            orthogonal_sigma = (
                _orthogonal_equal_norm_gaussian_effective_sigma_from_finite_mixture(
                    pair
                )
            )
            if orthogonal_sigma is not None:
                remove = build_gaussian_random_allocation_realization(orthogonal_sigma)
                justification = (
                    "exact finite Gaussian-mixture pair reduces to the exact random-allocation Gaussian "
                    "base realization on an orthogonal equal-norm one-hot mode family with zero slack"
                )
            else:
                if len(pair.centered_reference_mean) == 1:
                    remove = _build_finite_gaussian_mixture_realization_1d(pair)
                    justification = (
                        "exact finite Gaussian-mixture pair lowered to a 1-D privacy-loss realization "
                        "against the centered Gaussian reference law"
                    )
                else:
                    return build_deterministic_initial_package_from_ambient_quantitative_window(
                        pair
                    )
    else:
        effective_sigma = _gaussian_effective_sigma_from_pair(pair)
        remove = build_gaussian_random_allocation_realization(effective_sigma)
        justification = "exact common-covariance Gaussian pair lowers to an exact initial package with zero slack"

    remove_dual = _calc_pld_dual(remove)
    add = remove_dual

    return _RandomAllocationDeterministicInitialPackage(
        layer_kind="deterministic_initial_accountant_package",
        route="pair_driven_exact_initial_package",
        exact_law_route=pair.metadata.route,
        mechanism=pair.mechanism,
        theorem_alignment=(
            "PLDRandomAllocationNumerics.lean",
            "ConcreteRandomAllocationInitialApprox",
            "InitialBounds",
            "valid_of_pair_sources",
            "tight_of_pair_bounds",
        ),
        remove=_DeterministicInitialDirection(
            direction="remove",
            source_law_kind="forward",
            realization=remove,
        ),
        remove_dual=_DeterministicInitialDirection(
            direction="remove_dual",
            source_law_kind="reverse",
            realization=remove_dual,
        ),
        add=_DeterministicInitialDirection(
            direction="add",
            source_law_kind="reverse",
            realization=add,
        ),
        bounds=_DeterministicInitialBounds(
            alpha=0.0,
            beta=0.0,
            has_initial_bounds=True,
            justification=justification,
        ),
    )


def _build_deterministic_witness_family_package_from_exact_law(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> _RandomAllocationDeterministicWitnessFamilyPackage:
    reference_mean = tuple(float(v) for v in pair.centered_reference_mean)
    if len(pair.reverse_modes) != 1:
        raise NotImplementedError(
            "exact-family witness packages currently require a singleton reverse reference law"
        )

    reverse_mode = tuple(float(v) for v in pair.reverse_modes[0])
    if reverse_mode != reference_mean:
        raise NotImplementedError(
            "exact-family witness packages currently require the reverse singleton to equal the centered reference mean"
        )

    if len(pair.forward_modes) != len(pair.forward_weights):
        raise ValueError(
            "finite Gaussian-mixture pair has mismatched forward mode and weight counts"
        )

    members: list[_DeterministicWitnessFamilyMember] = []
    for idx, (mode, weight) in enumerate(zip(pair.forward_modes, pair.forward_weights)):
        effective_sigma = _gaussian_effective_sigma_from_mode(
            forward_mode=mode,
            centered_reference_mean=reference_mean,
            noise_multiplier=pair.noise_multiplier,
        )
        remove = build_gaussian_random_allocation_realization(effective_sigma)
        add = _calc_pld_dual(remove)
        members.append(
            _DeterministicWitnessFamilyMember(
                source_index=int(idx),
                weight=float(weight),
                source_law_kind="forward_component",
                forward_mode=tuple(float(v) for v in mode),
                remove=_DeterministicInitialDirection(
                    direction="remove",
                    source_law_kind="forward_component",
                    realization=remove,
                ),
                add=_DeterministicInitialDirection(
                    direction="add",
                    source_law_kind="reverse_reference",
                    realization=add,
                ),
            )
        )

    return _RandomAllocationDeterministicWitnessFamilyPackage(
        layer_kind="deterministic_exact_witness_family_package",
        route="pair_driven_exact_witness_family_package",
        exact_law_route=pair.metadata.route,
        mechanism=pair.mechanism,
        theorem_alignment=(
            "PoissonGaussianMixturePLD.lean",
            "ProductGaussianMixturePLD.lean",
            "AmbientFiniteGaussianMixtureInitialApprox.lean",
        ),
        centered_reference_mean=reference_mean,
        family=tuple(members),
        bounds=_DeterministicInitialBounds(
            alpha=0.0,
            beta=0.0,
            has_initial_bounds=True,
            justification=(
                "exact ambient finite Gaussian-mixture pair represented as a weighted family of exact "
                "common-covariance Gaussian source realizations against the shared centered reference law"
            ),
        ),
    )


def _build_exact_family_accountant_contract_from_exact_law(
    pair: FiniteGaussianMixtureNeighboringPair,
) -> _RandomAllocationExactFamilyAccountantContract:
    witness_family = _build_deterministic_witness_family_package_from_exact_law(pair)
    reference_mean = tuple(float(v) for v in pair.centered_reference_mean)
    family: list[_ExactFamilySourceLawMember] = []
    for member in witness_family.family:
        family.append(
            _ExactFamilySourceLawMember(
                source_index=int(member.source_index),
                weight=float(member.weight),
                source_law_kind="forward_component_gaussian_source_law",
                forward_mean=tuple(float(v) for v in member.forward_mode),
                reference_mean=reference_mean,
                covariance_kind=str(pair.covariance_kind),
                effective_sigma=_gaussian_effective_sigma_from_mode(
                    forward_mode=member.forward_mode,
                    centered_reference_mean=reference_mean,
                    noise_multiplier=pair.noise_multiplier,
                ),
            )
        )

    return _RandomAllocationExactFamilyAccountantContract(
        layer_kind="exact_family_deterministic_accountant_contract",
        route="pair_driven_exact_family_accountant_contract",
        exact_law_route=pair.metadata.route,
        mechanism=pair.mechanism,
        theorem_alignment=(
            "AmbientFiniteGaussianMixtureExactFamily.lean",
            "AmbientFiniteGaussianMixtureInitialApprox.lean",
            "PLDRandomAllocationNumerics.lean",
        ),
        centered_reference_mean=reference_mean,
        family=tuple(family),
        witness_family_route=witness_family.route,
        bounds=witness_family.bounds,
    )


def _noise_multiplier_from_exact_family_contract(
    contract: _RandomAllocationExactFamilyAccountantContract,
) -> float:
    if not contract.family:
        raise ValueError(
            "exact-family accountant contract requires at least one source-law member"
        )

    sigma: float | None = None
    reference = np.asarray(contract.centered_reference_mean, dtype=np.float64)
    for member in contract.family:
        delta = np.asarray(member.forward_mean, dtype=np.float64) - reference
        l2 = float(np.linalg.norm(delta))
        if l2 <= 0.0:
            raise ValueError(
                "exact-family accountant contract requires non-degenerate source-law members"
            )

        member_sigma = float(member.effective_sigma) * l2
        if not math.isfinite(member_sigma) or member_sigma <= 0.0:
            raise ValueError(
                "exact-family accountant contract requires positive finite reconstructed sigma"
            )

        if sigma is None:
            sigma = member_sigma

        elif not math.isclose(member_sigma, sigma, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "exact-family accountant contract requires a shared common-covariance sigma"
            )

    assert sigma is not None
    return float(sigma)


def _embed_round_mean(
    *,
    forward_mean: tuple[float, ...],
    reference_mean: tuple[float, ...],
    position: int,
    num_steps_per_round: int,
) -> tuple[float, ...]:
    dim = len(reference_mean)
    if len(forward_mean) != dim:
        raise ValueError(
            "forward_mean and reference_mean must have matching dimensions"
        )

    if int(num_steps_per_round) < 1:
        raise ValueError("num_steps_per_round must be >= 1")

    if int(position) < 0 or int(position) >= int(num_steps_per_round):
        raise ValueError("position must lie in [0, num_steps_per_round)")

    blocks = [tuple(reference_mean) for _ in range(int(num_steps_per_round))]
    blocks[int(position)] = tuple(forward_mean)

    return tuple(float(v) for block in blocks for v in block)


def _build_exact_family_round_pair_from_contract(
    *,
    contract: _RandomAllocationExactFamilyAccountantContract,
    num_steps_per_round: int,
) -> FiniteGaussianMixtureNeighboringPair:
    resolved_steps = int(num_steps_per_round)
    if resolved_steps < 1:
        raise ValueError("exact-family round pair requires num_steps_per_round >= 1")

    sigma = _noise_multiplier_from_exact_family_contract(contract)
    reference_mean = tuple(float(v) for v in contract.centered_reference_mean)
    round_reference_mean = tuple(
        float(v) for _ in range(resolved_steps) for v in reference_mean
    )
    forward_modes: list[tuple[float, ...]] = []
    forward_weights: list[float] = []
    for position in range(resolved_steps):
        for member in contract.family:
            forward_modes.append(
                _embed_round_mean(
                    forward_mean=member.forward_mean,
                    reference_mean=reference_mean,
                    position=position,
                    num_steps_per_round=resolved_steps,
                )
            )
            forward_weights.append(float(member.weight) / float(resolved_steps))
    total_weight = float(sum(forward_weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "exact-family round pair requires positive finite total weight"
        )

    normalized_weights = tuple(float(w / total_weight) for w in forward_weights)
    covariance_kind = contract.family[0].covariance_kind
    if any(member.covariance_kind != covariance_kind for member in contract.family):
        raise ValueError(
            "exact-family round pair requires a shared covariance kind across family members"
        )

    return FiniteGaussianMixtureNeighboringPair(
        metadata=ExactLawMetadata(
            layer_kind="exact_law_level_input",
            accountant_package_kind="not_initialized",
            sampler_bridge_kind="none",
            route="exact_family_random_allocation_round_pair",
            theorem_alignment=(
                "AmbientFiniteGaussianMixtureExactFamily.lean",
                "PLDRandomAllocation.lean",
                "PLDRandomAllocationNumerics.lean",
            ),
        ),
        mechanism=contract.mechanism,
        covariance_kind=covariance_kind,
        noise_multiplier=sigma,
        centered_reference_mean=round_reference_mean,
        forward_modes=tuple(forward_modes),
        forward_weights=normalized_weights,
        reverse_modes=(round_reference_mean,),
        reverse_weights=(1.0,),
    )


def _build_finite_gaussian_mixture_realization_1d(
    pair: FiniteGaussianMixtureNeighboringPair,
    *,
    loss_discretization: float = 5e-3,
    num_std: float = 12.0,
    y_points_per_sigma: int = 800,
) -> _PLDRealization:
    sigma = float(pair.noise_multiplier)
    if len(pair.centered_reference_mean) != 1:
        raise NotImplementedError(
            "finite-mixture lowering currently supports only 1-D controls; "
            f"got dim={len(pair.centered_reference_mean)} route={pair.metadata.route}"
        )

    means = np.asarray([mode[0] for mode in pair.forward_modes], dtype=np.float64)
    weights = np.asarray(pair.forward_weights, dtype=np.float64)
    if means.size == 0:
        raise ValueError("finite-mixture lowering requires at least one forward mode")

    reference_mean = float(pair.centered_reference_mean[0])
    y_min = min(float(np.min(means)), reference_mean) - num_std * sigma
    y_max = max(float(np.max(means)), reference_mean) + num_std * sigma
    num_points = max(
        int(np.ceil((y_max - y_min) / (sigma / y_points_per_sigma))) + 1, 20_001
    )
    y_grid = np.linspace(y_min, y_max, num_points, dtype=np.float64)
    dy = float(y_grid[1] - y_grid[0])
    forward_density = np.zeros_like(y_grid)
    for weight, mean in zip(weights, means):
        forward_density += float(weight) * _normal_pdf_1d(
            y_grid, mean=float(mean), sigma=sigma
        )

    reverse_density = _normal_pdf_1d(y_grid, mean=reference_mean, sigma=sigma)
    mask = forward_density > 0.0
    if not np.any(mask):
        raise ValueError(
            "finite-mixture lowering produced zero forward density on the integration grid"
        )

    loss_values = np.log(forward_density[mask]) - np.log(reverse_density[mask])
    masses = forward_density[mask] * dy
    masses = masses / float(np.sum(masses, dtype=np.float64))
    lower_index = int(np.floor(float(np.min(loss_values)) / loss_discretization))
    upper_index = int(np.ceil(float(np.max(loss_values)) / loss_discretization))
    if upper_index <= lower_index:
        upper_index = lower_index + 1

    pmf = np.zeros(upper_index - lower_index + 1, dtype=np.float64)
    # Push mass to the upper loss bin so the forward PLD discretization remains
    # pessimistic enough to satisfy E[exp(-L)] <= 1 after binning.
    bin_indices = (
        np.ceil(loss_values / loss_discretization).astype(np.int64) - lower_index
    )
    bin_indices = np.clip(bin_indices, 0, pmf.size - 1)
    np.add.at(pmf, bin_indices, masses)
    pmf = pmf / float(np.sum(pmf, dtype=np.float64))

    return _PLDRealization(
        x_min=float(lower_index * loss_discretization),
        x_gap=float(loss_discretization),
        pmf=pmf,
        p_loss_inf=0.0,
        p_loss_neg_inf=0.0,
    )


def _normal_pdf_1d(values: np.ndarray, *, mean: float, sigma: float) -> np.ndarray:
    centered = (values - mean) / sigma
    return np.exp(-0.5 * centered * centered) / (sigma * np.sqrt(2.0 * np.pi))


def resolve_pair_driven_random_allocation_inputs(
    *,
    pair: (
        RealizableGaussianOneStepNeighboringPair | FiniteGaussianMixtureNeighboringPair
    ),
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> _PairDrivenRandomAllocationInputs:
    """Resolve a pair-driven deterministic route from an exact law pair.

    The result records the repeated-accounting shape `(t, k, epochs)` together
    with the deterministic initial package built from `pair`.

    Source: `PLD`.

    Raises:
        ValueError: If the repeated shape is invalid.
    """

    if int(num_steps) < 1 or int(num_selected) < 1 or int(num_epochs) < 1:
        raise ValueError(
            "pair-driven random_allocation inputs require num_steps, num_selected, num_epochs >= 1"
        )

    if int(num_selected) > int(num_steps):
        raise ValueError(
            "pair-driven random_allocation inputs require num_steps >= num_selected"
        )

    return _PairDrivenRandomAllocationInputs(
        route="pair_driven_exact_initial_package",
        mechanism=pair.mechanism,
        num_steps=int(num_steps),
        num_selected=int(num_selected),
        num_epochs=int(num_epochs),
        initial_package=build_deterministic_initial_package_from_exact_law(pair),
    )


def _resolve_pair_driven_exact_family_random_allocation_inputs(
    *,
    pair: FiniteGaussianMixtureNeighboringPair,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> _PairDrivenExactFamilyRandomAllocationInputs:
    if int(num_steps) < 1 or int(num_selected) < 1 or int(num_epochs) < 1:
        raise ValueError(
            "pair-driven exact-family inputs require num_steps, num_selected, num_epochs >= 1"
        )

    if int(num_selected) > int(num_steps):
        raise ValueError(
            "pair-driven exact-family inputs require num_steps >= num_selected"
        )

    return _PairDrivenExactFamilyRandomAllocationInputs(
        route="pair_driven_exact_family_accountant_contract",
        mechanism=pair.mechanism,
        num_steps=int(num_steps),
        num_selected=int(num_selected),
        num_epochs=int(num_epochs),
        exact_family_contract=_build_exact_family_accountant_contract_from_exact_law(
            pair
        ),
    )


def resolve_pair_driven_ambient_quantitative_window_inputs(
    *,
    pair: FiniteGaussianMixtureNeighboringPair,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> _PairDrivenAmbientQuantitativeWindowInputs:
    """Resolve the direct ambient quantitative-window route for an exact law.

    This exposes the ambient metadata route even when the caller is not yet
    materializing a full deterministic realization package.

    Source: `PLD`.

    Raises:
        ValueError: If the repeated shape is invalid.
    """

    if int(num_steps) < 1 or int(num_selected) < 1 or int(num_epochs) < 1:
        raise ValueError(
            "pair-driven ambient quantitative-window inputs require num_steps, num_selected, num_epochs >= 1"
        )

    if int(num_selected) > int(num_steps):
        raise ValueError(
            "pair-driven ambient quantitative-window inputs require num_steps >= num_selected"
        )

    return _PairDrivenAmbientQuantitativeWindowInputs(
        route="pair_driven_ambient_quantitative_window_policy",
        mechanism=pair.mechanism,
        num_steps=int(num_steps),
        num_selected=int(num_selected),
        num_epochs=int(num_epochs),
        quantitative_window_package=build_ambient_quantitative_window_package_from_exact_law(
            pair,
            runtime_config=runtime_config,
        ),
    )


def _resolve_exact_family_round_pair_inputs(
    *,
    contract: _RandomAllocationExactFamilyAccountantContract,
    num_steps_per_round: int,
    num_rounds: int,
) -> _ExactFamilyRoundPairInputs:
    resolved_rounds = int(num_rounds)
    if resolved_rounds < 1:
        raise ValueError("exact-family round pair inputs require num_rounds >= 1")

    return _ExactFamilyRoundPairInputs(
        route="pair_driven_exact_family_round_pair",
        mechanism=contract.mechanism,
        num_steps_per_round=int(num_steps_per_round),
        num_rounds=resolved_rounds,
        exact_family_contract=contract,
        round_pair=_build_exact_family_round_pair_from_contract(
            contract=contract,
            num_steps_per_round=int(num_steps_per_round),
        ),
    )


def _resolve_exact_family_round_pair_package_inputs(
    *,
    contract: _RandomAllocationExactFamilyAccountantContract,
    num_steps_per_round: int,
    num_rounds: int,
) -> _ExactFamilyRoundPairPackageInputs:
    round_pair_inputs = _resolve_exact_family_round_pair_inputs(
        contract=contract,
        num_steps_per_round=num_steps_per_round,
        num_rounds=num_rounds,
    )
    pair_driven_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=round_pair_inputs.round_pair,
        num_steps=1,
        num_selected=1,
        num_epochs=round_pair_inputs.num_rounds,
    )

    return _ExactFamilyRoundPairPackageInputs(
        route="pair_driven_exact_family_round_pair_package",
        mechanism=contract.mechanism,
        num_steps_per_round=round_pair_inputs.num_steps_per_round,
        num_rounds=round_pair_inputs.num_rounds,
        exact_family_round_pair=round_pair_inputs,
        pair_driven_inputs=pair_driven_inputs,
    )


def estimate_epsilon_random_allocation_from_initial_package(
    *,
    inputs: _PairDrivenRandomAllocationInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> float:
    """Evaluate the dominating `ε(δ)` query from a deterministic package.

    This is the package-backed analog of `estimate_epsilon_random_allocation`
    and returns the conservative upper endpoint.

    Source: `PLD`.
    """

    return _estimate_epsilon_random_allocation_bound_from_package(
        inputs=inputs,
        target_delta=target_delta,
        runtime_config=runtime_config,
        bound_type=_BoundType.DOMINATES,
    )


def estimate_epsilon_range_random_allocation_from_initial_package(
    *,
    inputs: _PairDrivenRandomAllocationInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> tuple[float, float]:
    """Evaluate a conservative `(ε_upper, ε_lower)` interval from a package.

    The upper endpoint uses dominating realizations from the package. The lower
    endpoint uses dominated package realizations when available.

    Source: `PLD`.
    """

    runtime = runtime_config or resolve_random_allocation_gaussian_runtime_config(
        target_delta=target_delta
    )
    upper = _estimate_epsilon_random_allocation_bound_from_package(
        inputs=inputs,
        target_delta=target_delta,
        runtime_config=runtime,
        bound_type=_BoundType.DOMINATES,
    )
    lower = _estimate_epsilon_random_allocation_bound_from_package(
        inputs=inputs,
        target_delta=target_delta,
        runtime_config=runtime,
        bound_type=_BoundType.IS_DOMINATED,
    )
    if lower > upper:
        lower = upper

    return float(upper), float(lower)


def _estimate_epsilon_range_random_allocation_from_exact_family_round_pair_package(
    *,
    inputs: _ExactFamilyRoundPairPackageInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> tuple[float, float]:
    return estimate_epsilon_range_random_allocation_from_initial_package(
        inputs=inputs.pair_driven_inputs,
        target_delta=target_delta,
        runtime_config=runtime_config,
    )


def _estimate_epsilon_random_allocation_bound_from_package(
    *,
    inputs: _PairDrivenRandomAllocationInputs,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None,
    bound_type: _BoundType,
) -> float:
    @dataclass(frozen=True)
    class _PackageBackedInputs:
        route: str
        mechanism: str
        effective_sigma: float
        pld_num_steps: int
        pld_num_selected: int
        pld_num_epochs: int
        reduced_num_steps_per_round: int
        reduced_num_rounds: int

    runtime = runtime_config or resolve_random_allocation_gaussian_runtime_config(
        target_delta=target_delta
    )
    package = inputs.initial_package
    if bound_type == _BoundType.DOMINATES:
        package_remove = package.remove.realization
        package_remove_dual = package.remove_dual.realization
        package_add = package.add.realization
    else:
        package_remove = (
            package.remove_lower.realization
            if package.remove_lower is not None
            else package.remove.realization
        )
        package_remove_dual = (
            package.add_lower.realization
            if package.add_lower is not None
            else package.remove_dual.realization
        )
        package_add = (
            package.add_lower.realization
            if package.add_lower is not None
            else package.add.realization
        )

    if runtime.runtime_policy == "efficient_staged_grid":
        stages = _derive_repeated_runtime_stages(
            inputs=_PackageBackedInputs(
                route=f"{inputs.route}:{package.exact_law_route}",
                mechanism=inputs.mechanism,
                effective_sigma=1.0,
                pld_num_steps=int(inputs.num_steps),
                pld_num_selected=int(inputs.num_selected),
                pld_num_epochs=int(inputs.num_epochs),
                reduced_num_steps_per_round=int(inputs.num_steps // inputs.num_selected),
                reduced_num_rounds=int(inputs.num_selected * inputs.num_epochs),
            ),
            runtime=runtime,
        )
        inner_runtime = RandomAllocationGaussianRuntimeConfig(
            policy_name=f"{runtime.policy_name}:inner",
            runtime_policy=str(runtime.runtime_policy),
            loss_discretization=float(stages.inner_loss_discretization),
            tail_truncation=float(stages.inner_tail_truncation),
            max_grid_fft=int(runtime.max_grid_fft),
            max_grid_mult=int(runtime.max_grid_mult),
            convolution_method=str(stages.remove_convolution_method),
            matches_package_defaults=bool(runtime.matches_package_defaults),
            clamp_to_package_grid=False,
            remove_convolution_method=str(stages.remove_convolution_method),
            add_convolution_method=str(stages.add_convolution_method),
            refinement_rounds=int(stages.refinement_rounds),
        )
        package_remove_for_inner = package_remove
        if package_remove_for_inner.x_gap < inner_runtime.loss_discretization:
            package_remove_for_inner = _change_spacing_type(
                package_remove_for_inner,
                inner_runtime.tail_truncation,
                inner_runtime.loss_discretization,
                _SpacingType.LINEAR,
                bound_type,
            )
        package_remove_dual_for_inner = package_remove_dual
        if package_remove_dual_for_inner.x_gap < inner_runtime.loss_discretization:
            package_remove_dual_for_inner = _change_spacing_type(
                package_remove_dual_for_inner,
                inner_runtime.tail_truncation,
                inner_runtime.loss_discretization,
                _SpacingType.LINEAR,
                bound_type,
            )
        package_add_for_inner = package_add
        if package_add_for_inner.x_gap < inner_runtime.loss_discretization:
            package_add_for_inner = _change_spacing_type(
                package_add_for_inner,
                inner_runtime.tail_truncation,
                inner_runtime.loss_discretization,
                _SpacingType.LINEAR,
                bound_type,
            )

        remove_round = _allocation_pmf_remove_from_realization_with_dual(
            realization=package_remove_for_inner,
            dual_realization=package_remove_dual_for_inner,
            num_steps_per_round=int(inputs.num_steps // inputs.num_selected),
            config=inner_runtime,
            bound_type=bound_type,
        )
        add_runtime = RandomAllocationGaussianRuntimeConfig(
            policy_name=f"{runtime.policy_name}:inner_add",
            runtime_policy=str(runtime.runtime_policy),
            loss_discretization=float(stages.inner_loss_discretization),
            tail_truncation=float(stages.inner_tail_truncation),
            max_grid_fft=int(runtime.max_grid_fft),
            max_grid_mult=int(runtime.max_grid_mult),
            convolution_method=str(stages.add_convolution_method),
            matches_package_defaults=bool(runtime.matches_package_defaults),
            clamp_to_package_grid=False,
            remove_convolution_method=str(stages.remove_convolution_method),
            add_convolution_method=str(stages.add_convolution_method),
            refinement_rounds=int(stages.refinement_rounds),
        )
        add_round = _allocation_pmf_add_from_realization(
            package_add_for_inner,
            int(inputs.num_steps // inputs.num_selected),
            add_runtime,
            bound_type,
        )

        remove_round = _change_spacing_type(
            remove_round,
            float(stages.pre_composition_tail_truncation),
            float(stages.pre_composition_loss_discretization),
            _SpacingType.LINEAR,
            bound_type,
        )
        add_round = _change_spacing_type(
            add_round,
            float(stages.pre_composition_tail_truncation),
            float(stages.pre_composition_loss_discretization),
            _SpacingType.LINEAR,
            bound_type,
        )
        remove_final = _compose_linear_pmfs(
            remove_round,
            int(inputs.num_selected * inputs.num_epochs),
            float(stages.pre_composition_tail_truncation),
            bound_type,
            convolution_method=str(stages.remove_convolution_method),
        )
        add_final = _compose_linear_pmfs(
            add_round,
            int(inputs.num_selected * inputs.num_epochs),
            float(stages.pre_composition_tail_truncation),
            bound_type,
            convolution_method=(
                "direct"
                if str(stages.add_convolution_method) == "geometric"
                else str(stages.add_convolution_method)
            ),
        )
        remove_final = _change_spacing_type(
            remove_final,
            float(stages.output_tail_truncation),
            float(stages.output_loss_discretization),
            _SpacingType.LINEAR,
            bound_type,
        )
        add_final = _change_spacing_type(
            add_final,
            float(stages.output_tail_truncation),
            float(stages.output_loss_discretization),
            _SpacingType.LINEAR,
            bound_type,
        )
        return _epsilon_from_remove_add_pmfs_for_delta(
            remove_final,
            add_final,
            float(target_delta),
        )

    effective_loss_discretization = float(runtime.loss_discretization)
    if bool(runtime.clamp_to_package_grid):
        effective_loss_discretization = min(
            effective_loss_discretization,
            float(package_remove.x_gap),
            float(package_add.x_gap),
        )
    if effective_loss_discretization != float(runtime.loss_discretization):
        runtime = RandomAllocationGaussianRuntimeConfig(
            policy_name=str(runtime.policy_name),
            runtime_policy=str(runtime.runtime_policy),
            loss_discretization=effective_loss_discretization,
            tail_truncation=float(runtime.tail_truncation),
            max_grid_fft=int(runtime.max_grid_fft),
            max_grid_mult=int(runtime.max_grid_mult),
            convolution_method=str(runtime.convolution_method),
            matches_package_defaults=bool(runtime.matches_package_defaults),
            clamp_to_package_grid=bool(runtime.clamp_to_package_grid),
            remove_convolution_method=str(runtime.remove_convolution_method),
            add_convolution_method=str(runtime.add_convolution_method),
            refinement_rounds=int(runtime.refinement_rounds),
        )

    # Reuse the existing lower-level recurrence entry point but keep the public
    # repeated-k contract resolver out of this route entirely.
    dummy = _PackageBackedInputs(
        route=f"{inputs.route}:{package.exact_law_route}",
        mechanism=inputs.mechanism,
        effective_sigma=1.0,
        pld_num_steps=int(inputs.num_steps),
        pld_num_selected=int(inputs.num_selected),
        pld_num_epochs=int(inputs.num_epochs),
        reduced_num_steps_per_round=int(inputs.num_steps // inputs.num_selected),
        reduced_num_rounds=int(inputs.num_selected * inputs.num_epochs),
    )
    # The reused lower-level function ignores `effective_sigma` once supplied
    # with explicit remove/add realizations through the package-backed route.
    return _estimate_epsilon_random_allocation_bound(
        inputs=dummy,  # type: ignore[arg-type]
        target_delta=target_delta,
        runtime_config=runtime,
        bound_type=bound_type,
        package_remove_realization=package_remove,
        package_add_realization=package_add,
        package_remove_dual_realization=package_remove_dual,
    )
