"""Fixed-bin balls-in-bins bridge into deterministic random allocation.

This module owns the production fixed-bin bridge route used by the paper-facing
amplified workloads. It builds the exact fixed-bin Gaussian-mixture pair,
resolves the strongest deterministic package available above that pair, and
calibrates `σ` against the resulting deterministic random-allocation bounds.

See `random_allocation.__init__` for the package-level citation registry.

Traceability:
- `paper/balls-in-bins/mcaccounting.tex`
- `Mf/DP/BNBDeterministicRandomAllocationBridge.lean`
- `Mf/DP/BNBSamplerLink.lean`
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from opacus.accountants.utils import (
    MAX_NOISE_SEARCH_BINARY_STEPS,
    MAX_SIGMA,
    MIN_NOISE_SEARCH_SIGMA_INTERVAL,
    NoiseSearchConvergenceError,
)

from .accountant import (
    RandomAllocationGaussianRuntimeConfig,
    _allocation_pmf_add_from_realization,
    _allocation_pmf_remove_from_realization,
    _BoundType,
    _compose_linear_pmfs,
    resolve_random_allocation_gaussian_runtime_config,
)
from .exact_laws import ExactLawMetadata, FiniteGaussianMixtureNeighboringPair
from .initial_package import (
    _build_exact_family_accountant_contract_from_exact_law,
    _estimate_epsilon_range_random_allocation_from_exact_family_round_pair_package,
    _ExactFamilyRoundPairPackageInputs,
    _PairDrivenAmbientQuantitativeWindowInputs,
    _PairDrivenRandomAllocationInputs,
    _resolve_exact_family_round_pair_package_inputs,
    estimate_epsilon_range_random_allocation_from_initial_package,
    resolve_pair_driven_ambient_quantitative_window_inputs,
    resolve_pair_driven_random_allocation_inputs,
)


__all__ = [
    "FixedBinRandomAllocationBridgeInputs",
    "build_fixed_bin_exact_law_pair",
    "resolve_fixed_bin_random_allocation_bridge_inputs",
    "resolve_fixed_bin_random_allocation_bridge_runtime_config",
    "estimate_epsilon_range_fixed_bin_random_allocation",
    "get_noise_multiplier_fixed_bin_random_allocation",
]


@dataclass(frozen=True)
class FixedBinRandomAllocationBridgeInputs:
    """Resolved fixed-bin bridge metadata for the deterministic route.

    Source: `BSR`, `PLD`.

    Attributes:
        source_law_kind: Source-law family name for the bridge.
        accountant_engine_kind: Downstream accountant family selected by the
            bridge.
        route: Resolved bridge route name.
        mechanism: Mechanism name used for diagnostics.
        bins: Number of fixed bins in the balls-in-bins source law.
        epochs: Number of epochs induced by `horizon / bins`.
        horizon: Total repeated horizon represented by the workload.
        noise_multiplier: Gaussian noise multiplier `σ` attached to the exact pair.
        mode_family: Aggregated fixed-bin mode family used by the exact law.
        exact_law_route: Exact-law constructor route selected for the bridge.
        initial_package_route: Deterministic initial-package route, if one was
            resolved.
        pair_driven_route: Pair-driven evaluator route, if one was resolved.
    """

    source_law_kind: str
    accountant_engine_kind: str
    route: str
    mechanism: str
    bins: int
    epochs: int
    horizon: int
    noise_multiplier: float
    mode_family: tuple[tuple[float, ...], ...]
    exact_law_route: str
    initial_package_route: str | None
    pair_driven_route: str | None


@dataclass(frozen=True)
class _ResolvedFixedBinExactPackage:
    route: str
    exact_law_route: str
    initial_package_route: str | None
    pair_driven_route: str | None
    pair_inputs: _PairDrivenRandomAllocationInputs | None
    ambient_quantitative_window_inputs: (
        _PairDrivenAmbientQuantitativeWindowInputs | None
    )
    exact_family_round_pair_package_inputs: _ExactFamilyRoundPairPackageInputs | None
    blocker: str | None


@dataclass(frozen=True)
class _FixedBinGaussianMixturePair:
    forward_modes: tuple[tuple[float, ...], ...]
    reverse_modes: tuple[tuple[float, ...], ...]
    component_probs: tuple[float, ...]


@dataclass(frozen=True)
class _FixedBinAmbientNonfiniteUpperDiagnostic:
    route: str
    initial_package_route: str
    remove_round_pos_inf: float
    remove_final_pos_inf: float
    add_round_pos_inf: float
    add_final_pos_inf: float
    blocker: str


@dataclass
class _FixedBinNoiseSearchState:
    sigma_low: float = 0.0
    sigma_high: float = 10.0
    eps_high: float = float("inf")
    last_finite_sigma: float | None = None
    last_finite_epsilon: float | None = None
    last_nonfinite_sigma: float | None = None
    iterations: int = 0


def resolve_fixed_bin_random_allocation_bridge_runtime_config(
    *,
    target_delta: float,
    loss_discretization: float | None = None,
    tail_truncation: float | None = None,
    max_grid_fft: int | None = None,
    max_grid_mult: int | None = None,
    convolution_method: str | None = None,
) -> RandomAllocationGaussianRuntimeConfig:
    """Resolve the deterministic runtime policy for the fixed-bin bridge.

    The fixed-bin bridge intentionally uses a coarser default loss grid than
    the fine package-backed clamp kept for the public exact repeated route.
    Callers may still override any field explicitly.
    """

    base = resolve_random_allocation_gaussian_runtime_config(
        target_delta=target_delta,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        max_grid_fft=max_grid_fft,
        max_grid_mult=max_grid_mult,
        convolution_method=convolution_method,
    )
    if loss_discretization is not None:
        return base

    return RandomAllocationGaussianRuntimeConfig(
        policy_name="fixed_bin_bridge_candidate_grid_1e-2",
        runtime_policy=str(base.runtime_policy),
        loss_discretization=1e-2,
        tail_truncation=float(base.tail_truncation),
        max_grid_fft=int(base.max_grid_fft),
        max_grid_mult=int(base.max_grid_mult),
        convolution_method=str(base.convolution_method),
        matches_package_defaults=False,
        clamp_to_package_grid=bool(base.clamp_to_package_grid),
        remove_convolution_method=str(base.remove_convolution_method),
        add_convolution_method=str(base.add_convolution_method),
        refinement_rounds=int(base.refinement_rounds),
    )


def _aggregate_fixed_bin_mode_family(
    *, c_matrix: np.ndarray, bins: int
) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(c_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("c_matrix must be a 2-D array")

    horizon = int(matrix.shape[1])
    resolved_bins = int(bins)
    if resolved_bins < 1:
        raise ValueError("bins must be >= 1")

    if horizon % resolved_bins != 0:
        raise ValueError(
            f"fixed-bin bridge requires horizon divisible by bins (got horizon={horizon}, bins={resolved_bins})"
        )

    mode_family: list[tuple[float, ...]] = []
    absolute_matrix = np.abs(matrix)
    for bin_idx in range(resolved_bins):
        mode = np.sum(
            absolute_matrix[:, bin_idx::resolved_bins], axis=1, dtype=np.float64
        )
        mode_family.append(tuple(float(v) for v in mode))

    return tuple(mode_family)


def _build_fixed_bin_gaussian_mixture_pair(
    *,
    c_matrix: np.ndarray,
    bins: int,
) -> _FixedBinGaussianMixturePair:
    mode_family = _aggregate_fixed_bin_mode_family(c_matrix=c_matrix, bins=bins)
    if not mode_family:
        raise ValueError("fixed-bin bridge requires at least one mode")

    dim = len(mode_family[0])
    zero_mode = tuple(0.0 for _ in range(dim))
    probs = tuple(float(1.0 / len(mode_family)) for _ in range(len(mode_family)))

    return _FixedBinGaussianMixturePair(
        forward_modes=mode_family,
        reverse_modes=(zero_mode,),
        component_probs=probs,
    )


def build_fixed_bin_exact_law_pair(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    noise_multiplier: float,
) -> FiniteGaussianMixtureNeighboringPair:
    """Build the exact fixed-bin Gaussian-mixture pair for the bridge route.

    The forward/remove law is the exact fixed-bin mixture over aggregated mode
    placements. The reverse/add law is the centered single-mode Gaussian
    reference used by the deterministic bridge.

    Source: `BSR`.
    """

    sigma = float(noise_multiplier)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("fixed-bin bridge requires positive finite noise_multiplier")

    pair = _build_fixed_bin_gaussian_mixture_pair(c_matrix=c_matrix, bins=bins)
    dim = len(pair.forward_modes[0])

    return FiniteGaussianMixtureNeighboringPair(
        metadata=ExactLawMetadata(
            layer_kind="exact_law_level_input",
            accountant_package_kind="not_initialized",
            sampler_bridge_kind="fixed_bin_bridge",
            route="exact_fixed_bin_gaussian_mixture_pair",
            theorem_alignment=(
                "BNBDeterministicRandomAllocationBridge.lean",
                "BNBSamplerLink.lean",
                "mcaccounting.tex",
            ),
        ),
        mechanism=str(mechanism),
        covariance_kind="common_covariance_sigma_squared_identity",
        noise_multiplier=sigma,
        centered_reference_mean=tuple(0.0 for _ in range(dim)),
        forward_modes=pair.forward_modes,
        forward_weights=pair.component_probs,
        reverse_modes=pair.reverse_modes,
        reverse_weights=(1.0,),
    )


def _resolve_fixed_bin_exact_package(
    *,
    exact_pair: FiniteGaussianMixtureNeighboringPair,
    bins: int,
) -> _ResolvedFixedBinExactPackage:
    try:
        pair_inputs = resolve_pair_driven_random_allocation_inputs(
            pair=exact_pair,
            num_steps=int(bins),
            num_selected=1,
            num_epochs=1,
        )
        route = "fixed_bin_bridge_exact_pair_package"
        if (
            pair_inputs.initial_package.route
            == "pair_driven_ambient_quantitative_window_realization_package"
        ):
            route = "fixed_bin_bridge_ambient_quantitative_window_realization_package"

        return _ResolvedFixedBinExactPackage(
            route=route,
            exact_law_route=exact_pair.metadata.route,
            initial_package_route=pair_inputs.initial_package.route,
            pair_driven_route=pair_inputs.route,
            pair_inputs=pair_inputs,
            ambient_quantitative_window_inputs=None,
            exact_family_round_pair_package_inputs=None,
            blocker=None,
        )
    except NotImplementedError as direct_exc:
        try:
            quantitative_inputs = (
                resolve_pair_driven_ambient_quantitative_window_inputs(
                    pair=exact_pair,
                    num_steps=int(bins),
                    num_selected=1,
                    num_epochs=1,
                )
            )
            return _ResolvedFixedBinExactPackage(
                route="fixed_bin_bridge_ambient_quantitative_window_pending_realization",
                exact_law_route=exact_pair.metadata.route,
                initial_package_route=None,
                pair_driven_route=quantitative_inputs.route,
                pair_inputs=None,
                ambient_quantitative_window_inputs=quantitative_inputs,
                exact_family_round_pair_package_inputs=None,
                blocker=(
                    "direct exact-pair package blocker: "
                    f"{direct_exc}; ambient quantitative-window route is active but still lacks "
                    "an evaluable PLD realization package"
                ),
            )
        except NotImplementedError as quantitative_exc:
            quantitative_blocker = str(quantitative_exc)

        contract = _build_exact_family_accountant_contract_from_exact_law(exact_pair)
        try:
            round_pair_package_inputs = _resolve_exact_family_round_pair_package_inputs(
                contract=contract,
                num_steps_per_round=int(bins),
                num_rounds=1,
            )
            return _ResolvedFixedBinExactPackage(
                route="fixed_bin_bridge_exact_family_round_pair_package",
                exact_law_route=exact_pair.metadata.route,
                initial_package_route=round_pair_package_inputs.pair_driven_inputs.initial_package.route,
                pair_driven_route=round_pair_package_inputs.route,
                pair_inputs=None,
                ambient_quantitative_window_inputs=None,
                exact_family_round_pair_package_inputs=round_pair_package_inputs,
                blocker=None,
            )
        except NotImplementedError as round_exc:
            return _ResolvedFixedBinExactPackage(
                route="fixed_bin_bridge_exact_family_round_pair_pending_initial_package",
                exact_law_route=exact_pair.metadata.route,
                initial_package_route=None,
                pair_driven_route="pair_driven_exact_family_round_pair_package",
                pair_inputs=None,
                ambient_quantitative_window_inputs=None,
                exact_family_round_pair_package_inputs=None,
                blocker=(
                    "direct exact-pair package blocker: "
                    f"{direct_exc}; ambient quantitative-window blocker: {quantitative_blocker}; "
                    f"exact family round-pair package blocker: {round_exc}"
                ),
            )


def resolve_fixed_bin_random_allocation_bridge_inputs(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    noise_multiplier: float,
) -> FixedBinRandomAllocationBridgeInputs:
    """Resolve the fixed-bin bridge metadata for a live workload.

    This resolver validates the fixed-bin shape, builds the exact bridge pair,
    and records which deterministic package and evaluator routes are available
    above that pair.

    Source: `BSR`, `PLD`.

    Raises:
        ValueError: If `c_matrix` is not 2-D or the horizon is not divisible by
            `bins`.
    """

    matrix = np.asarray(c_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("c_matrix must be a 2-D array")

    horizon = int(matrix.shape[1])
    resolved_bins = int(bins)
    if resolved_bins < 1:
        raise ValueError("bins must be >= 1")

    if horizon % resolved_bins != 0:
        raise ValueError(
            f"fixed-bin bridge requires horizon divisible by bins (got horizon={horizon}, bins={resolved_bins})"
        )

    exact_pair = build_fixed_bin_exact_law_pair(
        mechanism=mechanism,
        c_matrix=matrix,
        bins=resolved_bins,
        noise_multiplier=noise_multiplier,
    )
    resolved_package = _resolve_fixed_bin_exact_package(
        exact_pair=exact_pair,
        bins=resolved_bins,
    )
    return FixedBinRandomAllocationBridgeInputs(
        source_law_kind="balls_in_bins_fixed_bin",
        accountant_engine_kind="deterministic_random_allocation",
        route=resolved_package.route,
        mechanism=str(mechanism),
        bins=resolved_bins,
        epochs=horizon // resolved_bins,
        horizon=horizon,
        noise_multiplier=float(noise_multiplier),
        mode_family=exact_pair.forward_modes,
        exact_law_route=exact_pair.metadata.route,
        initial_package_route=resolved_package.initial_package_route,
        pair_driven_route=resolved_package.pair_driven_route,
    )


def _resolve_fixed_bin_bridge_pair_inputs(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    noise_multiplier: float,
) -> _PairDrivenRandomAllocationInputs:
    matrix = np.asarray(c_matrix, dtype=np.float64)
    exact_pair = build_fixed_bin_exact_law_pair(
        mechanism=mechanism,
        c_matrix=matrix,
        bins=int(bins),
        noise_multiplier=noise_multiplier,
    )
    resolved_package = _resolve_fixed_bin_exact_package(
        exact_pair=exact_pair,
        bins=int(bins),
    )
    if resolved_package.pair_inputs is None:
        raise NotImplementedError(
            resolved_package.blocker
            or "fixed-bin bridge exact package did not resolve to pair-driven inputs"
        )

    return resolved_package.pair_inputs


def estimate_epsilon_range_fixed_bin_random_allocation(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    noise_multiplier: float,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> tuple[float, float]:
    """Evaluate a conservative `(ε_upper, ε_lower)` interval for the bridge.

    The returned interval comes from the strongest exact deterministic route
    currently available above the fixed-bin exact pair.

    Source: `BSR`, `PLD`.
    """

    runtime = runtime_config or resolve_fixed_bin_random_allocation_bridge_runtime_config(
        target_delta=target_delta
    )
    matrix = np.asarray(c_matrix, dtype=np.float64)
    exact_pair = build_fixed_bin_exact_law_pair(
        mechanism=mechanism,
        c_matrix=matrix,
        bins=int(bins),
        noise_multiplier=noise_multiplier,
    )
    resolved_package = _resolve_fixed_bin_exact_package(
        exact_pair=exact_pair,
        bins=int(bins),
    )

    if resolved_package.pair_inputs is not None:
        try:
            return estimate_epsilon_range_random_allocation_from_initial_package(
                inputs=resolved_package.pair_inputs,
                target_delta=target_delta,
                runtime_config=runtime,
            )
        except ValueError as exc:
            if (
                resolved_package.initial_package_route
                == "pair_driven_ambient_quantitative_window_realization_package"
            ):
                raise NotImplementedError(
                    "ambient quantitative-window realization package is constructed, but the "
                    f"current deterministic evaluator/runtime still fails on this route: {exc}"
                ) from exc
            raise

    if resolved_package.exact_family_round_pair_package_inputs is not None:
        return _estimate_epsilon_range_random_allocation_from_exact_family_round_pair_package(
            inputs=resolved_package.exact_family_round_pair_package_inputs,
            target_delta=target_delta,
            runtime_config=runtime,
        )

    raise NotImplementedError(
        resolved_package.blocker
        or "fixed-bin bridge exact package did not resolve to an evaluable exact route"
    )


def _diagnose_fixed_bin_ambient_nonfinite_upper_bound(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    noise_multiplier: float,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
) -> _FixedBinAmbientNonfiniteUpperDiagnostic | None:
    runtime = runtime_config or resolve_fixed_bin_random_allocation_bridge_runtime_config(
        target_delta=target_delta
    )
    pair_inputs = _resolve_fixed_bin_bridge_pair_inputs(
        mechanism=mechanism,
        c_matrix=c_matrix,
        bins=bins,
        noise_multiplier=noise_multiplier,
    )
    if (
        pair_inputs.initial_package.route
        != "pair_driven_ambient_quantitative_window_realization_package"
    ):
        return None

    package = pair_inputs.initial_package
    remove_round = _allocation_pmf_remove_from_realization(
        package.remove.realization,
        int(pair_inputs.num_steps // pair_inputs.num_selected),
        runtime,
        _BoundType.DOMINATES,
    )
    add_round = _allocation_pmf_add_from_realization(
        package.add.realization,
        int(pair_inputs.num_steps // pair_inputs.num_selected),
        runtime,
        _BoundType.DOMINATES,
    )
    num_rounds = int(pair_inputs.num_selected * pair_inputs.num_epochs)
    remove_final = _compose_linear_pmfs(
        remove_round, num_rounds, runtime.tail_truncation, _BoundType.DOMINATES
    )
    add_final = _compose_linear_pmfs(
        add_round, num_rounds, runtime.tail_truncation, _BoundType.DOMINATES
    )
    blocker = "ambient_nonfinite_upper_bound"

    if remove_final.p_pos_inf >= float(target_delta):
        blocker = "remove_composed_positive_infinity_mass_exceeds_delta"
    elif remove_round.p_pos_inf >= float(target_delta):
        blocker = "remove_round_positive_infinity_mass_exceeds_delta"
    elif remove_final.p_pos_inf >= 1.0 - 1e-12:
        blocker = "remove_composed_all_positive_infinity_mass"
    elif remove_round.p_pos_inf >= 1.0 - 1e-12:
        blocker = "remove_round_all_positive_infinity_mass"
    elif add_final.p_pos_inf >= 1.0 - 1e-12:
        blocker = "add_composed_all_positive_infinity_mass"
    elif add_round.p_pos_inf >= 1.0 - 1e-12:
        blocker = "add_round_all_positive_infinity_mass"

    return _FixedBinAmbientNonfiniteUpperDiagnostic(
        route="fixed_bin_bridge_ambient_quantitative_window_realization_package",
        initial_package_route=package.route,
        remove_round_pos_inf=float(remove_round.p_pos_inf),
        remove_final_pos_inf=float(remove_final.p_pos_inf),
        add_round_pos_inf=float(add_round.p_pos_inf),
        add_final_pos_inf=float(add_final.p_pos_inf),
        blocker=blocker,
    )


def _fixed_bin_bridge_noise_search_error(
    *,
    target_epsilon: float,
    target_delta: float,
    state: _FixedBinNoiseSearchState,
) -> NoiseSearchConvergenceError:
    return NoiseSearchConvergenceError(
        accountant="fixed_bin_random_allocation_bridge",
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        last_finite_sigma=state.last_finite_sigma,
        last_finite_epsilon=state.last_finite_epsilon,
        last_nonfinite_sigma=state.last_nonfinite_sigma,
        iterations=state.iterations,
    )


def _return_bracketed_sigma_or_raise(
    *,
    target_epsilon: float,
    target_delta: float,
    state: _FixedBinNoiseSearchState,
) -> float:
    # The fixed-bin bridge upper search only needs a valid finite sigma whose
    # upper epsilon is below target. On discretized ambient routes the upper
    # epsilon can plateau, so the binary search may hit its resolution floor
    # even though `sigma_high` is already a valid answer.
    if math.isfinite(state.eps_high) and state.eps_high <= float(target_epsilon):
        return float(state.sigma_high)

    raise _fixed_bin_bridge_noise_search_error(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        state=state,
    )


def _fixed_bin_search_eval(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    sigma: float,
    target_delta: float,
    runtime: RandomAllocationGaussianRuntimeConfig,
) -> float:
    eps_upper, _eps_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism=mechanism,
        c_matrix=c_matrix,
        bins=bins,
        noise_multiplier=float(sigma),
        target_delta=target_delta,
        runtime_config=runtime,
    )
    return float(eps_upper)


def _record_fixed_bin_search_epsilon(
    *,
    state: _FixedBinNoiseSearchState,
    sigma: float,
    epsilon: float,
) -> None:
    if math.isfinite(epsilon):
        state.last_finite_sigma = float(sigma)
        state.last_finite_epsilon = float(epsilon)
    else:
        state.last_nonfinite_sigma = float(sigma)


def _grow_fixed_bin_sigma_bracket(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    target_epsilon: float,
    target_delta: float,
    runtime: RandomAllocationGaussianRuntimeConfig,
    state: _FixedBinNoiseSearchState,
) -> float:
    while state.eps_high > float(target_epsilon):
        state.sigma_high = 2.0 * state.sigma_high
        state.eps_high = _fixed_bin_search_eval(
            mechanism=mechanism,
            c_matrix=c_matrix,
            bins=bins,
            sigma=state.sigma_high,
            target_delta=target_delta,
            runtime=runtime,
        )
        _record_fixed_bin_search_epsilon(
            state=state,
            sigma=state.sigma_high,
            epsilon=state.eps_high,
        )
        if state.sigma_high > MAX_SIGMA:
            return _return_bracketed_sigma_or_raise(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                state=state,
            )

    return float(state.sigma_high)


def _advance_fixed_bin_binary_search(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    target_epsilon: float,
    target_delta: float,
    runtime: RandomAllocationGaussianRuntimeConfig,
    state: _FixedBinNoiseSearchState,
) -> None:
    sigma = 0.5 * (state.sigma_low + state.sigma_high)
    if sigma <= state.sigma_low or sigma >= state.sigma_high:
        raise _fixed_bin_bridge_noise_search_error(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            state=state,
        )

    eps_mid = _fixed_bin_search_eval(
        mechanism=mechanism,
        c_matrix=c_matrix,
        bins=bins,
        sigma=sigma,
        target_delta=target_delta,
        runtime=runtime,
    )
    state.iterations += 1
    _record_fixed_bin_search_epsilon(
        state=state,
        sigma=sigma,
        epsilon=eps_mid,
    )
    if math.isfinite(eps_mid) and eps_mid < float(target_epsilon):
        state.sigma_high = float(sigma)
        state.eps_high = float(eps_mid)
    else:
        state.sigma_low = float(sigma)


def _refine_fixed_bin_sigma_bracket(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    target_epsilon: float,
    target_delta: float,
    epsilon_tolerance: float,
    runtime: RandomAllocationGaussianRuntimeConfig,
    state: _FixedBinNoiseSearchState,
) -> float:
    while float(target_epsilon) - state.eps_high > float(epsilon_tolerance):
        if state.iterations >= MAX_NOISE_SEARCH_BINARY_STEPS:
            return _return_bracketed_sigma_or_raise(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                state=state,
            )

        if state.sigma_high - state.sigma_low <= MIN_NOISE_SEARCH_SIGMA_INTERVAL:
            return _return_bracketed_sigma_or_raise(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                state=state,
            )

        _advance_fixed_bin_binary_search(
            mechanism=mechanism,
            c_matrix=c_matrix,
            bins=bins,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            runtime=runtime,
            state=state,
        )

    return float(state.sigma_high)


def get_noise_multiplier_fixed_bin_random_allocation(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    bins: int,
    target_epsilon: float,
    target_delta: float,
    runtime_config: RandomAllocationGaussianRuntimeConfig | None = None,
    epsilon_tolerance: float = 0.01,
) -> float:
    """Calibrate `σ` for the fixed-bin bridge at a target `ε`.

    The binary search uses the dominating bridge evaluator. The returned value
    is the current smallest known finite `σ_high` whose conservative upper bound
    is at or below the target `ε`, subject to the search tolerance.

    Source: `BSR`, `PLD`.

    Raises:
        NoiseSearchConvergenceError: If the search cannot establish a valid
            finite upper bracket.
    """

    runtime = runtime_config or resolve_fixed_bin_random_allocation_bridge_runtime_config(
        target_delta=target_delta
    )
    state = _FixedBinNoiseSearchState()
    _grow_fixed_bin_sigma_bracket(
        mechanism=mechanism,
        c_matrix=c_matrix,
        bins=bins,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        runtime=runtime,
        state=state,
    )
    return _refine_fixed_bin_sigma_bracket(
        mechanism=mechanism,
        c_matrix=c_matrix,
        bins=bins,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epsilon_tolerance=epsilon_tolerance,
        runtime=runtime,
        state=state,
    )
