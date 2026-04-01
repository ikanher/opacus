"""Exact neighboring-pair builders for the random-allocation subsystem.

This module builds the exact Gaussian law pairs that feed the higher-level
random-allocation routes:

- exact one-step common-covariance Gaussian pairs,
- exact finite Gaussian-mixture pairs against a centered Gaussian reference,
- no deterministic initial package construction,
- no fixed-bin bridge or repeated-accounting numerics.

See `random_allocation.__init__` for the package-level citation registry.

Traceability:
- `Mf/DP/RealizableGaussianOneStep.lean`
- `Mf/DP/PoissonGaussianMixturePLD.lean`
- `Mf/DP/ProductGaussianMixturePLD.lean`
- `paper/theory/numerical-composition/PRV.tex`
- `paper/theory/numerical-composition/numerical_composition.tex`
- `paper/theory/prv/arxiv_version.tex`
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


__all__ = [
    "ExactLawMetadata",
    "RealizableGaussianOneStepNeighboringPair",
    "FiniteGaussianMixtureNeighboringPair",
    "build_realizable_gaussian_one_step_neighboring_pair",
    "build_poisson_gaussian_mixture_neighboring_pair",
    "build_product_gaussian_mixture_neighboring_pair",
]


@dataclass(frozen=True)
class ExactLawMetadata:
    """Metadata carried by exact neighboring-pair inputs.

    Attributes:
        layer_kind: Stable identifier for the exact-law layer.
        accountant_package_kind: Whether a downstream deterministic package has
            already been constructed.
        sampler_bridge_kind: Name of any sampler/bridge layer already attached
            to the pair.
        route: Exact-law route name used for diagnostics and dispatch.
        theorem_alignment: Traceability anchors for the exact-law identity.
    """

    layer_kind: str
    accountant_package_kind: str
    sampler_bridge_kind: str
    route: str
    theorem_alignment: tuple[str, ...]


@dataclass(frozen=True)
class RealizableGaussianOneStepNeighboringPair:
    """Exact one-step Gaussian neighboring pair with shared covariance.

    This is the exact forward/reverse one-step law consumed directly by the
    repeated random-allocation accountant.

    Source: `PLD`.

    Attributes:
        metadata: Exact-law routing and traceability metadata.
        mechanism: Mechanism name used for diagnostics.
        covariance_kind: Covariance-family tag; currently shared `σ² I`.
        noise_multiplier: Gaussian noise multiplier `σ`.
        forward_mean: Mean vector of the remove/forward law.
        reverse_mean: Mean vector of the add/reverse law.
    """

    metadata: ExactLawMetadata
    mechanism: str
    covariance_kind: str
    noise_multiplier: float
    forward_mean: tuple[float, ...]
    reverse_mean: tuple[float, ...]


@dataclass(frozen=True)
class FiniteGaussianMixtureNeighboringPair:
    """Exact finite Gaussian-mixture neighboring pair with shared covariance.

    The forward side is a weighted finite Gaussian mixture. The reverse side is
    another weighted finite Gaussian mixture, often the centered single-mode
    reference law for random-allocation bridge routes.

    Source: `PLD`.

    Attributes:
        metadata: Exact-law routing and traceability metadata.
        mechanism: Mechanism name used for diagnostics.
        covariance_kind: Covariance-family tag; currently shared `σ² I`.
        noise_multiplier: Gaussian noise multiplier `σ`.
        centered_reference_mean: Mean of the centered Gaussian reference law.
        forward_modes: Mean vectors of the forward/remove Gaussian components.
        forward_weights: Weights of the forward/remove Gaussian mixture.
        reverse_modes: Mean vectors of the reverse/add Gaussian components.
        reverse_weights: Weights of the reverse/add Gaussian mixture.
    """

    metadata: ExactLawMetadata
    mechanism: str
    covariance_kind: str
    noise_multiplier: float
    centered_reference_mean: tuple[float, ...]
    forward_modes: tuple[tuple[float, ...], ...]
    forward_weights: tuple[float, ...]
    reverse_modes: tuple[tuple[float, ...], ...]
    reverse_weights: tuple[float, ...]


def _validate_noise_multiplier(noise_multiplier: float) -> float:
    sigma = float(noise_multiplier)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("exact law inputs require positive finite noise_multiplier")

    return sigma


def _vector_to_tuple(
    vector: np.ndarray | tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("law vectors must be 1-D")

    return tuple(float(v) for v in arr)


def _matrix_to_array(c_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(c_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("c_matrix must be a 2-D array")

    return matrix


def _constant_probability_vector(p: float, length: int) -> tuple[float, ...]:
    prob = float(p)
    if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError("sampling probabilities must lie in [0, 1]")

    return tuple(prob for _ in range(int(length)))


def _validate_probability_vector(
    probabilities: tuple[float, ...] | list[float] | np.ndarray, length: int
) -> tuple[float, ...]:
    probs = tuple(
        float(p) for p in np.asarray(probabilities, dtype=np.float64).tolist()
    )
    if len(probs) != int(length):
        raise ValueError(
            f"expected {int(length)} participation probabilities, got {len(probs)}"
        )

    for p in probs:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError("sampling probabilities must lie in [0, 1]")

    return probs


def _iter_traces(length: int) -> tuple[tuple[bool, ...], ...]:
    return tuple(itertools.product((False, True), repeat=int(length)))


def _trace_weight_poisson(trace: tuple[bool, ...], p: float) -> float:
    weight = 1.0
    for chosen in trace:
        weight *= p if chosen else (1.0 - p)

    return float(weight)


def _trace_weight_product(
    trace: tuple[bool, ...], probabilities: tuple[float, ...]
) -> float:
    weight = 1.0
    for chosen, p in zip(trace, probabilities):
        weight *= p if chosen else (1.0 - p)

    return float(weight)


def _sampled_column_sum(
    matrix: np.ndarray, trace: tuple[bool, ...]
) -> tuple[float, ...]:
    if not trace:
        return tuple(0.0 for _ in range(int(matrix.shape[0])))

    active_cols = [idx for idx, chosen in enumerate(trace) if chosen]
    if not active_cols:
        return tuple(0.0 for _ in range(int(matrix.shape[0])))

    mode = np.sum(matrix[:, active_cols], axis=1, dtype=np.float64)

    return tuple(float(v) for v in mode)


def _normalize_weights(weights: list[float]) -> tuple[float, ...]:
    total = float(sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("mixture weights must sum to a positive finite value")

    return tuple(float(w / total) for w in weights)


def build_realizable_gaussian_one_step_neighboring_pair(
    *,
    mechanism: str,
    forward_mean: np.ndarray | tuple[float, ...] | list[float],
    reverse_mean: np.ndarray | tuple[float, ...] | list[float],
    noise_multiplier: float,
) -> RealizableGaussianOneStepNeighboringPair:
    """Build the exact one-step Gaussian neighboring pair.

    The output is an exact forward/reverse Gaussian law pair with common
    covariance. Downstream code may treat it as an exact one-step substrate for
    repeated random-allocation PLD accounting.

    Source: `PLD`.

    Args:
        mechanism: Mechanism name used for diagnostics.
        forward_mean: Mean of the forward/remove Gaussian law.
        reverse_mean: Mean of the reverse/add Gaussian law.
        noise_multiplier: Shared Gaussian noise multiplier `σ`.

    Raises:
        ValueError: If `noise_multiplier` is not positive and finite, or if the
            supplied mean vectors are not 1-D.
    """

    return RealizableGaussianOneStepNeighboringPair(
        metadata=ExactLawMetadata(
            layer_kind="exact_law_level_input",
            accountant_package_kind="not_initialized",
            sampler_bridge_kind="none",
            route="exact_common_covariance_gaussian_one_step_pair",
            theorem_alignment=(
                "RealizableGaussianOneStep.lean",
                "PRV.tex",
                "numerical_composition.tex",
            ),
        ),
        mechanism=str(mechanism),
        covariance_kind="common_covariance_sigma_squared_identity",
        noise_multiplier=_validate_noise_multiplier(noise_multiplier),
        forward_mean=_vector_to_tuple(forward_mean),
        reverse_mean=_vector_to_tuple(reverse_mean),
    )


def build_poisson_gaussian_mixture_neighboring_pair(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    sampling_probability: float,
    noise_multiplier: float,
    participation_cap: int | None = None,
) -> FiniteGaussianMixtureNeighboringPair:
    """Build the exact Poisson-sampled Gaussian mixture pair from `c_matrix`.

    The forward law is the exact Poisson mixture over sampled column sums. The
    reverse law is the centered single-mode Gaussian reference. This is an exact
    remove/add neighboring-pair object, not an approximation.

    Source: `PLD`.

    Args:
        mechanism: Mechanism name used for diagnostics.
        c_matrix: Workload matrix whose sampled column sums define the modes.
        sampling_probability: Common per-column Poisson participation
            probability in `[0, 1]`.
        noise_multiplier: Shared Gaussian noise multiplier `σ`.
        participation_cap: Optional cap on the number of simultaneously
            participating columns; when present this builds the capped Poisson
            exact pair.

    Raises:
        ValueError: If `c_matrix` is not 2-D, `sampling_probability` is outside
            `[0, 1]`, `participation_cap` is negative, or `noise_multiplier` is
            not positive and finite.
    """

    matrix = _matrix_to_array(c_matrix)
    p = _constant_probability_vector(float(sampling_probability), int(matrix.shape[1]))[
        0
    ]
    cap = None if participation_cap is None else int(participation_cap)
    if cap is not None and cap < 0:
        raise ValueError("participation_cap must be >= 0")

    forward_modes: list[tuple[float, ...]] = []
    forward_weights: list[float] = []
    for trace in _iter_traces(int(matrix.shape[1])):
        if cap is not None and sum(bool(v) for v in trace) > cap:
            continue

        forward_modes.append(_sampled_column_sum(matrix, trace))
        forward_weights.append(_trace_weight_poisson(trace, p))

    return FiniteGaussianMixtureNeighboringPair(
        metadata=ExactLawMetadata(
            layer_kind="exact_law_level_input",
            accountant_package_kind="not_initialized",
            sampler_bridge_kind="none",
            route=(
                "exact_poisson_gaussian_mixture_pair"
                if cap is None
                else "exact_capped_poisson_gaussian_mixture_pair"
            ),
            theorem_alignment=(
                "PoissonGaussianMixturePLD.lean",
                "PRV.tex",
                "numerical_composition.tex",
            ),
        ),
        mechanism=str(mechanism),
        covariance_kind="common_covariance_sigma_squared_identity",
        noise_multiplier=_validate_noise_multiplier(noise_multiplier),
        centered_reference_mean=tuple(0.0 for _ in range(int(matrix.shape[0]))),
        forward_modes=tuple(forward_modes),
        forward_weights=_normalize_weights(forward_weights),
        reverse_modes=(tuple(0.0 for _ in range(int(matrix.shape[0]))),),
        reverse_weights=(1.0,),
    )


def build_product_gaussian_mixture_neighboring_pair(
    *,
    mechanism: str,
    c_matrix: np.ndarray,
    participation_probabilities: tuple[float, ...] | list[float] | np.ndarray,
    noise_multiplier: float,
) -> FiniteGaussianMixtureNeighboringPair:
    """Build the exact product-sampled Gaussian mixture pair from `c_matrix`.

    The forward law is the exact product-measure Gaussian mixture induced by the
    per-column participation probabilities. The reverse law is the centered
    single-mode Gaussian reference.

    Source: `PLD`.

    Args:
        mechanism: Mechanism name used for diagnostics.
        c_matrix: Workload matrix whose sampled column sums define the modes.
        participation_probabilities: Per-column product-measure participation
            probabilities. Its length must equal the number of columns in
            `c_matrix`.
        noise_multiplier: Shared Gaussian noise multiplier `σ`.

    Raises:
        ValueError: If `c_matrix` is not 2-D, the probability vector length does
            not match the number of columns, any participation probability lies
            outside `[0, 1]`, or `noise_multiplier` is not positive and finite.
    """

    matrix = _matrix_to_array(c_matrix)
    probs = _validate_probability_vector(
        participation_probabilities, int(matrix.shape[1])
    )
    forward_modes: list[tuple[float, ...]] = []
    forward_weights: list[float] = []
    for trace in _iter_traces(int(matrix.shape[1])):
        forward_modes.append(_sampled_column_sum(matrix, trace))
        forward_weights.append(_trace_weight_product(trace, probs))

    return FiniteGaussianMixtureNeighboringPair(
        metadata=ExactLawMetadata(
            layer_kind="exact_law_level_input",
            accountant_package_kind="not_initialized",
            sampler_bridge_kind="none",
            route="exact_product_gaussian_mixture_pair",
            theorem_alignment=(
                "ProductGaussianMixturePLD.lean",
                "PRV.tex",
                "numerical_composition.tex",
            ),
        ),
        mechanism=str(mechanism),
        covariance_kind="common_covariance_sigma_squared_identity",
        noise_multiplier=_validate_noise_multiplier(noise_multiplier),
        centered_reference_mean=tuple(0.0 for _ in range(int(matrix.shape[0]))),
        forward_modes=tuple(forward_modes),
        forward_weights=_normalize_weights(forward_weights),
        reverse_modes=(tuple(0.0 for _ in range(int(matrix.shape[0]))),),
        reverse_weights=(1.0,),
    )
