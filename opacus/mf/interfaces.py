from __future__ import annotations

"""
Engine-facing protocol surface for matrix-factorization mechanism families.

`PrivacyEngine` treats MF families as pluggable providers. These protocols
define the provider hooks it may call for canonicalization, runtime-state
augmentation, accountant-input resolution, and optional capability surfaces.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class MFFamily(Protocol):
    """
    Minimal engine-facing shell for MF families.

    Implementations own provider-layer translation only. They do not replace
    the underlying accountant or runtime noise-mechanism modules.
    """

    name: str

    def canonicalize(self, raw_state: Mapping[str, Any]) -> dict[str, Any]: ...

    def build_runtime(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]: ...

    def summarize(self, mechanism_state: Mapping[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class SupportsFixedBatch(Protocol):
    """Capability hook for fixed-batch/fixed-sampler accountant inputs."""

    def resolve_fixed_batch(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsCyclic(Protocol):
    """Capability hook for cyclic-poisson accountant inputs."""

    def resolve_cyclic(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsBallsInBins(Protocol):
    """Capability hook for balls-in-bins or related amplified accountant state."""

    def resolve_balls_in_bins(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsOptimization(Protocol):
    """Capability hook for family-local optimization surfaces such as BLT search."""

    def optimize(self, *, objective: Any, context: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class MFFamilyEntry:
    """Registry entry for one engine-facing MF family.

    The long-term target is family-local modules implementing this minimal shell,
    with optional capabilities provided only where the family supports them.

    Attributes:
        family: Provider object implementing the base family protocol.
        supports_fixed_batch: Whether the family exposes fixed-batch accountant
            input resolution through `resolve_fixed_batch`.
        supports_cyclic: Whether the family exposes cyclic-poisson accountant
            input resolution through `resolve_cyclic`.
        supports_balls_in_bins: Whether the family exposes amplified BNB-style
            state shaping through `resolve_balls_in_bins`.
        supports_optimization: Whether the family exposes a local optimization
            surface, currently used by BLT.
        needs_default_local_sampling_semantics: Whether target-epsilon
            calibration should synthesize a local sampling contract when the
            caller omitted one.
        accounting_requires_context: Whether epsilon queries must be given the
            persisted mechanism state and sampling semantics.
    """

    family: MFFamily
    supports_fixed_batch: bool = False
    supports_cyclic: bool = False
    supports_balls_in_bins: bool = False
    supports_optimization: bool = False
    needs_default_local_sampling_semantics: bool = False
    accounting_requires_context: bool = True
