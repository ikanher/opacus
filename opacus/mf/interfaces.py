from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class MFFamily(Protocol):
    """Minimal engine-facing shell for MF families."""

    name: str

    def canonicalize(self, raw_state: Mapping[str, Any]) -> dict[str, Any]: ...

    def build_runtime(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]: ...

    def summarize(self, mechanism_state: Mapping[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class SupportsFixedBatch(Protocol):
    def resolve_fixed_batch(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsCyclic(Protocol):
    def resolve_cyclic(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsBallsInBins(Protocol):
    def resolve_balls_in_bins(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsOptimization(Protocol):
    def optimize(self, *, objective: Any, context: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class MFFamilyEntry:
    """Registry entry for one engine-facing MF family.

    The long-term target is family-local modules implementing this minimal shell,
    with optional capabilities provided only where the family supports them.
    """

    family: MFFamily
    supports_fixed_batch: bool = False
    supports_cyclic: bool = False
    supports_balls_in_bins: bool = False
    supports_optimization: bool = False
    needs_default_local_sampling_semantics: bool = False
    accounting_requires_context: bool = True
