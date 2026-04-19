from __future__ import annotations

"""
Canonical input helpers for the BNB accountant bridge surface.

These helpers normalize the small but important pieces of state that the BNB
accountant expects from MF and BLT runtime/provider layers: coefficient
surfaces, canonical band counts, and canonical balls-in-bins cycle lengths.
"""

import copy
from typing import Any, Dict, Mapping


def attach_accountant_coeff_surface(
    runtime_state: Mapping[str, Any],
    *,
    coeff_key: str,
    coeff_source_key: str,
    coeffs: Mapping[str, Any] | list[float] | tuple[float, ...],
    coeff_source: str,
) -> Dict[str, Any]:
    """
    Persist an accountant-facing coefficient surface onto a runtime-state payload.
    """
    state = copy.deepcopy(dict(runtime_state))
    state[coeff_key] = [float(c) for c in coeffs]
    state[coeff_source_key] = str(coeff_source)
    return state


def resolve_canonical_bsr_bands(
    *,
    runtime_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    error_context: str,
) -> int:
    """
    Resolve the canonical band count shared by MF runtime state and accountant inputs.
    """
    metadata_bands = metadata.get("bands")
    explicit_bands = kwargs.get("bsr_bands")
    state_bands = runtime_state.get("bsr_bands")

    if explicit_bands is not None and metadata_bands is not None:
        # Reject mixed contracts early so the provider/runtime side and the
        # accountant side cannot silently disagree about the family bandwidth.
        if int(explicit_bands) != int(metadata_bands):
            raise ValueError(
                "conflicting canonical inputs: `bsr_bands` must match "
                "sampling_semantics privacy_metadata['bands']"
            )

    bands = explicit_bands
    if bands is None:
        bands = metadata_bands
    if bands is None:
        bands = state_bands
    if bands is None:
        raise ValueError(error_context)

    bands = int(bands)
    if bands < 1:
        raise ValueError("bands must be >= 1")
    return bands


def resolve_canonical_bnb_cycle_length(
    *,
    runtime_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    error_context: str,
) -> int:
    """
    Resolve the canonical balls-in-bins cycle length from runtime, metadata, or kwargs.
    """
    metadata_bins = metadata.get("bins", metadata.get("b"))
    explicit_cycle_length = kwargs.get(
        "bnb_cycle_length",
        runtime_state.get("bnb_cycle_length"),
    )
    explicit_bins = kwargs.get("bnb_b")
    legacy_bins = runtime_state.get("bnb_bins")

    if explicit_cycle_length is not None:
        explicit_cycle_length = int(explicit_cycle_length)
        if explicit_cycle_length < 1:
            raise ValueError("balls-in-bins cycle length must be >= 1")

        # `bins`, `b`, `bnb_b`, and `bnb_cycle_length` are all the same contract
        # under different public/runtime spellings, so keep them equal here.
        if metadata_bins is not None and int(metadata_bins) != explicit_cycle_length:
            raise ValueError(
                "conflicting canonical inputs: `bnb_cycle_length` must match "
                "sampling_semantics privacy_metadata['bins']"
            )

        if explicit_bins is not None and int(explicit_bins) != explicit_cycle_length:
            raise ValueError(
                "conflicting canonical inputs: `bnb_cycle_length` must match `bnb_b`"
            )

        if legacy_bins is not None and int(legacy_bins) != explicit_cycle_length:
            raise ValueError(
                "conflicting canonical inputs: `bnb_cycle_length` must match "
                "mechanism_state['bnb_bins']"
            )

    bins = explicit_bins
    if bins is None:
        bins = metadata_bins
    if bins is None:
        bins = explicit_cycle_length
    if bins is None:
        bins = runtime_state.get("bnb_cycle_length", runtime_state.get("bnb_bins"))
    if bins is None:
        raise ValueError(error_context)

    bins = int(bins)
    if bins < 1:
        raise ValueError("balls-in-bins bins must be >= 1")
    return bins


__all__ = [
    "attach_accountant_coeff_surface",
    "resolve_canonical_bnb_cycle_length",
    "resolve_canonical_bsr_bands",
]
