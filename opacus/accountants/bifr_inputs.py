from __future__ import annotations

"""
BIFR accountant/runtime input canonicalization helpers.

This module owns the canonical BIFR runtime/accountant state surface shared by
accountant code and the engine-facing BIFR family shell. It intentionally does
not live under `opacus.mf`: accountant-owned BIFR code should not depend on the
provider shell for state normalization.
"""

import copy
from typing import Any, Dict, Mapping

from opacus.accountants.analysis.bifr import validate_bifr_frac


def canonicalize_bifr_runtime_state(*, runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize a BIFR runtime/accountant state payload.

    This normalizes numeric fields to stable Python scalar/list types, injects
    the canonical `_noise_mechanism` tag, and supplies the default
    `bifr_frac = 0.5` when the caller left the interpolation parameter
    unspecified.

    Mapping type: implementation-contract canonicalization surface.
    """
    state = copy.deepcopy(dict(runtime_state))
    state["_noise_mechanism"] = "bifr"

    coeffs = state.get("coeffs")
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        state["coeffs"] = [float(c) for c in coeffs]

    inv_coeffs = state.get("bifr_inv_coeffs")
    if isinstance(inv_coeffs, (list, tuple)) and len(inv_coeffs) > 0:
        state["bifr_inv_coeffs"] = [float(c) for c in inv_coeffs]

    if state.get("bsr_bands") is not None:
        state["bsr_bands"] = int(state["bsr_bands"])

    if state.get("bifr_frac") is None:
        state["bifr_frac"] = 0.5

    if state.get("bifr_frac") is not None:
        # Keep the interpolation parameter normalized eagerly so all downstream
        # accountant helpers see the same canonical `γ` value.
        state["bifr_frac"] = float(
            validate_bifr_frac(float(state["bifr_frac"]))
        )

    for name in ("z_std", "bsr_mf_sensitivity"):
        if state.get(name) is not None:
            state[name] = float(state[name])

    for name in ("bsr_min_separation", "bsr_max_participations", "bsr_iterations_number", "bifr_horizon"):
        if state.get(name) is not None:
            state[name] = int(state[name])

    if (
        state.get("coeff_source") is None
        and isinstance(state.get("coeffs"), list)
        and len(state["coeffs"]) > 0
    ):
        state["coeff_source"] = "explicit_exact_factor"

    return state


def summarize_bifr_runtime_state(runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Summarize the canonical BIFR runtime/accountant surface for reports.

    The summary is intentionally shallow: it records counts, source tags, and
    resolved numeric metadata without duplicating the full coefficient payload.
    """
    state = canonicalize_bifr_runtime_state(runtime_state=runtime_state)
    coeffs = state.get("coeffs")
    return {
        "mechanism": "bifr",
        "coeff_count": len(coeffs) if isinstance(coeffs, list) else None,
        "coeff_source": state.get("coeff_source"),
        "z_std": state.get("z_std"),
        "bsr_mf_sensitivity": state.get("bsr_mf_sensitivity"),
        "bsr_min_separation": state.get("bsr_min_separation"),
        "bsr_max_participations": state.get("bsr_max_participations"),
        "bsr_iterations_number": state.get("bsr_iterations_number"),
        "bsr_bands": state.get("bsr_bands"),
        "bifr_frac": state.get("bifr_frac"),
        "bifr_horizon": state.get("bifr_horizon"),
        "bifr_inv_coeff_count": (
            len(state["bifr_inv_coeffs"])
            if isinstance(state.get("bifr_inv_coeffs"), list)
            else None
        ),
    }


__all__ = [
    "canonicalize_bifr_runtime_state",
    "summarize_bifr_runtime_state",
]
