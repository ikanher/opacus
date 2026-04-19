from __future__ import annotations

"""
BLT accountant-input boundary and report-surface helpers.

This module owns the canonicalization layer between loosely shaped runtime or
report inputs and the accountant-facing BLT parameter/configuration surfaces.
It is accountant-owned on purpose: `mf/blt_family.py` is only a provider layer
that forwards to these helpers.

Claim-type notes:
- canonicalization helpers are implementation-contract surfaces
- `resolve_blt_balls_in_bins_accountant_state` builds the accountant-side BLT
  bridge object for amplified `balls_in_bins` / `b_min_sep` consumers
- report-surface helpers describe current runtime/accounting outputs, not a
  formal proof surface
"""

import copy
import math
from typing import Any, Mapping

from opacus.accountants.analysis.blt import (
    BLTParams,
    BLTPairedParams,
    blt_pair_from_theta_pair,
    build_blt_amplified_bnb_inputs,
)
from opacus.accountants.bnb_inputs import (
    attach_accountant_coeff_surface,
    resolve_canonical_bnb_cycle_length,
)
from opacus.mechanism_contracts import SamplingSemantics
from opacus.accountants.blt_fixed_batch import (
    generate_blt_theta_pair_candidates,
    optimize_blt_fixed_batch,
)


def _pair_from_input_state(state: Mapping[str, Any]) -> BLTPairedParams:
    """
    Resolve a canonical BLT pair from either public input surface.

    Accepted shapes:
    - decay-pair input: `theta`, `theta_hat`
    - explicit paired input: `forward`, `inverse`
    """
    if "theta" in state or "theta_hat" in state:
        if "theta" not in state or "theta_hat" not in state:
            raise ValueError("blt decay-pair input requires both `theta` and `theta_hat`")
        pair = blt_pair_from_theta_pair(theta=state["theta"], theta_hat=state["theta_hat"])
    elif "forward" in state or "inverse" in state:
        if "forward" not in state or "inverse" not in state:
            raise ValueError("blt explicit input requires both `forward` and `inverse`")
        pair = BLTPairedParams(
            forward=BLTParams(
                theta=state["forward"]["theta"],
                omega=state["forward"]["omega"],
            ),
            inverse=BLTParams(
                theta=state["inverse"]["theta"],
                omega=state["inverse"]["omega"],
            ),
        )
    else:
        raise ValueError(
            "blt mechanism requires either decay-pair keys {'theta', 'theta_hat'} "
            "or explicit paired keys {'forward', 'inverse'}"
        )
    pair = pair.canonicalized()
    pair.validate()
    return pair


def _canonical_state_from_pair(
    *,
    pair: BLTPairedParams,
    z_std: float,
) -> dict[str, Any]:
    """Build the canonical BLT runtime/accountant state dictionary from a pair."""
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
        "z_std": float(z_std),
        "_blt_distributed_policy": "ddp_flat_only",
        "_blt_distributed_runtime": False,
    }


def _copy_blt_runtime_metadata(
    *,
    source_state: Mapping[str, Any],
    canonical_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve BLT runtime/accountant metadata that should survive canonicalization.

    Structural pair keys are rebuilt from the canonical pair, so only runtime
    metadata such as accountant references, BLT selection fields, and
    random-allocation attachments are copied through.
    """
    for key, value in source_state.items():
        if key in {"theta", "theta_hat", "forward", "inverse", "z_std"}:
            continue
        if (
            key == "noise_multiplier_ref"
            or key.startswith("blt_")
            or key.startswith("random_allocation_")
            or key.startswith("_")
        ):
            canonical_state[key] = copy.deepcopy(value)
    return canonical_state


def canonicalize_blt_public_or_runtime_state(
    mechanism_state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Canonicalize a BLT runtime or public state payload.

    Accepted inputs:
    - decay-pair surface: `theta`, `theta_hat`, `z_std`
    - explicit paired surface: `forward`, `inverse`, `z_std`

    Output:
    - canonical `forward` / `inverse` dictionaries with explicit `theta` and
      `omega`
    - preserved BLT runtime metadata such as `noise_multiplier_ref` or
      `blt_*` fields

    This is the main contract that separates the real BLT parameter surface from
    report-local labels like `lambda`.
    """
    state = copy.deepcopy(dict(mechanism_state))
    z_std = state.get("z_std")
    if z_std is None:
        raise ValueError("blt mechanism requires `mechanism_state['z_std']`")
    pair = _pair_from_input_state(state)
    canonical_state = _canonical_state_from_pair(pair=pair, z_std=float(z_std))
    return _copy_blt_runtime_metadata(
        source_state=state,
        canonical_state=canonical_state,
    )


def resolve_blt_fixed_batch_accountant_inputs(
    *,
    runtime_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    total_steps: int,
) -> tuple[BLTPairedParams, float, int, int, int]:
    """
    Resolve the fixed-batch BLT accountant inputs from runtime/public state.

    The resolved tuple is:
    - canonical BLT forward/inverse pair
    - accountant-side Gaussian reference sigma `noise_multiplier_ref`
    - fixed-batch workload parameters `max_participations`, `min_separation`
    - finite-horizon accounting length

    This is the fixed-batch accountant contract surface. It is not used for the
    amplified BNB bridge.
    """
    state = canonicalize_blt_public_or_runtime_state(runtime_state)

    noise_multiplier_ref = kwargs.get(
        "noise_multiplier_ref",
        metadata.get(
            "noise_multiplier_ref",
            state.get("noise_multiplier_ref"),
        ),
    )
    if noise_multiplier_ref is None:
        raise ValueError(
            "BLT fixed-batch accounting requires noise_multiplier_ref in mechanism_state or sampling metadata"
        )
    noise_multiplier_ref = float(noise_multiplier_ref)
    if not math.isfinite(noise_multiplier_ref) or noise_multiplier_ref <= 0.0:
        raise ValueError("noise_multiplier_ref must be finite and > 0")

    max_participations = kwargs.get(
        "blt_max_participations",
        metadata.get(
            "blt_max_participations",
            state.get("blt_max_participations"),
        ),
    )
    min_separation = kwargs.get(
        "blt_min_separation",
        metadata.get(
            "blt_min_separation",
            state.get("blt_min_separation"),
        ),
    )
    horizon = kwargs.get(
        "blt_horizon",
        metadata.get("blt_horizon", state.get("blt_horizon")),
    )
    if horizon is None:
        horizon = int(total_steps)

    if max_participations is None or min_separation is None:
        raise ValueError(
            "BLT fixed-batch accounting requires blt_max_participations and blt_min_separation"
        )

    pair = _pair_from_input_state(state)

    return (
        pair,
        float(noise_multiplier_ref),
        int(max_participations),
        int(min_separation),
        int(horizon),
    )


def summarize_blt_report_surface(runtime_state: Mapping[str, Any]) -> dict[str, Any]:
    """
    Summarize the BLT report-facing runtime/accounting surface.

    Returned quantities intentionally distinguish:
    - `computed_noise_multiplier`: the runtime BLT `z_std`
    - `accounting_noise_multiplier`: the accountant-side reference sigma, when
      present

    This function does not expose `lambda`; that is a report-script sweep label,
    not part of the canonical BLT runtime/accountant state.
    """
    state = canonicalize_blt_public_or_runtime_state(runtime_state)
    return {
        "computed_noise_multiplier": float(state["z_std"]),
        "accounting_noise_multiplier": (
            None
            if state.get("noise_multiplier_ref") is None
            else float(state["noise_multiplier_ref"])
        ),
        "blt_horizon": (
            None if state.get("blt_horizon") is None else int(state["blt_horizon"])
        ),
        "blt_min_separation": (
            None
            if state.get("blt_min_separation") is None
            else int(state["blt_min_separation"])
        ),
        "blt_max_participations": (
            None
            if state.get("blt_max_participations") is None
            else int(state["blt_max_participations"])
        ),
    }


def resolve_blt_balls_in_bins_accountant_state(
    *,
    runtime_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    total_steps: int,
) -> dict[str, Any]:
    """
    Resolve the amplified BLT `balls_in_bins` accountant state.

    This helper builds the accountant-side bridge object used by the current BNB
    path:
    - finite-horizon forward normalized `c_col`
    - Toeplitz `C` matrix and its metadata contract
    - explicit `bands`, `cycle_length`, and `horizon`

    Claim type: implementation-contract bridge for the current amplified BNB
    consumers. This state is used by the public `balls_in_bins` BLT route and by
    the direct `b_min_sep` analysis-side consumer path; it does not by itself
    claim a full amplified BLT accounting result.
    """
    from opacus.mf.state import BLTFamilyState

    state = BLTFamilyState.from_input_state(runtime_state)
    state.apply_explicit_overrides(metadata=metadata, kwargs=kwargs)
    canonical_state = state.to_state_dict()

    horizon = kwargs.get(
        "bnb_horizon",
        kwargs.get(
            "total_steps",
            metadata.get(
                "total_steps",
                canonical_state.get("bnb_horizon", canonical_state.get("blt_horizon")),
            ),
        ),
    )
    if horizon is None:
        horizon = int(total_steps)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("BLT balls_in_bins accounting requires horizon >= 1")

    bands = kwargs.get(
        "bnb_bands",
        metadata.get(
            "bands",
            canonical_state.get("bnb_bands", canonical_state.get("blt_min_separation")),
        ),
    )
    if bands is None:
        raise ValueError(
            "BLT balls_in_bins accounting requires `bnb_bands`, sampling metadata `bands`, "
            "or canonical BLT `blt_min_separation`"
        )
    bands = int(bands)

    cycle_length = resolve_canonical_bnb_cycle_length(
        runtime_state=canonical_state,
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            "BLT balls_in_bins accounting requires `bnb_cycle_length` or "
            "sampling metadata `bins`"
        ),
    )

    # Build the accountant-side BLT bridge object on the resolved visible
    # horizon before attaching the normalized coefficient surface.
    resolved = build_blt_amplified_bnb_inputs(
        pair=state.pair,
        bands=bands,
        horizon=horizon,
    )
    canonical_state = attach_accountant_coeff_surface(
        canonical_state,
        coeff_key="bnb_accountant_coeffs",
        coeff_source_key="bnb_accountant_coeffs_source",
        coeffs=resolved["bnb_accountant_coeffs"],
        coeff_source=str(resolved["bnb_accountant_coeffs_source"]),
    )
    canonical_state["bnb_c_matrix"] = resolved["bnb_c_matrix"]
    canonical_state["bnb_c_matrix_contract"] = resolved["bnb_c_matrix_contract"]
    canonical_state["bnb_bands"] = int(bands)
    canonical_state["bnb_horizon"] = int(horizon)
    canonical_state["bnb_cycle_length"] = int(cycle_length)
    canonical_state["bnb_bins"] = int(cycle_length)
    return canonical_state


def resolve_blt_workload_mechanism_state(
    *,
    total_steps: int,
    dataset_size: int,
    logical_batch_size: int,
    max_grad_norm: float,
    loss_reduction: str = "mean",
    sampling_semantics: SamplingSemantics | None = None,
    buffers: int | None = None,
    target_epsilon: float | None = None,
    target_delta: float | None = None,
    noise_multiplier_ref: float | None = None,
) -> dict[str, Any]:
    """
    Resolve a canonical BLT mechanism state from workload-shaped inputs.

    Supported modes:
    - fixed-batch target-epsilon calibration via the current accountant-backed
      BLT optimizer
    - deterministic workload-driven selection for explicit-noise or amplified
      BNB routes, without exposing diagnostic parameters like `lambda`
    """
    if total_steps < 1:
        raise ValueError("BLT workload resolution requires total_steps >= 1")
    if dataset_size < 1:
        raise ValueError("BLT workload resolution requires dataset_size >= 1")
    if logical_batch_size < 1:
        raise ValueError("BLT workload resolution requires logical_batch_size >= 1")
    if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0:
        raise ValueError("BLT workload resolution requires max_grad_norm > 0")

    resolved_buffers = 2 if buffers is None else int(buffers)
    if resolved_buffers < 1:
        raise ValueError("blt buffers must be >= 1")

    semantics = sampling_semantics or SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={},
    )

    if target_epsilon is not None:
        if target_delta is None:
            raise ValueError("BLT target-epsilon workload resolution requires target_delta")
        if semantics.sampling_mode == "torch_sampler":
            # Fixed-batch target-epsilon BLT uses the accountant-backed search
            # surface rather than the default deterministic candidate.
            return dict(
                optimize_blt_fixed_batch(
                    target_epsilon=float(target_epsilon),
                    target_delta=float(target_delta),
                    total_steps=int(total_steps),
                    dataset_size=int(dataset_size),
                    logical_batch_size=int(logical_batch_size),
                    max_grad_norm=float(max_grad_norm),
                    loss_reduction=str(loss_reduction),
                    sampling_semantics=semantics,
                    buffers=int(resolved_buffers),
                ).mechanism_state
            )

    # Outside the fixed-batch target-epsilon path, choose the maintained default
    # candidate deterministically from the canonical candidate family.
    candidates = generate_blt_theta_pair_candidates(buffers=int(resolved_buffers))
    theta_candidate, theta_hat_candidate = candidates[0]
    pair = blt_pair_from_theta_pair(
        theta=theta_candidate,
        theta_hat=theta_hat_candidate,
    ).canonicalized()
    pair.validate()

    steps_per_epoch = int(
        math.ceil(float(dataset_size) / float(logical_batch_size))
    )
    max_participations = int(
        math.ceil(float(total_steps) / float(steps_per_epoch))
    )

    if loss_reduction == "sum":
        denominator = 1.0
    else:
        denominator = float(logical_batch_size)
    if denominator <= 0.0:
        raise ValueError("BLT workload resolution requires a positive calibration denominator")

    if noise_multiplier_ref is None:
        z_std = 1.0
    else:
        if not math.isfinite(float(noise_multiplier_ref)) or float(noise_multiplier_ref) < 0.0:
            raise ValueError("noise_multiplier_ref must be finite and >= 0")
        # Convert the accountant-side Gaussian reference sigma into the runtime
        # BLT noise scale seen by the optimizer release.
        z_std = float(noise_multiplier_ref) * float(max_grad_norm) / float(denominator)

    mechanism_state = canonicalize_blt_public_or_runtime_state(
        {
            "theta": [float(x) for x in theta_candidate],
            "theta_hat": [float(x) for x in theta_hat_candidate],
            "z_std": float(z_std),
            "blt_horizon": int(total_steps),
            "blt_min_separation": int(steps_per_epoch),
            "blt_max_participations": int(max_participations),
            "blt_buffers": int(resolved_buffers),
            "blt_selection_mode": (
                "optimized_fixed_batch"
                if target_epsilon is not None and semantics.sampling_mode == "torch_sampler"
                else "implicit_workload_default"
            ),
            "blt_selected_candidate_index": 0,
            "blt_candidate_count": len(candidates),
            "blt_selected_theta": [float(x) for x in theta_candidate],
            "blt_selected_theta_hat": [float(x) for x in theta_hat_candidate],
        }
    )
    if noise_multiplier_ref is not None:
        mechanism_state["noise_multiplier_ref"] = float(noise_multiplier_ref)
    return mechanism_state


__all__ = [
    "canonicalize_blt_public_or_runtime_state",
    "resolve_blt_fixed_batch_accountant_inputs",
    "resolve_blt_balls_in_bins_accountant_state",
    "resolve_blt_workload_mechanism_state",
    "summarize_blt_report_surface",
]
