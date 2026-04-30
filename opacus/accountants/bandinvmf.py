from __future__ import annotations

"""
BandInvMF runtime/accountant helpers.

This module owns the runtime-facing BandInvMF logic that sits between:
- the pure analysis surface in `opacus.accountants.analysis.bandinvmf`, and
- the orchestration layer in `opacus.privacy_engine`.

BandInvMF is the optimized inverse-band method from the BISR paper. Its
production runtime contract is:
- inverse-band coefficients may be provided explicitly, or
- runtime Toeplitz coefficients may be provided explicitly and canonically
  converted back to the inverse-band view, or
- the inverse-band coefficients may be generated automatically by optimizing
  from the analytic BISR initialization for the resolved SGD workload.

This module does not implement new privacy mathematics. It resolves runtime
state and delegates fixed-batch / cyclic accounting reductions to the existing
BSR-family helpers where the contracts are shared.
"""

import copy
import math
from typing import Any, Dict, Optional

from torch import optim

from opacus.accountants.analysis.bandinvmf import (
    compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs,
    derive_bandinvmf_inv_coeffs_from_runtime_coeffs,
    derive_bandinvmf_runtime_coeffs_from_inv_coeffs,
    optimize_bandinvmf_inv_coeffs_for_sgd_workload,
)
from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer
from opacus.mechanism_contracts import NoiseMechanismConfig

from .bsr import (
    resolve_bsr_sensitivity_scale_for_cyclic,
)


def resolve_bandinvmf_sensitivity_scale_for_cyclic(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    kwargs: Dict[str, Any],
) -> float:
    """
    Resolve the cyclic BandInvMF sensitivity scale from runtime state.

    Cyclic BandInvMF reuses the BSR-family cyclic reduction after removing
    fixed-batch-only state fields. This keeps the runtime contract explicit
    while preserving the current accountant semantics.
    """
    filtered_state = dict(mechanism_state)
    filtered_state.pop("bsr_mf_sensitivity", None)
    filtered_state.pop("bsr_max_participations", None)
    filtered_state.pop("bsr_min_separation", None)
    return resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=filtered_state,
        sampling_semantics=sampling_semantics,
        steps=steps,
        kwargs=kwargs,
    )


def resolve_bandinvmf_mf_sensitivity_for_fixed_batch(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: Optional[float],
    kwargs: Dict[str, Any],
) -> float:
    """
    Resolve fixed-batch BandInvMF MF sensitivity from runtime state.

    BandInvMF shares the fixed-batch correlated Gaussian reduction with the
    BSR-family path. This helper exists so BandInvMF ownership stays local to
    its own accountant module even though the underlying accounting contract is
    reused.
    """
    metadata = (
        sampling_semantics.privacy_metadata
        if sampling_semantics is not None
        else {}
    )

    inv_coeffs = kwargs.get(
        "bandinvmf_inv_coeffs",
        mechanism_state.get("bandinvmf_inv_coeffs"),
    )
    if inv_coeffs is None:
        runtime_coeffs = kwargs.get("coeffs", mechanism_state.get("coeffs"))
        if runtime_coeffs is not None:
            inv_coeffs = derive_bandinvmf_inv_coeffs_from_runtime_coeffs(
                coeffs=runtime_coeffs
            )

    max_participations = kwargs.get(
        "bsr_max_participations",
        metadata.get(
            "bsr_max_participations",
            mechanism_state.get("bsr_max_participations"),
        ),
    )
    min_separation = kwargs.get(
        "bsr_min_separation",
        metadata.get(
            "bsr_min_separation",
            mechanism_state.get("bsr_min_separation"),
        ),
    )
    sensitivity_steps = kwargs.get(
        "bsr_iterations_number",
        metadata.get(
            "bsr_iterations_number",
            mechanism_state.get("bsr_iterations_number"),
        ),
    )
    if sensitivity_steps is None:
        sensitivity_steps = int(steps)

    mf_sensitivity = kwargs.get(
        "bsr_mf_sensitivity",
        metadata.get("bsr_mf_sensitivity", mechanism_state.get("bsr_mf_sensitivity")),
    )
    explicit_mf_sensitivity_override = kwargs.get("bsr_mf_sensitivity", None) is not None

    if max_participations is None:
        if sample_rate is None:
            max_participations = 1
        else:
            max_participations = max(1, int(math.ceil(float(sample_rate) * float(sensitivity_steps))))

    if min_separation is None:
        min_separation = 1

    sensitivity_steps = int(sensitivity_steps)
    max_participations = int(max_participations)
    min_separation = int(min_separation)

    if mf_sensitivity is None:
        if inv_coeffs is None:
            raise ValueError(
                "fixed-batch bandinvmf accounting requires inverse coefficients or "
                "enough data to derive MF sensitivity"
            )
        return float(
            compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
                inv_coeffs=inv_coeffs,
                steps=sensitivity_steps,
                max_participations=max_participations,
                min_separation=min_separation,
            )
        )

    mf_sensitivity = float(mf_sensitivity)
    if not math.isfinite(mf_sensitivity) or mf_sensitivity <= 0.0:
        raise ValueError("bsr_mf_sensitivity must be finite and > 0")

    if explicit_mf_sensitivity_override and inv_coeffs is not None:
        derived = float(
            compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
                inv_coeffs=inv_coeffs,
                steps=sensitivity_steps,
                max_participations=max_participations,
                min_separation=min_separation,
            )
        )
        if not math.isfinite(derived) or derived <= 0.0:
            raise ValueError(
                "derived bandinvmf bsr_mf_sensitivity must be finite and > 0 "
                "when validating explicit bsr_mf_sensitivity"
            )
        if not math.isclose(mf_sensitivity, derived, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "provided bsr_mf_sensitivity is inconsistent with "
                "bandinvmf inverse coefficients/max_participations/min_separation "
                "for the resolved bsr_iterations_number"
            )

    return float(mf_sensitivity)


def _resolve_bandinvmf_runtime_contract_inputs(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    kwargs: Dict[str, Any],
    steps_hint: int,
    sample_rate_hint: Optional[float],
) -> tuple[int, int, int, int]:
    """
    Resolve canonical runtime inputs needed to auto-generate BandInvMF state.

    The canonical inputs are:
    - `bands`
    - `max_participations`
    - `min_separation`
    - `optimizer_steps`

    They may come from explicit kwargs, sampling metadata, or persisted
    mechanism state, in that precedence order.
    """
    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    metadata_bands = metadata.get("bands")
    explicit_bands = kwargs.get("bsr_bands")
    if explicit_bands is not None and metadata_bands is not None:
        if int(explicit_bands) != int(metadata_bands):
            raise ValueError(
                "conflicting canonical inputs: `bsr_bands` must match "
                "sampling_semantics privacy_metadata['bands']"
            )

    bands = kwargs.get(
        "bsr_bands",
        metadata.get("bands", mechanism_state.get("bsr_bands")),
    )
    if bands is None:
        raise ValueError(
            "bandinvmf runtime state generation requires bands via "
            "`mechanism_state['bsr_bands']`, `sampling_semantics.privacy_metadata['bands']`, "
            "or `bsr_bands`"
        )

    bands = int(bands)
    if bands < 1:
        raise ValueError("bandinvmf bands must be >= 1")

    if int(steps_hint) < bands:
        raise ValueError(
            "bandinvmf runtime state generation requires steps >= bands; "
            f"got steps={int(steps_hint)}, bands={bands}"
        )

    default_k = 1
    if sample_rate_hint is not None:
        default_k = max(1, int(math.ceil(float(sample_rate_hint) * float(steps_hint))))

    default_min_separation = metadata.get("bins", 1)

    max_participations = int(
        kwargs.get(
            "bsr_max_participations",
            metadata.get(
                "bsr_max_participations",
                mechanism_state.get("bsr_max_participations", default_k),
            ),
        )
    )
    min_separation = int(
        kwargs.get(
            "bsr_min_separation",
            metadata.get(
                "bsr_min_separation",
                mechanism_state.get("bsr_min_separation", default_min_separation),
            ),
        )
    )
    optimizer_steps = int(
        kwargs.get(
            "bsr_iterations_number",
            metadata.get(
                "bsr_iterations_number",
                mechanism_state.get("bsr_iterations_number", 1000),
            ),
        )
    )
    if max_participations < 1:
        raise ValueError("bsr_max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("bsr_min_separation must be >= 1")

    if optimizer_steps < 1:
        raise ValueError("bsr_iterations_number must be >= 1")

    return bands, max_participations, min_separation, optimizer_steps


def ensure_bandinvmf_runtime_state(
    *,
    mechanism_config: NoiseMechanismConfig,
    optimizer: optim.Optimizer,
    sampling_semantics,
    steps_hint: int,
    sample_rate_hint: Optional[float],
    kwargs: Dict[str, Any],
) -> NoiseMechanismConfig:
    """
    Ensure that a BandInvMF mechanism config has canonical runtime state.

    Accepted input forms:
    - explicit inverse-band coefficients in `bandinvmf_inv_coeffs`
    - explicit runtime Toeplitz coefficients in `coeffs`
    - no coefficients, in which case optimized inverse-band coefficients are
      generated automatically from the resolved SGD workload

    The returned mechanism state always contains:
    - `bandinvmf_inv_coeffs`
    - runtime `coeffs`
    - the existing metadata keys needed for resume and accounting
    """
    if mechanism_config.mechanism != "bandinvmf":
        return mechanism_config

    state = copy.deepcopy(mechanism_config.mechanism_state)
    state["_noise_mechanism"] = mechanism_config.mechanism
    inv_coeffs = state.get("bandinvmf_inv_coeffs")
    coeffs = state.get("coeffs")

    if isinstance(inv_coeffs, (list, tuple)) and len(inv_coeffs) > 0:
        state["bandinvmf_inv_coeffs"] = [float(c) for c in inv_coeffs]
        if not (isinstance(coeffs, (list, tuple)) and len(coeffs) > 0):
            state["coeffs"] = derive_bandinvmf_runtime_coeffs_from_inv_coeffs(
                inv_coeffs=state["bandinvmf_inv_coeffs"]
            )
        state.setdefault("coeff_source", "optimized_inv_explicit")
    elif isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        state["coeffs"] = [float(c) for c in coeffs]
        state["bandinvmf_inv_coeffs"] = derive_bandinvmf_inv_coeffs_from_runtime_coeffs(
            coeffs=state["coeffs"]
        )
        state.setdefault("coeff_source", "runtime_coeffs_explicit")
    else:
        (
            bands,
            max_participations,
            min_separation,
            optimizer_steps,
        ) = _resolve_bandinvmf_runtime_contract_inputs(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            kwargs=kwargs,
            steps_hint=int(steps_hint),
            sample_rate_hint=sample_rate_hint,
        )
        momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(
            optimizer=optimizer
        )
        state["bandinvmf_inv_coeffs"] = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
            bands=bands,
            momentum=momentum,
            weight_decay=weight_decay,
            steps=int(steps_hint),
            max_participations=max_participations,
            min_separation=min_separation,
            optimizer_steps=optimizer_steps,
        )
        state["coeffs"] = derive_bandinvmf_runtime_coeffs_from_inv_coeffs(
            inv_coeffs=state["bandinvmf_inv_coeffs"]
        )
        state["coeff_source"] = "optimized_auto"
        state["bandinvmf_optimizer_steps"] = int(optimizer_steps)
        state["bsr_bands"] = int(bands)
        state["bsr_max_participations"] = int(max_participations)
        state["bsr_min_separation"] = int(min_separation)
        state["bandinvmf_steps_hint"] = int(steps_hint)
        if sample_rate_hint is not None:
            state["bandinvmf_sample_rate_hint"] = float(sample_rate_hint)

    return NoiseMechanismConfig(
        mechanism=mechanism_config.mechanism,
        accounting_mode=mechanism_config.accounting_mode,
        mechanism_state=state,
    )
