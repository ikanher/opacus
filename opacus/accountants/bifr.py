"""
BIFR accountant-side runtime/accounting contract helpers.

This module bridges engine/runtime BIFR state to the accountant-facing objects
used by the fixed-batch Gaussian path and the amplified BNB path. It resolves
three kinds of quantities:
- exact finite-horizon factor coefficients for runtime/accountant use
- fixed-batch sensitivity terms on the factor side
- amplified non-negative accountant coefficients and attached BNB state

Source: BIFR (Kalinin et al., 2026) for the `γ`-indexed inverse/factor family.

Claim-type notes:
- runtime-state augmentation and canonicalization are implementation-contract
  surfaces
- fixed-batch sensitivity resolution is an accountant input surface, not a full
  privacy statement by itself
- amplified BNB helpers build the accountant-side bridge object consumed by the
  current `balls_in_bins` / `b_min_sep` routes
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional

from torch import optim

from opacus.accountants.analysis.bifr import (
    build_bifr_amplified_bnb_inputs_from_factor_coeffs,
    compute_bifr_fixed_batch_sensitivity_from_sgd_workload,
    derive_bifr_amplified_accountant_coeffs_from_factor_coeffs,
    generate_bifr_inverse_coeffs_from_sgd_workload,
    resolve_bifr_exact_factor_coeffs_for_accounting,
    validate_bifr_frac,
)
from opacus.accountants.bifr_inputs import canonicalize_bifr_runtime_state
from opacus.accountants.bnb_inputs import (
    attach_accountant_coeff_surface,
    resolve_canonical_bnb_cycle_length,
    resolve_canonical_bsr_bands,
)
from opacus.accountants.analysis.toeplitz_family import ToeplitzMechanismFamily
from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer
from opacus.mechanism_contracts import NoiseMechanismConfig


def _resolve_positive_horizon(
    *,
    total_steps: int,
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    context: str,
) -> int:
    """
    Resolve the finite visible horizon for BIFR accountant-state construction.

    `PrivacyEngine` may call family augmentation more than once. In epochs-based
    runs, a later pass can receive `total_steps=0` even though the prepared
    mechanism state already carries `bifr_horizon`/`bnb_horizon`. Prefer an
    explicit positive argument when present, then fall back to persisted state
    and metadata.
    """
    candidates = (
        total_steps,
        kwargs.get("total_steps"),
        kwargs.get("steps"),
        metadata.get("total_steps"),
        metadata.get("steps"),
        state.get("bifr_horizon"),
        state.get("bnb_horizon"),
        state.get("bsr_iterations_number"),
    )
    for value in candidates:
        if value is None:
            continue
        resolved = int(value)
        if resolved > 0:
            return resolved
    raise ValueError(f"{context} requires a positive total_steps/horizon")


def ensure_bifr_exact_runtime_coeffs(
    *,
    mechanism_config: NoiseMechanismConfig,
    optimizer: optim.Optimizer,
    sampling_semantics,
    kwargs: Dict[str, Any],
) -> NoiseMechanismConfig:
    """
    Ensure a BIFR mechanism state carries inverse-side runtime coefficients and
    exact finite-horizon factor coefficients.

    BIFR runtime noising is inverse-side. A `coeffs`-only BIFR state is
    ambiguous because `coeffs` are factor-side/accountant coefficients in the
    canonical contract, so it is rejected rather than treated as a legacy
    factor-side runtime.

    Mapping type: implementation-contract.
    """
    if mechanism_config.mechanism != "bifr":
        return mechanism_config

    state = copy.deepcopy(mechanism_config.mechanism_state)
    inverse_coeffs = state.get("bifr_inv_coeffs")
    if not (isinstance(inverse_coeffs, (list, tuple)) and len(inverse_coeffs) > 0):
        coeffs = state.get("coeffs")
        if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
            raise ValueError(
                "bifr mechanism no longer accepts `coeffs`-only states; "
                "provide `bifr_inv_coeffs` or canonical inputs `bsr_bands`, "
                "`bifr_frac`, and `total_steps`"
            )

    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    bands = resolve_canonical_bsr_bands(
        runtime_state=state,
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            "bifr exact finite-horizon coeff generation requires bands via `mechanism_state['bsr_bands']`, "
            "`sampling_semantics.privacy_metadata['bands']`, or `bsr_bands`"
        ),
    )
    steps_hint = kwargs.get("total_steps", metadata.get("total_steps"))
    if steps_hint is None:
        raise ValueError("bifr exact finite-horizon coeff generation requires `total_steps`")

    if int(steps_hint) < int(bands):
        raise ValueError(
            f"bifr exact finite-horizon coeff generation requires steps >= bands; got steps={int(steps_hint)}, bands={int(bands)}"
        )

    frac = validate_bifr_frac(float(kwargs.get("bifr_frac", state.get("bifr_frac", metadata.get("bifr_frac", 0.5)))))
    if isinstance(inverse_coeffs, (list, tuple)) and len(inverse_coeffs) > 0:
        inv_coeffs = [float(c) for c in inverse_coeffs]
    else:
        # The exact finite-horizon BIFR runtime is parameterized by the SGD
        # workload, so auto-resolution starts from the optimizer-side `(momentum,
        # weight_decay)` pair.
        momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(optimizer=optimizer)
        inv_coeffs = generate_bifr_inverse_coeffs_from_sgd_workload(
            bands=int(bands),
            momentum=momentum,
            weight_decay=weight_decay,
            frac=float(frac),
        )
    factor_coeffs, factor_source = resolve_bifr_exact_factor_coeffs_for_accounting(
        inverse_coeffs=inv_coeffs,
        steps=int(steps_hint),
    )
    state["coeffs"] = factor_coeffs
    state["bifr_inv_coeffs"] = [float(c) for c in inv_coeffs]
    state["bifr_horizon"] = int(steps_hint)
    state["bsr_bands"] = int(bands)
    state["bifr_frac"] = float(frac)
    state["coeff_source"] = "exact_finite_horizon_auto"
    state["factor_coeff_source"] = str(factor_source)

    return NoiseMechanismConfig(
        mechanism=mechanism_config.mechanism,
        accounting_mode=mechanism_config.accounting_mode,
        mechanism_state=state,
    )


def resolve_bifr_mf_sensitivity_for_fixed_batch(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: Optional[float],
    kwargs: Dict[str, Any],
) -> float:
    """
    Resolve the BIFR fixed-batch factor-side sensitivity term.

    Resolution order:
    - explicit `bsr_mf_sensitivity`
    - explicit exact factor coefficients already stored in the mechanism state
    - exact workload parameters `(bands, steps, momentum, weight_decay, frac)`

    Returns:
        The fixed-batch sensitivity on the factor-side release object consumed
        by the Gaussian accountant reduction.

    Mapping type: accountant input surface.
    """
    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    explicit = kwargs.get(
        "bsr_mf_sensitivity",
        metadata.get("bsr_mf_sensitivity", mechanism_state.get("bsr_mf_sensitivity")),
    )
    if explicit is not None:
        value = float(explicit)
        if (not math.isfinite(value)) or value <= 0.0:
            raise ValueError("bsr_mf_sensitivity must be finite and > 0")
        return value

    bands = resolve_canonical_bsr_bands(
        runtime_state=mechanism_state,
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            "fixed-batch bifr accounting requires bands via `mechanism_state['bsr_bands']`, "
            "`sampling_semantics.privacy_metadata['bands']`, or `bsr_bands`"
        ),
    )
    sensitivity_steps = kwargs.get(
        "bsr_iterations_number",
        metadata.get("bsr_iterations_number", mechanism_state.get("bsr_iterations_number", steps)),
    )
    if sample_rate is None:
        raise ValueError("sample_rate must be provided for bifr fixed-batch sensitivity resolution")

    max_participations = kwargs.get(
        "bsr_max_participations",
        metadata.get("bsr_max_participations", mechanism_state.get("bsr_max_participations")),
    )
    if max_participations is None:
        max_participations = max(1, int(math.ceil(float(sample_rate) * float(sensitivity_steps))))

    min_separation = kwargs.get(
        "bsr_min_separation",
        metadata.get("bsr_min_separation", mechanism_state.get("bsr_min_separation", 1)),
    )
    coeffs = mechanism_state.get("coeffs")
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        # When exact factor coefficients are already available, reuse them
        # directly instead of regenerating the workload-dependent inverse side.
        return float(
            ToeplitzMechanismFamily(
                coeffs=[float(c) for c in coeffs],
                steps=int(sensitivity_steps),
                source="bifr",
            ).fixed_batch_sensitivity(
                max_participations=int(max_participations),
                min_separation=int(min_separation),
                allow_disjoint_fallback=True,
            )
        )

    momentum = kwargs.get("momentum")
    weight_decay = kwargs.get("weight_decay")
    if momentum is None or weight_decay is None:
        raise ValueError(
            "fixed-batch bifr accounting requires exact factor coefficients or exact workload parameters"
        )

    frac = validate_bifr_frac(float(kwargs.get("bifr_frac", mechanism_state.get("bifr_frac", metadata.get("bifr_frac", 0.5)))))

    return float(
        compute_bifr_fixed_batch_sensitivity_from_sgd_workload(
            bands=int(bands),
            steps=int(sensitivity_steps),
            max_participations=int(max_participations),
            min_separation=int(min_separation),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            frac=float(frac),
            allow_disjoint_fallback=True,
        )
    )


def resolve_bifr_amplified_accountant_coeffs(
    *,
    mechanism_state: Mapping[str, Any],
    sampling_semantics,
    optimizer: optim.Optimizer | None = None,
    kwargs: Mapping[str, Any],
    total_steps: int,
) -> tuple[list[float], str, dict[str, Any]]:
    """
    Resolve amplified BIFR accountant coefficients and the attached state.

    This helper canonicalizes the BIFR runtime state, resolves the exact
    visible-horizon factor column, converts it to the non-negative accountant
    first column used by the amplified BNB routes, and returns the updated
    state payload together with the coefficient-source tag.

    Returns:
        A tuple of `(accountant_coeffs, accountant_source, attached_state)`.

    Mapping type: implementation-contract bridge for amplified BNB accounting.
    """
    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    state = canonicalize_bifr_runtime_state(runtime_state=mechanism_state)
    resolved_total_steps = _resolve_positive_horizon(
        total_steps=int(total_steps),
        state=state,
        metadata=metadata,
        kwargs=kwargs,
        context="amplified bifr accounting",
    )
    bands = resolve_canonical_bsr_bands(
        runtime_state=state,
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            "amplified bifr accounting requires bands via `mechanism_state['bsr_bands']`, "
            "`sampling_semantics.privacy_metadata['bands']`, or `bsr_bands`"
        ),
    )

    frac = validate_bifr_frac(
        float(kwargs.get("bifr_frac", state.get("bifr_frac", metadata.get("bifr_frac", 0.5))))
    )
    momentum = kwargs.get("momentum", state.get("momentum", metadata.get("momentum")))
    weight_decay = kwargs.get(
        "weight_decay",
        state.get("weight_decay", metadata.get("weight_decay")),
    )
    if optimizer is not None and (momentum is None or weight_decay is None):
        momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(
            optimizer=optimizer
        )

    factor_coeffs, factor_source = resolve_bifr_exact_factor_coeffs_for_accounting(
        coeffs=state.get("coeffs"),
        inverse_coeffs=state.get("bifr_inv_coeffs"),
        steps=int(resolved_total_steps),
        bands=int(bands),
        momentum=None if momentum is None else float(momentum),
        weight_decay=None if weight_decay is None else float(weight_decay),
        frac=float(frac),
    )
    accountant_coeffs = derive_bifr_amplified_accountant_coeffs_from_factor_coeffs(
        coeffs=factor_coeffs
    )
    inputs = build_bifr_amplified_bnb_inputs_from_factor_coeffs(
        coeffs=factor_coeffs,
        bands=int(bands),
        horizon=int(resolved_total_steps),
    )
    state["coeffs"] = [float(c) for c in factor_coeffs]
    state["bsr_bands"] = int(bands)
    state["bifr_frac"] = float(frac)
    state["bifr_horizon"] = int(resolved_total_steps)

    if state.get("coeff_source") is None:
        state["coeff_source"] = str(factor_source)

    return accountant_coeffs, str(inputs["bnb_accountant_coeffs_source"]), {
        **state,
        **inputs,
    }


def resolve_bifr_bnb_accountant_state(
    *,
    mechanism_state: Mapping[str, Any],
    sampling_semantics,
    optimizer: optim.Optimizer | None = None,
    kwargs: Mapping[str, Any],
    total_steps: int,
) -> dict[str, Any]:
    """
    Attach the full BIFR amplified BNB accountant state surface.

    The returned dictionary includes the canonical BIFR runtime state together
    with the accountant-side non-negative coefficient column and the resolved
    BNB cycle metadata (`bands`, `horizon`, `cycle_length`, `bins`).
    """
    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    state0 = canonicalize_bifr_runtime_state(runtime_state=mechanism_state)
    resolved_total_steps = _resolve_positive_horizon(
        total_steps=int(total_steps),
        state=state0,
        metadata=metadata,
        kwargs=kwargs,
        context="bifr amplified BNB accountant state",
    )
    cycle_length = resolve_canonical_bnb_cycle_length(
        runtime_state=state0,
        metadata=metadata,
        kwargs=kwargs,
        error_context="bifr amplified accounting requires BNB cycle length or sampling bins metadata",
    )
    accountant_coeffs, accountant_source, state = resolve_bifr_amplified_accountant_coeffs(
        mechanism_state=state0,
        sampling_semantics=sampling_semantics,
        optimizer=optimizer,
        kwargs=kwargs,
        total_steps=int(resolved_total_steps),
    )
    state = attach_accountant_coeff_surface(
        state,
        coeff_key="bnb_accountant_coeffs",
        coeff_source_key="bnb_accountant_coeffs_source",
        coeffs=accountant_coeffs,
        coeff_source=accountant_source,
    )
    state["bnb_bands"] = int(state["bnb_bands"])
    state["bnb_horizon"] = int(resolved_total_steps)
    state["bnb_cycle_length"] = int(cycle_length)
    state["bnb_bins"] = int(cycle_length)

    return state
