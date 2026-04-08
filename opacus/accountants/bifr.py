from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

from torch import optim

from opacus.accountants.analysis.bifr import (
    compute_bifr_fixed_batch_sensitivity_from_sgd_workload,
    generate_bifr_factor_coeffs_from_sgd_workload,
    validate_bifr_frac,
)
from opacus.accountants.analysis.toeplitz_family import ToeplitzMechanismFamily
from opacus.mf.input_resolution import resolve_canonical_bsr_bands
from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer
from opacus.mechanism_contracts import NoiseMechanismConfig


def ensure_bifr_fixed_analytical_coeffs(
    *,
    mechanism_config: NoiseMechanismConfig,
    optimizer: optim.Optimizer,
    sampling_semantics,
    kwargs: Dict[str, Any],
) -> NoiseMechanismConfig:
    if mechanism_config.mechanism != "bifr":
        return mechanism_config

    state = copy.deepcopy(mechanism_config.mechanism_state)
    coeffs = state.get("coeffs")
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        return NoiseMechanismConfig(
            mechanism=mechanism_config.mechanism,
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )

    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    bands = resolve_canonical_bsr_bands(
        runtime_state=state,
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            "bifr analytical auto-coeff generation requires bands via `mechanism_state['bsr_bands']`, "
            "`sampling_semantics.privacy_metadata['bands']`, or `bsr_bands`"
        ),
    )
    steps_hint = kwargs.get("total_steps", metadata.get("total_steps"))
    if steps_hint is not None and int(steps_hint) < int(bands):
        raise ValueError(
            f"bifr analytical auto-coeff generation requires steps >= bands; got steps={int(steps_hint)}, bands={int(bands)}"
        )

    momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(optimizer=optimizer)
    frac = validate_bifr_frac(float(kwargs.get("bifr_frac", state.get("bifr_frac", metadata.get("bifr_frac", 0.5)))))
    state["coeffs"] = generate_bifr_factor_coeffs_from_sgd_workload(
        bands=int(bands),
        momentum=momentum,
        weight_decay=weight_decay,
        frac=float(frac),
    )
    state["bsr_bands"] = int(bands)
    state["bifr_frac"] = float(frac)
    state["coeff_source"] = "analytical_auto"
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
        return float(
            ToeplitzMechanismFamily(
                coeffs=coeffs,
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
            "fixed-batch bifr accounting requires either runtime coeffs or analytic workload parameters via auto-generated BIFR state"
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
