from __future__ import annotations

"""
Provider-layer family wrapper for the BSR/BISR/BandMF/BandInvMF stack.

This module owns family-local canonicalization, capability dispatch, and
query-state augmentation for the shared BSR-style provider surface used by
`PrivacyEngine`.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from torch import optim


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_optional_int(*values: Any, default: int = 0) -> int:
    value = _first_non_none(*values)
    if value is None:
        return int(default)
    return int(value)


def _noop_bsr_runtime_state_normalizer(*, state: Dict[str, Any]) -> None:
    """Leave plain BSR/BandMF runtime state untouched."""
    return None


def _normalize_bisr_runtime_state(*, state: Dict[str, Any]) -> None:
    """Normalize BISR inverse-side state and lazily materialize runtime coeffs."""
    inv_coeffs = state.get("bisr_inv_coeffs")
    if not (isinstance(inv_coeffs, (list, tuple)) and len(inv_coeffs) > 0):
        return

    from opacus.accountants.analysis.bisr import (
        derive_bisr_runtime_coeffs_from_inverse_coeffs,
    )

    state["bisr_inv_coeffs"] = [float(c) for c in inv_coeffs]
    if not (isinstance(state.get("coeffs"), list) and len(state["coeffs"]) > 0):
        # BISR may arrive as inverse coefficients only; derive the correlated
        # runtime coefficients once so later provider/accountant code can use a
        # single canonical state shape.
        state["coeffs"] = derive_bisr_runtime_coeffs_from_inverse_coeffs(
            coeffs=state["bisr_inv_coeffs"]
        )
    state.setdefault("coeff_source", "analytical_inv_explicit")


def _normalize_bandinvmf_runtime_state(*, state: Dict[str, Any]) -> None:
    """Normalize Band-Inv-MF inverse-side state and lazily materialize runtime coeffs."""
    inv_coeffs = state.get("bandinvmf_inv_coeffs")
    if not (isinstance(inv_coeffs, (list, tuple)) and len(inv_coeffs) > 0):
        return

    from opacus.accountants.analysis.bandinvmf import (
        derive_bandinvmf_runtime_coeffs_from_inv_coeffs,
    )

    state["bandinvmf_inv_coeffs"] = [float(c) for c in inv_coeffs]
    if not (isinstance(state.get("coeffs"), list) and len(state["coeffs"]) > 0):
        state["coeffs"] = derive_bandinvmf_runtime_coeffs_from_inv_coeffs(
            inv_coeffs=state["bandinvmf_inv_coeffs"]
        )
    state.setdefault("coeff_source", "analytical_inv_explicit")


def _resolve_bsr_cyclic_input(*, state: Dict[str, Any], sampling_semantics, steps: int, kwargs: Mapping[str, Any]) -> float:
    """Resolve the cyclic sensitivity-scale input for the BSR accountant path."""
    from opacus.accountants.bsr import resolve_bsr_sensitivity_scale_for_cyclic

    return float(
        resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            kwargs=dict(kwargs),
        )
    )


def _resolve_bisr_cyclic_input(*, state: Dict[str, Any], sampling_semantics, steps: int, kwargs: Mapping[str, Any]) -> float:
    """Resolve the cyclic sensitivity-scale input for the BISR accountant path."""
    from opacus.accountants.bsr import resolve_bisr_sensitivity_scale_for_cyclic

    return float(
        resolve_bisr_sensitivity_scale_for_cyclic(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            kwargs=dict(kwargs),
        )
    )


def _resolve_bandinvmf_cyclic_input(*, state: Dict[str, Any], sampling_semantics, steps: int, kwargs: Mapping[str, Any]) -> float:
    """Resolve the cyclic sensitivity-scale input for the Band-Inv-MF accountant path."""
    from opacus.accountants.bandinvmf import resolve_bandinvmf_sensitivity_scale_for_cyclic

    return float(
        resolve_bandinvmf_sensitivity_scale_for_cyclic(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            kwargs=dict(kwargs),
        )
    )


def _resolve_bsr_fixed_batch_input(
    *,
    state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> float:
    """Resolve the fixed-batch sensitivity input for the BSR accountant path."""
    from opacus.accountants.bsr import resolve_bsr_mf_sensitivity_for_fixed_batch

    return float(
        resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            sample_rate=float(sample_rate),
            kwargs=dict(kwargs),
        )
    )


def _resolve_bandmf_fixed_batch_input(
    *,
    state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> float:
    """Resolve the fixed-batch sensitivity input for the BandMF accountant path."""
    from opacus.accountants.bandmf import resolve_bandmf_mf_sensitivity_for_fixed_batch

    return float(
        resolve_bandmf_mf_sensitivity_for_fixed_batch(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            sample_rate=float(sample_rate),
            kwargs=dict(kwargs),
        )
    )


def _resolve_bisr_fixed_batch_input(
    *,
    state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> float:
    """Resolve the fixed-batch sensitivity input for the BISR accountant path."""
    from opacus.accountants.bsr import resolve_bisr_mf_sensitivity_for_fixed_batch

    return float(
        resolve_bisr_mf_sensitivity_for_fixed_batch(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            sample_rate=float(sample_rate),
            kwargs=dict(kwargs),
        )
    )


def _resolve_bandinvmf_fixed_batch_input(
    *,
    state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> float:
    """Resolve the fixed-batch sensitivity input for the Band-Inv-MF accountant path."""
    from opacus.accountants.bandinvmf import resolve_bandinvmf_mf_sensitivity_for_fixed_batch

    return float(
        resolve_bandinvmf_mf_sensitivity_for_fixed_batch(
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            steps=int(steps),
            sample_rate=float(sample_rate),
            kwargs=dict(kwargs),
        )
    )


_BSR_RUNTIME_STATE_NORMALIZERS: Dict[str, Callable[..., None]] = {
    "bsr": _noop_bsr_runtime_state_normalizer,
    "bandmf": _noop_bsr_runtime_state_normalizer,
    "bisr": _normalize_bisr_runtime_state,
    "bandinvmf": _normalize_bandinvmf_runtime_state,
}

_BSR_CYCLIC_ACCOUNTANT_INPUT_RESOLVERS: Dict[str, Callable[..., float]] = {
    "bsr": _resolve_bsr_cyclic_input,
    "bandmf": _resolve_bsr_cyclic_input,
    "bisr": _resolve_bisr_cyclic_input,
    "bandinvmf": _resolve_bandinvmf_cyclic_input,
}

_BSR_FIXED_BATCH_ACCOUNTANT_INPUT_RESOLVERS: Dict[str, Callable[..., float]] = {
    "bsr": _resolve_bsr_fixed_batch_input,
    "bandmf": _resolve_bandmf_fixed_batch_input,
    "bisr": _resolve_bisr_fixed_batch_input,
    "bandinvmf": _resolve_bandinvmf_fixed_batch_input,
}


def canonicalize_bsr_family_runtime_state(
    *,
    mechanism: str,
    runtime_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Canonicalize BSR-family runtime state into a single provider-layer payload shape.
    """
    state = copy.deepcopy(dict(runtime_state))
    state["_noise_mechanism"] = mechanism

    coeffs = state.get("coeffs")
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        state["coeffs"] = [float(c) for c in coeffs]
    try:
        normalizer = _BSR_RUNTIME_STATE_NORMALIZERS[mechanism]
    except KeyError as exc:
        raise ValueError(f"unsupported bsr-family mechanism for canonicalization: {mechanism}") from exc
    normalizer(state=state)

    if state.get("bsr_bands") is not None:
        state["bsr_bands"] = int(state["bsr_bands"])
    for name in (
        "z_std",
        "bsr_sensitivity_scale",
        "bsr_mf_sensitivity",
    ):
        if state.get(name) is not None:
            state[name] = float(state[name])
    for name in (
        "bsr_min_separation",
        "bsr_max_participations",
        "bsr_iterations_number",
    ):
        if state.get(name) is not None:
            state[name] = int(state[name])

    if state.get("coeff_source") is None and isinstance(state.get("coeffs"), list) and len(state["coeffs"]) > 0:
        state["coeff_source"] = "explicit_or_precomputed"

    return state


def resolve_bsr_family_cyclic_accountant_input(
    *,
    mechanism: str,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    steps: int,
    kwargs: Mapping[str, Any],
) -> float:
    """Resolve the cyclic accountant input for the selected BSR-family mechanism."""
    state = canonicalize_bsr_family_runtime_state(
        mechanism=mechanism,
        runtime_state=runtime_state,
    )
    try:
        resolver = _BSR_CYCLIC_ACCOUNTANT_INPUT_RESOLVERS[mechanism]
    except KeyError as exc:
        raise ValueError(f"unsupported bsr-family mechanism for cyclic accounting: {mechanism}") from exc
    return resolver(
        state=state,
        sampling_semantics=sampling_semantics,
        steps=int(steps),
        kwargs=kwargs,
    )


def resolve_bsr_family_fixed_batch_accountant_input(
    *,
    mechanism: str,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> float:
    """Resolve the fixed-batch accountant input for the selected BSR-family mechanism."""
    state = canonicalize_bsr_family_runtime_state(
        mechanism=mechanism,
        runtime_state=runtime_state,
    )
    try:
        resolver = _BSR_FIXED_BATCH_ACCOUNTANT_INPUT_RESOLVERS[mechanism]
    except KeyError as exc:
        raise ValueError(
            f"unsupported bsr-family mechanism for fixed-batch accounting: {mechanism}"
        ) from exc
    return resolver(
        state=state,
        sampling_semantics=sampling_semantics,
        steps=int(steps),
        sample_rate=float(sample_rate),
        kwargs=kwargs,
    )


def augment_bsr_family_cyclic_query_state(
    *,
    mechanism: str,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    steps: int,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Stamp cyclic sensitivity-scale metadata onto canonical BSR-family state."""
    state = canonicalize_bsr_family_runtime_state(
        mechanism=mechanism,
        runtime_state=runtime_state,
    )
    state["bsr_sensitivity_scale"] = resolve_bsr_family_cyclic_accountant_input(
        mechanism=mechanism,
        runtime_state=state,
        sampling_semantics=sampling_semantics,
        steps=int(steps),
        kwargs=kwargs,
    )
    return state


def augment_bsr_family_fixed_batch_query_state(
    *,
    mechanism: str,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Stamp fixed-batch sensitivity metadata onto canonical BSR-family state."""
    state = canonicalize_bsr_family_runtime_state(
        mechanism=mechanism,
        runtime_state=runtime_state,
    )
    state["bsr_mf_sensitivity"] = resolve_bsr_family_fixed_batch_accountant_input(
        mechanism=mechanism,
        runtime_state=state,
        sampling_semantics=sampling_semantics,
        steps=int(steps),
        sample_rate=float(sample_rate),
        kwargs=kwargs,
    )
    return state


def augment_bsr_family_balls_in_bins_query_state(
    *,
    mechanism: str,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    optimizer: optim.Optimizer,
    total_steps: int,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build the amplified balls-in-bins accountant bridge state for a BSR-family mechanism.
    """
    from opacus.accountants.analysis.bandinvmf import (
        derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs,
    )
    from opacus.accountants.analysis.bandmf import (
        derive_bandmf_amplified_accountant_coeffs_from_runtime_coeffs,
        generate_bandmf_coeffs_from_sgd_workload,
    )
    from opacus.accountants.analysis.bisr import (
        derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs,
        generate_bisr_coeffs_from_sgd_workload,
    )
    from opacus.accountants.analysis.bnb import (
        build_bnb_toeplitz_c_matrix_and_contract,
    )
    from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload
    from opacus.accountants.bnb_inputs import (
        attach_accountant_coeff_surface,
        resolve_canonical_bnb_cycle_length,
        resolve_canonical_bsr_bands,
    )
    from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer

    state = canonicalize_bsr_family_runtime_state(
        mechanism=mechanism,
        runtime_state=runtime_state,
    )
    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    bands = resolve_canonical_bsr_bands(
        runtime_state={"bsr_bands": state.get("bnb_bands", state.get("bsr_bands"))},
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            f"{mechanism} analytical auto-coeff generation requires bands via "
            "`mechanism_state['bsr_bands']`, `sampling_semantics.privacy_metadata['bands']`, or `bsr_bands`"
        ),
    )
    bins = resolve_canonical_bnb_cycle_length(
        runtime_state=state,
        metadata=metadata,
        kwargs=kwargs,
        error_context=(
            "balls-in-bins MF state generation requires `bnb_b` or "
            "sampling_semantics privacy_metadata['bins']"
        ),
    )
    if not (isinstance(state.get("coeffs"), (list, tuple)) and len(state["coeffs"]) > 0):
        momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(
            optimizer=optimizer
        )
        steps_hint = int(
            kwargs.get("total_steps", metadata.get("total_steps", state.get("bnb_horizon", bins)))
        )
        if mechanism == "bisr":
            state["bisr_inv_coeffs"] = generate_bisr_coeffs_from_sgd_workload(
                bands=bands,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            # Re-run canonicalization so the shared state shape picks up the
            # derived runtime coefficients from the fresh inverse-side payload.
            state = canonicalize_bsr_family_runtime_state(
                mechanism=mechanism,
                runtime_state=state,
            )
        elif mechanism == "bandmf":
            state["coeffs"] = generate_bandmf_coeffs_from_sgd_workload(
                bands=bands,
                momentum=momentum,
                weight_decay=weight_decay,
                steps=steps_hint,
            )
        elif mechanism == "bandinvmf":
            raise ValueError(
                "bandinvmf balls-in-bins state shaping requires explicit inverse/runtime coeffs before family augmentation"
            )
        else:
            state["coeffs"] = generate_bsr_coeffs_from_sgd_workload(
                bands=bands,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        state["coeff_source"] = "analytical_auto"

    horizon = int(state.get("bnb_horizon", kwargs.get("total_steps", kwargs.get("steps", bins))))
    if mechanism == "bsr":
        accountant_coeffs, accountant_source = list(state["coeffs"]), "raw_c_col"
    elif mechanism == "bisr":
        accountant_coeffs = derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs(
            coeffs=list(state.get("bisr_inv_coeffs", state["coeffs"])),
            steps=horizon,
        )
        accountant_source = "abs_factor_c_col"
    elif mechanism == "bandmf":
        accountant_coeffs = derive_bandmf_amplified_accountant_coeffs_from_runtime_coeffs(
            coeffs=list(state["coeffs"])
        )
        accountant_source = "runtime_c_col"
    else:
        accountant_coeffs = derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs(
            inv_coeffs=list(state["bandinvmf_inv_coeffs"]),
            steps=horizon,
        )
        accountant_source = "abs_factor_c_col"

    state["bsr_bands"] = int(bands)
    state["bnb_bands"] = int(bands)
    state["bnb_horizon"] = int(horizon)
    state["bnb_bins"] = int(bins)
    state["bnb_cycle_length"] = int(bins)
    # Persist both the accountant coefficient surface and the Toeplitz matrix
    # contract so later BNB epsilon queries can reconstruct the same bridge.
    state = attach_accountant_coeff_surface(
        state,
        coeff_key="bnb_accountant_coeffs",
        coeff_source_key="bnb_accountant_coeffs_source",
        coeffs=accountant_coeffs,
        coeff_source=accountant_source,
    )
    c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=list(accountant_coeffs),
        bands=int(bands),
        horizon=int(horizon),
    )
    state["bnb_c_matrix"] = c_matrix
    state["bnb_c_matrix_contract"] = c_matrix_contract
    return state


def summarize_bsr_runtime_state(runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a lightweight provider-facing summary of canonical BSR-family state."""
    mechanism = str(runtime_state.get("_noise_mechanism", "bsr"))
    state = canonicalize_bsr_family_runtime_state(
        mechanism=mechanism,
        runtime_state=runtime_state,
    )
    coeffs = state.get("coeffs")
    coeff_count = len(coeffs) if isinstance(coeffs, (list, tuple)) else None
    coeff_head = list(coeffs[:5]) if isinstance(coeffs, (list, tuple)) else None
    coeff_source = state.get("coeff_source")
    if coeff_source is None and coeff_count:
        coeff_source = "explicit_or_precomputed"
    return {
        "coeff_count": coeff_count,
        "coeff_head": coeff_head,
        "coeff_source": coeff_source,
        "z_std": state.get("z_std"),
        "bsr_sensitivity_scale": state.get("bsr_sensitivity_scale"),
        "bsr_mf_sensitivity": state.get("bsr_mf_sensitivity"),
        "bsr_min_separation": state.get("bsr_min_separation"),
        "bsr_max_participations": state.get("bsr_max_participations"),
        "bsr_iterations_number": state.get("bsr_iterations_number"),
        "bsr_bands": state.get("bsr_bands"),
        "has_inverse_coeffs": bool(
            state.get("bisr_inv_coeffs") is not None
            or state.get("bandinvmf_inv_coeffs") is not None
        ),
    }


@dataclass(frozen=True)
class BSRFamily:
    """Registry-facing provider for BSR, BISR, BandMF, and Band-Inv-MF."""
    name: str

    def validate_sampling_compatibility(
        self,
        *,
        mechanism_config,
        poisson_sampling: bool,
        sampling_semantics,
        validate_cyclic_poisson_mode: bool,
    ) -> None:
        """
        Enforce the runtime/accountant sampling contracts for the selected family member.
        """
        mechanism = mechanism_config.mechanism
        if poisson_sampling:
            raise ValueError(
                f"{mechanism} mechanism requires fixed-batch semantics; "
                "set poisson_sampling=False"
            )

        accounting_mode = mechanism_config.accounting_mode
        if self.name == "bandmf":
            if sampling_semantics is None:
                raise ValueError(
                    "bandmf mechanism requires explicit sampling_semantics "
                    "with sampling_mode in {'cyclic_poisson', 'balls_in_bins'}"
                )
            if (
                accounting_mode == "bandmf_accountant"
                and sampling_semantics.sampling_mode not in ("cyclic_poisson", "torch_sampler")
            ):
                raise ValueError(
                    "bandmf mechanism with bandmf_accountant supports sampling_mode in "
                    "{'cyclic_poisson', 'torch_sampler'} only"
                )
            if (
                accounting_mode == "bnb_accountant"
                and sampling_semantics.sampling_mode != "balls_in_bins"
            ):
                raise ValueError(
                    "bandmf mechanism with bnb_accountant requires sampling_mode='balls_in_bins'"
                )
            if (
                accounting_mode == "random_allocation_accountant"
                and sampling_semantics.sampling_mode != "k_out_of_t"
            ):
                raise ValueError(
                    "bandmf mechanism with random_allocation_accountant requires sampling_mode='k_out_of_t'"
                )
            return

        if self.name == "bandinvmf":
            if accounting_mode == "bnb_accountant":
                if sampling_semantics is None or sampling_semantics.sampling_mode != "balls_in_bins":
                    raise ValueError(
                        "bandinvmf mechanism with bnb_accountant requires sampling_mode='balls_in_bins'"
                    )
            elif accounting_mode == "random_allocation_accountant":
                if sampling_semantics is None or sampling_semantics.sampling_mode != "k_out_of_t":
                    raise ValueError(
                        "bandinvmf mechanism with random_allocation_accountant requires sampling_mode='k_out_of_t'"
                    )
            elif accounting_mode == "bsr_accountant":
                if (
                    sampling_semantics is not None
                    and sampling_semantics.sampling_mode not in ("torch_sampler", "cyclic_poisson")
                ):
                    raise ValueError(
                        "bandinvmf mechanism with bsr_accountant supports sampling_mode in "
                        "{'torch_sampler', 'cyclic_poisson'} only"
                    )
            return

    def canonicalize(self, raw_state: Mapping[str, Any]) -> dict[str, Any]:
        """Canonicalize raw family state through the shared BSR-family normalizer."""
        return canonicalize_bsr_family_runtime_state(
            mechanism=self.name,
            runtime_state=raw_state,
        )

    def build_runtime(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build provider-layer runtime state for the active sampling mode."""
        state = self.canonicalize(mechanism_state)
        mode = context.get("sampling_mode")
        if mode == "cyclic_poisson":
            return augment_bsr_family_cyclic_query_state(
                mechanism=self.name,
                runtime_state=state,
                sampling_semantics=context.get("sampling_semantics"),
                steps=int(context.get("steps", 0)),
                kwargs=context.get("kwargs", {}),
            )
        if mode == "torch_sampler":
            return augment_bsr_family_fixed_batch_query_state(
                mechanism=self.name,
                runtime_state=state,
                sampling_semantics=context.get("sampling_semantics"),
                steps=int(context.get("steps", 0)),
                sample_rate=float(context.get("sample_rate", 0.0)),
                kwargs=context.get("kwargs", {}),
            )
        return state

    def summarize(self, mechanism_state: Mapping[str, Any]) -> dict[str, Any]:
        """Return a lightweight provider-facing summary of family runtime state."""
        return summarize_bsr_runtime_state(mechanism_state)

    def resolve_fixed_batch(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> float:
        """Resolve the fixed-batch accountant input for the active family member."""
        return resolve_bsr_family_fixed_batch_accountant_input(
            mechanism=self.name,
            runtime_state=mechanism_state,
            sampling_semantics=context.get("sampling_semantics"),
            steps=int(context.get("steps", 0)),
            sample_rate=float(context.get("sample_rate", 0.0)),
            kwargs=context.get("kwargs", {}),
        )

    def resolve_cyclic(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> float:
        """Resolve the cyclic accountant input for the active family member."""
        return resolve_bsr_family_cyclic_accountant_input(
            mechanism=self.name,
            runtime_state=mechanism_state,
            sampling_semantics=context.get("sampling_semantics"),
            steps=int(context.get("steps", 0)),
            kwargs=context.get("kwargs", {}),
        )

    def resolve_balls_in_bins(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve the balls-in-bins amplified accountant bridge state when supported."""
        optimizer = context.get("optimizer")
        sampling_semantics = context.get("sampling_semantics")
        if optimizer is None or sampling_semantics is None:
            return self.canonicalize(mechanism_state)
        return augment_bsr_family_balls_in_bins_query_state(
            mechanism=self.name,
            runtime_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            optimizer=optimizer,
            total_steps=int(context.get("total_steps", 0)),
            kwargs=context.get("kwargs", {}),
        )

    def resolve_target_epsilon_terms(
        self,
        *,
        mechanism_config,
        sampling_semantics,
        steps: int,
        sample_rate: float,
        kwargs: Mapping[str, Any],
        phase: str,
        query_runtime_context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], Optional[float]]:
        """
        Resolve the query-time terms required by the active BSR-family epsilon solver.
        """
        mechanism = mechanism_config.mechanism
        nm_kwargs: Dict[str, Any] = {}
        bsr_mf_sensitivity: Optional[float] = None

        if sampling_semantics is not None and sampling_semantics.sampling_mode == "cyclic_poisson":
            if phase == "epochs" and mechanism not in ("bandmf", "bisr"):
                return nm_kwargs, bsr_mf_sensitivity
            nm_kwargs["bsr_sensitivity_scale"] = self.resolve_cyclic(
                mechanism_state=mechanism_config.mechanism_state,
                context={
                    "sampling_semantics": sampling_semantics,
                    "steps": int(steps),
                    "kwargs": kwargs,
                },
            )

        if sampling_semantics is not None and sampling_semantics.sampling_mode == "torch_sampler":
            effective_kwargs = dict(kwargs)
            context = dict(query_runtime_context or {})
            effective_kwargs.setdefault("bsr_iterations_number", int(steps))
            dataset_size = _resolve_optional_int(context.get("dataset_size"), default=0)
            logical_batch_size = _resolve_optional_int(
                context.get("logical_batch_size"),
                default=0,
            )
            global_steps_per_epoch = (
                math.ceil(dataset_size / logical_batch_size)
                if dataset_size > 0 and logical_batch_size > 0
                else 0
            )
            if global_steps_per_epoch < 1:
                global_steps_per_epoch = _resolve_optional_int(
                    context.get("dataloader_len"),
                    default=0,
                )
            if global_steps_per_epoch < 1:
                global_steps_per_epoch = 1

            effective_kwargs.setdefault("bsr_min_separation", global_steps_per_epoch)
            effective_kwargs.setdefault(
                "bsr_max_participations",
                math.ceil(
                    int(effective_kwargs["bsr_iterations_number"])
                    / int(effective_kwargs["bsr_min_separation"])
                ),
            )
            bsr_mf_sensitivity = self.resolve_fixed_batch(
                mechanism_state=mechanism_config.mechanism_state,
                context={
                    "sampling_semantics": sampling_semantics,
                    "steps": int(steps),
                    "sample_rate": float(sample_rate),
                    "kwargs": effective_kwargs,
                },
            )

        return nm_kwargs, bsr_mf_sensitivity

    def augment_query_mechanism_config(
        self,
        *,
        mechanism_config,
        local_sampling_semantics,
        total_steps: Optional[int],
        epochs: Optional[int],
        poisson_sampling: bool,
        data_loader,
        kwargs: Mapping[str, Any],
        resolve_total_steps_sample_rate,
        optimizer=None,
        query_runtime_context: Optional[Mapping[str, Any]] = None,
    ):
        """
        Augment provider-layer query config with cyclic or amplified BNB bridge state.
        """
        from opacus.mechanism_contracts import NoiseMechanismConfig

        mechanism = mechanism_config.mechanism
        context = dict(query_runtime_context or {})
        if (
            local_sampling_semantics is not None
            and local_sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            if total_steps is None and epochs is None:
                return mechanism_config
            if total_steps is not None:
                scale_steps = int(total_steps)
            else:
                dataloader_len = _resolve_optional_int(
                    context.get("dataloader_len"),
                    default=0,
                )
                if dataloader_len < 1:
                    dataloader_len = int(len(data_loader)) if data_loader is not None else 0
                sample_rate = 1.0 / float(dataloader_len)
                scale_steps = int(epochs / sample_rate)

            nm_kwargs, _ = self.resolve_target_epsilon_terms(
                mechanism_config=mechanism_config,
                sampling_semantics=local_sampling_semantics,
                steps=int(scale_steps),
                sample_rate=0.0,
                kwargs=kwargs,
                phase="total_steps",
            )
            resolved_scale = nm_kwargs.get("bsr_sensitivity_scale")
            if resolved_scale is None:
                return mechanism_config
            # Persist the resolved cyclic sensitivity scale on the mechanism
            # state so the later sigma solve and epsilon queries agree on the
            # same cyclic contract.
            state = augment_bsr_family_cyclic_query_state(
                mechanism=mechanism,
                runtime_state=mechanism_config.mechanism_state,
                sampling_semantics=local_sampling_semantics,
                steps=int(scale_steps),
                kwargs=kwargs,
            )

            return NoiseMechanismConfig(
                mechanism=mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        if (
            local_sampling_semantics is not None
            and local_sampling_semantics.sampling_mode == "balls_in_bins"
            and mechanism_config.accounting_mode == "bnb_accountant"
            and optimizer is not None
        ):
            resolved_total_steps = _resolve_optional_int(
                total_steps,
                context.get("total_steps"),
                default=0,
            )

            return NoiseMechanismConfig(
                mechanism=mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=self.resolve_balls_in_bins(
                    mechanism_state=mechanism_config.mechanism_state,
                    context={
                        "sampling_semantics": local_sampling_semantics,
                        "kwargs": kwargs,
                        "optimizer": optimizer,
                        "total_steps": resolved_total_steps,
                    },
                ),
            )

        if local_sampling_semantics is None or local_sampling_semantics.sampling_mode == "torch_sampler":
            if total_steps is None and epochs is None:
                return mechanism_config

            loader_dataset_size = (
                len(data_loader.dataset)
                if data_loader is not None and getattr(data_loader, "dataset", None) is not None
                else None
            )
            dataset_size = _resolve_optional_int(
                context.get("dataset_size"),
                loader_dataset_size,
                default=0,
            )
            loader_batch_size = (
                getattr(data_loader, "batch_size", None)
                if data_loader is not None
                else None
            )
            logical_batch_size = _resolve_optional_int(
                context.get("logical_batch_size"),
                loader_batch_size,
                default=0,
            )
            sample_rate = resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=local_sampling_semantics,
                mechanism=mechanism,
                batch_size=logical_batch_size,
                dataset_size=dataset_size,
            )
            if total_steps is not None:
                mf_steps = int(total_steps)
            else:
                mf_steps = int(epochs / sample_rate)

            _nm_kwargs, resolved_mf_sensitivity = self.resolve_target_epsilon_terms(
                mechanism_config=mechanism_config,
                sampling_semantics=local_sampling_semantics,
                steps=int(mf_steps),
                sample_rate=float(sample_rate),
                kwargs=kwargs,
                phase="total_steps",
            )
            if resolved_mf_sensitivity is None:
                return mechanism_config

            state = augment_bsr_family_fixed_batch_query_state(
                mechanism=mechanism,
                runtime_state=mechanism_config.mechanism_state,
                sampling_semantics=local_sampling_semantics,
                steps=int(mf_steps),
                sample_rate=float(sample_rate),
                kwargs=kwargs,
            )
            metadata = (
                local_sampling_semantics.privacy_metadata
                if local_sampling_semantics is not None
                else {}
            )
            global_steps_per_epoch = math.ceil(dataset_size / logical_batch_size)
            if global_steps_per_epoch < 1:
                global_steps_per_epoch = _resolve_optional_int(
                    context.get("dataloader_len"),
                    default=0,
                )
            if global_steps_per_epoch < 1:
                global_steps_per_epoch = 1
            state.setdefault(
                "bsr_iterations_number",
                int(kwargs.get("bsr_iterations_number", mf_steps)),
            )
            state.setdefault(
                "bsr_min_separation",
                int(
                    metadata.get(
                        "bsr_min_separation",
                        kwargs.get("bsr_min_separation", global_steps_per_epoch),
                    )
                ),
            )
            state.setdefault(
                "bsr_max_participations",
                int(
                    metadata.get(
                        "bsr_max_participations",
                        kwargs.get(
                            "bsr_max_participations",
                            math.ceil(
                                int(state["bsr_iterations_number"])
                                / int(state["bsr_min_separation"])
                            ),
                        ),
                    )
                ),
            )
            return NoiseMechanismConfig(
                mechanism=mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        return mechanism_config
