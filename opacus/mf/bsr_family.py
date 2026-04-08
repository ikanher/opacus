from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional


def _noop_bsr_runtime_state_normalizer(*, state: Dict[str, Any]) -> None:
    return None


def _normalize_bisr_runtime_state(*, state: Dict[str, Any]) -> None:
    inv_coeffs = state.get("bisr_inv_coeffs")
    if not (isinstance(inv_coeffs, (list, tuple)) and len(inv_coeffs) > 0):
        return

    from opacus.accountants.analysis.bisr import (
        derive_bisr_runtime_coeffs_from_inverse_coeffs,
    )

    state["bisr_inv_coeffs"] = [float(c) for c in inv_coeffs]
    if not (isinstance(state.get("coeffs"), list) and len(state["coeffs"]) > 0):
        state["coeffs"] = derive_bisr_runtime_coeffs_from_inverse_coeffs(
            coeffs=state["bisr_inv_coeffs"]
        )
    state.setdefault("coeff_source", "analytical_inv_explicit")


def _normalize_bandinvmf_runtime_state(*, state: Dict[str, Any]) -> None:
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


def summarize_bsr_runtime_state(runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
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
    name: str

    def validate_sampling_compatibility(
        self,
        *,
        mechanism_config,
        poisson_sampling: bool,
        sampling_semantics,
        validate_cyclic_poisson_mode: bool,
    ) -> None:
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
        return summarize_bsr_runtime_state(mechanism_state)

    def resolve_fixed_batch(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> float:
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
        return self.canonicalize(mechanism_state)

    def resolve_target_epsilon_terms(
        self,
        *,
        mechanism_config,
        sampling_semantics,
        steps: int,
        sample_rate: float,
        kwargs: Mapping[str, Any],
        phase: str,
    ) -> tuple[dict[str, Any], Optional[float]]:
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
            bsr_mf_sensitivity = self.resolve_fixed_batch(
                mechanism_state=mechanism_config.mechanism_state,
                context={
                    "sampling_semantics": sampling_semantics,
                    "steps": int(steps),
                    "sample_rate": float(sample_rate),
                    "kwargs": kwargs,
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
    ):
        from opacus.mechanism_contracts import NoiseMechanismConfig

        mechanism = mechanism_config.mechanism
        if (
            local_sampling_semantics is not None
            and local_sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            if total_steps is not None:
                scale_steps = int(total_steps)
            else:
                sample_rate = 1.0 / len(data_loader)
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

        if local_sampling_semantics is None or local_sampling_semantics.sampling_mode == "torch_sampler":
            sample_rate = resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=local_sampling_semantics,
                mechanism=mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
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
            return NoiseMechanismConfig(
                mechanism=mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        return mechanism_config
