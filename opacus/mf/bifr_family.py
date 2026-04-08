from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional


def _validate_bifr_frac(frac: float) -> float:
    resolved = float(frac)
    if not (0.0 <= resolved <= 1.0):
        raise ValueError("BIFR frac must satisfy 0 <= frac <= 1")
    return resolved


def canonicalize_bifr_runtime_state(*, runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(dict(runtime_state))
    state["_noise_mechanism"] = "bifr"

    coeffs = state.get("coeffs")
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        state["coeffs"] = [float(c) for c in coeffs]

    if state.get("bsr_bands") is not None:
        state["bsr_bands"] = int(state["bsr_bands"])

    if state.get("bifr_frac") is None:
        state["bifr_frac"] = 0.5
    state["bifr_frac"] = float(_validate_bifr_frac(float(state["bifr_frac"])))

    for name in ("z_std", "bsr_mf_sensitivity"):
        if state.get(name) is not None:
            state[name] = float(state[name])

    for name in ("bsr_min_separation", "bsr_max_participations", "bsr_iterations_number"):
        if state.get(name) is not None:
            state[name] = int(state[name])

    if state.get("coeff_source") is None and isinstance(state.get("coeffs"), list) and len(state["coeffs"]) > 0:
        state["coeff_source"] = "explicit_or_precomputed"

    return state


def augment_bifr_family_fixed_batch_query_state(
    *,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    from opacus.accountants.bifr import resolve_bifr_mf_sensitivity_for_fixed_batch

    state = canonicalize_bifr_runtime_state(runtime_state=runtime_state)
    state["bsr_mf_sensitivity"] = resolve_bifr_mf_sensitivity_for_fixed_batch(
        mechanism_state=state,
        sampling_semantics=sampling_semantics,
        steps=int(steps),
        sample_rate=float(sample_rate),
        kwargs=kwargs,
    )
    return state


def summarize_bifr_runtime_state(runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
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
    }


class BIFRFamily:
    name = "bifr"

    def validate_sampling_compatibility(
        self,
        *,
        mechanism_config,
        poisson_sampling: bool,
        sampling_semantics,
        validate_cyclic_poisson_mode: bool,
    ) -> None:
        del validate_cyclic_poisson_mode
        if poisson_sampling:
            raise ValueError("bifr mechanism requires fixed-batch semantics; set poisson_sampling=False")
        if mechanism_config.accounting_mode != "bsr_accountant":
            raise ValueError("bifr mechanism currently supports accounting_mode='bsr_accountant' only")
        if sampling_semantics is not None and sampling_semantics.sampling_mode not in (None, "torch_sampler"):
            raise ValueError("bifr mechanism currently supports sampling_mode in {None, 'torch_sampler'} only")

    def canonicalize(self, raw_state: Mapping[str, Any]) -> dict[str, Any]:
        return canonicalize_bifr_runtime_state(runtime_state=raw_state)

    def build_runtime(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        state = self.canonicalize(mechanism_state)
        mode = context.get("sampling_mode")
        if mode in (None, "torch_sampler"):
            return augment_bifr_family_fixed_batch_query_state(
                runtime_state=state,
                sampling_semantics=context.get("sampling_semantics"),
                steps=int(context.get("steps", 0)),
                sample_rate=float(context.get("sample_rate", 0.0)),
                kwargs=context.get("kwargs", {}),
            )
        raise ValueError("bifr runtime currently supports fixed-batch torch_sampler semantics only")

    def summarize(self, mechanism_state: Mapping[str, Any]) -> dict[str, Any]:
        return summarize_bifr_runtime_state(mechanism_state)

    def resolve_fixed_batch(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        from opacus.accountants.bifr import resolve_bifr_mf_sensitivity_for_fixed_batch

        return float(
            resolve_bifr_mf_sensitivity_for_fixed_batch(
                mechanism_state=mechanism_state,
                sampling_semantics=context.get("sampling_semantics"),
                steps=int(context.get("steps", 0)),
                sample_rate=float(context.get("sample_rate", 0.0)),
                kwargs=context.get("kwargs", {}),
            )
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
    ) -> tuple[dict[str, Any], Optional[float]]:
        del phase
        if sampling_semantics is not None and sampling_semantics.sampling_mode not in (None, "torch_sampler"):
            raise ValueError("bifr target-epsilon calibration currently supports fixed-batch torch_sampler semantics only")
        return {}, self.resolve_fixed_batch(
            mechanism_state=mechanism_config.mechanism_state,
            context={
                "sampling_semantics": sampling_semantics,
                "steps": int(steps),
                "sample_rate": float(sample_rate),
                "kwargs": kwargs,
            },
        )

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

        if local_sampling_semantics is not None and local_sampling_semantics.sampling_mode not in (None, "torch_sampler"):
            raise ValueError("bifr runtime currently supports fixed-batch torch_sampler semantics only")

        sample_rate = resolve_total_steps_sample_rate(
            poisson_sampling=poisson_sampling,
            sampling_semantics=local_sampling_semantics,
            mechanism="bifr",
            batch_size=data_loader.batch_size,
            dataset_size=len(data_loader.dataset),
        )
        mf_steps = int(total_steps) if total_steps is not None else int(float(epochs) / float(sample_rate))
        state = augment_bifr_family_fixed_batch_query_state(
            runtime_state=mechanism_config.mechanism_state,
            sampling_semantics=local_sampling_semantics,
            steps=int(mf_steps),
            sample_rate=float(sample_rate),
            kwargs=kwargs,
        )
        return NoiseMechanismConfig(
            mechanism="bifr",
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )
