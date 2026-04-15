from __future__ import annotations

"""
BLT provider/family wrapper.

This module is intentionally thin. It owns provider-layer translation for BLT
inside the matrix-factorization family registry, while the real BLT runtime,
accountant, and calibration logic live elsewhere.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from opacus.accountants.analysis.blt import BLTPairedParams
from opacus.accountants.blt_inputs import (
    canonicalize_blt_public_or_runtime_state as _canonicalize_blt_public_or_runtime_state,
    resolve_blt_balls_in_bins_accountant_state as _resolve_blt_balls_in_bins_accountant_state,
    resolve_blt_fixed_batch_accountant_inputs as _resolve_blt_fixed_batch_accountant_inputs,
    resolve_blt_workload_mechanism_state as _resolve_blt_workload_mechanism_state,
    summarize_blt_report_surface as _summarize_blt_report_surface,
)
from opacus.mf.state import BLTFamilyState
from opacus.accountants.blt_fixed_batch import optimize_blt_fixed_batch


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


def canonicalize_blt_public_or_runtime_state(
    mechanism_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Provider wrapper for the accountant-owned BLT canonicalization contract."""
    return _canonicalize_blt_public_or_runtime_state(mechanism_state)


def resolve_blt_fixed_batch_accountant_inputs(
    *,
    runtime_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    total_steps: int,
) -> tuple[BLTPairedParams, float, int, int, int]:
    """Provider wrapper for the fixed-batch BLT accountant-input contract."""
    return _resolve_blt_fixed_batch_accountant_inputs(
        runtime_state=runtime_state,
        metadata=metadata,
        kwargs=kwargs,
        total_steps=total_steps,
    )


def summarize_blt_runtime_state(runtime_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight provider-facing summary of BLT runtime metadata."""
    state = canonicalize_blt_public_or_runtime_state(runtime_state)
    return {
        "forward_theta_len": len(state.get("forward", {}).get("theta", [])),
        "inverse_theta_len": len(state.get("inverse", {}).get("theta", [])),
        "has_noise_multiplier_ref": state.get("noise_multiplier_ref") is not None,
        "distributed_policy": state.get("_blt_distributed_policy", "ddp_flat_only"),
        "distributed_runtime": bool(
            state.get("_blt_distributed_runtime", False)
        ),
    }


def summarize_blt_report_surface(runtime_state: Mapping[str, Any]) -> dict[str, Any]:
    """Provider wrapper for the BLT report-surface summary."""
    return _summarize_blt_report_surface(runtime_state)


def resolve_blt_balls_in_bins_accountant_state(
    *,
    runtime_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    total_steps: int,
) -> dict[str, Any]:
    """Provider wrapper for the amplified BLT `balls_in_bins` accountant bridge."""
    return _resolve_blt_balls_in_bins_accountant_state(
        runtime_state=runtime_state,
        metadata=metadata,
        kwargs=kwargs,
        total_steps=total_steps,
    )


def resolve_blt_workload_mechanism_state(**kwargs) -> dict[str, Any]:
    """Provider wrapper for workload-driven BLT state resolution."""
    return _resolve_blt_workload_mechanism_state(**kwargs)


def augment_blt_family_query_state(
    *,
    mechanism_state: Mapping[str, Any],
    sampling_semantics,
    total_steps: int,
    dataset_size: int,
    logical_batch_size: int,
    max_grad_norm: float | None,
    loss_reduction: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    state = BLTFamilyState.from_input_state(mechanism_state)
    metadata = (
        sampling_semantics.privacy_metadata
        if sampling_semantics is not None
        else {}
    )
    changed = state.apply_explicit_overrides(metadata=metadata, kwargs=kwargs)
    if (
        sampling_semantics is None
        or sampling_semantics.sampling_mode != "torch_sampler"
        or total_steps < 1
        or dataset_size < 1
        or logical_batch_size < 1
        or max_grad_norm is None
    ):
        return state.to_state_dict() if changed else dict(mechanism_state)

    calibration_denominator = (
        1.0 if loss_reduction == "sum" else float(logical_batch_size)
    )
    changed = (
        state.apply_torch_sampler_defaults(
            total_steps=int(total_steps),
            dataset_size=int(dataset_size),
            logical_batch_size=int(logical_batch_size),
            max_grad_norm=float(max_grad_norm),
            calibration_denominator=float(calibration_denominator),
        )
        or changed
    )
    return state.to_state_dict() if changed else dict(mechanism_state)


@dataclass(frozen=True)
class BLTFamily:
    """Registry-facing BLT family entry used by `PrivacyEngine` and MF dispatch."""
    name: str = "blt"

    def validate_sampling_compatibility(
        self,
        *,
        mechanism_config,
        poisson_sampling: bool,
        sampling_semantics,
        validate_cyclic_poisson_mode: bool,
    ) -> None:
        return None

    def canonicalize(self, raw_state: Mapping[str, Any]) -> dict[str, Any]:
        return canonicalize_blt_public_or_runtime_state(raw_state)

    def build_runtime(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = BLTFamilyState.from_input_state(mechanism_state)
        metadata = context.get("metadata", {})
        kwargs = context.get("kwargs", {})
        state.apply_explicit_overrides(metadata=metadata, kwargs=kwargs)
        return state.to_state_dict()

    def summarize(self, mechanism_state: Mapping[str, Any]) -> dict[str, Any]:
        return summarize_blt_runtime_state(mechanism_state)

    def resolve_fixed_batch(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Any:
        return resolve_blt_fixed_batch_accountant_inputs(
            runtime_state=mechanism_state,
            metadata=context.get("metadata", {}),
            kwargs=context.get("kwargs", {}),
            total_steps=int(context.get("total_steps", 0)),
        )

    def resolve_balls_in_bins(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return resolve_blt_balls_in_bins_accountant_state(
            runtime_state=mechanism_state,
            metadata=context.get("metadata", {}),
            kwargs=context.get("kwargs", {}),
            total_steps=int(context.get("total_steps", 0)),
        )

    def optimize(self, *, objective: Any, context: Mapping[str, Any]) -> Any:
        kwargs = dict(context)
        kwargs.update(dict(objective) if isinstance(objective, Mapping) else {})

        return optimize_blt_fixed_batch(**kwargs)

    def resolve_target_epsilon_terms(
        self,
        *,
        mechanism_config,
        sampling_semantics,
        steps: int,
        sample_rate: float,
        kwargs: Mapping[str, Any],
        phase: str,
        query_runtime_context: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], None]:
        del query_runtime_context

        return {}, None

    def augment_query_mechanism_config(
        self,
        *,
        mechanism_config,
        local_sampling_semantics,
        total_steps: int | None,
        epochs: int | None,
        poisson_sampling: bool,
        data_loader,
        kwargs: Mapping[str, Any],
        resolve_total_steps_sample_rate,
        optimizer=None,
        query_runtime_context: Mapping[str, Any] | None = None,
    ):
        from opacus.mechanism_contracts import NoiseMechanismConfig

        del resolve_total_steps_sample_rate
        del optimizer
        context = dict(query_runtime_context or {})
        sampling_mode = (
            None if local_sampling_semantics is None else local_sampling_semantics.sampling_mode
        )
        if sampling_mode in ("balls_in_bins", "b_min_sep"):
            resolved_total_steps = _resolve_optional_int(
                total_steps,
                context.get("total_steps"),
                default=0,
            )
            return NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=self.resolve_balls_in_bins(
                    mechanism_state=mechanism_config.mechanism_state,
                    context={
                        "metadata": local_sampling_semantics.privacy_metadata,
                        "kwargs": kwargs,
                        "total_steps": resolved_total_steps,
                    },
                ),
            )

        max_grad_norm = context.get("max_grad_norm")
        if isinstance(max_grad_norm, list):
            max_grad_norm = None

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
        resolved_total_steps = _resolve_optional_int(
            total_steps,
            context.get("total_steps"),
            default=0,
        )
        state = augment_blt_family_query_state(
            mechanism_state=mechanism_config.mechanism_state,
            sampling_semantics=local_sampling_semantics,
            total_steps=resolved_total_steps,
            dataset_size=dataset_size,
            logical_batch_size=logical_batch_size,
            max_grad_norm=(
                None if max_grad_norm is None else float(max_grad_norm)
            ),
            loss_reduction=str(context.get("loss_reduction", "mean")),
            kwargs=kwargs,
        )
        if state == mechanism_config.mechanism_state:
            return mechanism_config

        return NoiseMechanismConfig(
            mechanism=mechanism_config.mechanism,
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )
