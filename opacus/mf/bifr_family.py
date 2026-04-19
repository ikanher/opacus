from __future__ import annotations

"""
Provider-layer family wrapper for BIFR.

This module owns the BIFR-specific registry hooks used by `PrivacyEngine`. It
does not own the BIFR accountant formulas themselves; those remain in
`opacus.accountants.bifr` and `opacus.accountants.bifr_inputs`.
"""

from typing import Any, Dict, Mapping, Optional

from opacus.accountants.bifr_inputs import (
    canonicalize_bifr_runtime_state,
    summarize_bifr_runtime_state,
)
from opacus.mf.interfaces import SupportsBallsInBins


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


def augment_bifr_family_fixed_batch_query_state(
    *,
    runtime_state: Mapping[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float,
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Stamp fixed-batch BIFR sensitivity onto a canonical runtime state payload.
    """
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


class BIFRFamily(SupportsBallsInBins):
    """Registry-facing BIFR family provider."""
    name = "bifr"

    def validate_sampling_compatibility(
        self,
        *,
        mechanism_config,
        poisson_sampling: bool,
        sampling_semantics,
        validate_cyclic_poisson_mode: bool,
    ) -> None:
        """
        Enforce the runtime/accountant sampling contracts currently supported by BIFR.
        """
        del validate_cyclic_poisson_mode
        if poisson_sampling:
            raise ValueError("bifr mechanism requires fixed-batch semantics; set poisson_sampling=False")

        if mechanism_config.accounting_mode == "bsr_accountant":
            if sampling_semantics is not None and sampling_semantics.sampling_mode not in (None, "torch_sampler"):
                raise ValueError("bifr mechanism with bsr_accountant supports sampling_mode in {None, 'torch_sampler'} only")
            return

        if mechanism_config.accounting_mode == "bnb_accountant":
            if sampling_semantics is None or sampling_semantics.sampling_mode not in ("balls_in_bins", "b_min_sep"):
                raise ValueError(
                    "bifr mechanism with bnb_accountant requires sampling_mode in {'balls_in_bins', 'b_min_sep'}"
                )
            return

        if mechanism_config.accounting_mode == "random_allocation_accountant":
            if sampling_semantics is None or sampling_semantics.sampling_mode != "k_out_of_t":
                raise ValueError(
                    "bifr mechanism with random_allocation_accountant requires sampling_mode='k_out_of_t'"
                )
            return

        raise ValueError(
            "bifr mechanism currently supports accounting_mode in "
            "{'bsr_accountant', 'bnb_accountant', 'random_allocation_accountant'} only"
        )

    def canonicalize(self, raw_state: Mapping[str, Any]) -> dict[str, Any]:
        """Canonicalize raw BIFR runtime state through the accountant-owned input layer."""
        return canonicalize_bifr_runtime_state(runtime_state=raw_state)

    def build_runtime(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        """
        Build the provider-layer runtime/accountant state for the active BIFR sampling mode.
        """
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

        if mode in ("balls_in_bins", "b_min_sep"):
            # The amplified BNB path needs an accountant-state bridge rather than
            # the plain fixed-batch sensitivity scalar.
            return self.resolve_balls_in_bins(
                mechanism_state=state,
                context={
                    "metadata": (
                        context.get("sampling_semantics").privacy_metadata
                        if context.get("sampling_semantics") is not None
                        else context.get("metadata", {})
                    ),
                    "kwargs": context.get("kwargs", {}),
                    "total_steps": int(context.get("steps", 0)),
                    "optimizer": context.get("optimizer"),
                    "sampling_mode": mode,
                },
            )

        if mode == "k_out_of_t":
            return state

        raise ValueError("bifr runtime currently supports fixed-batch or supported BNB amplified semantics only")

    def summarize(self, mechanism_state: Mapping[str, Any]) -> dict[str, Any]:
        """Return a lightweight provider-facing summary of canonical BIFR state."""
        return summarize_bifr_runtime_state(mechanism_state)

    def resolve_balls_in_bins(
        self,
        *,
        mechanism_state: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve the amplified BNB accountant-state bridge for BIFR.
        """
        from opacus.accountants.bifr import resolve_bifr_bnb_accountant_state
        from opacus.mechanism_contracts import SamplingSemantics

        metadata = dict(context.get("metadata", {}))
        kwargs = dict(context.get("kwargs", {}))
        sampling_mode = str(context.get("sampling_mode", metadata.get("sampling_mode", "balls_in_bins")))

        return resolve_bifr_bnb_accountant_state(
            mechanism_state=mechanism_state,
            sampling_semantics=SamplingSemantics(
                sampling_mode=sampling_mode,
                privacy_metadata=metadata,
            ),
            optimizer=context.get("optimizer"),
            kwargs=kwargs,
            total_steps=int(context.get("total_steps", 0)),
        )

    def resolve_fixed_batch(self, *, mechanism_state: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        """Resolve the fixed-batch BIFR sensitivity used by the BSR accountant path."""
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
        query_runtime_context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], Optional[float]]:
        """
        Resolve the query-time terms needed by BIFR epsilon calibration.
        """
        del query_runtime_context
        del phase
        if sampling_semantics is not None and sampling_semantics.sampling_mode not in (None, "torch_sampler"):
            if sampling_semantics.sampling_mode in ("balls_in_bins", "b_min_sep"):
                return {}, None
            if sampling_semantics.sampling_mode == "k_out_of_t":
                return {}, None
            raise ValueError("bifr target-epsilon calibration currently supports fixed-batch or supported BNB amplified semantics only")

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
        optimizer=None,
        query_runtime_context: Optional[Mapping[str, Any]] = None,
    ):
        """
        Augment BIFR query config with the family-specific state required by the active accountant path.
        """
        from opacus.mechanism_contracts import NoiseMechanismConfig

        context = dict(query_runtime_context or {})

        if local_sampling_semantics is not None and local_sampling_semantics.sampling_mode in ("balls_in_bins", "b_min_sep"):
            resolved_total_steps = _resolve_optional_int(total_steps, default=0)
            return NoiseMechanismConfig(
                mechanism="bifr",
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=self.resolve_balls_in_bins(
                    mechanism_state=mechanism_config.mechanism_state,
                    context={
                        "metadata": local_sampling_semantics.privacy_metadata,
                        "kwargs": kwargs,
                        "sampling_mode": local_sampling_semantics.sampling_mode,
                        "total_steps": resolved_total_steps,
                        "optimizer": optimizer,
                    },
                ),
            )

        if local_sampling_semantics is not None and local_sampling_semantics.sampling_mode == "k_out_of_t":
            return mechanism_config

        if local_sampling_semantics is not None and local_sampling_semantics.sampling_mode not in (None, "torch_sampler"):
            raise ValueError("bifr runtime currently supports fixed-batch or supported BNB amplified semantics only")

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
            mechanism="bifr",
            batch_size=logical_batch_size,
            dataset_size=dataset_size,
        )
        # BIFR fixed-batch sensitivity still lives on the BSR-accountant side,
        # so query augmentation computes and persists that scalar here.
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
