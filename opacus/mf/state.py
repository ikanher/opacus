from __future__ import annotations

"""
Canonical state carriers for MF provider-layer runtime/accountant contracts.

These dataclasses do not implement privacy accounting. They normalize,
summarize, and update family-local mechanism state so provider modules and
`PrivacyEngine` can exchange one stable payload format.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from torch import optim

from opacus.accountants.analysis.blt import BLTParams, BLTPairedParams
from opacus.accountants.bnb_inputs import resolve_canonical_bsr_bands
from opacus.accountants.blt_inputs import (
    canonicalize_blt_public_or_runtime_state,
    resolve_blt_fixed_batch_accountant_inputs,
)
from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer


@dataclass
class BLTFamilyState:
    """
    Canonical BLT runtime/accountant state carried through the provider layer.

    Attributes:
        pair: Canonical BLT forward/inverse pair parameters.
        z_std: Runtime correlated-noise scale consumed by the BLT mechanism.
        metadata: Extra workload/accounting metadata such as horizon,
            min-separation, max participations, and calibration diagnostics.
    """

    pair: BLTPairedParams
    z_std: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input_state(cls, mechanism_state: Mapping[str, Any]) -> "BLTFamilyState":
        """
        Rebuild a canonical BLT family state from any accepted public/runtime payload.
        """
        state = canonicalize_blt_public_or_runtime_state(mechanism_state)
        pair = BLTPairedParams(
            forward=BLTParams(
                theta=state["forward"]["theta"],
                omega=state["forward"]["omega"],
            ),
            inverse=BLTParams(
                theta=state["inverse"]["theta"],
                omega=state["inverse"]["omega"],
            ),
        ).canonicalized()
        extras: Dict[str, Any] = {}
        for key, value in state.items():
            if key in {"forward", "inverse", "z_std"}:
                continue
            extras[key] = copy.deepcopy(value)
        return cls(pair=pair, z_std=float(state["z_std"]), metadata=extras)

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Serialize the canonical BLT family state back to the public/runtime payload shape.
        """
        forward = self.pair.forward.canonicalized()
        inverse = self.pair.inverse.canonicalized()
        state = {
            "forward": {
                "theta": [float(x) for x in forward.theta_array()],
                "omega": [float(x) for x in forward.omega_array()],
            },
            "inverse": {
                "theta": [float(x) for x in inverse.theta_array()],
                "omega": [float(x) for x in inverse.omega_array()],
            },
            "z_std": float(self.z_std),
        }
        state.update(copy.deepcopy(self.metadata))
        return state

    def apply_explicit_overrides(
        self,
        *,
        metadata: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> bool:
        """
        Fill missing BLT metadata from explicit metadata/kwargs overrides.
        """
        changed = False
        for field_name in (
            "blt_horizon",
            "blt_min_separation",
            "blt_max_participations",
            "noise_multiplier_ref",
        ):
            if self.metadata.get(field_name) is not None:
                continue
            explicit_value = kwargs.get(field_name)
            if explicit_value is None:
                explicit_value = metadata.get(field_name)
            if explicit_value is None:
                continue
            self.metadata[field_name] = explicit_value
            changed = True
        return changed

    def apply_torch_sampler_defaults(
        self,
        *,
        total_steps: int,
        dataset_size: int,
        logical_batch_size: int,
        max_grad_norm: float,
        calibration_denominator: float,
    ) -> bool:
        """
        Derive the fixed-batch BLT defaults implied by a torch-sampler workload.
        """
        if total_steps < 1 or dataset_size < 1 or logical_batch_size < 1:
            return False

        steps_per_epoch = int(math.ceil(float(dataset_size) / float(logical_batch_size)))
        if steps_per_epoch < 1:
            return False

        changed = False
        if self.metadata.get("blt_horizon") is None:
            self.metadata["blt_horizon"] = int(total_steps)
            changed = True
        if self.metadata.get("blt_min_separation") is None:
            # BLT uses one participation slot per epoch-sized interval under the
            # plain torch-sampler fallback, so min-separation defaults to the
            # derived steps-per-epoch quantity.
            self.metadata["blt_min_separation"] = int(steps_per_epoch)
            changed = True
        if self.metadata.get("blt_max_participations") is None:
            self.metadata["blt_max_participations"] = int(
                math.ceil(float(total_steps) / float(steps_per_epoch))
            )
            changed = True
        if self.metadata.get("noise_multiplier_ref") is None:
            self.metadata["noise_multiplier_ref"] = (
                float(self.z_std) * float(calibration_denominator) / float(max_grad_norm)
            )
            changed = True
        return changed

    def supports_fixed_batch_accountant(self) -> bool:
        """Return whether the state already carries the full fixed-batch BLT contract."""
        required_fields = (
            "noise_multiplier_ref",
            "blt_max_participations",
            "blt_min_separation",
            "blt_horizon",
        )
        return all(self.metadata.get(field_name) is not None for field_name in required_fields)

    def resolve_fixed_batch_accountant_inputs(
        self,
        *,
        metadata: Mapping[str, Any],
        kwargs: Mapping[str, Any],
        total_steps: int,
    ) -> tuple[BLTPairedParams, float, int, int, int]:
        """
        Resolve the accountant-owned BLT fixed-batch input tuple from canonical state.
        """
        return resolve_blt_fixed_batch_accountant_inputs(
            runtime_state=self.to_state_dict(),
            metadata=metadata,
            kwargs=kwargs,
            total_steps=int(total_steps),
        )


@dataclass
class BSRFamilyState:
    """
    Canonical provider-layer state carrier for the BSR/BISR/BandMF/BandInvMF family.

    Attributes:
        mechanism: Active family name.
        state: Canonicalized runtime/accountant state dictionary for that family.
    """

    mechanism: str
    state: Dict[str, Any]

    @classmethod
    def from_state(cls, *, mechanism: str, mechanism_state: Mapping[str, Any]) -> "BSRFamilyState":
        """
        Create a mutable family-state wrapper from a raw provider/runtime payload.
        """
        state = copy.deepcopy(dict(mechanism_state))
        state["_noise_mechanism"] = mechanism
        return cls(mechanism=mechanism, state=state)

    def to_state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def has_runtime_coeffs(self) -> bool:
        """Return whether runtime Toeplitz coefficients are already materialized."""
        coeffs = self.state.get("coeffs")
        return isinstance(coeffs, (list, tuple)) and len(coeffs) > 0

    def has_inverse_coeffs(self) -> bool:
        """Return whether inverse-side coefficients are already available."""
        coeffs = self.state.get("bisr_inv_coeffs")
        return isinstance(coeffs, (list, tuple)) and len(coeffs) > 0

    def ensure_bisr_runtime_coeffs_from_inverse(self) -> bool:
        """
        Materialize BISR runtime coefficients from explicit inverse coefficients when needed.
        """
        if self.mechanism != "bisr" or not self.has_inverse_coeffs():
            return False
        from opacus.accountants.analysis.bisr import (
            derive_bisr_runtime_coeffs_from_inverse_coeffs,
        )

        self.state["bisr_inv_coeffs"] = [float(c) for c in self.state["bisr_inv_coeffs"]]
        if not self.has_runtime_coeffs():
            # BISR may be specified on the inverse side only; the provider layer
            # fills in the correlated-runtime coefficients lazily.
            self.state["coeffs"] = derive_bisr_runtime_coeffs_from_inverse_coeffs(
                coeffs=self.state["bisr_inv_coeffs"],
            )
        return True

    def resolve_bands(self, *, metadata: Mapping[str, Any], kwargs: Mapping[str, Any]) -> int:
        """
        Resolve the canonical band count from runtime state, metadata, or kwargs.
        """
        return resolve_canonical_bsr_bands(
            runtime_state=self.state,
            metadata=metadata,
            kwargs=kwargs,
            error_context=(
                "auto coeff generation requires bands via `mechanism_state['bsr_bands']`, "
                "`sampling_semantics.privacy_metadata['bands']`, or `bsr_bands`"
            ),
        )

    def generate_analytical_coeffs(
        self,
        *,
        bands: int,
        optimizer: optim.Optimizer,
    ) -> None:
        """
        Generate analytical family coefficients from a uniform SGD workload.
        """
        from opacus.accountants.analysis.bisr import (
            derive_bisr_runtime_coeffs_from_inverse_coeffs,
            generate_bisr_coeffs_from_sgd_workload,
        )
        from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload

        momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(
            optimizer=optimizer
        )
        if self.mechanism == "bisr":
            self.state["bisr_inv_coeffs"] = generate_bisr_coeffs_from_sgd_workload(
                bands=bands,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            self.state["coeffs"] = derive_bisr_runtime_coeffs_from_inverse_coeffs(
                coeffs=self.state["bisr_inv_coeffs"],
            )
        else:
            self.state["coeffs"] = generate_bsr_coeffs_from_sgd_workload(
                bands=bands,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        self.state["bsr_bands"] = int(bands)
        self.state["coeff_source"] = "analytical_auto"
