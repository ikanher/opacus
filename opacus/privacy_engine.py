#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
import copy
import json
import math
import time
from itertools import chain
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union

import torch
from opacus.mechanism_contracts import NoiseMechanismConfig, SamplingSemantics
from opacus.accountants import create_accountant
from opacus.accountants.bnb import (
    resolve_bnb_b_min_sep_inputs as resolve_bnb_b_min_sep_inputs_helper,
    validate_bnb_accounting_runtime_consistency as validate_bnb_accounting_runtime_consistency_helper,
    validate_bnb_sampling_policy as validate_bnb_sampling_policy_helper,
)
from opacus.accountants.bsr import (
    ensure_bsr_family_cyclic_coeffs as ensure_bsr_family_cyclic_coeffs_helper,
    ensure_bsr_family_fixed_analytical_coeffs as ensure_bsr_family_fixed_analytical_coeffs_helper,
    resolve_bisr_mf_sensitivity_for_fixed_batch as resolve_bisr_mf_sensitivity_for_fixed_batch_helper,
    resolve_bisr_sensitivity_scale_for_cyclic as resolve_bisr_sensitivity_scale_for_cyclic_helper,
    resolve_bsr_mf_sensitivity_for_fixed_batch as resolve_bsr_mf_sensitivity_for_fixed_batch_helper,
    resolve_bsr_sensitivity_scale_for_cyclic as resolve_bsr_sensitivity_scale_for_cyclic_helper,
)
from opacus.accountants.bandinvmf import (
    ensure_bandinvmf_runtime_state as ensure_bandinvmf_runtime_state_helper,
    resolve_bandinvmf_mf_sensitivity_for_fixed_batch as resolve_bandinvmf_mf_sensitivity_for_fixed_batch_helper,
    resolve_bandinvmf_sensitivity_scale_for_cyclic as resolve_bandinvmf_sensitivity_scale_for_cyclic_helper,
)
from opacus.accountants.bandmf import (
    ensure_bandmf_fixed_analytical_coeffs as ensure_bandmf_fixed_analytical_coeffs_helper,
    resolve_bandmf_mf_sensitivity_for_fixed_batch as resolve_bandmf_mf_sensitivity_for_fixed_batch_helper,
)
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis.bsr import (
    calibrate_bsr_z_std,
    generate_bsr_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bandmf import (
    derive_bandmf_amplified_accountant_coeffs_from_runtime_coeffs,
    generate_bandmf_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bisr import (
    derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs,
    derive_bisr_runtime_coeffs_from_inverse_coeffs,
    generate_bisr_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bandinvmf import (
    derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs,
)
from opacus.accountants.analysis.bnb import (
    BNBCalibrationStatus,
    build_bnb_toeplitz_c_matrix_and_contract,
    normalize_bnb_accountant_coeffs,
    describe_bnb_calibration_report,
    make_bnb_calibration_report,
    parse_bnb_calibration_report,
    resolve_bnb_calibration_kwargs,
    sample_b_min_sep_llr,
    verify_hockey_stick_delta_hoeffding,
)
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import (
    AbstractGradSampleModule,
    GradSampleModule,
    get_gsm_class,
    wrap_model,
)
from opacus.optimizers import (
    CorrelatedNoiseMechanism,
    DPOptimizer,
    InverseBandNoiseMechanism,
    get_optimizer_class,
)
from opacus.schedulers import _GradClipScheduler, _NoiseScheduler
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.validators.module_validator import ModuleValidator
from opacus.utils.uniform_sampler import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    DistributedBallsInBinsSampler,
    DistributedBMinSepSampler,
    DistributedCyclicPoissonSampler,
    DistributedFixedSampler,
)
from torch import nn, optim
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PrivacyEngine:
    """
    Main entry point to the Opacus API - use ``PrivacyEngine``  to enable differential
    privacy for your model training.

    ``PrivacyEngine`` object encapsulates current privacy state (privacy budget +
    method it's been calculated) and exposes ``make_private`` method to wrap your
    PyTorch training objects with their private counterparts.

    Example:
        >>> dataloader = demo_dataloader
        >>> model = MyCustomModel()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> privacy_engine = PrivacyEngine()
        >>>
        >>> model, optimizer, dataloader = privacy_engine.make_private(
        ...    module=model,
        ...    optimizer=optimizer,
        ...    data_loader=dataloader,
        ...    noise_multiplier=1.0,
        ...    max_grad_norm=1.0,
        ... )
        >>> # continue training as normal
    """

    @staticmethod
    def _annotate_mechanism_state(
        *,
        mechanism: str,
        mechanism_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = copy.deepcopy(mechanism_state)
        state["_noise_mechanism"] = mechanism
        return state

    @staticmethod
    def _ensure_bnb_balls_in_bins_mf_state(
        *,
        mechanism_config: NoiseMechanismConfig,
        optimizer: optim.Optimizer,
        sampling_semantics: Optional[SamplingSemantics],
        kwargs: Dict[str, Any],
    ) -> NoiseMechanismConfig:
        """
        Ensure balls-in-bins BNB MF runs have analytical coeffs and Toeplitz state.

        This fills the gap between fixed/cyclic MF auto-coeff generation and the
        BNB accountant path: amplified MF runs may provide only
        `bsr_bands`, in which case we derive coefficients from the optimizer
        workload metadata and materialize the BNB Toeplitz contract before the
        accountant validates runtime inputs.
        """
        if mechanism_config.mechanism not in ("bsr", "bisr", "bandmf", "bandinvmf"):
            return mechanism_config
        if mechanism_config.accounting_mode != "bnb_accountant":
            return mechanism_config
        if (
            sampling_semantics is None
            or sampling_semantics.sampling_mode != "balls_in_bins"
        ):
            return mechanism_config

        state = copy.deepcopy(mechanism_config.mechanism_state)
        state["_noise_mechanism"] = mechanism_config.mechanism
        metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}

        coeffs = state.get("coeffs")
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
            metadata.get("bands", state.get("bnb_bands", state.get("bsr_bands"))),
        )
        if bands is None:
            raise ValueError(
                f"{mechanism_config.mechanism} analytical auto-coeff generation requires bands via "
                "`mechanism_state['bsr_bands']`, `sampling_semantics.privacy_metadata['bands']`, "
                "or `bsr_bands`"
            )
        bands = int(bands)
        if bands < 1:
            raise ValueError(f"{mechanism_config.mechanism} bands must be >= 1")

        metadata_bins = metadata.get("bins", metadata.get("b"))
        explicit_cycle_length = kwargs.get("bnb_cycle_length", state.get("bnb_cycle_length"))
        legacy_bins = state.get("bnb_bins")
        explicit_bins = kwargs.get("bnb_b")

        if explicit_cycle_length is not None:
            explicit_cycle_length = int(explicit_cycle_length)
            if explicit_cycle_length < 1:
                raise ValueError("balls-in-bins cycle length must be >= 1")

            if metadata_bins is not None and int(metadata_bins) != explicit_cycle_length:
                raise ValueError(
                    "conflicting canonical inputs: `bnb_cycle_length` must match "
                    "sampling_semantics privacy_metadata['bins']"
                )

            if explicit_bins is not None and int(explicit_bins) != explicit_cycle_length:
                raise ValueError(
                    "conflicting canonical inputs: `bnb_cycle_length` must match "
                    "`bnb_b`"
                )

            if legacy_bins is not None and int(legacy_bins) != explicit_cycle_length:
                raise ValueError(
                    "conflicting canonical inputs: `bnb_cycle_length` must match "
                    "mechanism_state['bnb_bins']"
                )

        bins = kwargs.get(
            "bnb_b",
            metadata_bins
            if metadata_bins is not None
            else (
                explicit_cycle_length
                if explicit_cycle_length is not None
                else state.get("bnb_cycle_length", state.get("bnb_bins"))
            ),
        )
        if bins is None:
            raise ValueError(
                "balls-in-bins MF state generation requires `bnb_b` or "
                "sampling_semantics privacy_metadata['bins']"
            )
        bins = int(bins)
        if bins < 1:
            raise ValueError("balls-in-bins bins must be >= 1")

        bisr_inv_coeffs = state.get("bisr_inv_coeffs")
        if (
            mechanism_config.mechanism == "bisr"
            and isinstance(bisr_inv_coeffs, (list, tuple))
            and len(bisr_inv_coeffs) > 0
            and not (isinstance(coeffs, (list, tuple)) and len(coeffs) > 0)
        ):
            state["bisr_inv_coeffs"] = [float(c) for c in bisr_inv_coeffs]
            state["coeffs"] = derive_bisr_runtime_coeffs_from_inverse_coeffs(
                coeffs=state["bisr_inv_coeffs"],
            )
            state.setdefault("coeff_source", "analytical_inv_explicit")
        elif not (isinstance(coeffs, (list, tuple)) and len(coeffs) > 0):
            momentum, weight_decay = PrivacyEngine._resolve_uniform_sgd_workload_from_optimizer(
                optimizer=optimizer
            )
            if mechanism_config.mechanism == "bisr":
                state["bisr_inv_coeffs"] = generate_bisr_coeffs_from_sgd_workload(
                    bands=bands,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
                state["coeffs"] = derive_bisr_runtime_coeffs_from_inverse_coeffs(
                    coeffs=state["bisr_inv_coeffs"],
                )
            elif mechanism_config.mechanism == "bandmf":
                steps_hint = kwargs.get(
                    "total_steps",
                    metadata.get("total_steps", state.get("bnb_horizon", bins)),
                )
                state["coeffs"] = generate_bandmf_coeffs_from_sgd_workload(
                    bands=bands,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    steps=int(steps_hint),
                )
            else:
                state["coeffs"] = generate_bsr_coeffs_from_sgd_workload(
                    bands=bands,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
            state["coeff_source"] = "analytical_auto"

        state["bsr_bands"] = bands
        state["bnb_bands"] = bands
        state["bnb_bins"] = bins
        state["bnb_cycle_length"] = bins

        accountant_coeffs = None
        horizon = state.get("bnb_horizon", kwargs.get("total_steps", kwargs.get("steps", bins)))
        if mechanism_config.mechanism == "bsr":
            accountant_coeffs = list(state["coeffs"])
            state["bnb_accountant_coeffs"] = list(accountant_coeffs)
            state["bnb_accountant_coeffs_source"] = "raw_c_col"
        elif mechanism_config.mechanism == "bisr":
            accountant_coeffs = (
                derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs(
                    coeffs=list(state.get("bisr_inv_coeffs", state["coeffs"])),
                    steps=int(horizon),
                )
            )
            state["bnb_accountant_coeffs"] = list(accountant_coeffs)
            state["bnb_accountant_coeffs_source"] = "abs_factor_c_col"
        elif mechanism_config.mechanism == "bandmf":
            accountant_coeffs = derive_bandmf_amplified_accountant_coeffs_from_runtime_coeffs(
                coeffs=list(state["coeffs"]),
            )
            state["bnb_accountant_coeffs"] = list(accountant_coeffs)
            state["bnb_accountant_coeffs_source"] = "runtime_c_col"
        elif mechanism_config.mechanism == "bandinvmf":
            accountant_coeffs = derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs(
                inv_coeffs=list(state["bandinvmf_inv_coeffs"]),
                steps=int(horizon),
            )
            state["bnb_accountant_coeffs"] = list(accountant_coeffs)
            state["bnb_accountant_coeffs_source"] = "abs_factor_c_col"

        if mechanism_config.mechanism in ("bsr", "bisr", "bandmf", "bandinvmf"):
            c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
                coeffs=list(accountant_coeffs if accountant_coeffs is not None else state["coeffs"]),
                bands=int(bands),
                horizon=int(horizon),
            )
            state["bnb_c_matrix"] = c_matrix
            state["bnb_c_matrix_contract"] = c_matrix_contract

        return NoiseMechanismConfig(
            mechanism=mechanism_config.mechanism,
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )

    @staticmethod
    def _ensure_bnb_balls_in_bins_gaussian_state(
        *,
        mechanism_config: NoiseMechanismConfig,
        sampling_semantics: Optional[SamplingSemantics],
        kwargs: Dict[str, Any],
    ) -> NoiseMechanismConfig:
        if (
            mechanism_config.mechanism != "gaussian"
            or mechanism_config.accounting_mode != "bnb_accountant"
        ):
            return mechanism_config
        if sampling_semantics is None or sampling_semantics.sampling_mode != "balls_in_bins":
            return mechanism_config

        state = copy.deepcopy(mechanism_config.mechanism_state)
        state["_noise_mechanism"] = "gaussian"
        metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
        metadata_bins = metadata.get("bins", metadata.get("b"))
        explicit_cycle_length = kwargs.get("bnb_cycle_length", state.get("bnb_cycle_length"))
        legacy_bins = state.get("bnb_bins")
        if explicit_cycle_length is not None:
            explicit_cycle_length = int(explicit_cycle_length)
            if explicit_cycle_length < 1:
                raise ValueError("balls-in-bins cycle length must be >= 1")

            if metadata_bins is not None and int(metadata_bins) != explicit_cycle_length:
                raise ValueError(
                    "conflicting canonical inputs: `bnb_cycle_length` must match "
                    "sampling_semantics privacy_metadata['bins']"
                )

            if legacy_bins is not None and int(legacy_bins) != explicit_cycle_length:
                raise ValueError(
                    "conflicting canonical inputs: `bnb_cycle_length` must match "
                    "mechanism_state['bnb_bins']"
                )

        bins = (
            metadata_bins
            if metadata_bins is not None
            else (
                explicit_cycle_length
                if explicit_cycle_length is not None
                else state.get("bnb_cycle_length", state.get("bnb_bins"))
            )
        )
        if bins is None:
            raise ValueError(
                "gaussian balls_in_bins accounting requires privacy_metadata['bins']"
            )

        bins = int(bins)
        if bins < 1:
            raise ValueError("gaussian balls_in_bins bins must be >= 1")

        state["coeffs"] = [1.0]
        state["bsr_bands"] = 1
        state["bnb_bands"] = 1
        state["bnb_bins"] = bins
        state["bnb_cycle_length"] = bins
        if state.get("bnb_c_matrix") is None:
            horizon = state.get("bnb_horizon", kwargs.get("total_steps", kwargs.get("steps", bins)))
            c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
                coeffs=[1.0],
                bands=1,
                horizon=horizon,
            )
            state["bnb_c_matrix"] = c_matrix
            state["bnb_c_matrix_contract"] = c_matrix_contract

        return NoiseMechanismConfig(
            mechanism=mechanism_config.mechanism,
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )

    @staticmethod
    def _summarize_bsr_state(mechanism_state: Dict[str, Any]) -> Dict[str, Any]:
        coeffs = mechanism_state.get("coeffs")
        coeff_count = len(coeffs) if isinstance(coeffs, (list, tuple)) else None
        coeff_head = list(coeffs[:5]) if isinstance(coeffs, (list, tuple)) else None
        coeff_source = mechanism_state.get("coeff_source")
        if coeff_source is None and coeff_count:
            coeff_source = "explicit_or_precomputed"
        return {
            "coeff_count": coeff_count,
            "coeff_head": coeff_head,
            "coeff_source": coeff_source,
            "z_std": mechanism_state.get("z_std"),
            "bsr_sensitivity_scale": mechanism_state.get("bsr_sensitivity_scale"),
            "bsr_mf_sensitivity": mechanism_state.get("bsr_mf_sensitivity"),
            "bsr_min_separation": mechanism_state.get("bsr_min_separation"),
            "bsr_max_participations": mechanism_state.get("bsr_max_participations"),
            "bsr_iterations_number": mechanism_state.get("bsr_iterations_number"),
            "bsr_bands": mechanism_state.get("bsr_bands"),
        }

    @staticmethod
    def _apply_default_fixed_batch_contract_for_torch_sampler(
        *,
        mechanism_config: NoiseMechanismConfig,
        sampling_semantics: Optional[SamplingSemantics],
        total_steps: int,
        dataloader_len: int,
        dataset_size: int,
        logical_batch_size: int,
        kwargs: Dict[str, Any],
    ) -> NoiseMechanismConfig:
        if mechanism_config.mechanism not in ("bandmf", "bsr", "bisr", "bandinvmf"):
            return mechanism_config

        if sampling_semantics is None or sampling_semantics.sampling_mode != "torch_sampler":
            return mechanism_config

        if total_steps < 1:
            return mechanism_config

        global_steps_per_epoch = 0
        if dataset_size > 0 and logical_batch_size > 0:
            global_steps_per_epoch = int(
                math.ceil(float(dataset_size) / float(logical_batch_size))
            )
        elif dataloader_len > 0:
            # Fallback for callers that do not preserve pre-DDP logical schedule inputs.
            global_steps_per_epoch = int(dataloader_len)

        if global_steps_per_epoch < 1:
            return mechanism_config

        metadata = (
            sampling_semantics.privacy_metadata
            if sampling_semantics is not None
            else {}
        )
        state = copy.deepcopy(mechanism_config.mechanism_state)
        changed = False

        for field_name in (
            "bsr_iterations_number",
            "bsr_min_separation",
            "bsr_max_participations",
        ):
            if state.get(field_name) is not None:
                continue
            explicit_value = kwargs.get(field_name)
            if explicit_value is None:
                explicit_value = metadata.get(field_name)
            if explicit_value is None:
                continue
            state[field_name] = int(explicit_value)
            changed = True

        if state.get("bsr_iterations_number") is None:
            state["bsr_iterations_number"] = int(total_steps)
            changed = True

        if state.get("bsr_min_separation") is None:
            state["bsr_min_separation"] = int(global_steps_per_epoch)
            changed = True

        if state.get("bsr_max_participations") is None:
            state["bsr_max_participations"] = int(
                math.ceil(float(total_steps) / float(global_steps_per_epoch))
            )
            changed = True

        if not changed:
            return mechanism_config

        return NoiseMechanismConfig(
            mechanism=mechanism_config.mechanism,
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )

    @staticmethod
    def _log_bsr_trace(
        *,
        stage: str,
        mechanism_config: NoiseMechanismConfig,
        sampling_semantics: Optional[SamplingSemantics],
        sample_rate: Optional[float],
        expected_batch_size: Optional[int],
        noise_multiplier: Optional[float],
        target_epsilon: Optional[float],
        target_delta: Optional[float],
        total_steps: Optional[int],
        epochs: Optional[int],
        loss_reduction: Optional[str],
        correlated_denominator: Optional[float],
    ) -> None:
        if mechanism_config.mechanism not in ("bsr", "bandmf", "bisr"):
            return

        metadata = (
            dict(sampling_semantics.privacy_metadata)
            if sampling_semantics is not None
            else None
        )
        payload = {
            "stage": stage,
            "mechanism": mechanism_config.mechanism,
            "accounting_mode": mechanism_config.accounting_mode,
            "sampling_mode": (
                sampling_semantics.sampling_mode
                if sampling_semantics is not None
                else None
            ),
            "sampling_metadata": metadata,
            "sample_rate": sample_rate,
            "expected_batch_size": expected_batch_size,
            "noise_multiplier": noise_multiplier,
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "total_steps": total_steps,
            "epochs": epochs,
            "loss_reduction": loss_reduction,
            "calibration_denominator": correlated_denominator,
            "mechanism_state": PrivacyEngine._summarize_bsr_state(
                mechanism_config.mechanism_state
            ),
        }
        logger.info("MF_TRACE %s", json.dumps(payload, sort_keys=True))

    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False):
        """

        Args:
            accountant: Accounting mechanism. Currently supported:
                - rdp (:class:`~opacus.accountants.RDPAccountant`)
                - gdp (:class:`~opacus.accountants.GaussianAccountant`)
                - prv (:class`~opacus.accountants.PRVAccountant`)
            secure_mode: Set to ``True`` if cryptographically strong DP guarantee is
                required. ``secure_mode=True`` uses secure random number generator for
                noise and shuffling (as opposed to pseudo-rng in vanilla PyTorch) and
                prevents certain floating-point arithmetic-based attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details.
                When set to ``True`` requires ``torchcsprng`` to be installed
        """
        self.default_accountant = create_accountant(mechanism=accountant)
        self.accountant = self.default_accountant
        self.secure_mode = secure_mode
        self.noise_mechanism_config = NoiseMechanismConfig()
        self.sampling_semantics = SamplingSemantics(sampling_mode="poisson")
        self.secure_rng = None
        self.dataset = None  # only used to detect switching to a different dataset
        if self.secure_mode:
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.secure_rng = csprng.create_random_device_generator("/dev/urandom")
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_mode`` turned on."
            )

    @staticmethod
    def _accountant_for_mechanism(*, mechanism_config: NoiseMechanismConfig, default_accountant):
        if mechanism_config.accounting_mode == "bnb_accountant":
            return create_accountant(mechanism="bnb")
        if mechanism_config.mechanism in ("bandmf", "bsr"):
            return create_accountant(mechanism=mechanism_config.mechanism)
        if mechanism_config.mechanism in ("bisr", "bandinvmf"):
            return create_accountant(mechanism="bsr")

        return default_accountant

    @staticmethod
    def _build_sampling_semantics(
        *,
        poisson_sampling: bool,
        sample_rate: float,
        expected_batch_size: int,
        distributed: bool,
        explicit_sampling_semantics: Optional[SamplingSemantics],
    ) -> SamplingSemantics:
        if explicit_sampling_semantics is not None:
            return explicit_sampling_semantics

        return SamplingSemantics(
            sampling_mode="poisson" if poisson_sampling else "torch_sampler",
            privacy_metadata={
                "sample_rate": sample_rate,
                "expected_batch_size": int(expected_batch_size),
                "distributed": distributed,
            },
        )

    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        normalize_clipping: bool = False,
        **kwargs,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            normalize_clipping=normalize_clipping,
            **kwargs,
        )

    @staticmethod
    def _validate_distributed_correlated_support(
        *,
        mechanism: str,
        clipping: str,
        grad_sample_mode: str,
        is_fsdp: bool,
    ) -> None:
        if is_fsdp:
            raise ValueError(
                f"{mechanism} noise mechanism is not yet supported with FSDP; "
                "supported distributed mode is DDP/DPDDP with flat clipping"
            )

        if clipping != "flat":
            raise ValueError(
                f"{mechanism} noise mechanism supports only distributed flat clipping; "
                f"got clipping={clipping!r}"
            )

        if grad_sample_mode not in ("hooks", "ew"):
            raise ValueError(
                f"{mechanism} noise mechanism supports distributed grad_sample_mode "
                "in {'hooks', 'ew'} only; "
                f"got grad_sample_mode={grad_sample_mode!r}"
            )

    @staticmethod
    def _build_noise_mechanism_from_config(config: NoiseMechanismConfig):
        if config.mechanism == "gaussian":
            return None

        state = config.mechanism_state
        mechanism_name = config.mechanism
        z_std = state.get("z_std")
        if z_std is None:
            raise ValueError(
                f"{mechanism_name} mechanism requires `mechanism_state['z_std']`"
            )

        if mechanism_name == "bisr":
            inverse_coeffs = state.get("bisr_inv_coeffs")
            if isinstance(inverse_coeffs, (list, tuple)) and len(inverse_coeffs) > 0:
                return InverseBandNoiseMechanism(
                    inverse_coeffs=inverse_coeffs,
                    z_std=float(z_std),
                )

        if mechanism_name == "bandinvmf":
            inverse_coeffs = state.get("bandinvmf_inv_coeffs")
            if isinstance(inverse_coeffs, (list, tuple)) and len(inverse_coeffs) > 0:
                return InverseBandNoiseMechanism(
                    inverse_coeffs=inverse_coeffs,
                    z_std=float(z_std),
                )

        coeffs = state.get("coeffs")
        if coeffs is None:
            raise ValueError(
                f"{mechanism_name} mechanism requires `mechanism_state['coeffs']`"
            )

        if not isinstance(coeffs, (list, tuple)):
            raise ValueError("`mechanism_state['coeffs']` must be a list or tuple")

        return CorrelatedNoiseMechanism(coeffs=coeffs, z_std=float(z_std))

    @staticmethod
    def _bsr_calibration_denominator(
        *,
        loss_reduction: str,
        expected_batch_size: int,
    ) -> float:
        if loss_reduction == "sum":
            return 1.0

        if expected_batch_size <= 0:
            raise ValueError(
                "bsr calibration requires expected batch size under loss_reduction='mean'"
            )
        return float(expected_batch_size)

    @staticmethod
    def _resolve_sample_rate_and_expected_batch_size(
        *,
        poisson_sampling: bool,
        sampling_semantics: Optional[SamplingSemantics],
        mechanism: str,
        batch_size: int,
        dataset_size: int,
        data_loader_len: int,
        total_steps: Optional[int],
        distributed: bool,
    ) -> Tuple[float, int]:
        if total_steps:
            sample_rate = PrivacyEngine._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=sampling_semantics,
                mechanism=mechanism,
                batch_size=batch_size,
                dataset_size=dataset_size,
            )
        else:
            sampling_mode = (
                sampling_semantics.sampling_mode
                if sampling_semantics is not None
                else None
            )
            if (
                not poisson_sampling
                and sampling_mode in ("cyclic_poisson", "balls_in_bins", "b_min_sep")
            ):
                sample_rate = PrivacyEngine._resolve_total_steps_sample_rate(
                    poisson_sampling=poisson_sampling,
                    sampling_semantics=sampling_semantics,
                    mechanism=mechanism,
                    batch_size=batch_size,
                    dataset_size=dataset_size,
                )
            else:
                sample_rate = 1 / data_loader_len

        expected_batch_size = int(dataset_size * sample_rate)

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size = int(expected_batch_size / world_size)

        return float(sample_rate), int(expected_batch_size)

    @staticmethod
    def _resolve_bnb_b_min_sep_inputs(
        *,
        mechanism_state: Dict[str, Any],
        sampling_semantics: Optional[SamplingSemantics],
        kwargs: Dict[str, Any],
    ) -> tuple[Any, int, Dict[str, Any]]:
        return resolve_bnb_b_min_sep_inputs_helper(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            kwargs=kwargs,
        )

    @staticmethod
    def _validate_bnb_accounting_runtime_consistency(
        *,
        mechanism_state: Dict[str, Any],
        sampling_semantics: Optional[SamplingSemantics],
        c_matrix: Any,
        bands: int,
        c_matrix_contract: Dict[str, Any],
    ) -> None:
        validate_bnb_accounting_runtime_consistency_helper(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            c_matrix=c_matrix,
            bands=bands,
            c_matrix_contract=c_matrix_contract,
        )

    def _sampler_generator(self, data_loader: DataLoader):
        return self.secure_rng if self.secure_mode else data_loader.generator

    def _rebuild_data_loader_with_batch_sampler(
        self, data_loader: DataLoader, sampler
    ) -> DataLoader:
        return DataLoader(
            dataset=data_loader.dataset,
            batch_sampler=sampler,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=self._sampler_generator(data_loader),
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
        )

    def _rebuild_data_loader_with_sampler(
        self,
        data_loader: DataLoader,
        sampler,
        *,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size if batch_size is None else int(batch_size),
            sampler=sampler,
            drop_last=data_loader.drop_last,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=self._sampler_generator(data_loader),
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
        )

    def _build_distributed_torch_sampler(self, *, data_loader: DataLoader):
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("distributed torch_sampler is not supported for IterableDataset")

        if data_loader.batch_size is None:
            raise ValueError("distributed torch_sampler requires data_loader.batch_size")

        shuffle = isinstance(data_loader.sampler, torch.utils.data.RandomSampler)
        return DistributedFixedSampler(
            total_size=len(data_loader.dataset),
            shuffle=shuffle,
            shuffle_seed=0,
        )

    def _build_cyclic_poisson_sampler(
        self,
        *,
        data_loader: DataLoader,
        distributed: bool,
        bands: int,
        total_steps: Optional[int],
    ):
        """
        Build cyclic-poisson sampler objects from runtime config.
        Math: partition ``P_r`` is active at step ``t`` iff ``r = t mod b``; each
        ``i ∈ P_r`` is sampled with ``q = m / |P_r|``.
        """
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("cyclic_poisson sampling is not supported for IterableDataset")

        if data_loader.batch_size is None:
            raise ValueError("cyclic_poisson sampling requires data_loader.batch_size")

        # `steps` is the global optimization horizon; must cover at least one full cycle of `bands`.
        steps = total_steps if total_steps is not None else len(data_loader)
        if int(steps) < int(bands):
            raise ValueError(
                "cyclic_poisson bandmf requires steps >= bands; "
                f"got steps={int(steps)}, bands={int(bands)}"
            )
        generator = self._sampler_generator(data_loader)
        if distributed:
            world_size = torch.distributed.get_world_size()
            local_batch_size = int(data_loader.batch_size / world_size)
            if local_batch_size <= 0:
                raise ValueError(
                    "cyclic_poisson distributed sampling requires "
                    "batch_size >= world_size"
                )
            return DistributedCyclicPoissonSampler(
                total_size=len(data_loader.dataset),
                batch_size=local_batch_size,
                bands=bands,
                generator=generator,
                steps=steps,
            )

        return CyclicPoissonSampler(
            num_samples=len(data_loader.dataset),
            batch_size=int(data_loader.batch_size),
            bands=bands,
            generator=generator,
            steps=steps,
        )

    def _build_b_min_sep_sampler(
        self,
        *,
        data_loader: DataLoader,
        distributed: bool,
        b: int,
        p: float,
        total_steps: Optional[int],
    ):
        """
        Build b-min-separation samplers.
        Math: item ``i`` is sampled with Bernoulli(``p``) only when ``τ_i(t)=0``; if sampled,
        ``τ_i←b−1``, else ``τ_i`` decreases by ``1``.
        """
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("b_min_sep sampling is not supported for IterableDataset")

        # `b` is min-separation; `p` is per-step participation probability.
        steps = total_steps if total_steps is not None else len(data_loader)
        generator = self._sampler_generator(data_loader)
        if distributed:
            return DistributedBMinSepSampler(
                total_size=len(data_loader.dataset),
                sample_rate=p,
                min_separation=b,
                generator=generator,
                steps=steps,
            )

        return BMinSepSampler(
            num_samples=len(data_loader.dataset),
            sample_rate=p,
            min_separation=b,
            generator=generator,
            steps=steps,
        )

    def _build_balls_in_bins_sampler(
        self,
        *,
        data_loader: DataLoader,
        distributed: bool,
        bins: int,
        total_steps: Optional[int],
    ):
        """
        Build balls-in-bins samplers.
        Math: assign ``u_i ~ Unif({0,…,B−1})`` once; item ``i`` is active at step ``t`` iff
        ``t mod B = u_i``.
        """
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("balls_in_bins sampling is not supported for IterableDataset")

        # `bins` is the modulo bucket count controlling periodic participation windows.
        steps = total_steps if total_steps is not None else len(data_loader)
        generator = self._sampler_generator(data_loader)
        if distributed:
            return DistributedBallsInBinsSampler(
                total_size=len(data_loader.dataset),
                bins=bins,
                generator=generator,
                steps=steps,
            )

        return BallsInBinsSampler(
            num_samples=len(data_loader.dataset),
            bins=bins,
            generator=generator,
            steps=steps,
        )

    @staticmethod
    def _validate_bnb_sampling_policy(
        *,
        sampling_semantics: Optional[SamplingSemantics],
        mechanism: str,
        accounting_mode: str,
    ) -> None:
        if accounting_mode != "bnb_accountant":
            return
        validate_bnb_sampling_policy_helper(
            sampling_semantics=sampling_semantics,
            mechanism=mechanism,
        )

    def _validate_mechanism_sampling_compatibility(
        self,
        *,
        mechanism_config: NoiseMechanismConfig,
        poisson_sampling: bool,
        sampling_semantics: Optional[SamplingSemantics],
        validate_cyclic_poisson_mode: bool,
    ) -> None:
        """
        Enforce mechanism/sampler compatibility constraints.
        Math: enforces accountant contracts against runtime sampling law, e.g.
        cyclic requires ``q = b·p ∈ (0,1]``, fixed-batch BSR uses ``S_{k,b}(C;T)``,
        and BNB requires b-min-sep/balls-in-bins semantics.

        Source: BandMF (Choquette-Choo et al., 2023), Section 5 and Theorems 4 and 5; and
        BMinSep (Dong and Ganesh, 2026):
            Section 4 and Section 5, Equations (2)-(4), Theorem 5.1.
        """
        mechanism = mechanism_config.mechanism
        accounting_mode = mechanism_config.accounting_mode
        if mechanism in ("bandmf", "bsr", "bisr", "bandinvmf") and poisson_sampling:
            raise ValueError(
                f"{mechanism} mechanism requires fixed-batch semantics; "
                "set poisson_sampling=False"
            )

        if mechanism == "bandmf":
            if sampling_semantics is None:
                raise ValueError(
                    f"{mechanism} mechanism requires explicit sampling_semantics "
                    "with sampling_mode in {'cyclic_poisson', 'balls_in_bins'}"
                )
            if (
                mechanism == "bandmf"
                and
                accounting_mode == "bandmf_accountant"
                and sampling_semantics.sampling_mode not in ("cyclic_poisson", "torch_sampler")
            ):
                raise ValueError(
                    "bandmf mechanism with bandmf_accountant supports sampling_mode in "
                    "{'cyclic_poisson', 'torch_sampler'} only"
                )
            if accounting_mode == "bnb_accountant" and sampling_semantics.sampling_mode != "balls_in_bins":
                raise ValueError(
                    f"{mechanism} mechanism with bnb_accountant requires sampling_mode='balls_in_bins'"
                )

        if mechanism == "bandinvmf":
            if accounting_mode == "bnb_accountant":
                if sampling_semantics is None or sampling_semantics.sampling_mode != "balls_in_bins":
                    raise ValueError(
                        "bandinvmf mechanism with bnb_accountant requires sampling_mode='balls_in_bins'"
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

        self._validate_bnb_sampling_policy(
            sampling_semantics=sampling_semantics,
            mechanism=mechanism,
            accounting_mode=accounting_mode,
        )

        if (
            sampling_semantics is not None
            and sampling_semantics.sampling_mode == "b_min_sep"
        ):
            raise ValueError(
                "b_min_sep sampling is temporarily disabled pending p-aware BNB accounting"
            )

        if (
            sampling_semantics is not None
            and sampling_semantics.sampling_mode == "balls_in_bins"
            and mechanism not in ("gaussian", "bandmf", "bsr", "bisr", "bandinvmf")
        ):
            raise ValueError(
                "balls_in_bins sampling is supported only for mechanism in {'gaussian', 'bandmf', 'bsr', 'bisr', 'bandinvmf'}"
            )

        if (
            validate_cyclic_poisson_mode
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "cyclic_poisson"
            and mechanism not in ("bandmf", "bsr", "bisr", "bandinvmf")
        ):
            raise ValueError(
                "cyclic_poisson sampling is supported only for mechanism in {'bandmf', 'bsr', 'bisr', 'bandinvmf'}"
            )

    @staticmethod
    def _resolve_total_steps_sample_rate(
        *,
        poisson_sampling: bool,
        sampling_semantics: Optional[SamplingSemantics],
        mechanism: str = "gaussian",
        batch_size: int,
        dataset_size: int,
    ) -> float:
        sampling_mode = (
            sampling_semantics.sampling_mode
            if sampling_semantics is not None
            else None
        )
        if not poisson_sampling and sampling_mode in (None, "torch_sampler"):
            if mechanism in ("bandmf", "bsr", "bisr", "bandinvmf"):
                return batch_size / dataset_size

            raise ValueError(
                "Setting total_steps with non-Poisson sampling requires "
                "explicit sampling_semantics in {'cyclic_poisson', 'b_min_sep', 'balls_in_bins'}"
            )

        if not poisson_sampling and sampling_mode == "b_min_sep":
            raise ValueError(
                "b_min_sep sampling is temporarily disabled pending p-aware BNB accounting"
            )

        if not poisson_sampling and sampling_mode == "balls_in_bins":
            bins = sampling_semantics.privacy_metadata.get("bins")
            if bins is None:
                bins = sampling_semantics.privacy_metadata.get("b")

            if bins is None:
                raise ValueError(
                    "balls_in_bins sampling requires privacy_metadata['bins'] "
                    "(or legacy key 'b')"
                )

            return 1.0 / float(int(bins))

        if not poisson_sampling and sampling_mode == "cyclic_poisson":
            bands = sampling_semantics.privacy_metadata.get("bands")
            if bands is None:
                raise ValueError(
                    "cyclic_poisson sampling requires privacy_metadata['bands']"
                )
            bands = int(bands)
            if bands <= 0:
                raise ValueError("cyclic_poisson bands must be > 0")
            partition_size = int(dataset_size) // bands
            if partition_size <= 0:
                raise ValueError(
                    "cyclic_poisson requires dataset_size // bands >= 1"
                )
            usable_size = partition_size * bands
            return float(batch_size) / float(usable_size)

        # For Poisson and cyclic_poisson, q follows the batch-size ratio.
        return batch_size / dataset_size

    def _resolve_local_sampling_semantics_for_epsilon(
        self,
        *,
        mechanism: str,
        sampling_semantics: Optional[SamplingSemantics],
        poisson_sampling: bool,
        total_steps: Optional[int],
        data_loader: DataLoader,
    ) -> Optional[SamplingSemantics]:
        local_sampling_semantics = sampling_semantics
        if mechanism in ("bsr", "bisr", "bandinvmf") and local_sampling_semantics is None:
            if total_steps:
                local_sample_rate = data_loader.batch_size / len(data_loader.dataset)
            else:
                local_sample_rate = 1 / len(data_loader)

            local_sampling_semantics = self._build_sampling_semantics(
                poisson_sampling=poisson_sampling,
                sample_rate=local_sample_rate,
                expected_batch_size=data_loader.batch_size,
                distributed=False,
                explicit_sampling_semantics=None,
            )

        return local_sampling_semantics

    def _resolve_bnb_runtime_inputs_for_epsilon(
        self,
        *,
        mechanism_config: NoiseMechanismConfig,
        sampling_semantics: Optional[SamplingSemantics],
        kwargs: Dict[str, Any],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[int],
        Optional[int],
        Optional[Dict[str, Any]],
    ]:
        if mechanism_config.accounting_mode != "bnb_accountant":
            return None, None, None, None

        if (
            sampling_semantics is None
            or sampling_semantics.sampling_mode not in ("balls_in_bins",)
        ):
            raise ValueError(
                "balls-in-bins calibration requires sampling_semantics with "
                "sampling_mode in {'balls_in_bins'}"
            )
        mechanism_state = copy.deepcopy(mechanism_config.mechanism_state)
        metadata = (
            sampling_semantics.privacy_metadata
            if sampling_semantics is not None
            else {}
        )
        bands = metadata.get("bands", mechanism_state.get("bnb_bands", mechanism_state.get("bsr_bands")))
        coeffs = mechanism_state.get("coeffs")
        if (
            mechanism_state.get("bnb_c_matrix") is None
            and isinstance(coeffs, (list, tuple))
            and len(coeffs) > 0
            and bands is not None
        ):
            horizon = kwargs.get("total_steps")
            if horizon is None:
                horizon = kwargs.get("steps")
            if horizon is not None:
                c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
                    coeffs=list(coeffs),
                    bands=int(bands),
                    horizon=int(horizon),
                )
                mechanism_state["bnb_c_matrix"] = c_matrix
                mechanism_state["bnb_c_matrix_contract"] = c_matrix_contract
                mechanism_state["bnb_bands"] = int(bands)

        bnb_c_matrix, bnb_bands, _bnb_cycle_length, bnb_c_matrix_contract = self._resolve_bnb_b_min_sep_inputs(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            kwargs=kwargs,
        )

        self._validate_bnb_accounting_runtime_consistency(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            c_matrix=bnb_c_matrix,
            bands=int(bnb_bands),
            c_matrix_contract=bnb_c_matrix_contract,
        )

        return bnb_c_matrix, int(bnb_bands), int(_bnb_cycle_length), bnb_c_matrix_contract

    def _resolve_non_bnb_noise_multiplier_for_target_epsilon(
        self,
        *,
        mechanism_config: NoiseMechanismConfig,
        active_accountant,
        target_epsilon: float,
        target_delta: float,
        total_steps: Optional[int],
        epochs: Optional[int],
        poisson_sampling: bool,
        data_loader: DataLoader,
        sampling_semantics: Optional[SamplingSemantics],
        kwargs: Dict[str, Any],
        nm_kwargs: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Resolve sigma for non-BNB mechanisms by delegating to accountant calibration.
        Math: non-BNB noise calibration solves for σ_ref such that composed ε(δ) meets target under selected accountant.
        """
        if total_steps:
            if epochs is not None:
                raise ValueError(
                    "make_private_with_epsilon takes as input EITHER a number of steps or a number of epochs"
                )

            # `sample_rate` is the accountant participation probability `q` input.
            sample_rate = self._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
            )

            bsr_mf_sensitivity = None
            if (
                mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf")
                and sampling_semantics is not None
                and sampling_semantics.sampling_mode == "cyclic_poisson"
            ):
                if mechanism_config.mechanism == "bisr":
                    nm_kwargs["bsr_sensitivity_scale"] = resolve_bisr_sensitivity_scale_for_cyclic_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        kwargs=kwargs,
                    )
                elif mechanism_config.mechanism == "bandinvmf":
                    nm_kwargs["bsr_sensitivity_scale"] = resolve_bandinvmf_sensitivity_scale_for_cyclic_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        kwargs=kwargs,
                    )
                else:
                    nm_kwargs["bsr_sensitivity_scale"] = resolve_bsr_sensitivity_scale_for_cyclic_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        kwargs=kwargs,
                    )
            if (
                mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf")
                and sampling_semantics is not None
                and sampling_semantics.sampling_mode == "torch_sampler"
            ):
                if mechanism_config.mechanism == "bandmf":
                    bsr_mf_sensitivity = resolve_bandmf_mf_sensitivity_for_fixed_batch_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        sample_rate=sample_rate,
                        kwargs=kwargs,
                    )
                elif mechanism_config.mechanism == "bisr":
                    bsr_mf_sensitivity = resolve_bisr_mf_sensitivity_for_fixed_batch_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        sample_rate=sample_rate,
                        kwargs=kwargs,
                    )
                elif mechanism_config.mechanism == "bandinvmf":
                    bsr_mf_sensitivity = resolve_bandinvmf_mf_sensitivity_for_fixed_batch_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        sample_rate=sample_rate,
                        kwargs=kwargs,
                    )
                else:
                    bsr_mf_sensitivity = resolve_bsr_mf_sensitivity_for_fixed_batch_helper(
                        mechanism_state=mechanism_config.mechanism_state,
                        sampling_semantics=sampling_semantics,
                        steps=int(total_steps),
                        sample_rate=sample_rate,
                        kwargs=kwargs,
                    )

            t0 = time.perf_counter()
            noise_multiplier = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                steps=total_steps,
                accountant=active_accountant.mechanism(),
                mechanism_state=self._annotate_mechanism_state(
                    mechanism=mechanism_config.mechanism,
                    mechanism_state=mechanism_config.mechanism_state,
                ),
                sampling_semantics=sampling_semantics,
                bsr_mf_sensitivity=bsr_mf_sensitivity,
                **nm_kwargs,
            )
            logger.info(
                "OPACUS_DP_TIMING %s",
                json.dumps(
                    {
                        "phase": "get_noise_multiplier_non_bnb_steps",
                        "elapsed_s": round(float(time.perf_counter() - t0), 6),
                        "mechanism": mechanism_config.mechanism,
                        "accountant": active_accountant.mechanism(),
                        "sampling_mode": (
                            sampling_semantics.sampling_mode
                            if sampling_semantics is not None
                            else None
                        ),
                        "steps": int(total_steps),
                        "sample_rate": float(sample_rate),
                        "target_epsilon": float(target_epsilon),
                        "target_delta": float(target_delta),
                    },
                    sort_keys=True,
                ),
            )
            return float(noise_multiplier), float(sample_rate)

        sample_rate = self._resolve_total_steps_sample_rate(
            poisson_sampling=poisson_sampling,
            sampling_semantics=sampling_semantics,
            mechanism=mechanism_config.mechanism,
            batch_size=data_loader.batch_size,
            dataset_size=len(data_loader.dataset),
        )
        implied_steps = int(float(epochs) * float(len(data_loader)))
        bsr_mf_sensitivity = None
        if (
            mechanism_config.mechanism in ("bandmf", "bisr")
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            if mechanism_config.mechanism == "bisr":
                nm_kwargs["bsr_sensitivity_scale"] = resolve_bisr_sensitivity_scale_for_cyclic_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=implied_steps,
                    kwargs=kwargs,
                )
            else:
                nm_kwargs["bsr_sensitivity_scale"] = resolve_bsr_sensitivity_scale_for_cyclic_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=implied_steps,
                    kwargs=kwargs,
                )
        if (
            mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf")
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "torch_sampler"
        ):
            if mechanism_config.mechanism == "bandmf":
                bsr_mf_sensitivity = resolve_bandmf_mf_sensitivity_for_fixed_batch_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=implied_steps,
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )
            elif mechanism_config.mechanism == "bisr":
                bsr_mf_sensitivity = resolve_bisr_mf_sensitivity_for_fixed_batch_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=implied_steps,
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )
            else:
                bsr_mf_sensitivity = resolve_bsr_mf_sensitivity_for_fixed_batch_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=implied_steps,
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )

        t0 = time.perf_counter()
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=implied_steps,
            accountant=active_accountant.mechanism(),
            mechanism_state=self._annotate_mechanism_state(
                mechanism=mechanism_config.mechanism,
                mechanism_state=mechanism_config.mechanism_state,
            ),
            sampling_semantics=sampling_semantics,
            bsr_mf_sensitivity=bsr_mf_sensitivity,
            **nm_kwargs,
        )
        logger.info(
            "OPACUS_DP_TIMING %s",
            json.dumps(
                {
                    "phase": "get_noise_multiplier_non_bnb_epochs",
                    "elapsed_s": round(float(time.perf_counter() - t0), 6),
                    "mechanism": mechanism_config.mechanism,
                    "accountant": active_accountant.mechanism(),
                    "sampling_mode": (
                        sampling_semantics.sampling_mode
                        if sampling_semantics is not None
                        else None
                    ),
                    "steps": int(implied_steps),
                    "sample_rate": float(sample_rate),
                    "target_epsilon": float(target_epsilon),
                    "target_delta": float(target_delta),
                },
                sort_keys=True,
            ),
        )
        return float(noise_multiplier), float(sample_rate)

    def _resolve_bnb_noise_multiplier_for_target_epsilon(
        self,
        *,
        mechanism_config: NoiseMechanismConfig,
        target_epsilon: float,
        target_delta: float,
        total_steps: Optional[int],
        epochs: Optional[int],
        poisson_sampling: bool,
        data_loader: DataLoader,
        sampling_semantics: Optional[SamplingSemantics],
        bnb_c_matrix: Optional[torch.Tensor],
        bnb_bands: Optional[int],
        bnb_cycle_length: Optional[int],
        kwargs: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Resolve sigma for BNB via Monte Carlo + EVR calibration.
        Math: BNB calibration searches σ using Monte Carlo δ estimates with EVR/Hoeffding confidence control.
        """
        if bnb_c_matrix is None or bnb_bands is None or bnb_cycle_length is None:
            raise ValueError(
                "bnb calibration requires resolved runtime inputs (`c_matrix`, `bands`, `cycle_length`)"
            )

        if total_steps:
            if epochs is not None:
                raise ValueError(
                    "make_private_with_epsilon takes as input EITHER a number of steps or a number of epochs"
                )

            # `sample_rate` is carried for runtime metadata; BNB sigma solve uses C/bands contract.
            sample_rate = self._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
            )
            logger.info(
                "bnb init: starting get_noise_multiplier (steps=%s, sample_rate=%.6g, eps=%.6g, delta=%.6g)",
                int(total_steps),
                float(sample_rate),
                float(target_epsilon),
                float(target_delta),
            )
        else:
            sample_rate = self._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
            )
            logger.info(
                "bnb init: starting get_noise_multiplier (epochs=%s, sample_rate=%.6g, eps=%.6g, delta=%.6g)",
                int(epochs),
                float(sample_rate),
                float(target_epsilon),
                float(target_delta),
            )

        calibration_cfg = resolve_bnb_calibration_kwargs(
            overrides=kwargs,
        )
        t0 = time.perf_counter()
        estimator_kwargs = {
            "mechanism_state": self._annotate_mechanism_state(
                mechanism=mechanism_config.mechanism,
                mechanism_state=mechanism_config.mechanism_state,
            ),
            "sampling_semantics": sampling_semantics,
            "bnb_calibration_mode": str(calibration_cfg["bnb_calibration_mode"]),
            "bnb_num_samples": int(calibration_cfg["bnb_num_samples"]),
            "bnb_seed": int(calibration_cfg["bnb_seed"]),
            "bnb_reduce_dimensionality": bool(
                calibration_cfg["bnb_reduce_dimensionality"]
            ),
            "bnb_tolerance": float(calibration_cfg["bnb_tolerance"]),
            "bnb_max_iterations": int(calibration_cfg["bnb_max_iterations"]),
            "bnb_chunk_size": calibration_cfg["bnb_chunk_size"],
            "bnb_num_workers": int(calibration_cfg["bnb_num_workers"]),
            "bnb_backend": str(calibration_cfg["bnb_backend"]),
            "bnb_device": calibration_cfg["bnb_device"],
            "bnb_distributed_mode": str(calibration_cfg["bnb_distributed_mode"]),
            "bnb_distributed_dp_runtime": bool(
                calibration_cfg["bnb_distributed_dp_runtime"]
            ),
        }
        if total_steps:
            noise_multiplier = get_noise_multiplier(
                target_epsilon=float(target_epsilon),
                target_delta=float(target_delta),
                sample_rate=float(sample_rate),
                steps=int(total_steps),
                accountant="bnb",
                **estimator_kwargs,
            )
        else:
            noise_multiplier = get_noise_multiplier(
                target_epsilon=float(target_epsilon),
                target_delta=float(target_delta),
                sample_rate=float(sample_rate),
                epochs=int(epochs),
                accountant="bnb",
                **estimator_kwargs,
            )
        logger.info(
            "OPACUS_DP_TIMING %s",
            json.dumps(
                {
                    "phase": "get_noise_multiplier_bnb_monte_carlo",
                    "elapsed_s": round(float(time.perf_counter() - t0), 6),
                    "mechanism": mechanism_config.mechanism,
                    "accountant": "bnb",
                    "sampling_mode": (
                        sampling_semantics.sampling_mode
                        if sampling_semantics is not None
                        else None
                    ),
                    "bands": int(bnb_bands),
                    "cycle_length": int(bnb_cycle_length),
                    "target_epsilon": float(target_epsilon),
                    "target_delta": float(target_delta),
                    "num_samples": int(calibration_cfg["bnb_num_samples"]),
                    "bnb_backend": str(calibration_cfg["bnb_backend"]),
                    "bnb_distributed_mode": str(
                        calibration_cfg["bnb_distributed_mode"]
                    ),
                    "bnb_distributed_dp_runtime": bool(
                        calibration_cfg["bnb_distributed_dp_runtime"]
                    ),
                },
                sort_keys=True,
            ),
        )
        logger.info(
            "bnb init: get_noise_multiplier done -> sigma=%.6g",
            float(noise_multiplier),
        )
        return float(noise_multiplier), float(sample_rate)

    def _resolve_noise_multiplier_for_target_epsilon(
        self,
        *,
        mechanism_config: NoiseMechanismConfig,
        active_accountant,
        target_epsilon: float,
        target_delta: float,
        total_steps: Optional[int],
        epochs: Optional[int],
        poisson_sampling: bool,
        data_loader: DataLoader,
        sampling_semantics: Optional[SamplingSemantics],
        bnb_c_matrix: Optional[torch.Tensor],
        bnb_bands: Optional[int],
        bnb_cycle_length: Optional[int],
        kwargs: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Dispatch epsilon->sigma calibration based on mechanism/accountant path.
        Math: dispatcher picks the correct ε→σ calibration path (BNB Monte Carlo vs non-BNB accountant solve).
        """
        nm_kwargs = dict(kwargs)
        nm_kwargs.pop("bsr_mf_sensitivity", None)

        if active_accountant.mechanism() == "bnb":
            return self._resolve_bnb_noise_multiplier_for_target_epsilon(
                mechanism_config=mechanism_config,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                total_steps=total_steps,
                epochs=epochs,
                poisson_sampling=poisson_sampling,
                data_loader=data_loader,
                sampling_semantics=sampling_semantics,
                bnb_c_matrix=bnb_c_matrix,
                bnb_bands=bnb_bands,
                bnb_cycle_length=bnb_cycle_length,
                kwargs=kwargs,
            )

        return self._resolve_non_bnb_noise_multiplier_for_target_epsilon(
            mechanism_config=mechanism_config,
            active_accountant=active_accountant,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            total_steps=total_steps,
            epochs=epochs,
            poisson_sampling=poisson_sampling,
            data_loader=data_loader,
            sampling_semantics=sampling_semantics,
            kwargs=kwargs,
            nm_kwargs=nm_kwargs,
        )

    def _build_bnb_calibration_report(
        self,
        *,
        mechanism: str,
        bnb_c_matrix: Optional[torch.Tensor],
        bnb_bands: Optional[int],
        bnb_cycle_length: Optional[int],
        noise_multiplier: float,
        target_epsilon: float,
        target_delta: float,
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if bnb_c_matrix is None or bnb_bands is None or bnb_cycle_length is None:
            return None

        calibration_cfg = resolve_bnb_calibration_kwargs(
            overrides=kwargs,
        )
        num_samples = int(calibration_cfg["bnb_num_samples"])
        seed = int(calibration_cfg["bnb_seed"])
        reduce_dimensionality = bool(calibration_cfg["bnb_reduce_dimensionality"])
        confidence_alpha = float(calibration_cfg["bnb_confidence_alpha"])

        llr_samples = sample_b_min_sep_llr(
            c_matrix=bnb_c_matrix,
            bands=int(bnb_bands),
            cycle_length=int(bnb_cycle_length),
            num_samples=num_samples,
            seed=seed,
            sigma=float(noise_multiplier),
            reduce_dimensionality=reduce_dimensionality,
        )
        verification = verify_hockey_stick_delta_hoeffding(
            epsilon=float(target_epsilon),
            llr_samples=llr_samples,
            target_delta=float(target_delta),
            confidence_alpha=float(confidence_alpha),
        )

        if bool(calibration_cfg["bnb_require_evr_pass"]) and not verification.accepted:
            raise ValueError(
                "bnb verification guard failed for calibrated noise multiplier; "
                "increase noise, sample budget, or loosen targets"
            )

        return make_bnb_calibration_report(
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            noise_multiplier=float(noise_multiplier),
            num_samples=int(num_samples),
            seed=int(seed),
            bands=int(bnb_bands),
            verification=verification,
            evr_confidence_alpha_total=float(confidence_alpha),
            evr_num_checks=1,
            evr_per_check_alpha=float(confidence_alpha),
            evr_pass_count=1 if verification.accepted else 0,
        ).to_dict()

    @staticmethod
    def _build_bnb_accounting_kwargs_for_state(
        *,
        mechanism: str,
        kwargs: Dict[str, Any],
        distributed_dp_runtime: bool,
    ) -> Optional[Dict[str, Any]]:
        if mechanism not in ("gaussian", "bandmf", "bsr", "bisr", "bandinvmf"):
            return None

        calibration_cfg = resolve_bnb_calibration_kwargs(
            overrides=kwargs,
        )
        explicit_distributed_mode = kwargs.get("bnb_distributed_mode")
        resolved_distributed_mode = (
            explicit_distributed_mode
            if explicit_distributed_mode is not None
            else ("chunk_shard" if distributed_dp_runtime else "none")
        )
        return {
            "bnb_calibration_mode": str(calibration_cfg["bnb_calibration_mode"]),
            "bnb_num_samples": int(calibration_cfg["bnb_num_samples"]),
            "bnb_seed": int(calibration_cfg["bnb_seed"]),
            "bnb_reduce_dimensionality": bool(
                calibration_cfg["bnb_reduce_dimensionality"]
            ),
            "bnb_tolerance": float(calibration_cfg["bnb_tolerance"]),
            "bnb_max_iterations": int(calibration_cfg["bnb_max_iterations"]),
            "bnb_chunk_size": calibration_cfg["bnb_chunk_size"],
            "bnb_num_workers": int(calibration_cfg["bnb_num_workers"]),
            "bnb_backend": str(calibration_cfg["bnb_backend"]),
            "bnb_device": calibration_cfg["bnb_device"],
            "bnb_distributed_mode": str(resolved_distributed_mode),
            "bnb_distributed_dp_runtime": bool(distributed_dp_runtime),
        }

    @staticmethod
    def _apply_correlated_runtime_calibration(
        *,
        mechanism_config: NoiseMechanismConfig,
        max_grad_norm: Union[float, List[float]],
        noise_multiplier: float,
        correlated_denominator: Optional[float],
        bnb_calibration_report: Optional[Dict[str, Any]],
        bnb_accounting_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NoiseMechanismConfig:
        if mechanism_config.mechanism not in ("bandmf", "bsr", "bisr", "bandinvmf"):
            return mechanism_config

        if isinstance(max_grad_norm, list):
            raise ValueError(
                f"{mechanism_config.mechanism} calibration requires scalar "
                "max_grad_norm under flat clipping"
            )

        state = copy.deepcopy(mechanism_config.mechanism_state)
        state["z_std"] = calibrate_bsr_z_std(
            noise_multiplier_ref=float(noise_multiplier),
            max_grad_norm=float(max_grad_norm),
            denominator=float(correlated_denominator),
        )

        if bnb_calibration_report is not None:
            state["_bnb_calibration_report"] = bnb_calibration_report

        if bnb_accounting_kwargs is not None:
            state["_bnb_accounting_kwargs"] = dict(bnb_accounting_kwargs)

        return NoiseMechanismConfig(
            mechanism=mechanism_config.mechanism,
            accounting_mode=mechanism_config.accounting_mode,
            mechanism_state=state,
        )

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        *,
        poisson_sampling: bool,
        distributed: bool,
        sampling_semantics: Optional[SamplingSemantics] = None,
        total_steps: int = None,
    ) -> DataLoader:
        if self.dataset is None:
            self.dataset = data_loader.dataset
        elif self.dataset != data_loader.dataset:
            warnings.warn(
                f"PrivacyEngine detected new dataset object. "
                f"Was: {self.dataset}, got: {data_loader.dataset}. "
                f"Privacy accounting works per dataset, please initialize "
                f"new PrivacyEngine if you're using different dataset. "
                f"You can ignore this warning if two datasets above "
                f"represent the same logical dataset"
            )

        if sampling_semantics is not None and sampling_semantics.sampling_mode == "cyclic_poisson":
            bands = sampling_semantics.privacy_metadata.get("bands")
            if bands is None:
                raise ValueError(
                    "cyclic_poisson sampling requires privacy_metadata['bands']"
                )

            sampler = self._build_cyclic_poisson_sampler(
                data_loader=data_loader,
                distributed=distributed,
                bands=int(bands),
                total_steps=total_steps,
            )

            return self._rebuild_data_loader_with_batch_sampler(data_loader, sampler)

        if sampling_semantics is not None and sampling_semantics.sampling_mode == "b_min_sep":
            raise ValueError(
                "b_min_sep sampling is temporarily disabled pending p-aware BNB accounting"
            )

        if sampling_semantics is not None and sampling_semantics.sampling_mode == "balls_in_bins":
            bins = sampling_semantics.privacy_metadata.get("bins")
            if bins is None:
                bins = sampling_semantics.privacy_metadata.get("b")

            if bins is None:
                raise ValueError(
                    "balls_in_bins sampling requires privacy_metadata['bins'] "
                    "(or legacy key 'b')"
                )

            sampler = self._build_balls_in_bins_sampler(
                data_loader=data_loader,
                distributed=distributed,
                bins=int(bins),
                total_steps=total_steps,
            )

            return self._rebuild_data_loader_with_batch_sampler(data_loader, sampler)

        if (
            distributed
            and not poisson_sampling
            and (
                sampling_semantics is None
                or sampling_semantics.sampling_mode == "torch_sampler"
            )
        ):
            sampler = self._build_distributed_torch_sampler(data_loader=data_loader)
            world_size = torch.distributed.get_world_size()
            local_batch_size = int(data_loader.batch_size / world_size)
            if local_batch_size <= 0:
                raise ValueError(
                    "distributed torch_sampler requires batch_size >= world_size"
                )
            return self._rebuild_data_loader_with_sampler(
                data_loader,
                sampler,
                batch_size=local_batch_size,
            )

        if poisson_sampling:
            return DPDataLoader.from_data_loader(
                data_loader,
                generator=self.secure_rng,
                distributed=distributed,
                total_steps=total_steps,
            )
        elif self.secure_mode:
            return switch_generator(data_loader=data_loader, generator=self.secure_rng)
        else:
            return data_loader

    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        max_grad_norm: Union[float, List[float]] = 1.0,
        loss_reduction: str = "mean",
        grad_sample_mode: str = "hooks",
    ) -> AbstractGradSampleModule:
        # Ideally, validation should have been taken care of by calling
        # `get_compatible_module()`
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, AbstractGradSampleModule):
            if (
                module.batch_first != batch_first
                or module.loss_reduction != loss_reduction
                or type(module) is not get_gsm_class(grad_sample_mode)
            ):
                raise ValueError(
                    f"Pre-existing GradSampleModule doesn't match new arguments."
                    f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}, type(module): {type(module)}"
                    f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}, grad_sample_mode: {grad_sample_mode} "
                    f"Please pass vanilla nn.Module instead"
                )

            return module
        else:
            if grad_sample_mode in ["ghost", "ghost_fsdp"]:
                return wrap_model(
                    module,
                    grad_sample_mode=grad_sample_mode,
                    batch_first=batch_first,
                    loss_reduction=loss_reduction,
                    max_grad_norm=max_grad_norm,
                )
            else:
                return wrap_model(
                    module,
                    grad_sample_mode=grad_sample_mode,
                    batch_first=batch_first,
                    loss_reduction=loss_reduction,
                )

    def _prepare_criterion(
        self,
        *,
        module: GradSampleModule,
        optimizer: DPOptimizer,
        criterion=nn.CrossEntropyLoss(),
        loss_reduction: str = "mean",
        **kwargs,
    ) -> DPLossFastGradientClipping:
        """
        Args:
            module: GradSampleModule used for training,
            optimizer: DPOptimizer used for training,
            criterion: Loss function used for training,
            loss_reduction: "mean" or "sum", indicates if the loss reduction (for aggregating the gradients)

        Prepare the DP loss class, which packages the two backward passes for fast gradient clipping.
        """
        return DPLossFastGradientClipping(module, optimizer, criterion, loss_reduction)

    def is_compatible(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ) -> bool:
        """
        Check if task components are compatible with DP.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Returns:
            ``True`` if compatible, ``False`` otherwise
        """
        return ModuleValidator.is_valid(module)

    def validate(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ):
        """
        Validate that task components are compatible with DP.
        Same as ``is_compatible()``, but raises error instead of returning bool.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Raises:
            UnsupportedModuleError
                If one or more modules found to be incompatible
        """
        ModuleValidator.validate(module, strict=True)

    @classmethod
    def get_compatible_module(cls, module: nn.Module) -> nn.Module:
        """
        Return a privacy engine compatible module. Also validates the module after
        running registered fixes.

        Args:
            module: module to be modified

        Returns:
            Module with some submodules replaced for their deep copies or
            close equivalents.
            See :class:`~opacus.validators.module_validator.ModuleValidator` for
            more details
        """
        module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, strict=True)
        return module

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # Added deafult for backward compatibility
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        normalize_clipping: bool = False,
        total_steps: int = None,
        noise_mechanism_config: Optional[NoiseMechanismConfig] = None,
        sampling_semantics: Optional[SamplingSemantics] = None,
        **kwargs,
    ) -> Union[
        Tuple[GradSampleModule, DPOptimizer, DataLoader],
        Tuple[GradSampleModule, DPOptimizer, DPLossFastGradientClipping, DataLoader],
    ]:
        """
        Add privacy-related responsibilities to the main PyTorch training objects:
        model, optimizer, and the data loader.

        All of the returned objects act just like their non-private counterparts
        passed as arguments, but with added DP tasks.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - Criterion is a wrapper around the original criterion that packages the two backward passes for fast gradient clipping.
        - DataLoader is updated to perform Poisson sampling.

        Notes:
            Using any other models, optimizers, or data sources during training
            will invalidate stated privacy guarantees.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise
            grad_sample_mode: mode for computing per sample gradients. Determines the
                implementation class for the wrapped ``module``. See
                :class:`~opacus.grad_sample.gsm_base.AbstractGradSampleModule` for more
                details
            normalize_clipping: Decouples the learning rate and max_grad_norm by normalizing the
                clipped gradients by 1/C as described in the paper "Unlocking High-Accuracy
                Differentially Private Image Classification through Scale" by De et al. (2022)
                - https://arxiv.org/pdf/2204.13650.pdf
            total_steps: Instead of stepping through once the dataloader for once expected epoch,
            we will step through it `total_steps` times. This will set the sample rate to
            batch_size/data_size. The parameter total_steps is any positive integer.
        Returns:
            Tuple of  (model, optimizer, data_loader) or (model, optimizer, criterion, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
             gradient clipping and noise addition to the gradients
            Criterion is a wrapper around the original criterion that packages the two backward passes for fast gradient clipping.
                Only returned when grad_sample_mode is "ghost".
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        mechanism_config = noise_mechanism_config or NoiseMechanismConfig()
        self._validate_mechanism_sampling_compatibility(
            mechanism_config=mechanism_config,
            poisson_sampling=poisson_sampling,
            sampling_semantics=sampling_semantics,
            validate_cyclic_poisson_mode=True,
        )

        if noise_mechanism_config is not None and "noise_mechanism" in kwargs:
            raise ValueError(
                "pass either noise_mechanism_config or noise_mechanism, not both"
            )

        active_accountant = self._accountant_for_mechanism(
            mechanism_config=mechanism_config,
            default_accountant=self.default_accountant,
        )

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        is_dpddp = isinstance(module, DPDDP)
        is_ddp = isinstance(module, DDP)
        is_fsdp = isinstance(module, FSDPModule)
        distributed = is_dpddp or is_ddp or is_fsdp

        requested_noise_mechanism = kwargs.get("noise_mechanism")
        if distributed and (
            mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf")
            or isinstance(requested_noise_mechanism, CorrelatedNoiseMechanism)
        ):
            self._validate_distributed_correlated_support(
                mechanism=mechanism_config.mechanism,
                clipping=clipping,
                grad_sample_mode=grad_sample_mode,
                is_fsdp=is_fsdp,
            )

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )

        if poisson_sampling:
            module.forbid_grad_accumulation()

        # save batch size from original data loader
        batch_size = data_loader.batch_size

        data_loader = self._prepare_data_loader(
            data_loader,
            distributed=distributed,
            poisson_sampling=poisson_sampling,
            sampling_semantics=sampling_semantics,
            total_steps=total_steps,
        )

        sample_rate, expected_batch_size = self._resolve_sample_rate_and_expected_batch_size(
            poisson_sampling=poisson_sampling,
            sampling_semantics=sampling_semantics,
            mechanism=mechanism_config.mechanism,
            batch_size=batch_size,
            dataset_size=len(data_loader.dataset),
            data_loader_len=len(data_loader),
            total_steps=total_steps,
            distributed=distributed,
        )
        semantics = self._build_sampling_semantics(
            poisson_sampling=poisson_sampling,
            sample_rate=sample_rate,
            expected_batch_size=expected_batch_size,
            distributed=distributed,
            explicit_sampling_semantics=sampling_semantics,
        )

        coeff_resolution_kwargs = dict(kwargs)
        if total_steps is not None:
            coeff_resolution_kwargs["total_steps"] = int(total_steps)
        mechanism_config = ensure_bsr_family_fixed_analytical_coeffs_helper(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = ensure_bandmf_fixed_analytical_coeffs_helper(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = self._ensure_bnb_balls_in_bins_gaussian_state(
            mechanism_config=mechanism_config,
            sampling_semantics=sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = self._apply_default_fixed_batch_contract_for_torch_sampler(
            mechanism_config=mechanism_config,
            sampling_semantics=semantics,
            total_steps=int(total_steps) if total_steps is not None else 0,
            dataloader_len=int(len(data_loader)),
            dataset_size=int(len(data_loader.dataset)),
            logical_batch_size=int(batch_size) if batch_size is not None else 0,
            kwargs=kwargs,
        )
        mechanism_config = ensure_bandinvmf_runtime_state_helper(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=semantics,
            steps_hint=int(total_steps) if total_steps is not None else int(len(data_loader)),
            sample_rate_hint=float(sample_rate),
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = self._ensure_bnb_balls_in_bins_mf_state(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=semantics,
            kwargs=coeff_resolution_kwargs,
        )
        if (
            mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf")
            and mechanism_config.mechanism_state.get("z_std") is None
        ):
            state = copy.deepcopy(mechanism_config.mechanism_state)
            state["z_std"] = calibrate_bsr_z_std(
                noise_multiplier_ref=float(noise_multiplier),
                max_grad_norm=float(max_grad_norm),
                denominator=float(expected_batch_size),
            )
            mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        bnb_accounting_kwargs = self._build_bnb_accounting_kwargs_for_state(
            mechanism=mechanism_config.mechanism,
            kwargs=kwargs,
            distributed_dp_runtime=bool(distributed),
        )
        if bnb_accounting_kwargs is not None:
            state = copy.deepcopy(mechanism_config.mechanism_state)
            state["_bnb_accounting_kwargs"] = dict(bnb_accounting_kwargs)
            mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        if (
            mechanism_config.mechanism in ("bandmf", "bisr")
            or (
                mechanism_config.mechanism == "bsr"
                and sampling_semantics is not None
                and sampling_semantics.sampling_mode == "cyclic_poisson"
            )
        ):
            strategy_steps = int(total_steps) if total_steps is not None else int(len(data_loader))
            mechanism_config = ensure_bsr_family_cyclic_coeffs_helper(
                mechanism_config=mechanism_config,
                sampling_semantics=sampling_semantics,
                steps=strategy_steps,
                optimizer=optimizer,
                kwargs=kwargs,
            )

        configured_noise_mechanism = self._build_noise_mechanism_from_config(
            mechanism_config
        )

        optimizer_prepare_kwargs = dict(kwargs)
        if configured_noise_mechanism is not None:
            optimizer_prepare_kwargs["noise_mechanism"] = configured_noise_mechanism

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            normalize_clipping=normalize_clipping,
            **optimizer_prepare_kwargs,
        )

        self._log_bsr_trace(
            stage="make_private",
            mechanism_config=mechanism_config,
            sampling_semantics=semantics,
            sample_rate=float(sample_rate),
            expected_batch_size=int(expected_batch_size),
            noise_multiplier=float(noise_multiplier),
            target_epsilon=None,
            target_delta=None,
            total_steps=int(total_steps) if total_steps is not None else None,
            epochs=None,
            loss_reduction=loss_reduction,
            correlated_denominator=None,
        )

        self.noise_mechanism_config = mechanism_config
        self.sampling_semantics = semantics
        self.accountant = active_accountant
        setattr(optimizer, "accounting_mode", mechanism_config.accounting_mode)
        setattr(optimizer, "noise_mechanism_config", mechanism_config)
        setattr(optimizer, "sampling_semantics", semantics)

        optimizer.attach_step_hook(
            active_accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )
        if "ghost" in grad_sample_mode:
            criterion = self._prepare_criterion(
                module=module,
                optimizer=optimizer,
                criterion=criterion,
                loss_reduction=loss_reduction,
                **kwargs,
            )

            return module, optimizer, criterion, data_loader

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # Added deafult for backward compatibility
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: Optional[int] = None,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        normalize_clipping: bool = False,
        total_steps: int = None,
        noise_mechanism_config: Optional[NoiseMechanismConfig] = None,
        sampling_semantics: Optional[SamplingSemantics] = None,
        **kwargs,
    ) -> Union[
        Tuple[GradSampleModule, DPOptimizer, DataLoader],
        Tuple[GradSampleModule, DPOptimizer, DPLossFastGradientClipping, DataLoader],
    ]:
        """
        Version of :meth:`~opacus.privacy_engine.PrivacyEngine.make_private`,
        that calculates privacy parameters based on a given privacy budget.

        For the full documentation see
        :meth:`~opacus.privacy_engine.PrivacyEngine.make_private`

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            target_epsilon: Target epsilon to be achieved, a metric of privacy loss at differential changes in data.
            target_delta: Target delta to be achieved. Probability of information being leaked.
            epochs: Number of training epochs you intend to perform; noise_multiplier relies on this to calculate
                an appropriate sigma to ensure privacy budget of (target_epsilon, target_delta) at the end
                of epochs. Must be provided when ``total_steps`` is not set.
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise
            grad_sample_mode: mode for computing per sample gradients. Determines the
                implementation class for the wrapped ``module``. See
                :class:`~opacus.grad_sample.gsm_base.AbstractGradSampleModule` for more
                details
            total_steps: Instead of stepping through once the dataloader for once expected epoch,
            we will step through it `total_steps` times. This will set the sample rate to
            batch_size/data_size. The parameter total_steps is any positive integer.
        Returns:
            Tuple of (model, optimizer, data_loader) or (model, optimizer, criterion, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
                gradient clipping and noise addition to the gradients
            Criterion is a wrapper around the original criterion that packages the two backward passes for fast gradient clipping.
                Only returned when grad_sample_mode is "ghost".
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        mechanism_config = noise_mechanism_config or NoiseMechanismConfig()
        if total_steps is None and epochs is None:
            raise ValueError(
                "make_private_with_epsilon requires either `epochs` or `total_steps`"
            )

        active_accountant = self._accountant_for_mechanism(
            mechanism_config=mechanism_config,
            default_accountant=self.default_accountant,
        )

        self._validate_mechanism_sampling_compatibility(
            mechanism_config=mechanism_config,
            poisson_sampling=poisson_sampling,
            sampling_semantics=sampling_semantics,
            validate_cyclic_poisson_mode=False,
        )

        local_sampling_semantics = self._resolve_local_sampling_semantics_for_epsilon(
            mechanism=mechanism_config.mechanism,
            sampling_semantics=sampling_semantics,
            poisson_sampling=poisson_sampling,
            total_steps=total_steps,
            data_loader=data_loader,
        )

        coeff_resolution_kwargs = dict(kwargs)
        if total_steps is not None:
            coeff_resolution_kwargs["total_steps"] = int(total_steps)
        elif epochs is not None:
            coeff_resolution_kwargs["total_steps"] = int(float(epochs) * float(len(data_loader)))
        mechanism_config = ensure_bsr_family_fixed_analytical_coeffs_helper(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=local_sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = ensure_bandmf_fixed_analytical_coeffs_helper(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=local_sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = self._ensure_bnb_balls_in_bins_gaussian_state(
            mechanism_config=mechanism_config,
            sampling_semantics=local_sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )
        batch_size = data_loader.batch_size
        sample_rate_hint = (
            self._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=local_sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
            )
            if total_steps is not None
            else (1.0 / float(len(data_loader)))
        )
        mechanism_config = self._apply_default_fixed_batch_contract_for_torch_sampler(
            mechanism_config=mechanism_config,
            sampling_semantics=local_sampling_semantics,
            total_steps=(
                int(total_steps)
                if total_steps is not None
                else int(float(epochs) * float(len(data_loader)))
            ),
            dataloader_len=int(len(data_loader)),
            dataset_size=int(len(data_loader.dataset)),
            logical_batch_size=int(batch_size) if batch_size is not None else 0,
            kwargs=kwargs,
        )
        mechanism_config = ensure_bandinvmf_runtime_state_helper(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=local_sampling_semantics,
            steps_hint=(
                int(total_steps)
                if total_steps is not None
                else int(float(epochs) * float(len(data_loader)))
            ),
            sample_rate_hint=float(sample_rate_hint),
            kwargs=coeff_resolution_kwargs,
        )
        mechanism_config = self._ensure_bnb_balls_in_bins_mf_state(
            mechanism_config=mechanism_config,
            optimizer=optimizer,
            sampling_semantics=local_sampling_semantics,
            kwargs=coeff_resolution_kwargs,
        )

        is_dpddp = isinstance(module, DPDDP)
        is_ddp = isinstance(module, DDP)
        is_fsdp = isinstance(module, FSDPModule)
        distributed = is_dpddp or is_ddp or is_fsdp

        correlated_denominator = None
        if mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf"):
            _, calibration_expected_batch_size = self._resolve_sample_rate_and_expected_batch_size(
                poisson_sampling=poisson_sampling,
                sampling_semantics=local_sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=int(data_loader.batch_size),
                dataset_size=len(data_loader.dataset),
                data_loader_len=len(data_loader),
                total_steps=total_steps,
                distributed=distributed,
            )
            correlated_denominator = self._bsr_calibration_denominator(
                loss_reduction=loss_reduction,
                expected_batch_size=int(calibration_expected_batch_size),
            )
            self._log_bsr_trace(
                stage="make_private_with_epsilon_pre_calibration",
                mechanism_config=mechanism_config,
                sampling_semantics=local_sampling_semantics,
                sample_rate=None,
                expected_batch_size=int(calibration_expected_batch_size),
                noise_multiplier=None,
                target_epsilon=float(target_epsilon),
                target_delta=float(target_delta),
                total_steps=int(total_steps) if total_steps is not None else None,
                epochs=int(epochs) if epochs is not None else None,
                loss_reduction=loss_reduction,
                correlated_denominator=float(correlated_denominator),
            )

        bnb_c_matrix, bnb_bands, bnb_cycle_length, _ = self._resolve_bnb_runtime_inputs_for_epsilon(
            mechanism_config=mechanism_config,
            sampling_semantics=local_sampling_semantics,
            kwargs=kwargs,
        )

        if (
            mechanism_config.mechanism in ("bandmf", "bisr", "bandinvmf")
            or (
                mechanism_config.mechanism == "bsr"
                and local_sampling_semantics is not None
                and local_sampling_semantics.sampling_mode == "cyclic_poisson"
            )
        ):
            t0 = time.perf_counter()
            if total_steps is not None:
                strategy_steps = int(total_steps)
            else:
                strategy_steps = int(float(epochs) * float(len(data_loader)))
            mechanism_config = ensure_bsr_family_cyclic_coeffs_helper(
                mechanism_config=mechanism_config,
                sampling_semantics=local_sampling_semantics,
                steps=strategy_steps,
                optimizer=optimizer,
                kwargs=kwargs,
            )
            logger.info(
                "OPACUS_DP_TIMING %s",
                json.dumps(
                    {
                        "phase": "ensure_bsr_cyclic_coeffs",
                        "elapsed_s": round(float(time.perf_counter() - t0), 6),
                        "mechanism": mechanism_config.mechanism,
                        "sampling_mode": "cyclic_poisson",
                        "steps": int(strategy_steps),
                    },
                    sort_keys=True,
                ),
            )

        t0 = time.perf_counter()
        noise_multiplier, _ = self._resolve_noise_multiplier_for_target_epsilon(
            mechanism_config=mechanism_config,
            active_accountant=active_accountant,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            total_steps=total_steps,
            epochs=epochs,
            poisson_sampling=poisson_sampling,
            data_loader=data_loader,
            sampling_semantics=local_sampling_semantics,
            bnb_c_matrix=bnb_c_matrix,
            bnb_bands=bnb_bands,
            bnb_cycle_length=bnb_cycle_length,
            kwargs=kwargs,
        )
        logger.info(
            "OPACUS_DP_TIMING %s",
            json.dumps(
                {
                    "phase": "resolve_noise_multiplier_for_target_epsilon_total",
                    "elapsed_s": round(float(time.perf_counter() - t0), 6),
                    "mechanism": mechanism_config.mechanism,
                    "sampling_mode": (
                        local_sampling_semantics.sampling_mode
                        if local_sampling_semantics is not None
                        else None
                    ),
                    "target_epsilon": float(target_epsilon),
                    "target_delta": float(target_delta),
                },
                sort_keys=True,
            ),
        )

        if (
            mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf")
            and local_sampling_semantics is not None
            and local_sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            if total_steps is not None:
                scale_steps = int(total_steps)
            else:
                sample_rate = 1.0 / len(data_loader)
                scale_steps = int(epochs / sample_rate)

            if mechanism_config.mechanism == "bisr":
                resolved_scale = resolve_bisr_sensitivity_scale_for_cyclic_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=local_sampling_semantics,
                    steps=scale_steps,
                    kwargs=kwargs,
                )
            elif mechanism_config.mechanism == "bandinvmf":
                resolved_scale = resolve_bandinvmf_sensitivity_scale_for_cyclic_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=local_sampling_semantics,
                    steps=scale_steps,
                    kwargs=kwargs,
                )
            else:
                resolved_scale = resolve_bsr_sensitivity_scale_for_cyclic_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=local_sampling_semantics,
                    steps=scale_steps,
                    kwargs=kwargs,
                )
            state = copy.deepcopy(mechanism_config.mechanism_state)
            state["_noise_mechanism"] = mechanism_config.mechanism
            state["bsr_sensitivity_scale"] = float(resolved_scale)

            mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )
        elif (
            mechanism_config.mechanism in ("bsr", "bisr", "bandinvmf")
            and (
                local_sampling_semantics is None
                or local_sampling_semantics.sampling_mode == "torch_sampler"
            )
        ):
            sample_rate = self._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=local_sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
            )
            if total_steps is not None:
                mf_steps = int(total_steps)
            else:
                mf_steps = int(epochs / sample_rate)

            if mechanism_config.mechanism == "bisr":
                resolved_mf_sensitivity = resolve_bisr_mf_sensitivity_for_fixed_batch_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=local_sampling_semantics,
                    steps=mf_steps,
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )
            elif mechanism_config.mechanism == "bandinvmf":
                resolved_mf_sensitivity = resolve_bandinvmf_mf_sensitivity_for_fixed_batch_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=local_sampling_semantics,
                    steps=mf_steps,
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )
            else:
                resolved_mf_sensitivity = resolve_bsr_mf_sensitivity_for_fixed_batch_helper(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=local_sampling_semantics,
                    steps=mf_steps,
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )

            state = copy.deepcopy(mechanism_config.mechanism_state)
            state["_noise_mechanism"] = mechanism_config.mechanism
            state["bsr_mf_sensitivity"] = float(resolved_mf_sensitivity)

            mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        bnb_calibration_report = self._build_bnb_calibration_report(
            mechanism=mechanism_config.mechanism,
            bnb_c_matrix=bnb_c_matrix,
            bnb_bands=bnb_bands,
            bnb_cycle_length=bnb_cycle_length,
            noise_multiplier=float(noise_multiplier),
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            kwargs=kwargs,
        )

        bnb_accounting_kwargs = self._build_bnb_accounting_kwargs_for_state(
            mechanism=mechanism_config.mechanism,
            kwargs=kwargs,
            distributed_dp_runtime=bool(distributed),
        )

        mechanism_config = self._apply_correlated_runtime_calibration(
            mechanism_config=mechanism_config,
            max_grad_norm=max_grad_norm,
            noise_multiplier=float(noise_multiplier),
            correlated_denominator=correlated_denominator,
            bnb_calibration_report=bnb_calibration_report,
            bnb_accounting_kwargs=bnb_accounting_kwargs,
        )

        self._log_bsr_trace(
            stage="make_private_with_epsilon_post_calibration",
            mechanism_config=mechanism_config,
            sampling_semantics=local_sampling_semantics,
            sample_rate=None,
            expected_batch_size=None,
            noise_multiplier=float(noise_multiplier),
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            total_steps=int(total_steps) if total_steps is not None else None,
            epochs=int(epochs) if epochs is not None else None,
            loss_reduction=loss_reduction,
            correlated_denominator=(
                float(correlated_denominator)
                if correlated_denominator is not None
                else None
            ),
        )

        if len(active_accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            normalize_clipping=normalize_clipping,
            total_steps=total_steps,
            noise_mechanism_config=mechanism_config,
            sampling_semantics=local_sampling_semantics,
            **kwargs,
        )

    def get_epsilon(self, delta, **kwargs):
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        Args:
            delta: The target delta.

        Returns:
            Privacy budget (epsilon) expended so far.
        """
        if self.accountant.mechanism() in ("bandmf", "bsr", "bisr", "bnb", "bandinvmf"):
            kwargs.setdefault(
                "mechanism_state", self.noise_mechanism_config.mechanism_state
            )
            kwargs.setdefault("sampling_semantics", self.sampling_semantics)
        return self.accountant.get_epsilon(delta, **kwargs)

    def get_accounting_telemetry(self, *, delta: Optional[float] = None) -> Dict[str, Any]:
        """
        Returns a compact runtime snapshot of accounting configuration and state.
        """
        mechanism_config = getattr(self, "noise_mechanism_config", None)
        sampling_semantics = getattr(self, "sampling_semantics", None)
        accountant = getattr(self, "accountant", None)

        mechanism_state = (
            mechanism_config.mechanism_state
            if mechanism_config is not None and isinstance(mechanism_config.mechanism_state, dict)
            else {}
        )
        coeffs = mechanism_state.get("coeffs")
        coeff_count = len(coeffs) if isinstance(coeffs, (list, tuple)) else None
        coeff_head = list(coeffs[:5]) if isinstance(coeffs, (list, tuple)) else None

        payload: Dict[str, Any] = {
            "mechanism": mechanism_config.mechanism if mechanism_config is not None else None,
            "accounting_mode": (
                mechanism_config.accounting_mode if mechanism_config is not None else None
            ),
            "accountant": accountant.mechanism() if accountant is not None else None,
            "sampling_mode": (
                sampling_semantics.sampling_mode if sampling_semantics is not None else None
            ),
            "sampling_metadata": (
                dict(sampling_semantics.privacy_metadata)
                if sampling_semantics is not None
                else None
            ),
            "events_recorded": int(len(accountant)) if accountant is not None else None,
            "z_std": mechanism_state.get("z_std"),
            "coeff_count": coeff_count,
            "coeff_head": coeff_head,
            "coeff_source": mechanism_state.get("coeff_source"),
            "ts_unix": round(time.time(), 6),
        }

        bnb_accounting_kwargs = mechanism_state.get("_bnb_accounting_kwargs")
        if isinstance(bnb_accounting_kwargs, dict):
            payload["bnb_accounting_kwargs"] = dict(bnb_accounting_kwargs)
            payload["bnb_calibration_mode"] = bnb_accounting_kwargs.get(
                "bnb_calibration_mode"
            )

        if mechanism_config is not None:
            if mechanism_config.mechanism in ("bandmf", "bsr", "bisr", "bandinvmf"):
                payload["mechanism_state_summary"] = self._summarize_bsr_state(
                    mechanism_state
                )

        if delta is not None and accountant is not None:
            payload["target_delta"] = float(delta)
            try:
                payload["epsilon_at_target_delta"] = float(self.get_epsilon(delta))
            except Exception as exc:
                payload["epsilon_at_target_delta_error"] = str(exc)

        return payload

    def get_bnb_calibration_report(self):
        """
        Returns parsed BNB calibration report from mechanism_state, if present.
        """
        state = self.noise_mechanism_config.mechanism_state
        if not isinstance(state, dict):
            return None

        payload = state.get("_bnb_calibration_report")
        if payload is None:
            return None

        return parse_bnb_calibration_report(payload)

    def get_bnb_calibration_summary(self) -> str | None:
        """
        Returns compact BNB calibration summary, if report exists.
        """
        report = self.get_bnb_calibration_report()
        if report is None:
            return None

        return describe_bnb_calibration_report(report)

    def get_bnb_calibration_status(self) -> BNBCalibrationStatus | None:
        """
        Returns structured BNB calibration diagnostics, if report exists.
        """
        report = self.get_bnb_calibration_report()
        if report is None:
            return None

        return BNBCalibrationStatus(
            mechanism=self.noise_mechanism_config.mechanism,
            accounting_mode=self.noise_mechanism_config.accounting_mode,
            sampling_mode=self.sampling_semantics.sampling_mode
            if self.sampling_semantics is not None
            else None,
            report=report,
            summary=describe_bnb_calibration_report(report),
        )

    def save_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModule,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        module_state_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_save_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves the state_dict of module, optimizer, and accountant at path.
        Args:
            path: Path to save the state dict objects.
            module: GradSampleModule to save; wrapped module's state_dict is saved.
            optimizer: DPOptimizer to save; wrapped optimizer's state_dict is saved.
            noise_scheduler: _NoiseScheduler whose state we should save.
            grad_clip_scheduler: _GradClipScheduler whose state we should save.
            checkpoint_dict: Dict[str, Any]; an already-filled checkpoint dict.
            module_state_dict_kwargs: dict of kwargs to pass to ``module.state_dict()``
            torch_save_kwargs: dict of kwargs to pass to ``torch.save()``

        """
        if optimizer is not None:
            mech = getattr(optimizer, "noise_mechanism", None)
            rank = getattr(optimizer, "rank", None)
            world_size = getattr(optimizer, "world_size", None)
            if (
                isinstance(mech, CorrelatedNoiseMechanism)
                and rank is not None
                and world_size is not None
                and int(world_size) > 1
                and int(rank) != 0
            ):
                raise ValueError(
                    "distributed correlated-noise checkpoint save is supported only on rank 0"
                )

        checkpoint_dict = checkpoint_dict or {}
        checkpoint_dict["module_state_dict"] = module.state_dict(
            **(module_state_dict_kwargs or {})
        )
        checkpoint_dict["privacy_accountant_state_dict"] = self.accountant.state_dict()
        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint_dict["noise_mechanism_config"] = {
            "mechanism": self.noise_mechanism_config.mechanism,
            "accounting_mode": self.noise_mechanism_config.accounting_mode,
            "mechanism_state": copy.deepcopy(self.noise_mechanism_config.mechanism_state),
        }
        checkpoint_dict["sampling_semantics"] = {
            "sampling_mode": self.sampling_semantics.sampling_mode,
            "privacy_metadata": copy.deepcopy(self.sampling_semantics.privacy_metadata),
        }
        if noise_scheduler is not None:
            checkpoint_dict["noise_scheduler_state_dict"] = noise_scheduler.state_dict()
        if grad_clip_scheduler is not None:
            checkpoint_dict["grad_clip_scheduler_state_dict"] = (
                grad_clip_scheduler.state_dict()
            )

        torch.save(checkpoint_dict, path, **(torch_save_kwargs or {}))

    def load_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModule,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        module_load_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        checkpoint = torch.load(path, **(torch_load_kwargs or {}), weights_only=False)
        module.load_state_dict(
            checkpoint["module_state_dict"], **(module_load_dict_kwargs or {})
        )
        self.accountant.load_state_dict(checkpoint["privacy_accountant_state_dict"])
        mechanism_cfg_payload = checkpoint.get("noise_mechanism_config")

        if isinstance(mechanism_cfg_payload, dict):
            mechanism_state = copy.deepcopy(mechanism_cfg_payload.get("mechanism_state", {}))

            if isinstance(mechanism_state, dict) and "_bnb_calibration_report" in mechanism_state:
                parsed = parse_bnb_calibration_report(
                    mechanism_state["_bnb_calibration_report"]
                )
                mechanism_state["_bnb_calibration_report"] = parsed.to_dict()
                checkpoint["bnb_calibration_summary"] = describe_bnb_calibration_report(
                    parsed
                )

            self.noise_mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_cfg_payload.get("mechanism", "gaussian"),
                accounting_mode=mechanism_cfg_payload.get(
                    "accounting_mode", "standard_step_accountant"
                ),
                mechanism_state=mechanism_state,
            )

        sampling_payload = checkpoint.get("sampling_semantics")
        if isinstance(sampling_payload, dict):
            self.sampling_semantics = SamplingSemantics(
                sampling_mode=sampling_payload.get("sampling_mode", "poisson"),
                privacy_metadata=copy.deepcopy(
                    sampling_payload.get("privacy_metadata", {})
                ),
            )

        optimizer_state_dict = checkpoint.pop("optimizer_state_dict", {})
        if optimizer is not None and len(optimizer_state_dict) > 0:
            optimizer.load_state_dict(optimizer_state_dict)
        elif (optimizer is not None) ^ (len(optimizer_state_dict) > 0):
            # warn if only one of them is available
            warnings.warn(
                f"optimizer_state_dict has {len(optimizer_state_dict)} items"
                f" but optimizer is {'' if optimizer else 'not'} provided."
            )

        noise_scheduler_state_dict = checkpoint.pop("noise_scheduler_state_dict", {})
        if noise_scheduler is not None and len(noise_scheduler_state_dict) > 0:
            noise_scheduler.load_state_dict(noise_scheduler_state_dict)

        grad_clip_scheduler_state_dict = checkpoint.pop(
            "grad_clip_scheduler_state_dict", {}
        )
        if grad_clip_scheduler is not None and len(grad_clip_scheduler_state_dict) > 0:
            grad_clip_scheduler.load_state_dict(grad_clip_scheduler_state_dict)

        return checkpoint
