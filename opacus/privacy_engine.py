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
from itertools import chain
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union

import torch
from opacus.mechanism_contracts import NoiseMechanismConfig, SamplingSemantics
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis.bsr import (
    calibrate_bsr_z_std,
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
)
from opacus.accountants.analysis.bnb import (
    BNBCalibrationStatus,
    calibrate_b_min_sep_noise_multiplier_monte_carlo,
    describe_bnb_calibration_report,
    make_bnb_calibration_report,
    parse_bnb_calibration_report,
    resolve_bnb_calibration_kwargs,
    sample_b_min_sep_llr,
    verify_hockey_stick_delta_hoeffding,
    validate_bnb_c_matrix_contract,
)
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import (
    AbstractGradSampleModule,
    GradSampleModule,
    get_gsm_class,
    wrap_model,
)
from opacus.optimizers import CorrelatedNoiseMechanism, DPOptimizer, get_optimizer_class
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
    def _optimize_bsr_cyclic_coeffs(
        *,
        bands: int,
        steps: int,
        max_optimizer_steps: int = 250,
    ) -> list[float]:
        if bands < 1:
            raise ValueError("cyclic_poisson bands must be > 0")
        if steps < 1:
            raise ValueError("cyclic_poisson steps must be >= 1")
        if steps < bands:
            raise ValueError(
                "cyclic_poisson BSR requires steps >= bands; "
                f"got steps={steps}, bands={bands}"
            )

        dtype = torch.float64
        k = torch.arange(int(bands), dtype=dtype)
        init = torch.cumprod(
            torch.where(k == 0, torch.ones_like(k), (2 * k - 1) / (2 * k)),
            dim=0,
        )
        params = torch.nn.Parameter(init.clone())
        opt = torch.optim.LBFGS(
            [params],
            max_iter=int(max_optimizer_steps),
            line_search_fn="strong_wolfe",
        )

        def _reconcile(c: torch.Tensor, n: int) -> torch.Tensor:
            if c.numel() >= n:
                return c[:n]
            return torch.cat(
                [c, torch.zeros((n - c.numel(),), dtype=c.dtype, device=c.device)],
                dim=0,
            )

        def _mean_error_from_strategy(c: torch.Tensor, n: int) -> torch.Tensor:
            rhs_val = torch.ones((), dtype=c.dtype, device=c.device)
            b_vals: list[torch.Tensor] = []
            for t in range(n):
                acc = rhs_val
                max_lag = min(t, c.numel() - 1)
                for lag in range(1, max_lag + 1):
                    acc = acc - c[lag] * b_vals[t - lag]
                b_vals.append(acc / c[0])
            b = torch.stack(b_vals)
            return torch.cumsum(b * b, dim=0).mean()

        def _loss(v: torch.Tensor) -> torch.Tensor:
            c = _reconcile(v, int(steps))
            penalty = torch.relu(torch.tensor(1e-9, dtype=c.dtype, device=c.device) - c[0]) * 1e9
            return _mean_error_from_strategy(c, int(steps)) * torch.sum(c * c) + penalty

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = _loss(params)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            c = params.detach() / (torch.linalg.norm(params.detach()) + 1e-24)
            if c[0] < 0:
                c = -c
        return [float(x) for x in c.tolist()]

    @staticmethod
    def _ensure_bsr_cyclic_coeffs(
        *,
        mechanism_config: NoiseMechanismConfig,
        sampling_semantics: Optional[SamplingSemantics],
        steps: int,
        kwargs: Dict[str, Any],
    ) -> NoiseMechanismConfig:
        if mechanism_config.mechanism != "bsr":
            return mechanism_config

        if (
            sampling_semantics is None
            or sampling_semantics.sampling_mode != "cyclic_poisson"
        ):
            return mechanism_config

        state = copy.deepcopy(mechanism_config.mechanism_state)
        coeffs = state.get("coeffs")
        if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
            return mechanism_config

        metadata = sampling_semantics.privacy_metadata or {}
        bands = metadata.get("bands", state.get("bands"))
        if bands is None:
            raise ValueError(
                "cyclic_poisson sampling requires privacy_metadata['bands']"
            )
        bands = int(bands)

        max_optimizer_steps = int(kwargs.get("bsr_strategy_max_optimizer_steps", 250))
        state["coeffs"] = PrivacyEngine._optimize_bsr_cyclic_coeffs(
            bands=bands,
            steps=int(steps),
            max_optimizer_steps=max_optimizer_steps,
        )

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
        return {
            "coeff_count": coeff_count,
            "coeff_head": coeff_head,
            "z_std": mechanism_state.get("z_std"),
            "sensitivity_scale": mechanism_state.get("sensitivity_scale"),
            "mf_sensitivity": mechanism_state.get("mf_sensitivity"),
            "min_separation": mechanism_state.get("min_separation"),
            "max_participations": mechanism_state.get("max_participations"),
            "iterations_number": mechanism_state.get("iterations_number"),
            "bands": mechanism_state.get("bands"),
        }

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
        if mechanism_config.mechanism != "bsr":
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
        logger.info("BSR_TRACE %s", json.dumps(payload, sort_keys=True))

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
    def _accountant_for_mechanism(*, mechanism: str, default_accountant):
        if mechanism in ("bsr", "bnb"):
            return create_accountant(mechanism=mechanism)

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
        coeffs = state.get("coeffs")
        if coeffs is None:
            raise ValueError(
                f"{mechanism_name} mechanism requires `mechanism_state['coeffs']`"
            )

        if not isinstance(coeffs, (list, tuple)):
            raise ValueError("`mechanism_state['coeffs']` must be a list or tuple")

        z_std = state.get("z_std")
        if z_std is None:
            raise ValueError(
                f"{mechanism_name} mechanism requires `mechanism_state['z_std']`"
            )

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
    def _resolve_bsr_mf_sensitivity_for_fixed_batch(
        *,
        mechanism_state: Dict[str, Any],
        sampling_semantics: Optional[SamplingSemantics],
        steps: int,
        sample_rate: Optional[float],
        kwargs: Dict[str, Any],
    ) -> float:
        metadata = (
            sampling_semantics.privacy_metadata
            if sampling_semantics is not None
            else {}
        )
        cyclic_only_params = []
        if kwargs.get("bsr_sensitivity_scale") is not None:
            cyclic_only_params.append("bsr_sensitivity_scale")
        if metadata.get("sensitivity_scale") is not None:
            cyclic_only_params.append("privacy_metadata['sensitivity_scale']")
        if mechanism_state.get("sensitivity_scale") is not None:
            cyclic_only_params.append("mechanism_state['sensitivity_scale']")
        if cyclic_only_params:
            raise ValueError(
                "fixed-batch bsr accounting received cyclic-only parameters: "
                + ", ".join(cyclic_only_params)
            )

        coeffs = mechanism_state.get("coeffs")
        max_participations = kwargs.get(
            "bsr_max_participations",
            metadata.get("max_participations", mechanism_state.get("max_participations")),
        )
        min_separation = kwargs.get(
            "bsr_min_separation",
            metadata.get(
                "min_separation",
                mechanism_state.get("min_separation", metadata.get("bands")),
            ),
        )
        sensitivity_steps = kwargs.get(
            "bsr_iterations_number",
            metadata.get("iterations_number", mechanism_state.get("iterations_number")),
        )
        if sensitivity_steps is None:
            sensitivity_steps = steps

        sensitivity_steps = int(sensitivity_steps)
        if sensitivity_steps < 1:
            raise ValueError("bsr_iterations_number must be >= 1")
        if max_participations is None and sample_rate is not None:
            max_participations = int(math.ceil(float(sample_rate) * float(sensitivity_steps)))
            max_participations = max(1, int(max_participations))
        if min_separation is None:
            # Conservative default for fixed-batch schedule when no stronger contract is provided.
            min_separation = 1

        mf_sensitivity = kwargs.get(
            "bsr_mf_sensitivity",
            metadata.get("mf_sensitivity", mechanism_state.get("mf_sensitivity")),
        )
        if mf_sensitivity is not None:
            resolved = float(mf_sensitivity)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError("bsr_mf_sensitivity must be finite and > 0")
            if (
                coeffs is not None
                and max_participations is not None
                and min_separation is not None
            ):
                derived = float(
                    compute_bsr_mf_sensitivity_from_coeffs(
                        coeffs=coeffs,
                        steps=sensitivity_steps,
                        max_participations=int(max_participations),
                        min_separation=int(min_separation),
                    )
                )
                if not math.isfinite(derived) or derived <= 0.0:
                    raise ValueError(
                        "derived bsr_mf_sensitivity must be finite and > 0 "
                        "when validating explicit bsr_mf_sensitivity"
                    )
                if not math.isclose(
                    resolved, derived, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise ValueError(
                        "provided bsr_mf_sensitivity is inconsistent with "
                        "coeffs/max_participations/min_separation for the resolved "
                        "bsr_iterations_number"
                    )
            return resolved

        if coeffs is None or max_participations is None or min_separation is None:
            raise ValueError(
                "fixed-batch bsr accounting requires MF sensitivity or "
                "enough metadata to derive it: "
                "`mechanism_state['coeffs']`, "
                "`max_participations`, `min_separation`"
            )

        resolved = float(
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=sensitivity_steps,
                max_participations=int(max_participations),
                min_separation=int(min_separation),
            )
        )
        if not math.isfinite(resolved) or resolved <= 0.0:
            raise ValueError("resolved bsr_mf_sensitivity must be finite and > 0")
        return resolved

    @staticmethod
    def _resolve_bsr_sensitivity_scale_for_cyclic(
        *,
        mechanism_state: Dict[str, Any],
        sampling_semantics: Optional[SamplingSemantics],
        steps: int,
        kwargs: Dict[str, Any],
    ) -> float:
        metadata = (
            sampling_semantics.privacy_metadata
            if sampling_semantics is not None
            else {}
        )
        fixed_only_params = []
        if kwargs.get("bsr_mf_sensitivity") is not None:
            fixed_only_params.append("bsr_mf_sensitivity")
        if kwargs.get("bsr_max_participations") is not None:
            fixed_only_params.append("bsr_max_participations")
        if kwargs.get("bsr_min_separation") is not None:
            fixed_only_params.append("bsr_min_separation")
        if metadata.get("mf_sensitivity") is not None:
            fixed_only_params.append("privacy_metadata['mf_sensitivity']")
        if metadata.get("max_participations") is not None:
            fixed_only_params.append("privacy_metadata['max_participations']")
        if metadata.get("min_separation") is not None:
            fixed_only_params.append("privacy_metadata['min_separation']")
        if mechanism_state.get("mf_sensitivity") is not None:
            fixed_only_params.append("mechanism_state['mf_sensitivity']")
        if mechanism_state.get("max_participations") is not None:
            fixed_only_params.append("mechanism_state['max_participations']")
        if mechanism_state.get("min_separation") is not None:
            fixed_only_params.append("mechanism_state['min_separation']")
        if fixed_only_params:
            raise ValueError(
                "cyclic-poisson bsr accounting received fixed-batch-only parameters: "
                + ", ".join(fixed_only_params)
            )

        explicit_scale = kwargs.get(
            "bsr_sensitivity_scale",
            metadata.get("sensitivity_scale", mechanism_state.get("sensitivity_scale")),
        )
        if explicit_scale is not None:
            scale = float(explicit_scale)
            if (not math.isfinite(scale)) or scale <= 0.0:
                raise ValueError("bsr_sensitivity_scale must be finite and > 0")
            return scale

        coeffs = mechanism_state.get("coeffs")
        if coeffs is None:
            raise ValueError(
                "cyclic-poisson bsr accounting requires either `bsr_sensitivity_scale` "
                "or `mechanism_state['coeffs']`"
            )

        scale_steps = kwargs.get(
            "bsr_iterations_number",
            metadata.get("iterations_number", mechanism_state.get("iterations_number")),
        )
        if scale_steps is None:
            scale_steps = steps

        scale_steps = int(scale_steps)
        if scale_steps < 1:
            raise ValueError("bsr_iterations_number must be >= 1")

        bands = metadata.get("bands", mechanism_state.get("bands"))
        if bands is None and coeffs is not None:
            bands = len(coeffs)

        if bands is not None:
            resolved_bands = int(bands)
            if resolved_bands <= 0:
                raise ValueError("cyclic_poisson bands must be > 0")

            if scale_steps < resolved_bands:
                raise ValueError(
                    "cyclic_poisson BSR requires steps >= bands; "
                    f"got steps={scale_steps}, bands={resolved_bands}"
                )

        scale = float(
            compute_bsr_kappa_from_coeffs(
                coeffs=coeffs,
                steps=scale_steps,
            )
        )
        if (not math.isfinite(scale)) or scale <= 0.0:
            raise ValueError("resolved bsr_sensitivity_scale must be finite and > 0")

        return scale

    @staticmethod
    def _resolve_bnb_b_min_sep_inputs(
        *,
        mechanism_state: Dict[str, Any],
        sampling_semantics: Optional[SamplingSemantics],
        kwargs: Dict[str, Any],
    ) -> tuple[Any, int, Dict[str, Any]]:
        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
        c_matrix = kwargs.get("bnb_c_matrix", state.get("c_matrix"))
        bands = kwargs.get("bnb_bands", metadata.get("bands", state.get("bands")))

        c_matrix_contract = kwargs.get(
            "bnb_c_matrix_contract",
            state.get("c_matrix_contract"),
        )

        if c_matrix is None or bands is None or c_matrix_contract is None:
            raise ValueError(
                "bnb calibration requires b_min_sep/balls_in_bins inputs: "
                "`c_matrix`, `bands`, and `c_matrix_contract`"
            )

        return c_matrix, int(bands), c_matrix_contract

    @staticmethod
    def _validate_bnb_accounting_runtime_consistency(
        *,
        mechanism_state: Dict[str, Any],
        sampling_semantics: Optional[SamplingSemantics],
        c_matrix: Any,
        bands: int,
        c_matrix_contract: Dict[str, Any],
    ) -> None:
        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        coeffs = state.get("coeffs")

        if coeffs is None or not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
            raise ValueError(
                "bnb mechanism_state consistency check requires non-empty `coeffs`"
            )

        if int(bands) != len(coeffs):
            raise ValueError(
                "bnb consistency check failed: `bands` must match len(coeffs); "
                f"got bands={int(bands)} and len(coeffs)={len(coeffs)}"
            )

        metadata = (
            sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
        )

        metadata_bands = metadata.get("bands")
        if metadata_bands is not None and int(metadata_bands) != int(bands):
            raise ValueError(
                "bnb consistency check failed: sampling_semantics privacy_metadata['bands'] "
                f"({int(metadata_bands)}) != accounting bands ({int(bands)})"
            )

        if not torch.is_tensor(c_matrix):
            raise ValueError(
                "bnb consistency check requires `c_matrix` to be a torch.Tensor"
            )

        if c_matrix.ndim != 2:
            raise ValueError("bnb consistency check requires `c_matrix` with shape [d, m]")

        d, _m = c_matrix.shape
        if d < int(bands):
            raise ValueError(
                "bnb consistency check failed: c_matrix must have at least `bands` rows; "
                f"got rows={d}, bands={int(bands)}"
            )

        validate_bnb_c_matrix_contract(
            c_matrix=c_matrix,
            coeffs=coeffs,
            bands=int(bands),
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

    def _build_cyclic_poisson_sampler(
        self,
        *,
        data_loader: DataLoader,
        distributed: bool,
        bands: int,
        total_steps: Optional[int],
    ):
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("cyclic_poisson sampling is not supported for IterableDataset")

        if data_loader.batch_size is None:
            raise ValueError("cyclic_poisson sampling requires data_loader.batch_size")

        steps = total_steps if total_steps is not None else len(data_loader)
        if int(steps) < int(bands):
            raise ValueError(
                "cyclic_poisson BSR requires steps >= bands; "
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
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("b_min_sep sampling is not supported for IterableDataset")

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
        if isinstance(data_loader.dataset, torch.utils.data.IterableDataset):
            raise ValueError("balls_in_bins sampling is not supported for IterableDataset")

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
    ) -> None:
        if mechanism != "bnb" or sampling_semantics is None:
            return

        mode = sampling_semantics.sampling_mode
        if mode == "b_min_sep":
            raise ValueError(
                "b_min_sep sampling is temporarily disabled pending p-aware BNB accounting; "
                "use sampling_mode='balls_in_bins'"
            )

        if mode not in ("balls_in_bins",):
            raise ValueError(
                "bnb mechanism requires sampling_semantics in "
                "{'balls_in_bins'}"
            )

    def _validate_mechanism_sampling_compatibility(
        self,
        *,
        mechanism_config: NoiseMechanismConfig,
        poisson_sampling: bool,
        sampling_semantics: Optional[SamplingSemantics],
        validate_cyclic_poisson_mode: bool,
    ) -> None:
        mechanism = mechanism_config.mechanism
        if mechanism == "bsr" and poisson_sampling:
            raise ValueError(
                "bsr mechanism requires fixed-batch semantics; "
                "set poisson_sampling=False"
            )

        if mechanism == "bnb" and poisson_sampling:
            raise ValueError(
                "bnb mechanism requires explicit non-Poisson sampling semantics; "
                "set poisson_sampling=False and provide sampling_semantics"
            )

        if mechanism == "bnb" and sampling_semantics is None:
            raise ValueError(
                "bnb mechanism requires explicit sampling_semantics "
                "in {'b_min_sep', 'balls_in_bins'}"
            )

        self._validate_bnb_sampling_policy(
            sampling_semantics=sampling_semantics,
            mechanism=mechanism,
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
            and mechanism != "bnb"
        ):
            raise ValueError(
                "balls_in_bins sampling is supported only for mechanism='bnb'"
            )

        if (
            validate_cyclic_poisson_mode
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "cyclic_poisson"
            and mechanism != "bsr"
        ):
            raise ValueError(
                "cyclic_poisson sampling is supported only for mechanism='bsr'"
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
            if mechanism == "bsr":
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
        if mechanism == "bsr" and local_sampling_semantics is None:
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
    ) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[Dict[str, Any]]]:
        if mechanism_config.mechanism != "bnb":
            return None, None, None

        if (
            sampling_semantics is None
            or sampling_semantics.sampling_mode not in ("balls_in_bins",)
        ):
            raise ValueError(
                "bnb calibration requires sampling_semantics with "
                "sampling_mode in {'balls_in_bins'}"
            )

        bnb_c_matrix, bnb_bands, bnb_c_matrix_contract = self._resolve_bnb_b_min_sep_inputs(
            mechanism_state=mechanism_config.mechanism_state,
            sampling_semantics=sampling_semantics,
            kwargs=kwargs,
        )

        self._validate_bnb_accounting_runtime_consistency(
            mechanism_state=mechanism_config.mechanism_state,
            sampling_semantics=sampling_semantics,
            c_matrix=bnb_c_matrix,
            bands=int(bnb_bands),
            c_matrix_contract=bnb_c_matrix_contract,
        )

        return bnb_c_matrix, int(bnb_bands), bnb_c_matrix_contract

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
        if total_steps:
            if epochs is not None:
                raise ValueError(
                    "make_private_with_epsilon takes as input EITHER a number of steps or a number of epochs"
                )

            sample_rate = self._resolve_total_steps_sample_rate(
                poisson_sampling=poisson_sampling,
                sampling_semantics=sampling_semantics,
                mechanism=mechanism_config.mechanism,
                batch_size=data_loader.batch_size,
                dataset_size=len(data_loader.dataset),
            )

            bsr_mf_sensitivity = None
            if (
                mechanism_config.mechanism == "bsr"
                and sampling_semantics is not None
                and sampling_semantics.sampling_mode == "cyclic_poisson"
            ):
                nm_kwargs["bsr_sensitivity_scale"] = self._resolve_bsr_sensitivity_scale_for_cyclic(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=int(total_steps),
                    kwargs=kwargs,
                )
            if (
                mechanism_config.mechanism == "bsr"
                and sampling_semantics is not None
                and sampling_semantics.sampling_mode == "torch_sampler"
            ):
                bsr_mf_sensitivity = self._resolve_bsr_mf_sensitivity_for_fixed_batch(
                    mechanism_state=mechanism_config.mechanism_state,
                    sampling_semantics=sampling_semantics,
                    steps=int(total_steps),
                    sample_rate=sample_rate,
                    kwargs=kwargs,
                )

            noise_multiplier = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                steps=total_steps,
                accountant=active_accountant.mechanism(),
                mechanism_state=mechanism_config.mechanism_state,
                sampling_semantics=sampling_semantics,
                bsr_mf_sensitivity=bsr_mf_sensitivity,
                **nm_kwargs,
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
            mechanism_config.mechanism == "bsr"
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            nm_kwargs["bsr_sensitivity_scale"] = self._resolve_bsr_sensitivity_scale_for_cyclic(
                mechanism_state=mechanism_config.mechanism_state,
                sampling_semantics=sampling_semantics,
                steps=implied_steps,
                kwargs=kwargs,
            )
        if (
            mechanism_config.mechanism == "bsr"
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "torch_sampler"
        ):
            bsr_mf_sensitivity = self._resolve_bsr_mf_sensitivity_for_fixed_batch(
                mechanism_state=mechanism_config.mechanism_state,
                sampling_semantics=sampling_semantics,
                steps=implied_steps,
                sample_rate=sample_rate,
                kwargs=kwargs,
            )

        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=implied_steps,
            accountant=active_accountant.mechanism(),
            mechanism_state=mechanism_config.mechanism_state,
            sampling_semantics=sampling_semantics,
            bsr_mf_sensitivity=bsr_mf_sensitivity,
            **nm_kwargs,
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
        kwargs: Dict[str, Any],
    ) -> Tuple[float, float]:
        if bnb_c_matrix is None or bnb_bands is None:
            raise ValueError(
                "bnb calibration requires resolved runtime inputs (`c_matrix`, `bands`)"
            )

        if total_steps:
            if epochs is not None:
                raise ValueError(
                    "make_private_with_epsilon takes as input EITHER a number of steps or a number of epochs"
                )

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
        noise_multiplier = calibrate_b_min_sep_noise_multiplier_monte_carlo(
            c_matrix=bnb_c_matrix,
            bands=int(bnb_bands),
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            num_samples=int(calibration_cfg["bnb_num_samples"]),
            seed=int(calibration_cfg["bnb_seed"]),
            reduce_dimensionality=bool(calibration_cfg["bnb_reduce_dimensionality"]),
            sigma_low=float(kwargs.get("bnb_sigma_low", 1e-7)),
            sigma_high=float(kwargs.get("bnb_sigma_high", 100.0)),
            tolerance=float(calibration_cfg["bnb_tolerance"]),
            max_iterations=int(calibration_cfg["bnb_max_iterations"]),
            max_sigma=float(kwargs.get("bnb_max_sigma", 1e6)),
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
        kwargs: Dict[str, Any],
    ) -> Tuple[float, float]:
        nm_kwargs = dict(kwargs)
        nm_kwargs.pop("bsr_mf_sensitivity", None)

        if mechanism_config.mechanism == "bnb":
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
        noise_multiplier: float,
        target_epsilon: float,
        target_delta: float,
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if (
            mechanism != "bnb"
            or bnb_c_matrix is None
            or bnb_bands is None
        ):
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
    ) -> Optional[Dict[str, Any]]:
        if mechanism != "bnb":
            return None

        calibration_cfg = resolve_bnb_calibration_kwargs(
            overrides=kwargs,
        )
        return {
            "bnb_num_samples": int(calibration_cfg["bnb_num_samples"]),
            "bnb_seed": int(calibration_cfg["bnb_seed"]),
            "bnb_reduce_dimensionality": bool(
                calibration_cfg["bnb_reduce_dimensionality"]
            ),
            "bnb_tolerance": float(calibration_cfg["bnb_tolerance"]),
            "bnb_max_iterations": int(calibration_cfg["bnb_max_iterations"]),
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
        if mechanism_config.mechanism not in ("bsr", "bnb"):
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
            mechanism=mechanism_config.mechanism,
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
            mechanism_config.mechanism in ("bsr", "bnb")
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

        if (
            mechanism_config.mechanism == "bsr"
            and sampling_semantics is not None
            and sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            strategy_steps = int(total_steps) if total_steps is not None else int(len(data_loader))
            mechanism_config = self._ensure_bsr_cyclic_coeffs(
                mechanism_config=mechanism_config,
                sampling_semantics=sampling_semantics,
                steps=strategy_steps,
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

        semantics = self._build_sampling_semantics(
            poisson_sampling=poisson_sampling,
            sample_rate=sample_rate,
            expected_batch_size=expected_batch_size,
            distributed=distributed,
            explicit_sampling_semantics=sampling_semantics,
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
            mechanism=mechanism_config.mechanism,
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

        is_dpddp = isinstance(module, DPDDP)
        is_ddp = isinstance(module, DDP)
        is_fsdp = isinstance(module, FSDPModule)
        distributed = is_dpddp or is_ddp or is_fsdp

        correlated_denominator = None
        if mechanism_config.mechanism in ("bsr", "bnb"):
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

        bnb_c_matrix, bnb_bands, _ = self._resolve_bnb_runtime_inputs_for_epsilon(
            mechanism_config=mechanism_config,
            sampling_semantics=local_sampling_semantics,
            kwargs=kwargs,
        )

        if (
            mechanism_config.mechanism == "bsr"
            and local_sampling_semantics is not None
            and local_sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            if total_steps is not None:
                strategy_steps = int(total_steps)
            else:
                strategy_steps = int(float(epochs) * float(len(data_loader)))
            mechanism_config = self._ensure_bsr_cyclic_coeffs(
                mechanism_config=mechanism_config,
                sampling_semantics=local_sampling_semantics,
                steps=strategy_steps,
                kwargs=kwargs,
            )

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
            kwargs=kwargs,
        )

        if (
            mechanism_config.mechanism == "bsr"
            and local_sampling_semantics is not None
            and local_sampling_semantics.sampling_mode == "cyclic_poisson"
        ):
            if total_steps is not None:
                scale_steps = int(total_steps)
            else:
                sample_rate = 1.0 / len(data_loader)
                scale_steps = int(epochs / sample_rate)

            resolved_scale = self._resolve_bsr_sensitivity_scale_for_cyclic(
                mechanism_state=mechanism_config.mechanism_state,
                sampling_semantics=local_sampling_semantics,
                steps=scale_steps,
                kwargs=kwargs,
            )
            state = copy.deepcopy(mechanism_config.mechanism_state)
            state["sensitivity_scale"] = float(resolved_scale)

            mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )
        elif (
            mechanism_config.mechanism == "bsr"
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

            resolved_mf_sensitivity = self._resolve_bsr_mf_sensitivity_for_fixed_batch(
                mechanism_state=mechanism_config.mechanism_state,
                sampling_semantics=local_sampling_semantics,
                steps=mf_steps,
                sample_rate=sample_rate,
                kwargs=kwargs,
            )

            state = copy.deepcopy(mechanism_config.mechanism_state)
            state["mf_sensitivity"] = float(resolved_mf_sensitivity)

            mechanism_config = NoiseMechanismConfig(
                mechanism=mechanism_config.mechanism,
                accounting_mode=mechanism_config.accounting_mode,
                mechanism_state=state,
            )

        bnb_calibration_report = self._build_bnb_calibration_report(
            mechanism=mechanism_config.mechanism,
            bnb_c_matrix=bnb_c_matrix,
            bnb_bands=bnb_bands,
            noise_multiplier=float(noise_multiplier),
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            kwargs=kwargs,
        )

        bnb_accounting_kwargs = self._build_bnb_accounting_kwargs_for_state(
            mechanism=mechanism_config.mechanism,
            kwargs=kwargs,
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
        )

    def get_epsilon(self, delta, **kwargs):
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        Args:
            delta: The target delta.

        Returns:
            Privacy budget (epsilon) expended so far.
        """
        if self.accountant.mechanism() in ("bsr", "bnb"):
            kwargs.setdefault(
                "mechanism_state", self.noise_mechanism_config.mechanism_state
            )
            kwargs.setdefault("sampling_semantics", self.sampling_semantics)
        return self.accountant.get_epsilon(delta, **kwargs)

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

            if (
                mechanism_cfg_payload.get("mechanism") == "bnb"
                and isinstance(mechanism_state, dict)
                and "_bnb_calibration_report" in mechanism_state
            ):
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
