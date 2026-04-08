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

from __future__ import annotations

"""
Accountant adapter for BandMF fixed-batch and cyclic accounting.

Paper mapping:
- `bands` -> band width / minimum-separation parameter `b`
- `steps` -> finite horizon `T`
- `sample_rate` -> base participation probability `p`
- `bsr_mf_sensitivity` -> fixed-batch Toeplitz sensitivity `S_{k,b}(C;T)`

Source lineage:
- Multi-Epoch MF (Choquette-Choo et al., 2023) for banded MF strategy context
- BSR (Kalinin and Lampert, 2024) for the reduced fixed-batch Gaussian path
  after BandMF has been packaged as a Toeplitz family.
"""

import copy
import math
from typing import Any, Dict

from torch import optim

from opacus.accountants.analysis.bandmf import (
    compute_bandmf_mf_sensitivity_from_coeffs,
    generate_bandmf_coeffs_from_sgd_workload,
    optimize_bandmf_strategy_coeffs,
)
from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    bsr_fixed_batch_epsilon_upper_bound,
    resolve_bsr_fixed_batch_gaussian_contract,
    resolve_bsr_cyclic_gaussian_contract,
)
from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer
from opacus.mechanism_contracts import NoiseMechanismConfig

from .accountant import IAccountant


def optimize_cyclic_bandmf_strategy_coeffs(
    *,
    bands: int,
    steps: int,
    max_optimizer_steps: int = 250,
) -> list[float]:
    return optimize_bandmf_strategy_coeffs(
        steps=int(steps),
        bands=int(bands),
        max_optimizer_steps=int(max_optimizer_steps),
    )


def ensure_bandmf_fixed_analytical_coeffs(
    *,
    mechanism_config: NoiseMechanismConfig,
    optimizer: optim.Optimizer,
    sampling_semantics,
    kwargs: Dict[str, Any],
) -> NoiseMechanismConfig:
    if mechanism_config.mechanism != "bandmf":
        return mechanism_config

    state = copy.deepcopy(mechanism_config.mechanism_state)
    coeffs = state.get("coeffs")
    if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
        return mechanism_config

    metadata = (
        sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    )
    metadata_bands = metadata.get("bands")
    explicit_bands = kwargs.get("bsr_bands")
    if explicit_bands is not None and metadata_bands is not None:
        if int(explicit_bands) != int(metadata_bands):
            raise ValueError(
                "conflicting canonical inputs: `bsr_bands` must match "
                "sampling_semantics privacy_metadata['bands']"
            )

    bands = kwargs.get("bsr_bands", metadata.get("bands", state.get("bsr_bands")))
    if bands is None:
        raise ValueError(
            "bandmf analytical auto-coeff generation requires bands via "
            "`mechanism_state['bsr_bands']`, `sampling_semantics.privacy_metadata['bands']`, "
            "or `bsr_bands`"
        )

    bands = int(bands)
    if bands < 1:
        raise ValueError("bandmf bands must be >= 1")

    steps_hint = kwargs.get("total_steps", metadata.get("total_steps"))
    if steps_hint is None:
        raise ValueError(
            "paper-faithful bandmf auto-coeff generation requires total_steps "
            "or sampling metadata total_steps"
        )

    if int(steps_hint) < bands:
        raise ValueError(
            "bandmf analytical auto-coeff generation requires steps >= bands; "
            f"got steps={int(steps_hint)}, bands={bands}"
        )

    momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(
        optimizer=optimizer
    )
    state["coeffs"] = generate_bandmf_coeffs_from_sgd_workload(
        bands=bands,
        momentum=momentum,
        weight_decay=weight_decay,
        steps=int(steps_hint),
    )
    state["bsr_bands"] = bands
    state["coeff_source"] = "analytical_auto"

    return NoiseMechanismConfig(
        mechanism=mechanism_config.mechanism,
        accounting_mode=mechanism_config.accounting_mode,
        mechanism_state=state,
    )


def resolve_bandmf_mf_sensitivity_for_fixed_batch(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: float | None,
    kwargs: Dict[str, Any],
) -> float:
    metadata = (
        sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    )
    explicit = kwargs.get(
        "bsr_mf_sensitivity",
        metadata.get(
            "bsr_mf_sensitivity",
            mechanism_state.get("bsr_mf_sensitivity"),
        ),
    )
    if explicit is not None:
        return float(explicit)

    (
        coeffs,
        max_participations,
        min_separation,
        sensitivity_steps,
        mf_sensitivity,
        explicit_mf_sensitivity_override,
    ) = BandMFAccountant._resolve_fixed_batch_contract_inputs(
        state=mechanism_state,
        metadata=metadata,
        kwargs=kwargs,
        total_steps=int(steps),
        sample_rate=float(sample_rate) if sample_rate is not None else 0.0,
    )
    return BandMFAccountant._resolve_fixed_batch_mf_sensitivity(
        coeffs=coeffs,
        max_participations=max_participations,
        min_separation=min_separation,
        sensitivity_steps=sensitivity_steps,
        mf_sensitivity=mf_sensitivity,
        explicit_mf_sensitivity_override=explicit_mf_sensitivity_override,
    )


class BandMFAccountant(IAccountant):
    """
    Accountant adapter for BandMF accounting.

    BandMF supports two accounting reductions:
    1. fixed-batch Toeplitz sensitivity under ``(k, b)`` participation;
    2. cyclic-poisson reduction via ``q = bands * sample_rate`` and a
       sensitivity normalization scale.

    Source: BandMF (Choquette-Choo et al., 2023), fixed-participation and
    cyclic Toeplitz accounting reductions.
    """

    def __init__(self):
        super().__init__()
        self.last_contract = None

    @staticmethod
    def _resolve_cyclic_contract(*, sample_rate: float, steps: int, bands: int) -> dict:
        """
        Resolve and validate the cyclic accountant contract ``(q, cycles)``.

        This helper exists so contract logic stays explicit and testable before
        calling privacy accounting routines.
        """
        if bands <= 0:
            raise ValueError("bands must be > 0")
        if steps < bands:
            raise ValueError(f"steps must be >= bands; got steps={steps}, bands={bands}")

        # `q` is effective cyclic participation probability.
        q = float(sample_rate) * float(bands)
        if not math.isfinite(q) or q <= 0.0 or q > 1.0:
            raise ValueError(
                f"derived q = bands * sample_rate must be in (0, 1]; got {q}"
            )

        # `cycles` is the finite horizon count ceil(steps / bands).
        cycles = int(math.ceil(float(steps) / float(bands)))
        if cycles <= 0:
            raise ValueError(f"derived cycles must be > 0; got {cycles}")

        return {"q": q, "cycles": cycles, "bands": int(bands), "steps": int(steps)}

    @staticmethod
    def _resolve_fixed_batch_contract_inputs(
        *,
        state: dict,
        metadata: dict,
        kwargs: dict,
        total_steps: int,
        sample_rate: float,
    ) -> tuple[object, int, int, int, object, bool]:
        mf_sensitivity = kwargs.get(
            "bsr_mf_sensitivity",
            metadata.get(
                "bsr_mf_sensitivity",
                state.get("bsr_mf_sensitivity"),
            ),
        )
        explicit_mf_sensitivity_override = "bsr_mf_sensitivity" in kwargs
        coeffs = state.get("coeffs")
        max_participations = kwargs.get(
            "bsr_max_participations",
            metadata.get(
                "bsr_max_participations",
                state.get("bsr_max_participations"),
            ),
        )
        min_separation = kwargs.get(
            "bsr_min_separation",
            metadata.get(
                "bsr_min_separation",
                state.get("bsr_min_separation"),
            ),
        )
        sensitivity_steps = kwargs.get(
            "bsr_iterations_number",
            metadata.get(
                "bsr_iterations_number",
                state.get("bsr_iterations_number"),
            ),
        )

        if sensitivity_steps is None:
            sensitivity_steps = total_steps

        sensitivity_steps = int(sensitivity_steps)
        if sensitivity_steps < 1:
            raise ValueError("bsr_iterations_number must be >= 1")

        if max_participations is None:
            max_participations = int(math.ceil(float(sample_rate) * float(sensitivity_steps)))
            max_participations = max(1, int(max_participations))

        if min_separation is None:
            min_separation = 1

        return (
            coeffs,
            int(max_participations),
            int(min_separation),
            int(sensitivity_steps),
            mf_sensitivity,
            bool(explicit_mf_sensitivity_override),
        )

    @staticmethod
    def _resolve_fixed_batch_mf_sensitivity(
        *,
        coeffs,
        max_participations: int,
        min_separation: int,
        sensitivity_steps: int,
        mf_sensitivity,
        explicit_mf_sensitivity_override: bool,
    ) -> float:
        def _derive_from_coeffs() -> float:
            return float(
                compute_bandmf_mf_sensitivity_from_coeffs(
                    coeffs=coeffs,
                    steps=sensitivity_steps,
                    max_participations=int(max_participations),
                    min_separation=int(min_separation),
                )
            )

        if mf_sensitivity is None:
            if coeffs is None:
                raise ValueError(
                    "fixed-batch bandmf accounting requires MF sensitivity or "
                    "enough data to derive it: `coeffs`, `max_participations`, `min_separation`"
                )
            return _derive_from_coeffs()

        mf_sensitivity = float(mf_sensitivity)
        if not math.isfinite(mf_sensitivity) or mf_sensitivity <= 0.0:
            raise ValueError("bsr_mf_sensitivity must be finite and > 0")

        if explicit_mf_sensitivity_override and coeffs is not None:
            derived = _derive_from_coeffs()
            if not math.isclose(mf_sensitivity, derived, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "provided bsr_mf_sensitivity is inconsistent with "
                    "coeffs/max_participations/min_separation for the resolved "
                    "bsr_iterations_number"
                )

        return float(mf_sensitivity)

    def _get_epsilon_fixed_batch(
        self,
        *,
        delta: float,
        noise_multiplier: float,
        sample_rate: float,
        total_steps: int,
        state: dict,
        metadata: dict,
        kwargs: dict,
    ) -> float:
        (
            coeffs,
            max_participations,
            min_separation,
            sensitivity_steps,
            mf_sensitivity,
            explicit_mf_sensitivity_override,
        ) = self._resolve_fixed_batch_contract_inputs(
            state=state,
            metadata=metadata,
            kwargs=kwargs,
            total_steps=total_steps,
            sample_rate=sample_rate,
        )
        resolved_mf_sensitivity = self._resolve_fixed_batch_mf_sensitivity(
            coeffs=coeffs,
            max_participations=max_participations,
            min_separation=min_separation,
            sensitivity_steps=sensitivity_steps,
            mf_sensitivity=mf_sensitivity,
            explicit_mf_sensitivity_override=explicit_mf_sensitivity_override,
        )
        reduced_contract = resolve_bsr_fixed_batch_gaussian_contract(
            noise_multiplier=float(noise_multiplier),
            mf_sensitivity=float(resolved_mf_sensitivity),
        )
        self.last_contract = {
            "mechanism": "bandmf",
            "accounting_mode": "fixed_batch_prv",
            "sampling_mode": "torch_sampler",
            "global_steps": int(total_steps),
            "sample_rate": float(sample_rate),
            "sensitivity_steps": int(sensitivity_steps),
            "max_participations": int(max_participations),
            "min_separation": int(min_separation),
            "mf_sensitivity": float(resolved_mf_sensitivity),
            "effective_noise_multiplier": float(
                reduced_contract["effective_noise_multiplier"]
            ),
            "accountant_backend": "prv",
        }
        return float(
            bsr_fixed_batch_epsilon_upper_bound(
                noise_multiplier=float(noise_multiplier),
                target_delta=float(delta),
                mf_sensitivity=float(resolved_mf_sensitivity),
            )
        )

    def step(self, *, noise_multiplier: float, sample_rate: float):
        # `noise_multiplier` and `sample_rate` are recorded per step for composition.
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))
        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(
        self,
        delta: float,
        *,
        mechanism_state=None,
        sampling_semantics=None,
        **kwargs,
    ) -> float:
        """
        Compute epsilon for BandMF under the sampling-selected accounting branch.
        """
        if not self.history:
            return 0.0

        noise_multiplier, sample_rate, _ = self.history[0]
        total_steps = 0
        for nm_i, sr_i, steps_i in self.history:
            if nm_i != noise_multiplier or sr_i != sample_rate:
                raise ValueError(
                    "bandmf accountant currently expects constant "
                    "noise_multiplier and sample_rate across steps"
                )

            total_steps += int(steps_i)

        metadata = (
            sampling_semantics.privacy_metadata
            if sampling_semantics is not None
            else {}
        )

        sampling_mode = (
            sampling_semantics.sampling_mode
            if sampling_semantics is not None
            else None
        )
        state = mechanism_state if isinstance(mechanism_state, dict) else {}

        if sampling_mode != "cyclic_poisson":
            return self._get_epsilon_fixed_batch(
                delta=delta,
                noise_multiplier=float(noise_multiplier),
                sample_rate=float(sample_rate),
                total_steps=int(total_steps),
                state=state,
                metadata=metadata,
                kwargs=kwargs,
            )

        bands = metadata.get("bands", None)
        if bands is None:
            raise ValueError(
                "cyclic_poisson sampling requires privacy_metadata['bands']"
            )
        bands = int(bands)

        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        # `bsr_sensitivity_scale` corresponds to the mechanism sensitivity normalization.
        sensitivity_scale = kwargs.get(
            "bsr_sensitivity_scale",
            metadata.get(
                "bsr_sensitivity_scale",
                state.get("bsr_sensitivity_scale", 1.0),
            ),
        )

        sensitivity_scale = float(sensitivity_scale)
        if not math.isfinite(sensitivity_scale) or sensitivity_scale <= 0.0:
            raise ValueError("bandmf_sensitivity_scale must be > 0")

        contract = self._resolve_cyclic_contract(
            sample_rate=float(sample_rate),
            steps=int(total_steps),
            bands=bands,
        )
        self.last_contract = {
            "mechanism": "bandmf",
            "accounting_mode": "bandmf_accountant",
            "accountant_backend": "prv",
            "sampling_mode": sampling_mode,
            "sample_rate": float(sample_rate),
            "bands": int(contract["bands"]),
            "steps": int(contract["steps"]),
            "q": float(contract["q"]),
            "cycles": int(contract["cycles"]),
            "sensitivity_scale": float(sensitivity_scale),
        }
        reduced_contract = resolve_bsr_cyclic_gaussian_contract(
            noise_multiplier=float(noise_multiplier) / float(sensitivity_scale),
            steps=int(total_steps),
            sample_rate=float(sample_rate),
            bands=int(contract["bands"]),
        )
        self.last_contract["effective_noise_multiplier"] = float(
            reduced_contract["effective_noise_multiplier"]
        )

        return float(
            bsr_cyclic_poisson_epsilon_upper_bound(
                noise_multiplier=float(noise_multiplier) / sensitivity_scale,
                target_delta=float(delta),
                steps=int(total_steps),
                sample_rate=float(sample_rate),
                bands=int(contract["bands"]),
            )
        )

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "bandmf"
