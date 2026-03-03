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

import math

from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
)

from .accountant import IAccountant


class BandMFAccountant(IAccountant):
    """
    Accountant adapter for amplified cyclic BandMF mechanisms.

    This path is intentionally separate from fixed-batch BSR accounting.
    """

    def __init__(self):
        super().__init__()
        self.last_contract = None

    @staticmethod
    def _resolve_cyclic_contract(*, sample_rate: float, steps: int, bands: int) -> dict:
        if bands <= 0:
            raise ValueError("bands must be > 0")
        if steps < bands:
            raise ValueError(f"steps must be >= bands; got steps={steps}, bands={bands}")

        q = float(sample_rate) * float(bands)
        if not math.isfinite(q) or q <= 0.0 or q > 1.0:
            raise ValueError(
                f"derived q = bands * sample_rate must be in (0, 1]; got {q}"
            )

        cycles = int(math.ceil(float(steps) / float(bands)))
        if cycles <= 0:
            raise ValueError(f"derived cycles must be > 0; got {cycles}")

        return {"q": q, "cycles": cycles, "bands": int(bands), "steps": int(steps)}

    def step(self, *, noise_multiplier: float, sample_rate: float):
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
 
        if sampling_mode != "cyclic_poisson":
            raise ValueError(
                "bandmf accountant requires cyclic_poisson sampling semantics"
            )

        bands = metadata.get("bands", None)
        if bands is None:
            raise ValueError(
                "cyclic_poisson sampling requires privacy_metadata['bands']"
            )
        bands = int(bands)

        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        sensitivity_scale = kwargs.get(
            "sensitivity_scale",
            metadata.get("sensitivity_scale", state.get("sensitivity_scale", 1.0)),
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
            "sampling_mode": sampling_mode,
            "sample_rate": float(sample_rate),
            "bands": int(contract["bands"]),
            "steps": int(contract["steps"]),
            "q": float(contract["q"]),
            "cycles": int(contract["cycles"]),
            "sensitivity_scale": float(sensitivity_scale),
        }

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
