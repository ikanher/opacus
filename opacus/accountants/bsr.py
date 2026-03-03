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
    compute_bsr_mf_sensitivity_from_coeffs,
    bsr_fixed_batch_epsilon_upper_bound,
)

from .accountant import IAccountant


class BSRAccountant(IAccountant):
    """
    Accountant adapter for bsr mechanisms.
    """

    def __init__(self):
        super().__init__()

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

        # Current Opacus MF v1 uses a constant sample_rate and noise multiplier.
        noise_multiplier, sample_rate, _ = self.history[0]
        total_steps = 0

        for nm_i, sr_i, steps_i in self.history:
            if nm_i != noise_multiplier or sr_i != sample_rate:
                raise ValueError(
                    "bsr accountant currently expects constant "
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
            else "torch_sampler"
        )
        if sampling_mode == "cyclic_poisson":
            raise ValueError(
                "bsr accountant does not support cyclic_poisson semantics in this phase; "
                "use mechanism/accountant family 'bandmf' instead"
            )

        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        mf_sensitivity = kwargs.get(
            "bsr_mf_sensitivity",
            metadata.get("mf_sensitivity", state.get("mf_sensitivity")),
        )
        explicit_mf_sensitivity_override = "bsr_mf_sensitivity" in kwargs
        coeffs = state.get("coeffs")
        max_participations = kwargs.get(
            "bsr_max_participations",
            metadata.get(
                "max_participations",
                state.get("max_participations"),
            ),
        )
        min_separation = kwargs.get(
            "bsr_min_separation",
            metadata.get(
                "min_separation",
                state.get("min_separation", metadata.get("bands")),
            ),
        )
        sensitivity_steps = kwargs.get(
            "bsr_iterations_number",
            metadata.get("iterations_number", state.get("iterations_number")),
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

        if mf_sensitivity is None:
            if coeffs is None:
                raise ValueError(
                    "fixed-batch bsr accounting requires MF sensitivity or "
                    "enough data to derive it: "
                    "`coeffs`, `max_participations`, `min_separation`"
                )

            mf_sensitivity = compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=sensitivity_steps,
                max_participations=int(max_participations),
                min_separation=int(min_separation),
            )
        else:
            mf_sensitivity = float(mf_sensitivity)
            if not math.isfinite(mf_sensitivity) or mf_sensitivity <= 0.0:
                raise ValueError("bsr_mf_sensitivity must be finite and > 0")
            if (
                explicit_mf_sensitivity_override
                and
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
                    mf_sensitivity, derived, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise ValueError(
                        "provided bsr_mf_sensitivity is inconsistent with "
                        "coeffs/max_participations/min_separation for the resolved "
                        "bsr_iterations_number"
                    )

        return float(
            bsr_fixed_batch_epsilon_upper_bound(
                noise_multiplier=float(noise_multiplier),
                target_delta=float(delta),
                mf_sensitivity=float(mf_sensitivity),
            )
        )

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "bsr"
