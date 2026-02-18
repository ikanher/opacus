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

from opacus.accountants.analysis.bnb import (
    estimate_b_min_sep_epsilon_monte_carlo,
)
from opacus.accountants.analysis.bnb_preflight import validate_bnb_runtime_consistency
from opacus.bnb_defaults import resolve_bnb_calibration_kwargs

from .accountant import IAccountant


class BNBAccountant(IAccountant):
    """
    Accountant adapter shell for bnb mechanisms.

    Full Monte Carlo accounting implementation is introduced in later milestones.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_builtin_b_min_sep_consistency(
        *,
        mechanism_state,
        sampling_semantics,
        c_matrix,
        bands: int,
        c_matrix_contract,
    ) -> None:
        validate_bnb_runtime_consistency(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            c_matrix=c_matrix,
            bands=int(bands),
            c_matrix_contract=c_matrix_contract,
            coeffs_error_prefix="bnb consistency check",
        )

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
        epsilon_fn=None,
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
                    "bnb accountant currently expects constant "
                    "noise_multiplier and sample_rate across steps"
                )
            total_steps += int(steps_i)

        if epsilon_fn is None:
            state = mechanism_state if isinstance(mechanism_state, dict) else {}
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

            c_matrix = kwargs.get("bnb_c_matrix", state.get("c_matrix"))
            bands = kwargs.get("bnb_bands", metadata.get("bands", state.get("bands")))
            c_matrix_contract = kwargs.get(
                "bnb_c_matrix_contract",
                state.get("c_matrix_contract"),
            )
            calibration_cfg = resolve_bnb_calibration_kwargs(
                profile="opacus_strict",
                overrides=kwargs,
            )
            num_samples = int(calibration_cfg["bnb_num_samples"])
            seed = int(calibration_cfg["bnb_seed"])
            reduce_dimensionality = bool(calibration_cfg["bnb_reduce_dimensionality"])
            tolerance = float(calibration_cfg["bnb_tolerance"])
            max_iterations = int(calibration_cfg["bnb_max_iterations"])
            if (
                sampling_mode == "b_min_sep"
                and c_matrix is not None
                and bands is not None
                and c_matrix_contract is not None
            ):
                self._validate_builtin_b_min_sep_consistency(
                    mechanism_state=state,
                    sampling_semantics=sampling_semantics,
                    c_matrix=c_matrix,
                    bands=int(bands),
                    c_matrix_contract=c_matrix_contract,
                )
                # Conservative built-in composition:
                # 1) convert (steps, sample_rate) to an expected-participation count;
                # 2) split delta across those effective participations;
                # 3) estimate one-step epsilon by MC and compose linearly.
                effective_steps = max(
                    1,
                    int(
                        math.ceil(
                            float(total_steps)
                            * min(max(float(sample_rate), 0.0), 1.0)
                        )
                    ),
                )
                horizon = c_matrix_contract.get("horizon")
                if horizon is not None and int(effective_steps) > int(horizon):
                    raise ValueError(
                        "bnb consistency check failed: c_matrix_contract['horizon'] "
                        f"({int(horizon)}) must be >= effective_steps ({int(effective_steps)})"
                    )
                delta_per_step = float(delta) / float(effective_steps)
                if delta_per_step <= 0.0 or delta_per_step >= 1.0:
                    raise ValueError(
                        "target delta is incompatible with built-in bnb composition "
                        f"(delta={delta}, effective_steps={effective_steps})"
                    )

                epsilon_per_step = float(
                    estimate_b_min_sep_epsilon_monte_carlo(
                        c_matrix=c_matrix,
                        bands=int(bands),
                        noise_multiplier=float(noise_multiplier),
                        target_delta=delta_per_step,
                        num_samples=num_samples,
                        seed=seed,
                        reduce_dimensionality=reduce_dimensionality,
                        tolerance=tolerance,
                        max_iterations=max_iterations,
                    )
                )
                return float(
                    float(effective_steps) * epsilon_per_step
                )

            raise ValueError(
                "bnb accountant requires epsilon_fn, or built-in b_min_sep "
                "inputs (`c_matrix`, `bands`, `c_matrix_contract`, and "
                "sampling_mode='b_min_sep')"
            )

        return float(
            epsilon_fn(
                noise_multiplier=float(noise_multiplier),
                target_delta=float(delta),
                sample_rate=float(sample_rate),
                steps=int(total_steps),
                mechanism=self.mechanism(),
                mechanism_state=mechanism_state,
                sampling_semantics=sampling_semantics,
                **kwargs,
            )
        )

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "bnb"
