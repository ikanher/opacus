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
    estimate_balls_in_bins_epsilon_monte_carlo,
    validate_bnb_c_matrix_contract,
)

from .accountant import IAccountant


class BNBAccountant(IAccountant):
    """
    Accountant adapter shell for bnb mechanisms.

    Full Monte Carlo accounting implementation is introduced in later milestones.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_builtin_balls_in_bins_consistency(
        *,
        mechanism_state,
        sampling_semantics,
        c_matrix,
        bands: int,
        c_matrix_contract,
    ) -> None:
        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        coeffs = state.get("coeffs")
        if coeffs is None or not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
            raise ValueError(
                "bnb consistency check requires non-empty mechanism_state['coeffs']"
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

        if not hasattr(c_matrix, "ndim") or c_matrix.ndim != 2:
            raise ValueError("bnb consistency check requires c_matrix with shape [d, m]")
        if int(c_matrix.shape[0]) < int(bands):
            raise ValueError(
                "bnb consistency check failed: c_matrix must have at least `bands` rows; "
                f"got rows={int(c_matrix.shape[0])}, bands={int(bands)}"
            )
        validate_bnb_c_matrix_contract(
            c_matrix=c_matrix,
            coeffs=coeffs,
            bands=int(bands),
            c_matrix_contract=c_matrix_contract,
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
            num_samples = int(kwargs.get("bnb_num_samples", 100_000))
            seed = int(kwargs.get("bnb_seed", 0))
            reduce_dimensionality = bool(kwargs.get("bnb_reduce_dimensionality", False))
            tolerance = float(kwargs.get("bnb_tolerance", 1e-4))
            max_iterations = int(kwargs.get("bnb_max_iterations", 200))
            if (
                sampling_mode == "balls_in_bins"
                and c_matrix is not None
                and bands is not None
                and c_matrix_contract is not None
            ):
                self._validate_builtin_balls_in_bins_consistency(
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
                    estimate_balls_in_bins_epsilon_monte_carlo(
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
                "bnb accountant requires epsilon_fn, or built-in balls_in_bins "
                "inputs (`c_matrix`, `bands`, `c_matrix_contract`, and "
                "sampling_mode='balls_in_bins')"
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
