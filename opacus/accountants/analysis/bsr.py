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


def calibrate_bsr_z_std(
    *,
    noise_multiplier_ref: float,
    max_grad_norm: float,
    denominator: float,
) -> float:
    """
    Fixed-batch BSR runtime mapping (proof-aligned):
    z_std = noise_multiplier_ref * max_grad_norm / denominator
    """
    if noise_multiplier_ref <= 0.0:
        raise ValueError("noise_multiplier_ref must be > 0")

    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be > 0")

    if denominator <= 0.0:
        raise ValueError("denominator must be > 0")

    return float(noise_multiplier_ref) * float(max_grad_norm) / float(denominator)


def bsr_fixed_batch_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    steps: int,
    denominator: float,
) -> float:
    """
    Conservative fixed-batch composed epsilon upper bound:
    eps = steps * denominator / noise_multiplier * sqrt(2 * log(1.25 / delta))
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if target_delta <= 0.0 or target_delta > 1.0:
        raise ValueError("target_delta must be in (0, 1]")

    if steps < 0:
        raise ValueError("steps must be >= 0")

    if denominator <= 0.0:
        raise ValueError("denominator must be > 0")

    gaussian_term = math.sqrt(2.0 * math.log(1.25 / float(target_delta)))
    return (
        float(steps)
        * float(denominator)
        * gaussian_term
        / float(noise_multiplier)
    )

