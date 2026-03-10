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
from typing import Iterable


def compute_bandmf_mf_sensitivity_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compute fixed-batch BandMF sensitivity from Toeplitz coefficients.

    For lower-triangular Toeplitz strategies with nonnegative non-increasing
    coefficients, the worst-case participation pattern under a ``(k, b)``
    contract uses up to ``k`` participations separated by exactly ``b`` steps.
    This lets us evaluate fixed-batch matrix-factorization sensitivity in
    linear time using a difference-of-cumsums formulation.

    Math:
    ``S_{k,b}(C;T)^2 = sum_i v_i^2``, where ``v`` is the row-wise sum induced by
    the worst-case separated participation pattern.

    Source: BandMF (Choquette-Choo et al., 2023), fixed-participation Toeplitz
    sensitivity for nonnegative non-increasing coefficients.
    """
    coeff_list = [float(c) for c in coeffs]
    if not coeff_list:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    if any(c < 0.0 for c in coeff_list):
        raise ValueError("coeffs must be nonnegative")

    if steps < 1:
        raise ValueError("steps must be >= 1")

    if max_participations < 1:
        raise ValueError("max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")

    for prev, cur in zip(coeff_list, coeff_list[1:]):
        if cur > prev + 1e-12:
            raise ValueError("coeffs must be non-increasing")

    k_eff = min(int(max_participations), (int(steps) - 1) // int(min_separation) + 1)

    padding = (int(min_separation) - int(steps)) % int(min_separation)
    padded = coeff_list + [0.0] * max(0, int(steps) - len(coeff_list) + padding)

    vector: list[float] = [0.0] * len(padded)
    for block_start in range(0, len(padded), int(min_separation)):
        running = 0.0
        block_end = min(block_start + int(min_separation), len(padded))

        for idx in range(block_start, block_end):
            running += padded[idx]
            vector[idx] = running

    stride = int(min_separation) * int(k_eff)
    for idx in range(stride, len(vector)):
        vector[idx] -= vector[idx - stride]

    total_sq = sum(v * v for v in vector[: int(steps)])

    return math.sqrt(total_sq)

