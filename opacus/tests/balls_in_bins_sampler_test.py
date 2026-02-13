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

import unittest

import numpy as np
import torch
from opacus.utils.uniform_sampler import (
    BMinSepSampler,
    BallsInBinsSampler,
    WarmStartBMinSepSampler,
)


class WarmStartAndBallsInBinsSamplerTest(unittest.TestCase):
    def _generator(self, seed: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    def test_warm_start_reduces_initial_transient_bias(self) -> None:
        num_samples = 100
        sample_rate = 0.5
        min_sep = 4
        steps = 20

        cold = BMinSepSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=min_sep,
            steps=steps,
            generator=self._generator(7),
        )
        warm = WarmStartBMinSepSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=min_sep,
            steps=steps,
            generator=self._generator(7),
        )

        expected_rate = sample_rate / (1 + sample_rate * (min_sep - 1))
        expected_mean = num_samples * expected_rate

        cold_initial = float(np.mean([len(b) for b in list(cold)[:5]]))
        warm_initial = float(np.mean([len(b) for b in list(warm)[:5]]))

        self.assertLess(abs(warm_initial - expected_mean), abs(cold_initial - expected_mean))

    def test_warm_start_stationary_batch_size_stability(self) -> None:
        num_samples = 120
        sample_rate = 0.4
        min_sep = 3
        steps = 1500

        warm = WarmStartBMinSepSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=min_sep,
            steps=steps,
            generator=self._generator(9),
        )

        expected_rate = sample_rate / (1 + sample_rate * (min_sep - 1))
        expected_mean = num_samples * expected_rate
        empirical_mean = float(np.mean([len(batch) for batch in warm]))
        self.assertAlmostEqual(empirical_mean, expected_mean, delta=2.0)

    def test_balls_in_bins_wrapper_equivalent_to_warm_start_bminsep(self) -> None:
        num_samples = 96
        batch_size = 12
        bands = 4
        steps = 80
        seed = 11

        wrapped = BallsInBinsSampler(
            num_samples=num_samples,
            batch_size=batch_size,
            bands=bands,
            steps=steps,
            generator=self._generator(seed),
        )
        direct = WarmStartBMinSepSampler(
            num_samples=num_samples,
            sample_rate=wrapped.sample_rate,
            min_separation=bands,
            steps=steps,
            generator=self._generator(seed),
        )
        self.assertEqual(list(wrapped), list(direct))

    def test_warm_start_b1_matches_cold_sampler(self) -> None:
        num_samples = 60
        sample_rate = 0.2
        steps = 100
        seed = 15

        cold = BMinSepSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=1,
            steps=steps,
            generator=self._generator(seed),
        )
        warm = WarmStartBMinSepSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=1,
            steps=steps,
            generator=self._generator(seed),
        )
        self.assertEqual(list(cold), list(warm))

