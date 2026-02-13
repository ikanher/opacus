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
from opacus.utils.uniform_sampler import BMinSepSampler


class BMinSepSamplerTest(unittest.TestCase):
    def _init_sampler(
        self,
        *,
        seed: int,
        num_samples: int = 50,
        sample_rate: float = 0.2,
        min_separation: int = 3,
        steps: int = 200,
    ) -> BMinSepSampler:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return BMinSepSampler(
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=min_separation,
            steps=steps,
            generator=generator,
        )

    def test_length(self) -> None:
        sampler = self._init_sampler(seed=7, steps=123)
        self.assertEqual(len(sampler), 123)

    def test_min_separation_gap_invariant(self) -> None:
        min_sep = 4
        sampler = self._init_sampler(
            seed=7,
            num_samples=40,
            sample_rate=0.5,
            min_separation=min_sep,
            steps=250,
        )
        participations = {i: [] for i in range(40)}
        for step, batch in enumerate(sampler):
            for idx in batch:
                participations[idx].append(step)

        for history in participations.values():
            for prev, nxt in zip(history, history[1:]):
                self.assertGreaterEqual(nxt - prev, min_sep)

    def test_same_seed(self) -> None:
        sampler1 = self._init_sampler(seed=7)
        sampler2 = self._init_sampler(seed=7)
        self.assertEqual(list(sampler1), list(sampler2))

    def test_different_seed(self) -> None:
        sampler1 = self._init_sampler(seed=7)
        sampler2 = self._init_sampler(seed=8)
        self.assertNotEqual(list(sampler1), list(sampler2))

    def test_expected_batch_size_sanity(self) -> None:
        num_samples = 80
        sample_rate = 0.3
        min_sep = 3
        steps = 800
        sampler = self._init_sampler(
            seed=9,
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=min_sep,
            steps=steps,
        )
        batch_sizes = [len(batch) for batch in sampler]
        empirical_mean = float(np.mean(batch_sizes))
        expected_rate = sample_rate / (1 + sample_rate * (min_sep - 1))
        expected_mean = num_samples * expected_rate
        self.assertAlmostEqual(empirical_mean, expected_mean, delta=2.0)

    def test_b_equals_one_matches_bernoulli_rate(self) -> None:
        num_samples = 120
        sample_rate = 0.15
        sampler = self._init_sampler(
            seed=11,
            num_samples=num_samples,
            sample_rate=sample_rate,
            min_separation=1,
            steps=600,
        )
        batch_sizes = [len(batch) for batch in sampler]
        self.assertAlmostEqual(float(np.mean(batch_sizes)), num_samples * sample_rate, delta=3.0)

    def test_exhausted_cooldown_behavior_with_prob_one(self) -> None:
        sampler = self._init_sampler(
            seed=13,
            num_samples=3,
            sample_rate=1.0,
            min_separation=5,
            steps=7,
        )
        batches = list(sampler)
        self.assertEqual(batches[0], [0, 1, 2])
        self.assertEqual(batches[1], [])
        self.assertEqual(batches[2], [])
        self.assertEqual(batches[3], [])
        self.assertEqual(batches[4], [])
        self.assertEqual(batches[5], [0, 1, 2])

