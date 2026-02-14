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
from opacus.utils.uniform_sampler import BallsInBinsSampler, DistributedBallsInBinsSampler
from unittest.mock import patch


class BallsInBinsSamplerTest(unittest.TestCase):
    def _init_sampler(
        self,
        *,
        seed: int,
        num_samples: int = 120,
        bins: int = 12,
        steps: int = 24,
    ) -> BallsInBinsSampler:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return BallsInBinsSampler(
            num_samples=num_samples,
            bins=bins,
            steps=steps,
            generator=generator,
        )

    def test_length(self) -> None:
        sampler = self._init_sampler(seed=1, steps=17)
        self.assertEqual(len(sampler), 17)

    def test_same_seed(self) -> None:
        sampler1 = self._init_sampler(seed=7)
        sampler2 = self._init_sampler(seed=7)
        self.assertEqual(list(sampler1), list(sampler2))

    def test_different_seed(self) -> None:
        sampler1 = self._init_sampler(seed=7)
        sampler2 = self._init_sampler(seed=8)
        self.assertNotEqual(list(sampler1), list(sampler2))

    def test_each_example_participates_once_per_cycle(self) -> None:
        num_samples = 200
        bins = 10
        sampler = self._init_sampler(
            seed=21,
            num_samples=num_samples,
            bins=bins,
            steps=2 * bins,
        )
        batches = list(sampler)

        first_cycle = batches[:bins]
        second_cycle = batches[bins:]

        counts_first = np.zeros(num_samples, dtype=np.int64)
        counts_second = np.zeros(num_samples, dtype=np.int64)

        for batch in first_cycle:
            counts_first[np.array(batch, dtype=np.int64)] += 1
        for batch in second_cycle:
            counts_second[np.array(batch, dtype=np.int64)] += 1

        self.assertTrue(np.all(counts_first == 1))
        self.assertTrue(np.all(counts_second == 1))

    def test_empirical_batch_size_matches_n_over_bins(self) -> None:
        num_samples = 1000
        bins = 20
        sampler = self._init_sampler(
            seed=33,
            num_samples=num_samples,
            bins=bins,
            steps=5 * bins,
        )
        batch_sizes = [len(batch) for batch in sampler]
        empirical_mean = float(np.mean(batch_sizes))
        expected_mean = num_samples / bins
        self.assertAlmostEqual(empirical_mean, expected_mean, delta=3.0)


class DistributedBallsInBinsSamplerTest(unittest.TestCase):
    def _init_sampler(
        self,
        *,
        seed: int,
        total_size: int = 120,
        bins: int = 12,
        steps: int = 24,
        world_size: int = 2,
        rank: int = 0,
    ) -> DistributedBallsInBinsSampler:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DistributedBallsInBinsSampler(
            total_size=total_size,
            bins=bins,
            steps=steps,
            generator=generator,
            shuffle=True,
            shuffle_seed=17,
        )

    def test_each_local_example_once_per_cycle_even_with_set_epoch(self) -> None:
        bins = 8
        total_size = 64
        with patch("torch.distributed.get_world_size", return_value=2), patch(
            "torch.distributed.get_rank", return_value=0
        ):
            sampler = self._init_sampler(
                seed=123,
                total_size=total_size,
                bins=bins,
                steps=2 * bins,
                world_size=2,
                rank=0,
            )

            # First pass
            batches_epoch0 = list(sampler)

            # Change epoch; balls-in-bins assignment should remain example-stable
            sampler.set_epoch(5)
            batches_epoch5 = list(sampler)

        local_indices = np.arange(total_size)[0:total_size:2]
        local_n = len(local_indices)

        def counts_over_cycle(batches):
            counts = np.zeros(total_size, dtype=np.int64)
            for batch in batches[:bins]:
                counts[np.array(batch, dtype=np.int64)] += 1
            return counts

        c0 = counts_over_cycle(batches_epoch0)
        c5 = counts_over_cycle(batches_epoch5)

        self.assertTrue(np.all(c0[local_indices] == 1))
        self.assertTrue(np.all(c5[local_indices] == 1))

        # Non-local indices never appear on this rank.
        mask_non_local = np.ones(total_size, dtype=bool)
        mask_non_local[local_indices] = False
        self.assertTrue(np.all(c0[mask_non_local] == 0))
        self.assertTrue(np.all(c5[mask_non_local] == 0))


if __name__ == "__main__":
    unittest.main()
