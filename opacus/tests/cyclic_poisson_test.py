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

import torch
from opacus.utils.uniform_sampler import CyclicPoissonSampler


class CyclicPoissonSamplerTest(unittest.TestCase):
    def _init_sampler(self, *, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return CyclicPoissonSampler(
            num_samples=20,
            batch_size=2,
            bands=4,
            steps=8,
            generator=generator,
        )

    def test_length(self) -> None:
        sampler = self._init_sampler(seed=7)
        self.assertEqual(len(sampler), 8)

    def test_cyclic_band_membership(self) -> None:
        sampler = self._init_sampler(seed=7)

        band_size = 20 // 4
        partitions = []
        for j in range(4):
            start = j * band_size
            partitions.append(set(range(start, start + band_size)))

        for step, batch in enumerate(sampler):
            active_band = step % 4
            self.assertTrue(set(batch).issubset(partitions[active_band]))
    
    def test_batch_size_is_not_forced_fixed(self) -> None:
        sampler = CyclicPoissonSampler(
            num_samples=40,
            batch_size=5,
            bands=4,
            steps=20,
            generator=torch.Generator().manual_seed(11),
        )
        sizes = [len(b) for b in sampler]
        self.assertGreater(len(set(sizes)), 1)

    def test_same_seed(self) -> None:
        sampler1 = self._init_sampler(seed=7)
        sampler2 = self._init_sampler(seed=7)
        self.assertEqual(list(sampler1), list(sampler2))

    def test_different_seed(self) -> None:
        sampler1 = self._init_sampler(seed=7)
        sampler2 = self._init_sampler(seed=8)
        self.assertNotEqual(list(sampler1), list(sampler2))
