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
from unittest import mock

import math
import numpy as np
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
        # Use q=1 so every partition member appears whenever its band is active.
        sampler = CyclicPoissonSampler(
            num_samples=20,
            batch_size=5,  # partition_size = 20 // 4 = 5 => q=1
            bands=4,
            steps=8,
            generator=torch.Generator().manual_seed(7),
            shuffle=True,
            shuffle_seed=123,
        )

        residue_by_index = {}
        for step, batch in enumerate(sampler):
            residue = step % 4
            for idx in batch:
                previous = residue_by_index.setdefault(idx, residue)
                self.assertEqual(previous, residue)

        self.assertEqual(len(residue_by_index), 20)
    
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

    def test_set_epoch_changes_partition(self) -> None:
        sampler = CyclicPoissonSampler(
            num_samples=20,
            batch_size=5,  # q=1
            bands=4,
            steps=4,
            generator=torch.Generator().manual_seed(9),
            shuffle=True,
            shuffle_seed=42,
        )
        epoch0 = list(sampler)
        sampler.set_epoch(1)
        epoch1 = list(sampler)
        self.assertNotEqual(epoch0, epoch1)
