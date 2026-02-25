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
from opacus.utils.uniform_sampler import DistributedCyclicPoissonSampler


class DistributedCyclicPoissonSamplerTest(unittest.TestCase):
    def _init_samplers(self, *, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        samplers = []
        torch.distributed.get_world_size = lambda: self.world_size
        for rank in range(self.world_size):
            torch.distributed.get_rank = lambda: rank
            samplers.append(
                DistributedCyclicPoissonSampler(
                    total_size=self.total_size,
                    batch_size=self.local_batch_size,
                    bands=self.bands,
                    steps=self.steps,
                    generator=generator,
                )
            )
        return samplers

    def setUp(self) -> None:
        self.world_size = 2
        self.total_size = 24
        # partition_size = 24 // 3 = 8, local shard size = 4, choose q=1.
        self.local_batch_size = 4
        self.bands = 3
        self.steps = 6
        self.samplers = self._init_samplers(seed=7)

    def test_length(self) -> None:
        for sampler in self.samplers:
            self.assertEqual(len(sampler), self.steps)

    def test_local_batches_follow_cyclic_band_membership(self) -> None:
        # With q=1 every local shard member appears whenever its band is active.
        for sampler in self.samplers:
            residue_by_index = {}
            for step, batch in enumerate(sampler):
                residue = step % self.bands
                for idx in batch:
                    previous = residue_by_index.setdefault(idx, residue)
                    self.assertEqual(previous, residue)

    def test_per_step_cross_rank_indices_are_disjoint(self) -> None:
        rank_batches = [list(sampler) for sampler in self.samplers]
        for step in range(self.steps):
            b0 = set(rank_batches[0][step])
            b1 = set(rank_batches[1][step])
            self.assertEqual(len(b0 & b1), 0)
