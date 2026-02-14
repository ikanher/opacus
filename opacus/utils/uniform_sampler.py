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

from typing import List

import torch
from torch.utils.data import Sampler


class UniformWithReplacementSampler(Sampler[List[int]]):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    The sampler generates ``steps`` number of batches, that defaults to 1/``sample_rate``.
    """

    def __init__(
        self, *, num_samples: int, sample_rate: float, generator=None, steps=None
    ):
        r"""
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling.
            steps: Number of steps (iterations of the Sampler)
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator
        self.steps = steps

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

        if steps is not None:
            self.steps = steps
        else:
            self.steps = int(1 / self.sample_rate)

    def __len__(self):
        return self.steps

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices

            num_batches -= 1


class DistributedUniformWithReplacementSampler(Sampler):
    """
    Distributed batch sampler.

    Each batch is sampled as follows:
        1. Shuffle the dataset (enabled by default)
        2. Split the dataset among the replicas into chunks of equal size
           (plus or minus one sample)
        3. Each replica selects each sample of its chunk independently
           with probability `sample_rate`
        4. Each replica outputs the selected samples, which form a local batch

    The sum of the lengths of the local batches follows a Poisson distribution.
    In particular, the expected length of each local batch is:
    `sample_rate * total_size / num_replicas`
    """

    def __init__(
        self,
        *,
        total_size: int,
        sample_rate: float,
        shuffle: bool = True,
        shuffle_seed: int = 0,
        steps: int = None,
        generator=None,
    ):
        """

        Args:
            total_size: total number of samples to sample from
            sample_rate: number of samples to draw.
            shuffle: Flag indicating whether apply shuffle when dividing elements
                between workers
            shuffle_seed: Random seed used to shuffle when dividing elements across workers
            generator: torch.Generator() object used as a source of randomness
                when selecting items for the next round on a given worker
        """
        self.total_size = total_size
        self.sample_rate = sample_rate
        self.generator = generator
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.epoch = 0
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.steps = steps

        if self.total_size <= 0:
            raise ValueError(
                "total_size should be a positive integer "
                "value, but got total_size={}".format(self.total_size)
            )

        # Size of the local dataset specific to the current replica
        self.num_samples = self.total_size // self.num_replicas
        if self.rank < self.total_size % self.num_replicas:
            # The first replicas get an extra datapoint if necessary (balanced)
            self.num_samples += 1

        # Number of batches: same as non-distributed Poisson sampling, but each batch is smaller
        if steps is not None:
            self.num_batches = steps
        else:
            self.num_batches = int(1 / self.sample_rate)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.shuffle_seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g)  # type: ignore
        else:
            indices = torch.arange(self.total_size)  # type: ignore

        # Subset of the dataset assigned to this replica
        # NOTE: the first replicas might have 1 more sample.
        # (Different from the regular distributed loader that pads with more samples)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Now, select a batch with Poisson subsampling
        for _ in range(self.num_batches):
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            selected_examples = mask.nonzero(as_tuple=False).reshape(-1)
            if len(selected_examples) > 0:
                yield indices[selected_examples]

    def __len__(self) -> int:
        """
        Expected number of batches.
        """
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class BMinSepSampler(Sampler[List[int]]):
    r"""
    Non-distributed b-min-separation sampler.

    Each example tracks a cooldown counter:
    - counter == 0: example is eligible and participates with probability `sample_rate`;
    - counter > 0: example is in cooldown and cannot participate.

    When an eligible example participates, its cooldown is reset to
    `min_separation - 1`. Cooldown decrements by one at each step.

    "Privacy Amplification for BandMF via b-Min-Sep Subsampling" (Dong et al., 2026)
    """

    def __init__(
        self,
        *,
        num_samples: int,
        sample_rate: float,
        min_separation: int,
        generator=None,
        steps: int = None,
    ):
        self.num_samples = int(num_samples)
        self.sample_rate = float(sample_rate)
        self.min_separation = int(min_separation)
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(f"num_samples should be positive, got {self.num_samples}")

        if self.sample_rate <= 0.0 or self.sample_rate > 1.0:
            raise ValueError(f"sample_rate should be in (0, 1], got {self.sample_rate}")

        if self.min_separation <= 0:
            raise ValueError(
                f"min_separation should be positive, got {self.min_separation}"
            )

        self.steps = int(steps) if steps is not None else int(1 / self.sample_rate)
        if self.steps <= 0:
            raise ValueError(f"steps should be positive, got {self.steps}")

    def __len__(self):
        return self.steps

    def __iter__(self):
        cooldown = torch.zeros(self.num_samples, dtype=torch.int64)

        for _ in range(self.steps):
            eligible = cooldown == 0

            draws = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )

            selected_mask = eligible & draws
            indices = selected_mask.nonzero(as_tuple=False).reshape(-1).tolist()

            yield indices

            cooldown = torch.where(cooldown > 0, cooldown - 1, cooldown)
            if self.min_separation > 1:
                cooldown[selected_mask] = self.min_separation - 1


class DistributedBMinSepSampler(Sampler[List[int]]):
    r"""
    Distributed b-min-separation sampler.

    The global index set is sharded across ranks (optionally after deterministic
    shuffle per epoch). Each rank then runs local b-min-separation sampling on
    its shard and yields global indices selected on that rank.
    """

    def __init__(
        self,
        *,
        total_size: int,
        sample_rate: float,
        min_separation: int,
        shuffle: bool = True,
        shuffle_seed: int = 0,
        generator=None,
        steps: int = None,
    ):
        self.total_size = int(total_size)
        self.sample_rate = float(sample_rate)
        self.min_separation = int(min_separation)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.generator = generator
        self.epoch = 0
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        if self.total_size <= 0:
            raise ValueError(f"total_size should be positive, got {self.total_size}")

        if self.sample_rate <= 0.0 or self.sample_rate > 1.0:
            raise ValueError(f"sample_rate should be in (0, 1], got {self.sample_rate}")

        if self.min_separation <= 0:
            raise ValueError(
                f"min_separation should be positive, got {self.min_separation}"
            )

        if self.num_replicas <= 0:
            raise ValueError(
                f"num_replicas should be positive, got {self.num_replicas}"
            )

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"invalid rank {self.rank} for world size {self.num_replicas}"
            )

        # Size of the local shard for this rank.
        self.num_samples = self.total_size // self.num_replicas
        if self.rank < self.total_size % self.num_replicas:
            self.num_samples += 1

        self.steps = int(steps) if steps is not None else int(1 / self.sample_rate)
        if self.steps <= 0:
            raise ValueError(f"steps should be positive, got {self.steps}")

    def __len__(self):
        return self.steps

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.shuffle_seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g)
        else:
            indices = torch.arange(self.total_size)

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        cooldown = torch.zeros(self.num_samples, dtype=torch.int64)
        for _ in range(self.steps):
            eligible = cooldown == 0

            draws = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )

            selected_mask = eligible & draws
            selected_local = selected_mask.nonzero(as_tuple=False).reshape(-1)

            yield indices[selected_local].tolist()

            cooldown = torch.where(cooldown > 0, cooldown - 1, cooldown)
            if self.min_separation > 1:
                cooldown[selected_mask] = self.min_separation - 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class CyclicPoissonSampler(Sampler[List[int]]):
    r"""
    Cyclic partitioned fixed-size sampler.

    Implements the Banded-MF cyclic sampling pattern:
    - partition dataset indices into ``bands`` disjoint subsets of equal size
      (extra tail indices are discarded);
    - at step ``t`` sample a uniform-size ``batch_size`` subset from partition
      ``t % bands``.

    Reference:
    "(Amplified) Banded Matrix Factorization: A unified approach to private
    training" (Choquette-Choo et al., 2023)
    """

    def __init__(
        self,
        *,
        num_samples: int,
        batch_size: int,
        bands: int,
        generator=None,
        steps: int = None,
    ):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.bands = int(bands)
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be positive, got {self.num_samples}"
            )

        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be positive, got {self.batch_size}"
            )

        if self.bands <= 0:
            raise ValueError(f"bands should be positive, got {self.bands}")

        self.partition_size = self.num_samples // self.bands
        if self.partition_size <= 0:
            raise ValueError(
                "bands is too large for dataset size: partition_size is zero"
            )

        if self.batch_size > self.partition_size:
            raise ValueError(
                "batch_size must be <= partition size in cyclic_poisson sampler"
            )

        self.usable_size = self.partition_size * self.bands
        self.steps = int(steps) if steps is not None else int(self.num_samples / self.batch_size)

        if self.steps <= 0:
            raise ValueError(f"steps should be positive, got {self.steps}")

        self._partitions = [
            list(range(j * self.partition_size, (j + 1) * self.partition_size))
            for j in range(self.bands)
        ]

    def __len__(self):
        return self.steps

    def __iter__(self):
        for step in range(self.steps):
            partition = self._partitions[step % self.bands]
            perm = torch.randperm(self.partition_size, generator=self.generator)
            selected = perm[: self.batch_size].tolist()

            yield [partition[i] for i in selected]


class DistributedCyclicPoissonSampler(Sampler[List[int]]):
    r"""
    Distributed cyclic partitioned fixed-size sampler.

    At each step, every rank samples from the same active cyclic partition,
    but only from its local shard of that partition (sharded by rank).
    """

    def __init__(
        self,
        *,
        total_size: int,
        batch_size: int,
        bands: int,
        generator=None,
        steps: int = None,
    ):
        self.total_size = int(total_size)
        self.batch_size = int(batch_size)
        self.bands = int(bands)
        self.generator = generator
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        if self.total_size <= 0:
            raise ValueError(f"total_size should be positive, got {self.total_size}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size should be positive, got {self.batch_size}")

        if self.bands <= 0:
            raise ValueError(f"bands should be positive, got {self.bands}")

        if self.num_replicas <= 0:
            raise ValueError(
                f"num_replicas should be positive, got {self.num_replicas}"
            )

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"invalid rank {self.rank} for world size {self.num_replicas}"
            )

        self.partition_size = self.total_size // self.bands
        if self.partition_size <= 0:
            raise ValueError(
                "bands is too large for dataset size: partition_size is zero"
            )

        self._global_partitions = [
            list(range(j * self.partition_size, (j + 1) * self.partition_size))
            for j in range(self.bands)
        ]

        self._local_partitions = [
            part[self.rank :: self.num_replicas] for part in self._global_partitions
        ]

        min_local = min(len(part) for part in self._local_partitions)
        if self.batch_size > min_local:
            raise ValueError(
                "batch_size must be <= local shard size for every cyclic partition"
            )

        self.steps = (
            int(steps) if steps is not None else int(self.total_size / self.batch_size)
        )
        if self.steps <= 0:
            raise ValueError(f"steps should be positive, got {self.steps}")

    def __len__(self):
        return self.steps

    def __iter__(self):
        for step in range(self.steps):
            local_partition = self._local_partitions[step % self.bands]
            perm = torch.randperm(len(local_partition), generator=self.generator)
            selected = perm[: self.batch_size].tolist()

            yield [local_partition[i] for i in selected]
