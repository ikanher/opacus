import unittest
from unittest.mock import patch

import numpy as np
import torch

from opacus.utils.uniform_sampler import DistributedKOutOfTSampler, KOutOfTSampler


class KOutOfTSamplerTest(unittest.TestCase):
    def _init_sampler(self, *, seed: int, num_samples: int = 60, num_steps: int = 10, num_selected: int = 3):
        g = torch.Generator()
        g.manual_seed(seed)
        return KOutOfTSampler(
            num_samples=num_samples,
            num_steps=num_steps,
            num_selected=num_selected,
            generator=g,
        )

    def test_length(self) -> None:
        sampler = self._init_sampler(seed=1, num_steps=17)
        self.assertEqual(len(sampler), 17)

    def test_each_example_appears_exactly_k_times(self) -> None:
        num_samples = 80
        num_steps = 11
        num_selected = 4
        sampler = self._init_sampler(
            seed=7,
            num_samples=num_samples,
            num_steps=num_steps,
            num_selected=num_selected,
        )
        counts = np.zeros(num_samples, dtype=np.int64)
        for batch in sampler:
            batch_arr = np.array(batch, dtype=np.int64)
            self.assertEqual(len(batch_arr), len(set(batch)))
            counts[batch_arr] += 1
        self.assertTrue(np.all(counts == num_selected))

    def test_set_epoch_resamples_deterministically(self) -> None:
        sampler_a = self._init_sampler(seed=5)
        sampler_b = self._init_sampler(seed=5)
        sampler_a.set_epoch(3)
        sampler_b.set_epoch(3)
        self.assertEqual(list(sampler_a), list(sampler_b))
        sampler_c = self._init_sampler(seed=5)
        before = list(sampler_c)
        sampler_c.set_epoch(1)
        after = list(sampler_c)
        self.assertNotEqual(before, after)


class DistributedKOutOfTSamplerTest(unittest.TestCase):
    def _init_sampler(self, *, seed: int, total_size: int = 60, num_steps: int = 10, num_selected: int = 3):
        g = torch.Generator()
        g.manual_seed(seed)
        return DistributedKOutOfTSampler(
            total_size=total_size,
            num_steps=num_steps,
            num_selected=num_selected,
            generator=g,
            shuffle=True,
            shuffle_seed=17,
        )

    def test_rank_shards_are_disjoint_and_local_counts_equal_k(self) -> None:
        total_size = 50
        num_steps = 9
        num_selected = 3
        rank_batches = []
        with patch('torch.distributed.get_world_size', return_value=2):
            for rank in [0, 1]:
                with patch('torch.distributed.get_rank', return_value=rank):
                    sampler = self._init_sampler(
                        seed=11,
                        total_size=total_size,
                        num_steps=num_steps,
                        num_selected=num_selected,
                    )
                    rank_batches.append(list(sampler))

        all_seen = [set() for _ in range(2)]
        counts = [np.zeros(total_size, dtype=np.int64) for _ in range(2)]
        for step in range(num_steps):
            step_sets = [set(rank_batches[r][step]) for r in range(2)]
            self.assertEqual(step_sets[0] & step_sets[1], set())
            for rank in [0, 1]:
                all_seen[rank].update(step_sets[rank])
                counts[rank][np.array(list(step_sets[rank]), dtype=np.int64)] += 1
        self.assertEqual(all_seen[0] & all_seen[1], set())
        self.assertEqual(len(all_seen[0] | all_seen[1]), total_size)
        for rank in [0, 1]:
            local = list(all_seen[rank])
            self.assertTrue(np.all(counts[rank][np.array(local, dtype=np.int64)] == num_selected))

    def test_set_epoch_resamples(self) -> None:
        with patch('torch.distributed.get_world_size', return_value=2), patch('torch.distributed.get_rank', return_value=0):
            sampler = self._init_sampler(seed=13)
            first = list(sampler)
            sampler.set_epoch(4)
            second = list(sampler)
        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
