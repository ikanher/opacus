#!/usr/bin/env python3

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from opacus.accountants.analysis.bnb import (
    build_b_min_sep_gaussian_mixture,
    build_bnb_toeplitz_c_matrix_and_contract,
    normalize_bnb_accountant_coeffs,
)

try:
    from jax_privacy import batch_selection as jax_batch_selection
    from jax_privacy.experimental.monte_carlo import sample_generation as jax_sample_generation
except Exception:  # pragma: no cover - optional oracle dependency
    jax_batch_selection = None
    jax_sample_generation = None

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "local-scripts"
    / "replicate_bisr_paper_cifar_noise_multipliers.py"
)
_SPEC = importlib.util.spec_from_file_location("replicate_bisr_paper_cifar_noise_multipliers", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@unittest.skipIf(
    jax_batch_selection is None or jax_sample_generation is None,
    "jax_privacy Monte Carlo reference is unavailable",
)
class BNBJaxBSRParityTest(unittest.TestCase):
    def test_amplified_bsr_modes_match_jax_balls_in_bins_reference(self) -> None:
        raw_coeffs = [1.0, 0.5]
        accountant_coeffs = normalize_bnb_accountant_coeffs(coeffs=raw_coeffs)
        c_matrix, _ = build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=accountant_coeffs,
            bands=len(accountant_coeffs),
            horizon=6,
        )
        gm = build_b_min_sep_gaussian_mixture(c_matrix=c_matrix, cycle_length=3)
        opacus_modes = sorted(
            tuple(round(float(x), 6) for x in row)
            for row in gm.modes.to(dtype=torch.float64).cpu().numpy()
        )

        strategy = jax_batch_selection.BallsInBinsSampling(cycle_length=3, iterations=6)
        samples = jax_sample_generation.generate_sample(
            strategy,
            noise_multiplier=1e-9,
            c_col=np.asarray(accountant_coeffs, dtype=np.float64),
            positive_sample=True,
            num_samples=120,
            seed=0,
        )
        jax_modes = sorted(
            {
                tuple(round(float(x), 6) for x in samples[:, i])
                for i in range(samples.shape[1])
            }
        )

        self.assertEqual(opacus_modes, jax_modes)

    def test_oracle_chunk_counts_cover_sample_budget(self) -> None:
        counts = _MODULE._jax_oracle_chunk_counts(total_samples=12_500, chunk_size=5_000)
        self.assertEqual(counts, [5_000, 5_000, 2_500])
        self.assertEqual(sum(counts), 12_500)

    def test_discretize_chunk_preserves_shape(self) -> None:
        samples = np.asarray([[0.123456, 0.333333], [0.888888, -0.111111]], dtype=np.float64)
        discretized = _MODULE._maybe_discretize_chunk(samples, bin_width=1e-2)
        self.assertEqual(discretized.shape, samples.shape)
        self.assertTrue(np.allclose(discretized * 100.0, np.round(discretized * 100.0)))


if __name__ == "__main__":
    unittest.main()
