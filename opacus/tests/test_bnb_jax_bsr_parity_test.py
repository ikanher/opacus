#!/usr/bin/env python3

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from opacus import SamplingSemantics
from opacus.accountants.analysis.bnb import (
    estimate_balls_in_bins_epsilon_monte_carlo,
    get_bnb_base_delta,
)
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis.bnb import (
    build_b_min_sep_gaussian_mixture,
    build_bnb_toeplitz_c_matrix_and_contract,
    normalize_bnb_accountant_coeffs,
)

try:
    from jax_privacy import batch_selection as jax_batch_selection
    from jax_privacy.experimental.monte_carlo import delta_calculation as jax_delta_calculation
    from jax_privacy.experimental.monte_carlo import sample_generation as jax_sample_generation
except Exception:  # pragma: no cover - optional oracle dependency
    jax_batch_selection = None
    jax_delta_calculation = None
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
    jax_batch_selection is None or jax_sample_generation is None or jax_delta_calculation is None,
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

    def test_amplified_bsr_candidate_acceptance_matches_jax_oracle(self) -> None:
        epsilon = 0.8
        delta = 0.2
        sample_budget = 4_000
        coeffs = normalize_bnb_accountant_coeffs(coeffs=[1.0, 0.5])
        strategy = jax_batch_selection.BallsInBinsSampling(cycle_length=3, iterations=6)
        base_delta = float(get_bnb_base_delta(num_samples=sample_budget, target_delta=delta))
        nm_sweep = [3.0, 2.0, 1.5, 1.2]

        positive_samples = []
        negative_samples = []
        for idx, nm in enumerate(nm_sweep):
            positive_samples.append(
                jax_sample_generation.get_privacy_loss_sample(
                    strategy,
                    noise_multiplier=float(nm),
                    c_col=np.asarray(coeffs, dtype=np.float64),
                    seed=1000 + idx,
                    positive_sample=True,
                    num_samples=sample_budget,
                )
            )
            negative_samples.append(
                jax_sample_generation.get_privacy_loss_sample(
                    strategy,
                    noise_multiplier=float(nm),
                    c_col=np.asarray(coeffs, dtype=np.float64),
                    seed=2000 + idx,
                    positive_sample=False,
                    num_samples=sample_budget,
                )
            )

        jax_passes, jax_best_index = jax_delta_calculation.perform_calibration_from_samples(
            epsilon,
            delta,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
        )
        self.assertTrue(jax_passes)

        opacus_accepts = []
        for idx, nm in enumerate(nm_sweep):
            eps_est = estimate_balls_in_bins_epsilon_monte_carlo(
                coeffs=coeffs,
                cycle_length=3,
                horizon=6,
                noise_multiplier=float(nm),
                target_delta=delta,
                num_samples=sample_budget,
                seed=10_000 + idx,
                chunk_size=1_000,
                num_workers=0,
            )
            opacus_accepts.append(float(eps_est) <= float(epsilon))

        self.assertTrue(opacus_accepts[int(jax_best_index)])
        if int(jax_best_index) + 1 < len(nm_sweep):
            self.assertFalse(opacus_accepts[int(jax_best_index) + 1])
        self.assertAlmostEqual(base_delta, float(jax_delta_calculation.get_base_delta(sample_budget, delta)), places=8)

    def test_amplified_bsr_noise_calibration_tracks_jax_oracle(self) -> None:
        original_budget = _MODULE.JAX_MAX_CALIBRATION_SAMPLES
        original_chunk = _MODULE.JAX_ORACLE_CHUNK_SIZE
        original_bw = _MODULE.JAX_ORACLE_DISCRETIZATION_BIN_WIDTH
        original_seed = _MODULE.JAX_ORACLE_SEED
        try:
            _MODULE.JAX_MAX_CALIBRATION_SAMPLES = 50_000
            _MODULE.JAX_ORACLE_CHUNK_SIZE = 10_000
            _MODULE.JAX_ORACLE_DISCRETIZATION_BIN_WIDTH = 1e-4
            _MODULE.JAX_ORACLE_SEED = 123
            oracle = _MODULE._jax_amplified_bsr_noise_multiplier(bands=4)
        finally:
            _MODULE.JAX_MAX_CALIBRATION_SAMPLES = original_budget
            _MODULE.JAX_ORACLE_CHUNK_SIZE = original_chunk
            _MODULE.JAX_ORACLE_DISCRETIZATION_BIN_WIDTH = original_bw
            _MODULE.JAX_ORACLE_SEED = original_seed

        if oracle.reference_noise_multiplier is None:
            self.skipTest(f"JAX oracle did not produce a reference sigma: {oracle.status}")
        coeffs = normalize_bnb_accountant_coeffs(
            coeffs=_MODULE.generate_bsr_coeffs_from_sgd_workload(
                bands=4,
                momentum=_MODULE.MOMENTUM,
                weight_decay=_MODULE.WEIGHT_DECAY,
            )
        )
        c_matrix, contract = _MODULE.build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=coeffs,
            bands=4,
            horizon=_MODULE.TOTAL_STEPS,
        )
        sigma = get_noise_multiplier(
            target_epsilon=_MODULE.EPSILON,
            target_delta=_MODULE.DELTA,
            sample_rate=_MODULE.SAMPLE_RATE,
            steps=_MODULE.TOTAL_STEPS,
            accountant="bnb",
            epsilon_tolerance=0.2,
            mechanism_state={
                "coeffs": list(coeffs),
                "bnb_accountant_coeffs": list(coeffs),
                "bsr_bands": 4,
                "bnb_cycle_length": _MODULE.STEPS_PER_EPOCH,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": contract,
                "_noise_mechanism": "bsr",
                "_bnb_accounting_kwargs": {
                    "bnb_num_samples": 50_000,
                    "bnb_seed": 123,
                    "bnb_chunk_size": 10_000,
                    "bnb_num_workers": 0,
                },
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": _MODULE.STEPS_PER_EPOCH, "bands": 4},
            ),
        )
        tolerance = 0.25 if oracle.status == "jax_computed_verified" else 0.5
        self.assertLess(abs(float(sigma) - float(oracle.reference_noise_multiplier)), tolerance)


if __name__ == "__main__":
    unittest.main()
