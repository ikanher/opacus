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
import math
import itertools
import warnings
from unittest.mock import patch

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from opacus import SamplingSemantics
from opacus.accountants import (
    BandMFAccountant,
    BNBAccountant,
    GaussianAccountant,
    IAccountant,
    BSRAccountant,
    PRVAccountant,
    RDPAccountant,
    create_accountant,
    register_accountant,
    registry,
)
from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    bsr_fixed_batch_epsilon_upper_bound,
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
)
from opacus.accountants.analysis.bandmf import (
    compute_bandmf_mf_sensitivity_from_coeffs,
)
from opacus.accountants.analysis.bnb import (
    build_bnb_toeplitz_c_matrix_and_contract,
    normalize_bnb_accountant_coeffs,
)
from opacus.accountants.utils import get_noise_multiplier


def _materialize_lower_triangular_from_coeffs(coeffs: list[float], n: int) -> list[list[float]]:
    c = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for lag, v in enumerate(coeffs):
            j = i - lag
            if j < 0:
                break
            c[i][j] = float(v)
    return c


def _participation_bruteforce_sensitivity(coeffs: list[float], n: int, k: int, b: int) -> float:
    c = _materialize_lower_triangular_from_coeffs(coeffs, n)
    best_sq = 0.0
    indices = list(range(n))
    for r in range(0, k + 1):
        for subset in itertools.combinations(indices, r):
            if any(subset[t + 1] - subset[t] < b for t in range(len(subset) - 1)):
                continue
            sq = 0.0
            for i in range(n):
                row_sum = 0.0
                for j in subset:
                    row_sum += c[i][j]
                sq += row_sum * row_sum
            best_sq = max(best_sq, sq)
    return math.sqrt(best_sq)


def _participation_sensitivity_for_set(coeffs: list[float], n: int, subset: tuple[int, ...]) -> float:
    c = _materialize_lower_triangular_from_coeffs(coeffs, n)
    sq = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in subset:
            row_sum += c[i][j]
        sq += row_sum * row_sum
    return math.sqrt(sq)


def _bnb_c_matrix_contract(*, c_matrix: torch.Tensor, bands: int) -> dict:
    return {
        "sampling_mode": "b_min_sep",
        "bands": int(bands),
        "granularity": "single_participation",
        "matrix_columns": int(c_matrix.shape[1]),
    }


def _lower_toeplitz_from_coeffs(coeffs: list[float], horizon: int) -> torch.Tensor:
    c = torch.zeros((horizon, horizon), dtype=torch.float64)
    for i in range(horizon):
        max_lag = min(i, len(coeffs) - 1)
        for lag in range(max_lag + 1):
            c[i, i - lag] = float(coeffs[lag])
    return c


class AccountantRegistryTest(unittest.TestCase):

    class DummyAccountant(IAccountant):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def step(self, **kwargs):
            pass

        def get_epsilon(self, **kwargs):
            return 0.0

        def mechanism(cls) -> str:
            return "dummy"

    class Dummy2Accountant(DummyAccountant):
        pass

    def test_register_accountant(self) -> None:
        try:
            register_accountant("dummy", AccountantRegistryTest.DummyAccountant)
            self.assertIsInstance(
                create_accountant("dummy"), AccountantRegistryTest.DummyAccountant
            )
            self.assertEqual(create_accountant("dummy").mechanism(), "dummy")
        finally:
            if "dummy" in registry._ACCOUNTANTS:
                del registry._ACCOUNTANTS["dummy"]

    def test_create_accountant_not_registered(self) -> None:
        with self.assertRaises(ValueError):
            create_accountant("not_registered")

    def test_create_bsr_accountant(self) -> None:
        self.assertIsInstance(create_accountant("bsr"), BSRAccountant)

    def test_create_bandmf_accountant(self) -> None:
        self.assertIsInstance(create_accountant("bandmf"), BandMFAccountant)

    def test_create_bnb_accountant(self) -> None:
        self.assertIsInstance(create_accountant("bnb"), BNBAccountant)

    def test_register_existing_accountant(self):
        try:
            register_accountant("dummy", AccountantRegistryTest.DummyAccountant)

            with self.assertRaises(ValueError):
                register_accountant("rdp", AccountantRegistryTest.DummyAccountant)
        finally:
            if "dummy" in registry._ACCOUNTANTS:
                del registry._ACCOUNTANTS["dummy"]

    def test_force_register_existing_accountant(self) -> None:
        try:
            register_accountant("dummy", AccountantRegistryTest.DummyAccountant)

            register_accountant(
                "dummy", AccountantRegistryTest.Dummy2Accountant, force=True
            )
            self.assertIsInstance(
                create_accountant("dummy"), AccountantRegistryTest.Dummy2Accountant
            )
        finally:
            if "dummy" in registry._ACCOUNTANTS:
                del registry._ACCOUNTANTS["dummy"]


class AccountingTest(unittest.TestCase):
    def test_bnb_accountant_requires_builtin_inputs(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 10)]
        with self.assertRaisesRegex(
            ValueError, "built-in calibration requires"
        ):
            accountant.get_epsilon(delta=1e-5)

    def test_bnb_accountant_rejects_nonconstant_history(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.05, 10), (1.2, 0.05, 1)]
        with self.assertRaisesRegex(ValueError, "constant noise_multiplier and sample_rate"):
            accountant.get_epsilon(delta=1e-5)

    def test_bnb_accountant_builtin_monte_carlo_path(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.05, 5)]
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        eps = accountant.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=10_000,
            bnb_seed=123,
        )
        self.assertGreaterEqual(eps, 0.0)

    def test_bnb_accountant_builtin_forwards_target_delta_directly(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.4, 7)]  # effective_steps = ceil(7 * 0.4) = 3
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )

        calls = []

        def _mock_estimator(**kwargs):
            calls.append(kwargs)
            return 0.5

        with patch(
            "opacus.accountants.bnb.estimate_b_min_sep_epsilon_monte_carlo",
            side_effect=_mock_estimator,
        ):
            eps = accountant.get_epsilon(
                delta=0.2,
                mechanism_state={
                    "bnb_c_matrix": c_matrix,
                    "coeffs": [1.0, 0.2],
                    "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
                sampling_semantics=sampling_semantics,
            )

        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(float(calls[0]["target_delta"]), 0.2, places=12)
        self.assertAlmostEqual(float(eps), 0.5, places=12)

    def test_bnb_accountant_builtin_rejects_out_of_domain_delta(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.8, 5)]
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        with self.assertRaisesRegex(ValueError, "target_delta must be in \\[0, 1\\)"):
            accountant.get_epsilon(
                delta=4.0,
                mechanism_state={
                    "bnb_c_matrix": c_matrix,
                    "coeffs": [1.0, 0.2],
                    "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
                sampling_semantics=sampling_semantics,
                bnb_tolerance=1e-6,
                bnb_max_iterations=10,
            )

    def test_bnb_accountant_builtin_supports_balls_in_bins_sampling_mode(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 5)]
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bands": 2},
        )
        epsilon = accountant.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
        )
        self.assertTrue(float(epsilon) > 0.0)

    def test_bnb_accountant_builtin_monte_carlo_noise_monotonicity_smoke(self) -> None:
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )

        accountant_low_noise = BNBAccountant()
        accountant_low_noise.history = [(0.9, 0.05, 5)]
        eps_low_noise = accountant_low_noise.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=99,
        )

        accountant_high_noise = BNBAccountant()
        accountant_high_noise.history = [(1.8, 0.05, 5)]
        eps_high_noise = accountant_high_noise.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=99,
        )

        self.assertLessEqual(eps_high_noise, eps_low_noise)

    def test_bnb_accountant_builtin_monte_carlo_invariant_to_steps_and_sample_rate(self) -> None:
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )

        accountant_a = BNBAccountant()
        accountant_a.history = [(1.2, 0.05, 5)]
        eps_a = accountant_a.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=77,
        )

        accountant_b = BNBAccountant()
        accountant_b.history = [(1.2, 0.4, 25)]
        eps_b = accountant_b.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=77,
        )

        self.assertAlmostEqual(float(eps_a), float(eps_b), places=12)

    def test_bnb_accountant_builtin_rejects_bands_coeffs_mismatch(self) -> None:
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 5)]
        with self.assertRaisesRegex(ValueError, "bands.*len\\(coeffs\\)"):
            accountant.get_epsilon(
                delta=0.2,
                mechanism_state={
                    "bnb_c_matrix": c_matrix,
                    "coeffs": [1.0],
                    "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
                sampling_semantics=sampling_semantics,
                bnb_num_samples=5_000,
                bnb_seed=1,
            )

    def test_bnb_accountant_builtin_rejects_metadata_bands_mismatch(self) -> None:
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 3},
        )
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 5)]
        with self.assertRaisesRegex(
            ValueError, "privacy_metadata\\['bands'\\].*accounting bands"
        ):
            accountant.get_epsilon(
                delta=0.2,
                mechanism_state={
                    "bnb_c_matrix": c_matrix,
                    "coeffs": [1.0, 0.2],
                    "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
                sampling_semantics=sampling_semantics,
                bnb_bands=2,
                bnb_num_samples=5_000,
                bnb_seed=1,
            )

    def test_bnb_accountant_builtin_rejects_contract_mismatch(self) -> None:
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 5)]
        bad_contract = _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2)
        bad_contract["granularity"] = "full_horizon"
        with self.assertRaisesRegex(ValueError, "c_matrix_contract\\['granularity'\\]"):
            accountant.get_epsilon(
                delta=0.2,
                mechanism_state={
                    "bnb_c_matrix": c_matrix,
                    "coeffs": [1.0, 0.2],
                    "bnb_c_matrix_contract": bad_contract,
                },
                sampling_semantics=sampling_semantics,
                bnb_num_samples=5_000,
                bnb_seed=1,
            )

    def test_bnb_accountant_builtin_rejects_toeplitz_derivation_mismatch(self) -> None:
        coeffs = [1.0, 0.2]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)
        c_matrix[3, 2] += 0.3
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 5)]
        contract = _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2)
        contract["derivation"] = "lower_toeplitz_from_coeffs"
        contract["horizon"] = 4
        contract["atol"] = 1e-12

        with self.assertRaisesRegex(ValueError, "lower_toeplitz_from_coeffs derivation"):
            accountant.get_epsilon(
                delta=0.2,
                mechanism_state={
                    "bnb_c_matrix": c_matrix,
                    "coeffs": coeffs,
                    "bnb_c_matrix_contract": contract,
                },
                sampling_semantics=sampling_semantics,
                bnb_num_samples=5_000,
                bnb_seed=1,
            )

    def test_bnb_accountant_builtin_accepts_toeplitz_derivation_contract(self) -> None:
        coeffs = [1.0, 0.2]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 5)]
        contract = _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2)
        contract["derivation"] = "lower_toeplitz_from_coeffs"
        contract["horizon"] = 4
        contract["atol"] = 1e-12

        eps = accountant.get_epsilon(
            delta=0.2,
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": coeffs,
                "bnb_c_matrix_contract": contract,
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=5_000,
            bnb_seed=1,
        )
        self.assertGreater(eps, 0.0)

    def test_bsr_mf_sensitivity_matches_bruteforce_small_cases(self) -> None:
        cases = [
            ([1.0], 4, 1, 1),
            ([1.0, 0.5], 5, 2, 1),
            ([1.0, 0.7, 0.2], 6, 2, 2),
            ([1.0, 0.8, 0.4, 0.1], 7, 3, 2),
        ]
        for coeffs, n, k, b in cases:
            expected = _participation_bruteforce_sensitivity(coeffs, n, k, b)
            actual = compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=n,
                max_participations=k,
                min_separation=b,
            )
            self.assertAlmostEqual(actual, expected, places=10)

    def test_bsr_mf_sensitivity_identity_formula(self) -> None:
        # For coeffs=[1], b=1 under this Toeplitz convention:
        # sensitivity^2 = min(n, k)
        n = 6
        k = 3
        expected_sq = min(n, k)
        actual = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=[1.0],
            steps=n,
            max_participations=k,
            min_separation=1,
        )
        self.assertAlmostEqual(actual * actual, float(expected_sq), places=10)

    def test_bsr_mf_sensitivity_identity_formula_general_b(self) -> None:
        # Lean cross-check (Mf/DP/Sensitivity): for coeffs=[1], sensitivity^2
        # equals k_eff = min(k, floor((n-1)/b)+1).
        n = 17
        k = 7
        b = 3
        expected_sq = min(k, (n - 1) // b + 1)
        actual = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=[1.0],
            steps=n,
            max_participations=k,
            min_separation=b,
        )
        self.assertAlmostEqual(actual * actual, float(expected_sq), places=10)

    def test_bsr_mf_sensitivity_rejects_bad_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "coeffs must be non-empty"):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[],
                steps=4,
                max_participations=1,
                min_separation=1,
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[1.0, float("nan")],
                steps=4,
                max_participations=1,
                min_separation=1,
            )
        with self.assertRaisesRegex(ValueError, "max_participations must be >= 1"):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[1.0],
                steps=4,
                max_participations=0,
                min_separation=1,
            )
        with self.assertRaisesRegex(ValueError, "min_separation must be >= 1"):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[1.0],
                steps=4,
                max_participations=1,
                min_separation=0,
            )
        with self.assertRaisesRegex(
            ValueError, "requires nonnegative decreasing coefficients"
        ):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[1.0, 1.2],
                steps=4,
                max_participations=1,
                min_separation=1,
            )

    def test_bsr_mf_sensitivity_coeff_domain_tolerance_edges(self) -> None:
        # Tiny numerical drift within tolerance should be accepted.
        near_monotone = [1.0, 1.0 + 5e-13, 0.7, 0.4]
        near_nonnegative = [1.0, 0.5, -5e-13]
        for coeffs in (near_monotone, near_nonnegative):
            got = compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=6,
                max_participations=2,
                min_separation=2,
            )
            self.assertGreater(got, 0.0)

        # Meaningful violations must still be rejected.
        with self.assertRaisesRegex(
            ValueError, "requires nonnegative decreasing coefficients"
        ):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[1.0, 1.0 + 5e-11, 0.7],
                steps=6,
                max_participations=2,
                min_separation=2,
            )

        with self.assertRaisesRegex(
            ValueError, "requires nonnegative decreasing coefficients"
        ):
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=[1.0, 0.5, -5e-11],
                steps=6,
                max_participations=2,
                min_separation=2,
            )

    def test_bsr_mf_sensitivity_fixed_epoch_max_participations_invariant(self) -> None:
        # Matches JAX MF intent: when k exceeds true max for (n, min_sep),
        # sensitivity should be unchanged.
        for n, epochs in [(2, 1), (2, 2), (10, 1), (10, 5), (10, 10)]:
            min_sep = n // epochs
            coeffs = [1.0 - 0.05 * i for i in range(min(n, 8))]
            base = compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=n,
                max_participations=epochs,
                min_separation=min_sep,
            )
            larger_k = compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=n,
                max_participations=epochs + 10,
                min_separation=min_sep,
            )
            self.assertAlmostEqual(base, larger_k, places=10)

    def test_bsr_mf_sensitivity_matches_fixed_epoch_pattern(self) -> None:
        # For nonnegative decreasing Toeplitz coefficients, the fixed-epoch
        # pattern (0, b, 2b, ...) attains the min-sep sensitivity.
        for n, epochs in [(8, 4), (10, 2), (10, 5), (12, 3)]:
            min_sep = n // epochs
            coeffs = [1.0 - 0.05 * i for i in range(min(n, 8))]
            fixed_epoch_set = tuple(t * min_sep for t in range(epochs))
            expected = _participation_sensitivity_for_set(coeffs, n, fixed_epoch_set)
            actual = compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=n,
                max_participations=epochs,
                min_separation=min_sep,
            )
            self.assertAlmostEqual(actual, expected, places=10)

    def test_bsr_kappa_from_coeffs_identity_and_two_tap(self) -> None:
        # Identity mechanism C=I has unit column norm at every finite horizon.
        self.assertAlmostEqual(
            compute_bsr_kappa_from_coeffs(coeffs=[1.0], steps=32),
            1.0,
            places=12,
        )

        # For coeffs=[1, 2], finite-horizon kappa is sqrt(1^2 + 2^2).
        self.assertAlmostEqual(
            compute_bsr_kappa_from_coeffs(coeffs=[1.0, 2.0], steps=32),
            math.sqrt(5.0),
            places=12,
        )

    def test_bsr_kappa_matches_direct_toeplitz_column_norm(self) -> None:
        # Lean/definition cross-check: kappa = max_i ||C e_i||_2.
        coeffs = [1.0, 0.6, -0.2]
        steps = 7
        c = torch.zeros((steps, steps), dtype=torch.float64)
        for i in range(steps):
            for j in range(i + 1):
                lag = i - j
                if lag < len(coeffs):
                    c[i, j] = float(coeffs[lag])

        expected = float(torch.linalg.norm(c, ord=2, dim=0).max().item())
        actual = compute_bsr_kappa_from_coeffs(coeffs=coeffs, steps=steps)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_bsr_kappa_from_coeffs_rejects_bad_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "coeffs must be non-empty"):
            compute_bsr_kappa_from_coeffs(coeffs=[], steps=4)

        with self.assertRaisesRegex(ValueError, "coeffs must be finite"):
            compute_bsr_kappa_from_coeffs(coeffs=[1.0, float("nan")], steps=4)

        with self.assertRaisesRegex(ValueError, "steps must be >= 1"):
            compute_bsr_kappa_from_coeffs(coeffs=[1.0], steps=0)

    def test_bandmf_accountant_cyclic_poisson_respects_sensitivity_scale_override(self) -> None:
        delta = 1e-5
        accountant = BandMFAccountant()
        accountant.history = [(1.0, 0.01, 100)]
        sampling_semantics = SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 10},
        )

        eps_unit = accountant.get_epsilon(
            delta=delta,
            sampling_semantics=sampling_semantics,
            bsr_sensitivity_scale=1.0,
        )
        eps_larger_scale = accountant.get_epsilon(
            delta=delta,
            sampling_semantics=sampling_semantics,
            bsr_sensitivity_scale=2.0,
        )

        self.assertGreater(eps_larger_scale, eps_unit)

    def test_bandmf_accountant_exposes_resolved_contract_metadata(self) -> None:
        accountant = BandMFAccountant()
        accountant.history = [(1.25, 0.02, 120)]
        sampling_semantics = SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 10},
        )

        eps = accountant.get_epsilon(
            delta=1e-5,
            sampling_semantics=sampling_semantics,
            bsr_sensitivity_scale=1.0,
        )
        self.assertGreaterEqual(eps, 0.0)
        self.assertIsInstance(accountant.last_contract, dict)
        assert accountant.last_contract is not None
        self.assertEqual(accountant.last_contract["mechanism"], "bandmf")
        self.assertEqual(accountant.last_contract["accounting_mode"], "bandmf_accountant")
        self.assertEqual(accountant.last_contract["sampling_mode"], "cyclic_poisson")
        self.assertEqual(accountant.last_contract["bands"], 10)
        self.assertEqual(accountant.last_contract["steps"], 120)
        self.assertAlmostEqual(accountant.last_contract["sample_rate"], 0.02, places=12)
        self.assertAlmostEqual(accountant.last_contract["q"], 0.2, places=12)
        self.assertEqual(accountant.last_contract["cycles"], 12)

    def test_bandmf_accountant_rejects_steps_below_bands(self) -> None:
        accountant = BandMFAccountant()
        accountant.history = [(1.0, 0.01, 5)]
        sampling_semantics = SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 8},
        )
        with self.assertRaisesRegex(ValueError, "steps must be >= bands"):
            accountant.get_epsilon(
                delta=1e-5,
                sampling_semantics=sampling_semantics,
            )

    def test_bandmf_accountant_rejects_invalid_derived_q(self) -> None:
        accountant = BandMFAccountant()
        accountant.history = [(1.0, 0.2, 100)]
        sampling_semantics = SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 6},
        )
        with self.assertRaisesRegex(ValueError, "derived q = bands \\* sample_rate"):
            accountant.get_epsilon(
                delta=1e-5,
                sampling_semantics=sampling_semantics,
            )

    def test_bandmf_accountant_ignores_legacy_sensitivity_scale_kwarg(self) -> None:
        accountant = BandMFAccountant()
        accountant.history = [(1.0, 0.01, 100)]
        sampling_semantics = SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 10},
        )
        epsilon = accountant.get_epsilon(
            delta=1e-5,
            sampling_semantics=sampling_semantics,
            sensitivity_scale=1.0,
        )
        self.assertGreater(epsilon, 0.0)

    def test_bandmf_accountant_fixed_batch_matches_direct_prv_contract(self) -> None:
        coeffs = [1.0, 0.5, 0.25]
        sample_rate = 0.2
        steps = 10
        accountant = BandMFAccountant()
        accountant.history = [(1.3, sample_rate, steps)]
        sampling_semantics = SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        )
        mf_sensitivity = compute_bandmf_mf_sensitivity_from_coeffs(
            coeffs=coeffs,
            steps=steps,
            max_participations=2,
            min_separation=1,
        )
        epsilon = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state={
                "coeffs": coeffs,
                "bsr_max_participations": 2,
                "bsr_min_separation": 1,
            },
            sampling_semantics=sampling_semantics,
        )
        direct = bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=1.3,
            target_delta=1e-5,
            mf_sensitivity=mf_sensitivity,
        )
        self.assertAlmostEqual(epsilon, direct, places=12)
        self.assertEqual(accountant.last_contract["sampling_mode"], "torch_sampler")

    def test_bsr_accountant_legacy_fixed_batch_alias_is_not_used(self) -> None:
        accountant = BSRAccountant()
        accountant.history = [(1.0, 0.1, 10)]
        with self.assertRaisesRegex(
            ValueError,
            "requires MF sensitivity or enough data to derive it",
        ):
            accountant.get_epsilon(
                delta=1e-5,
                sampling_semantics=SamplingSemantics(
                    sampling_mode="torch_sampler",
                    privacy_metadata={},
                ),
                mf_sensitivity=1.0,  # legacy key: no longer consumed
            )

    def test_bnb_accountant_legacy_runtime_aliases_are_not_used(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.05, 5)]
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires b_min_sep/balls_in_bins inputs",
        ):
            accountant.get_epsilon(
                delta=0.2,
                sampling_semantics=SamplingSemantics(
                    sampling_mode="b_min_sep",
                    privacy_metadata={"bands": 2},
                ),
                c_matrix=c_matrix,  # legacy key: no longer consumed
                bnb_bands=2,
                bnb_c_matrix_contract=_bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            )

    def test_rdp_accountant(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 7.32911117143)

    def test_gdp_accountant(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 // 0.04)

        accountant = GaussianAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertLess(6.59, epsilon)
        self.assertLess(epsilon, 6.6)

    def test_prv_accountant(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 // 0.04)

        accountant = PRVAccountant()

        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 6.777395712150674)

    def test_get_noise_multiplier_rdp_epochs(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp",
        )

        self.assertAlmostEqual(noise_multiplier, 1.416, places=4)

    def test_get_noise_multiplier_rdp_steps(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        steps = 2000

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
        )

        self.assertAlmostEqual(noise_multiplier, 1.3562, places=4)

    def test_get_noise_multiplier_prv_epochs(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="prv",
        )

        self.assertAlmostEqual(noise_multiplier, 1.34765625, places=4)

    def test_get_noise_multiplier_prv_steps(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        steps = 2000

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="prv",
        )

        self.assertAlmostEqual(noise_multiplier, 1.2915, places=4)

    @given(
        epsilon=st.floats(1.0, 10.0),
        epochs=st.integers(10, 100),
        sample_rate=st.sampled_from(
            [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
        ),
        delta=st.sampled_from([1e-4, 1e-5, 1e-6]),
    )
    @settings(deadline=60000)
    def test_get_noise_multiplier_overshoot(self, epsilon, epochs, sample_rate, delta):
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )

        accountant = create_accountant(mechanism="rdp")
        accountant.history = [
            (noise_multiplier, sample_rate, int(epochs / sample_rate))
        ]

        actual_epsilon = accountant.get_epsilon(delta=delta)
        self.assertLess(actual_epsilon, epsilon)

    def test_get_noise_multiplier_gdp(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="gdp",
        )

        self.assertAlmostEqual(noise_multiplier, 1.3232421875)

    def test_get_noise_multiplier_bsr_epochs(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 0.5
        epochs = 1

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bsr",
            bsr_mf_sensitivity=1.0,
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_get_noise_multiplier_bnb_epochs(self) -> None:
        delta = 0.2
        sample_rate = 0.04
        epsilon = 0.5
        epochs = 1
        c_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "bnb_c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"bands": 2},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_get_noise_multiplier_bsr_balls_in_bins_uses_bnb_accountant(self) -> None:
        delta = 0.2
        sample_rate = 0.25
        epsilon = 0.5
        epochs = 1
        coeffs = [1.0, 0.2]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "coeffs": coeffs,
                "bsr_bands": 2,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                "_noise_mechanism": "bsr",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 2},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_get_noise_multiplier_bisr_balls_in_bins_uses_bnb_accountant(self) -> None:
        delta = 0.2
        sample_rate = 0.25
        epsilon = 0.5
        epochs = 1
        coeffs = [1.0, -0.5]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "coeffs": coeffs,
                "bsr_bands": 2,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                "_noise_mechanism": "bisr",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 2},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_get_noise_multiplier_bandmf_balls_in_bins_uses_bnb_accountant(self) -> None:
        delta = 0.2
        sample_rate = 0.25
        epsilon = 0.5
        epochs = 1
        coeffs = [1.0, 0.2]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "coeffs": coeffs,
                "bsr_bands": 2,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                "_noise_mechanism": "bandmf",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 2},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_get_noise_multiplier_bsr_balls_in_bins_separates_cycle_length_from_matrix_bandwidth(self) -> None:
        delta = 0.2
        sample_rate = 0.25
        epsilon = 0.5
        epochs = 1
        coeffs = [1.0, 0.2]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=8)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "coeffs": coeffs,
                "bsr_bands": 2,
                "bnb_bands": 2,
                "bnb_cycle_length": 4,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                "_noise_mechanism": "bsr",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 2},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_bsr_balls_in_bins_can_validate_normalized_accountant_coeffs(self) -> None:
        coeffs = [1.0, 0.2]
        accountant_coeffs = normalize_bnb_accountant_coeffs(coeffs=coeffs)
        c_matrix, contract = build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=accountant_coeffs,
            bands=2,
            horizon=8,
        )

        noise_multiplier = get_noise_multiplier(
            target_epsilon=0.5,
            target_delta=0.2,
            sample_rate=0.25,
            epochs=1,
            accountant="bnb",
            epsilon_tolerance=0.2,
            bnb_num_samples=200,
            bnb_max_iterations=8,
            mechanism_state={
                "coeffs": coeffs,
                "bnb_accountant_coeffs": accountant_coeffs,
                "bsr_bands": 2,
                "bnb_bands": 2,
                "bnb_cycle_length": 4,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": contract,
                "_noise_mechanism": "bsr",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 2},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_get_noise_multiplier_gaussian_balls_in_bins_uses_bnb_accountant(self) -> None:
        delta = 0.2
        sample_rate = 0.25
        epsilon = 0.5
        epochs = 1
        coeffs = [1.0]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "coeffs": coeffs,
                "bnb_bands": 1,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=1),
                "_noise_mechanism": "gaussian",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 1},
            ),
        )

        self.assertGreater(noise_multiplier, 0.0)

    def test_gaussian_balls_in_bins_noise_differs_from_prv_poisson(self) -> None:
        target_epsilon = 0.5
        target_delta = 0.2
        sample_rate = 0.25
        epochs = 1
        coeffs = [1.0]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)

        bnb_noise = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            mechanism_state={
                "coeffs": coeffs,
                "bnb_bands": 1,
                "bnb_c_matrix": c_matrix,
                "bnb_c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=1),
                "_noise_mechanism": "gaussian",
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 1},
            ),
        )
        prv_noise = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="prv",
        )

        self.assertGreater(bnb_noise, 0.0)
        self.assertGreater(prv_noise, 0.0)
        self.assertFalse(math.isclose(bnb_noise, prv_noise, rel_tol=1e-6, abs_tol=1e-9))

    def test_bsr_accountant_default_calibration(self) -> None:
        target_epsilon = 1.0
        target_delta = 1e-5
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=0.1,
            steps=1,
            accountant="bsr",
            bsr_mf_sensitivity=1.0,
        )
        actual_epsilon = bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            mf_sensitivity=1.0,
        )
        self.assertLessEqual(actual_epsilon, target_epsilon)

    def test_bsr_accountant_mf_sensitivity_controls_default_rdp_bound(self) -> None:
        lower_sens_noise = get_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.1,
            steps=20,
            accountant="bsr",
            bsr_mf_sensitivity=1.0,
        )
        higher_sens_noise = get_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.1,
            steps=20,
            accountant="bsr",
            bsr_mf_sensitivity=2.0,
        )
        self.assertGreater(higher_sens_noise, lower_sens_noise)

    def test_bsr_accountant_default_requires_mf_sensitivity_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires MF sensitivity or enough data"):
            get_noise_multiplier(
                target_epsilon=1.0,
                target_delta=1e-5,
                sample_rate=0.1,
                steps=1,
                accountant="bsr",
            )

    def test_bsr_fixed_batch_noise_multiplier_invariant_to_sample_rate(self) -> None:
        noise_a = get_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.05,
            steps=50,
            accountant="bsr",
            bsr_mf_sensitivity=1.0,
        )
        noise_b = get_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.20,
            steps=50,
            accountant="bsr",
            bsr_mf_sensitivity=1.0,
        )
        self.assertLess(abs(noise_a - noise_b), 1e-6)

    def test_bsr_accountant_supports_cyclic_poisson(self) -> None:
        accountant = BSRAccountant()
        accountant.history = [(1.0, 0.01, 100)]
        eps = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state={"bsr_sensitivity_scale": 1.0},
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 10},
            ),
        )
        self.assertTrue(math.isfinite(eps))
        self.assertGreater(eps, 0.0)

    def test_bandmf_cyclic_runtime_matches_direct_prv_contract(self) -> None:
        accountant = BandMFAccountant()
        accountant.history = [(1.1, 0.02, 120)]
        eps = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state={"bsr_sensitivity_scale": 1.4},
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 10},
            ),
        )
        contract = accountant.last_contract
        direct = PRVAccountant()
        direct.history = [
            (
                float(contract["effective_noise_multiplier"]),
                float(contract["q"]),
                int(contract["cycles"]),
            )
        ]
        direct_eps = direct.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(eps, direct_eps, places=12)

    def test_bsr_cyclic_runtime_matches_direct_prv_contract(self) -> None:
        accountant = BSRAccountant()
        accountant.history = [(1.1, 0.02, 120)]
        eps = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state={"bsr_sensitivity_scale": 1.4},
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 10},
            ),
        )
        contract = accountant.last_contract
        direct = PRVAccountant()
        direct.history = [
            (
                float(contract["effective_noise_multiplier"]),
                float(contract["q"]),
                int(contract["cycles"]),
            )
        ]
        direct_eps = direct.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(eps, direct_eps, places=12)

    def test_bsr_fixed_batch_runtime_matches_direct_prv_contract(self) -> None:
        accountant = BSRAccountant()
        accountant.history = [(1.25, 0.1, 20)]
        eps = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state={"bsr_mf_sensitivity": 1.4},
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )
        contract = accountant.last_contract
        direct = PRVAccountant()
        direct.history = [
            (
                float(contract["effective_noise_multiplier"]),
                1.0,
                1,
            )
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"opacus\.accountants\.analysis\.prv\.prvs",
            )
            direct_eps = direct.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(eps, direct_eps, places=12)

    def test_bsr_cyclic_poisson_epsilon_invariant_within_same_round_bucket(self) -> None:
        # With fixed (q, noise, delta, bands), epsilon depends on rounds=ceil(steps/bands).
        eps_a = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            target_delta=1e-5,
            steps=11,
            sample_rate=0.02,
            bands=10,
        )
        eps_b = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            target_delta=1e-5,
            steps=20,
            sample_rate=0.02,
            bands=10,
        )
        self.assertAlmostEqual(eps_a, eps_b, places=12)

    def test_bsr_cyclic_poisson_epsilon_depends_on_q_and_rounds(self) -> None:
        # Pair 1: q=0.1, rounds=10
        eps_a = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            target_delta=1e-5,
            steps=100,
            sample_rate=0.01,
            bands=10,
        )
        # Pair 2: q=0.1, rounds=10
        eps_b = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            target_delta=1e-5,
            steps=50,
            sample_rate=0.02,
            bands=5,
        )
        self.assertAlmostEqual(eps_a, eps_b, places=12)

    def test_bsr_cyclic_prv_is_tighter_than_explicit_rdp(self) -> None:
        common = {
            "target_delta": 1e-5,
            "steps": 120,
            "sample_rate": 0.02,
            "bands": 10,
        }
        prv_eps = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            accountant="prv",
            **common,
        )
        rdp_eps = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            accountant="rdp",
            **common,
        )
        self.assertLessEqual(prv_eps, rdp_eps)

    def test_bsr_cyclic_poisson_no_amplification_boundary_matches_gaussian(self) -> None:
        # No amplification boundary from JAX tests:
        # bands == dataset_size / batch_size  => q = bands * sample_rate = 1.
        # With steps == bands, cyclic composition is a single non-subsampled Gaussian step.
        nm = 1.7
        delta = 1e-6
        steps = 10
        sample_rate = 0.1
        bands = 10

        cyclic_eps = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=nm,
            target_delta=delta,
            steps=steps,
            sample_rate=sample_rate,
            bands=bands,
        )
        fixed_eps = bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=nm,
            target_delta=delta,
            mf_sensitivity=1.0,
        )
        self.assertAlmostEqual(cyclic_eps, fixed_eps, places=10)

    def test_bandmf_cyclic_poisson_no_amplification_boundary_calibration_matches_gaussian(
        self,
    ) -> None:
        target_epsilon = 1.0
        target_delta = 1e-5
        steps = 10
        sample_rate = 0.1
        bands = 10

        cyclic_noise = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="bandmf",
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": bands},
            ),
        )
        fixed_noise = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="bsr",
            bsr_mf_sensitivity=1.0,
        )
        self.assertAlmostEqual(cyclic_noise, fixed_noise, places=8)

    def test_bsr_fixed_batch_epsilon_golden_small_case(self) -> None:
        eps = bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=1.25,
            target_delta=1e-5,
            mf_sensitivity=1.4,
            accountant="rdp",
            rdp_orders=[1.5, 2, 3, 4, 8, 16, 32],
        )
        self.assertAlmostEqual(eps, 5.596661628831665, places=12)

    def test_bsr_cyclic_poisson_epsilon_golden_small_case(self) -> None:
        eps = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=1.1,
            target_delta=1e-5,
            steps=120,
            sample_rate=0.02,
            bands=10,
            accountant="rdp",
            rdp_orders=[1.5, 2, 3, 4, 8, 16, 32],
        )
        self.assertAlmostEqual(eps, 5.218712005463466, places=12)

    def test_bandmf_cyclic_poisson_default_calibration(self) -> None:
        target_epsilon = 1.0
        target_delta = 1e-5
        steps = 100
        sample_rate = 0.01
        sampling_semantics = SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 10},
        )

        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="bandmf",
            sampling_semantics=sampling_semantics,
        )

        accountant = BandMFAccountant()
        accountant.history = [(noise_multiplier, sample_rate, steps)]
        actual_epsilon = accountant.get_epsilon(
            delta=target_delta,
            sampling_semantics=sampling_semantics,
        )
        self.assertLessEqual(actual_epsilon, target_epsilon)

    def test_bsr_iterations_number_override_changes_fixed_batch_epsilon(self) -> None:
        accountant = BSRAccountant()
        accountant.history = [(1.0, 0.01, 200)]

        mechanism_state = {
            "coeffs": [1.0, 0.5],
            "bsr_max_participations": 50,
            "bsr_min_separation": 1,
        }
        sampling_semantics = SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        )

        eps_default = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
        )
        eps_override = accountant.get_epsilon(
            delta=1e-5,
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            bsr_iterations_number=20,
        )

        self.assertNotAlmostEqual(eps_default, eps_override, places=10)

    def test_bsr_get_epsilon_fixed_batch_uses_calibrated_sensitivity_under_partial_progress(
        self,
    ) -> None:
        coeffs = [1.0, 0.8, 0.4, 0.1]
        calibration_steps = 16
        executed_steps = 4
        noise_multiplier = 1.3
        delta = 1e-5

        calibrated_mf_sensitivity = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=coeffs,
            steps=calibration_steps,
            max_participations=4,
            min_separation=2,
        )

        accountant = BSRAccountant()
        accountant.history = [(noise_multiplier, 0.125, executed_steps)]

        eps = accountant.get_epsilon(
            delta=delta,
            mechanism_state={
                "coeffs": coeffs,
                "bsr_max_participations": 4,
                "bsr_min_separation": 2,
                "bsr_mf_sensitivity": calibrated_mf_sensitivity,
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )

        expected = bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=noise_multiplier,
            target_delta=delta,
            mf_sensitivity=calibrated_mf_sensitivity,
        )
        self.assertAlmostEqual(eps, expected, places=12)

    def test_get_noise_multiplier_accepts_bsr_iterations_number_override(self) -> None:
        noise = get_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.05,
            steps=100,
            accountant="bsr",
            mechanism_state={
                "coeffs": [1.0, 0.5, 0.25],
                "bsr_max_participations": 20,
                "bsr_min_separation": 2,
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
            bsr_iterations_number=40,
        )
        self.assertGreater(noise, 0.0)

    def test_accountant_state_dict(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        dummy_dest = {"dummy_k": "dummy_v"}
        # history should be equal but not the same instance
        self.assertEqual(accountant.state_dict()["history"], accountant.history)
        self.assertFalse(accountant.state_dict()["history"] is accountant.history)
        # mechanism populated to supplied dict
        self.assertEqual(
            accountant.state_dict(dummy_dest)["mechanism"], accountant.mechanism()
        )
        # existing values in supplied dict unchanged
        self.assertEqual(
            accountant.state_dict(dummy_dest)["dummy_k"], dummy_dest["dummy_k"]
        )

    def test_accountant_load_state_dict(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps - 1000):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        new_rdp_accountant = RDPAccountant()
        new_gdp_accountant = GaussianAccountant()
        # check corner cases
        with self.assertRaises(ValueError):
            new_rdp_accountant.load_state_dict({})
        with self.assertRaises(ValueError):
            new_rdp_accountant.load_state_dict({"1": 2})
        with self.assertRaises(ValueError):
            new_rdp_accountant.load_state_dict({"history": []})
        with self.assertRaises(ValueError):
            new_gdp_accountant.load_state_dict(accountant.state_dict())
        # check loading logic
        self.assertNotEqual(new_rdp_accountant.state_dict(), accountant.state_dict())
        new_rdp_accountant.load_state_dict(accountant.state_dict())
        self.assertEqual(new_rdp_accountant.state_dict(), accountant.state_dict())

        # ensure correct output after completion
        for _ in range(steps - 1000, steps):
            new_rdp_accountant.step(
                noise_multiplier=noise_multiplier, sample_rate=sample_rate
            )

        epsilon = new_rdp_accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 7.32911117143)
