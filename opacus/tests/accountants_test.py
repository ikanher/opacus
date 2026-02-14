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

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from opacus import SamplingSemantics
from opacus.accountants import (
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
    compute_bsr_mf_sensitivity_from_coeffs,
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
    def test_bnb_accountant_requires_epsilon_fn(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.1, 10)]
        with self.assertRaisesRegex(
            ValueError, "requires epsilon_fn|currently disabled"
        ):
            accountant.get_epsilon(delta=1e-5)

    def test_bnb_accountant_callback_path(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.5, 0.05, 12), (1.5, 0.05, 8)]
        calls = []

        def epsilon_fn(
            *,
            noise_multiplier: float,
            target_delta: float,
            sample_rate: float,
            steps: int,
            mechanism: str,
            **kwargs,
        ) -> float:
            calls.append((noise_multiplier, target_delta, sample_rate, steps, mechanism))
            return 0.123

        eps = accountant.get_epsilon(delta=1e-5, epsilon_fn=epsilon_fn)
        self.assertAlmostEqual(eps, 0.123)
        self.assertTrue(calls)
        self.assertEqual(calls[-1][-1], "bnb")
        self.assertEqual(calls[-1][3], 20)

    def test_bnb_accountant_rejects_nonconstant_history(self) -> None:
        accountant = BNBAccountant()
        accountant.history = [(1.0, 0.05, 10), (1.2, 0.05, 1)]
        with self.assertRaisesRegex(ValueError, "constant noise_multiplier and sample_rate"):
            accountant.get_epsilon(delta=1e-5, epsilon_fn=lambda **_: 1.0)

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
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=10_000,
            bnb_seed=123,
        )
        self.assertGreaterEqual(eps, 0.0)

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
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
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
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=99,
        )

        self.assertLessEqual(eps_high_noise, eps_low_noise)

    def test_bnb_accountant_builtin_monte_carlo_steps_monotonicity_smoke(self) -> None:
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

        accountant_few_steps = BNBAccountant()
        accountant_few_steps.history = [(1.2, 0.2, 5)]
        eps_few = accountant_few_steps.get_epsilon(
            delta=0.2,
            mechanism_state={
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=77,
        )

        accountant_more_steps = BNBAccountant()
        accountant_more_steps.history = [(1.2, 0.2, 25)]
        eps_more = accountant_more_steps.get_epsilon(
            delta=0.2,
            mechanism_state={
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=77,
        )

        self.assertGreaterEqual(eps_more, eps_few)

    def test_bnb_accountant_builtin_monte_carlo_sample_rate_monotonicity_smoke(self) -> None:
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

        accountant_low_rate = BNBAccountant()
        accountant_low_rate.history = [(1.2, 0.05, 20)]
        eps_low_rate = accountant_low_rate.get_epsilon(
            delta=0.2,
            mechanism_state={
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=88,
        )

        accountant_high_rate = BNBAccountant()
        accountant_high_rate.history = [(1.2, 0.4, 20)]
        eps_high_rate = accountant_high_rate.get_epsilon(
            delta=0.2,
            mechanism_state={
                "c_matrix": c_matrix,
                "coeffs": [1.0, 0.2],
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=20_000,
            bnb_seed=88,
        )

        self.assertGreaterEqual(eps_high_rate, eps_low_rate)

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
                    "c_matrix": c_matrix,
                    "coeffs": [1.0],
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
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
                    "c_matrix": c_matrix,
                    "coeffs": [1.0, 0.2],
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
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
                    "c_matrix": c_matrix,
                    "coeffs": [1.0, 0.2],
                    "c_matrix_contract": bad_contract,
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
                    "c_matrix": c_matrix,
                    "coeffs": coeffs,
                    "c_matrix_contract": contract,
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
                "c_matrix": c_matrix,
                "coeffs": coeffs,
                "c_matrix_contract": contract,
            },
            sampling_semantics=sampling_semantics,
            bnb_num_samples=5_000,
            bnb_seed=1,
        )
        self.assertGreater(eps, 0.0)

    def test_bnb_accountant_builtin_rejects_horizon_effective_steps_mismatch(self) -> None:
        coeffs = [1.0, 0.2]
        c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)
        sampling_semantics = SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"bands": 2},
        )
        accountant = BNBAccountant()
        accountant.history = [(1.0, 1.0, 5)]  # effective_steps = ceil(5 * 1.0) = 5
        contract = _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2)
        contract["derivation"] = "lower_toeplitz_from_coeffs"
        contract["horizon"] = 4
        contract["atol"] = 1e-12

        with self.assertRaisesRegex(
            ValueError, "c_matrix_contract\\['horizon'\\].*>= effective_steps"
        ):
            accountant.get_epsilon(
                delta=0.2,
                mechanism_state={
                    "c_matrix": c_matrix,
                    "coeffs": coeffs,
                    "c_matrix_contract": contract,
                },
                sampling_semantics=sampling_semantics,
                bnb_num_samples=5_000,
                bnb_seed=1,
            )

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
        calls = []

        def epsilon_fn(
            *,
            noise_multiplier: float,
            target_delta: float,
            sample_rate: float,
            steps: int,
            mechanism: str,
            **kwargs,
        ) -> float:
            calls.append((noise_multiplier, target_delta, sample_rate, steps, mechanism))
            return 1.0 / noise_multiplier

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bsr",
            epsilon_fn=epsilon_fn,
        )

        self.assertLess(abs(noise_multiplier - 2.0), 0.1)
        self.assertTrue(len(calls) > 0)
        self.assertEqual(calls[-1][-1], "bsr")

    def test_get_noise_multiplier_bnb_epochs(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 0.5
        epochs = 1
        calls = []

        def epsilon_fn(
            *,
            noise_multiplier: float,
            target_delta: float,
            sample_rate: float,
            steps: int,
            mechanism: str,
            **kwargs,
        ) -> float:
            calls.append((noise_multiplier, target_delta, sample_rate, steps, mechanism))
            return 1.0 / noise_multiplier

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="bnb",
            epsilon_fn=epsilon_fn,
        )

        self.assertLess(abs(noise_multiplier - 2.0), 0.1)
        self.assertTrue(len(calls) > 0)
        self.assertEqual(calls[-1][-1], "bnb")

    def test_bsr_accountant_default_calibration_without_epsilon_fn(self) -> None:
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

    def test_bsr_accountant_branches_torch_sampler_vs_cyclic_poisson(self) -> None:
        delta = 1e-5
        accountant = BSRAccountant()
        accountant.history = [(1.0, 0.01, 100)]

        fixed_eps = accountant.get_epsilon(
            delta=delta,
            mechanism_state={"mf_sensitivity": 1.0},
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )
        cyclic_eps = accountant.get_epsilon(
            delta=delta,
            mechanism_state={"mf_sensitivity": 1.0},
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 10},
            ),
        )

        self.assertNotAlmostEqual(fixed_eps, cyclic_eps, places=6)

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

    def test_bsr_cyclic_poisson_no_amplification_boundary_calibration_matches_gaussian(
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
            accountant="bsr",
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
            rdp_orders=[1.5, 2, 3, 4, 8, 16, 32],
        )
        self.assertAlmostEqual(eps, 5.218712005463466, places=12)

    def test_bsr_cyclic_poisson_default_calibration_without_epsilon_fn(self) -> None:
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
            accountant="bsr",
            sampling_semantics=sampling_semantics,
        )

        accountant = BSRAccountant()
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
            "max_participations": 50,
            "min_separation": 1,
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

    def test_get_noise_multiplier_accepts_bsr_iterations_number_override(self) -> None:
        noise = get_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            sample_rate=0.05,
            steps=100,
            accountant="bsr",
            mechanism_state={
                "coeffs": [1.0, 0.5, 0.25],
                "max_participations": 20,
                "min_separation": 2,
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
