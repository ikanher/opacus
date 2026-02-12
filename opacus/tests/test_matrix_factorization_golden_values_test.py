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

"""
JAX-derived BSR/MF golden numeric checks.

Golden values are hard-coded to keep this test dependency-free at
runtime (no JAX/jax_privacy imports in CI).
"""

import math

from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    bsr_fixed_batch_epsilon_upper_bound,
    compute_bsr_mf_sensitivity_from_coeffs,
)

RDP_ORDERS = [1.5, 2, 3, 4, 8, 16, 32]
RTOL = 1e-7
ATOL = 1e-9


def _assert_close(actual: float, expected: float, *, rtol: float, atol: float) -> None:
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (
        f"actual={actual} expected={expected} rtol={rtol} atol={atol}"
    )


def test_bsr_sensitivity_matches_jax_golden_cases() -> None:
    # Values below were generated from jax_privacy Toeplitz sensitivity
    # (`minsep_sensitivity_squared`) for the given (coeffs, steps, k, b) tuples.
    cases = [
        {
            "coeffs": [1.0, 0.7, 0.2],
            "steps": 6,
            "max_participations": 2,
            "min_separation": 2,
            "expected_sensitivity": 1.8601075340277973,
        },
        {
            "coeffs": [1.0, 0.8, 0.4, 0.1],
            "steps": 7,
            "max_participations": 3,
            "min_separation": 2,
            "expected_sensitivity": 2.7092433769875184,
        },
        {
            # Identity-like band (coeffs=[1]) with k=3 over n=6.
            "coeffs": [1.0],
            "steps": 6,
            "max_participations": 3,
            "min_separation": 1,
            "expected_sensitivity": 1.7320508075688772,
        },
        {
            # No-amplification-shaped participation layout: b=4, n=12, k=3.
            "coeffs": [1.0, 0.5, 0.25, 0.125],
            "steps": 12,
            "max_participations": 3,
            "min_separation": 4,
            "expected_sensitivity": 1.996089927833914,
        },
    ]
    for case in cases:
        got = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=case["coeffs"],
            steps=case["steps"],
            max_participations=case["max_participations"],
            min_separation=case["min_separation"],
        )
        _assert_close(
            got,
            float(case["expected_sensitivity"]),
            rtol=RTOL,
            atol=ATOL,
        )


def test_bsr_fixed_batch_epsilon_matches_jax_golden_cases() -> None:
    # Values below were computed in the JAX stack using dp_accounting RDP
    # accountant over a single Gaussian event with sigma_eff = nm / sensitivity.
    cases = [
        {
            "noise_multiplier": 1.25,
            "target_delta": 1e-5,
            "mf_sensitivity": 1.4,
            "expected_epsilon": 5.596661628831665,
        },
        {
            "noise_multiplier": 0.9,
            "target_delta": 1e-6,
            "mf_sensitivity": 1.0,
            "expected_epsilon": 6.32452579563215,
        },
        {
            "noise_multiplier": 2.0,
            "target_delta": 1e-5,
            "mf_sensitivity": 2.5,
            "expected_epsilon": 6.212861628831664,
        },
    ]
    for case in cases:
        got = bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=case["noise_multiplier"],
            target_delta=case["target_delta"],
            mf_sensitivity=case["mf_sensitivity"],
            rdp_orders=RDP_ORDERS,
        )
        _assert_close(
            got,
            float(case["expected_epsilon"]),
            rtol=RTOL,
            atol=ATOL,
        )


def test_bsr_cyclic_poisson_epsilon_matches_jax_golden_cases() -> None:
    # Values below were computed in the JAX stack via sampled-Gaussian RDP
    # composition using q = bands * sample_rate and cycles = ceil(steps / bands).
    cases = [
        {
            "noise_multiplier": 1.1,
            "target_delta": 1e-5,
            "steps": 120,
            "sample_rate": 0.02,
            "bands": 10,
            "expected_epsilon": 5.218712005463466,
        },
        {
            "noise_multiplier": 0.8,
            "target_delta": 1e-6,
            "steps": 300,
            "sample_rate": 0.01,
            "bands": 20,
            "expected_epsilon": 11.943206607668474,
        },
        {
            # Boundary where q=1 and steps=bands (no amplification).
            "noise_multiplier": 1.7,
            "target_delta": 1e-6,
            "steps": 10,
            "sample_rate": 0.1,
            "bands": 10,
            "expected_epsilon": 2.9271329403988107,
        },
    ]
    for case in cases:
        got = bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=case["noise_multiplier"],
            target_delta=case["target_delta"],
            steps=case["steps"],
            sample_rate=case["sample_rate"],
            bands=case["bands"],
            rdp_orders=RDP_ORDERS,
        )
        _assert_close(
            got,
            float(case["expected_epsilon"]),
            rtol=RTOL,
            atol=ATOL,
        )
