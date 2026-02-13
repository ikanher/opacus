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

from opacus.accountants.analysis.bnb import (
    BNBCalibrationReport,
    DeltaVerificationResult,
    GaussianMixture,
    build_balls_in_bins_gaussian_mixture,
    calibrate_sigma_evr_binary_search,
    compute_llr_samples,
    describe_bnb_calibration_report,
    estimate_balls_in_bins_epsilon_monte_carlo,
    estimate_epsilon_from_llr_samples,
    estimate_hockey_stick_delta_from_llr_samples,
    find_sigma_binary_search,
    make_bnb_calibration_report,
    parse_bnb_calibration_report,
    generate_mixture_samples,
    sample_balls_in_bins_llr,
    select_evr_candidate_ladder,
    select_evr_candidate_ladder_two_sided,
    split_confidence_alpha,
    verify_evr_confidence_split,
    verify_hockey_stick_delta_hoeffding,
)


class BNBAnalysisTest(unittest.TestCase):
    def test_gaussian_mixture_requires_probs_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            GaussianMixture(
                modes=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                probs=torch.tensor([0.9, 0.2]),
            )

    def test_build_balls_in_bins_mixture_expected_modes(self) -> None:
        c = torch.tensor(
            [
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                [4.0, 40.0, 5.0, 50.0, 6.0, 60.0],
            ]
        )
        gm = build_balls_in_bins_gaussian_mixture(c_matrix=c, bands=3)
        expected_modes = torch.tensor([[6.0, 15.0], [60.0, 150.0]])
        self.assertTrue(torch.allclose(gm.modes, expected_modes))
        self.assertTrue(torch.allclose(gm.probs, torch.tensor([0.5, 0.5])))

    def test_build_balls_in_bins_reduced_dimensionality(self) -> None:
        # d > k so reduction should project to k-dimensional subspace.
        c = torch.arange(1, 11, dtype=torch.float32).reshape(5, 2)
        gm = build_balls_in_bins_gaussian_mixture(
            c_matrix=c,
            bands=1,
            reduce_dimensionality=True,
        )
        self.assertEqual(gm.modes.shape[0], 2)
        self.assertEqual(gm.modes.shape[1], 2)

    def test_generate_mixture_samples_is_seed_reproducible(self) -> None:
        gm = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            probs=torch.tensor([0.3, 0.7]),
        )
        g1 = torch.Generator().manual_seed(12345)
        g2 = torch.Generator().manual_seed(12345)
        ids1, samples1 = generate_mixture_samples(
            gm=gm, sigma=0.5, num_samples=64, generator=g1
        )
        ids2, samples2 = generate_mixture_samples(
            gm=gm, sigma=0.5, num_samples=64, generator=g2
        )
        self.assertTrue(torch.equal(ids1, ids2))
        self.assertTrue(torch.allclose(samples1, samples2))

    def test_compute_llr_samples_requires_matching_dimension(self) -> None:
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            probs=torch.tensor([0.5, 0.5]),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0, 0.0]]),
            probs=torch.tensor([1.0]),
        )
        with self.assertRaisesRegex(ValueError, "same dimension"):
            compute_llr_samples(up_gm=up, lo_gm=lo, sigma=1.0, num_samples=8)

    def test_compute_llr_samples_is_seed_reproducible(self) -> None:
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            probs=torch.tensor([0.4, 0.6]),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [0.2, 0.0]]),
            probs=torch.tensor([0.4, 0.6]),
        )
        g1 = torch.Generator().manual_seed(2026)
        g2 = torch.Generator().manual_seed(2026)
        llr_1 = compute_llr_samples(
            up_gm=up,
            lo_gm=lo,
            sigma=0.8,
            num_samples=128,
            generator=g1,
        )
        llr_2 = compute_llr_samples(
            up_gm=up,
            lo_gm=lo,
            sigma=0.8,
            num_samples=128,
            generator=g2,
        )
        self.assertTrue(torch.allclose(llr_1, llr_2))

    def test_compute_llr_samples_matches_reference_notebook_formula(self) -> None:
        # Parity against examples/monte_carlo_accountant.py formulas:
        #   component_llrs = (points @ modes.T - ||modes||^2 / 2) / sigma^2
        #   log p(points) = logsumexp(component_llrs, b=probs)
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, -0.5], [0.2, 1.3]], dtype=torch.float64),
            probs=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [0.5, -0.2]], dtype=torch.float64),
            probs=torch.tensor([0.4, 0.6], dtype=torch.float64),
        )
        sigma = 0.9
        num_samples = 1024
        seed = 4242

        g_impl = torch.Generator().manual_seed(seed)
        g_ref = torch.Generator().manual_seed(seed)

        llr_impl = compute_llr_samples(
            up_gm=up,
            lo_gm=lo,
            sigma=sigma,
            num_samples=num_samples,
            generator=g_impl,
        )

        # Reference sampling path used in the notebook.
        component_ids = torch.multinomial(
            up.probs.to(dtype=torch.float64),
            num_samples=num_samples,
            replacement=True,
            generator=g_ref,
        ).to(dtype=torch.long)
        points = up.modes[component_ids] + sigma * torch.randn(
            num_samples, up.modes.shape[1], generator=g_ref, dtype=up.modes.dtype
        )

        def ref_logpdf(points: torch.Tensor, gm: GaussianMixture, sigma: float) -> torch.Tensor:
            component_llrs = (
                points @ gm.modes.T - (gm.modes * gm.modes).sum(dim=1)[None, :] / 2.0
            ) / (sigma * sigma)
            return torch.logsumexp(component_llrs + torch.log(gm.probs)[None, :], dim=1)

        llr_ref = ref_logpdf(points, up, sigma) - ref_logpdf(points, lo, sigma)
        self.assertTrue(torch.allclose(llr_impl, llr_ref, atol=1e-10, rtol=1e-10))

    def test_hockey_stick_delta_is_in_unit_interval_and_eps_monotone(self) -> None:
        llr = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
        d1 = estimate_hockey_stick_delta_from_llr_samples(epsilon=0.1, llr_samples=llr)
        d2 = estimate_hockey_stick_delta_from_llr_samples(epsilon=0.9, llr_samples=llr)
        self.assertGreaterEqual(d1, 0.0)
        self.assertLessEqual(d1, 1.0)
        self.assertGreaterEqual(d2, 0.0)
        self.assertLessEqual(d2, 1.0)
        self.assertGreaterEqual(d1, d2)

    def test_find_sigma_binary_search_tracks_target_for_monotone_delta(self) -> None:
        target_delta = 0.2

        def delta_fn(sigma: float) -> float:
            # Monotone decreasing in sigma.
            return 1.0 / (1.0 + sigma)

        sigma = find_sigma_binary_search(
            delta_fn=delta_fn,
            target_delta=target_delta,
            sigma_low=1e-4,
            sigma_high=100.0,
            tolerance=1e-8,
            max_iterations=200,
        )
        self.assertAlmostEqual(delta_fn(sigma), target_delta, places=6)

    def test_estimate_epsilon_from_llr_samples_inverts_target_delta(self) -> None:
        llr = torch.tensor([-2.0, -0.5, 0.2, 1.0, 2.5, 3.0], dtype=torch.float64)
        true_epsilon = 0.7
        target_delta = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=true_epsilon, llr_samples=llr
        )
        estimated = estimate_epsilon_from_llr_samples(
            target_delta=target_delta,
            llr_samples=llr,
            tolerance=1e-8,
        )
        self.assertAlmostEqual(estimated, true_epsilon, places=6)

    def test_estimate_epsilon_from_llr_samples_returns_zero_when_feasible(self) -> None:
        llr = torch.tensor([0.1, 0.3, 1.2], dtype=torch.float64)
        delta_at_zero = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=0.0, llr_samples=llr
        )
        estimated = estimate_epsilon_from_llr_samples(
            target_delta=delta_at_zero + 1e-6,
            llr_samples=llr,
        )
        self.assertEqual(estimated, 0.0)

    def test_estimate_balls_in_bins_epsilon_is_seed_reproducible(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        kwargs = dict(
            c_matrix=c,
            bands=2,
            noise_multiplier=1.2,
            target_delta=0.2,
            num_samples=10_000,
            seed=7,
        )
        eps_1 = estimate_balls_in_bins_epsilon_monte_carlo(**kwargs)
        eps_2 = estimate_balls_in_bins_epsilon_monte_carlo(**kwargs)
        self.assertAlmostEqual(eps_1, eps_2, places=12)
        self.assertGreaterEqual(eps_1, 0.0)

    def test_verify_hockey_stick_delta_hoeffding_accepts_and_rejects(self) -> None:
        llr = torch.full((20_000,), 2.0, dtype=torch.float64)
        ok = verify_hockey_stick_delta_hoeffding(
            epsilon=1.0,
            llr_samples=llr,
            target_delta=0.7,
            confidence_alpha=0.5,
        )
        fail = verify_hockey_stick_delta_hoeffding(
            epsilon=1.0,
            llr_samples=llr,
            target_delta=0.1,
            confidence_alpha=0.5,
        )
        self.assertIsInstance(ok, DeltaVerificationResult)
        self.assertTrue(ok.accepted)
        self.assertFalse(fail.accepted)
        self.assertLessEqual(ok.delta_estimate, ok.upper_confidence_bound)

    def test_calibrate_sigma_evr_binary_search_finds_feasible_sigma(self) -> None:
        def llr_samples_fn(sigma: float) -> torch.Tensor:
            # Deterministic monotone setup: larger sigma -> smaller llr -> smaller delta.
            return torch.full((20_000,), 1.0 / sigma, dtype=torch.float64)

        sigma, verification = calibrate_sigma_evr_binary_search(
            llr_samples_fn=llr_samples_fn,
            target_epsilon=1.0,
            target_delta=0.1,
            confidence_alpha=0.5,
            sigma_low=0.1,
            sigma_high=3.0,
            tolerance=1e-4,
            max_iterations=80,
        )
        self.assertGreaterEqual(sigma, 0.1)
        self.assertLessEqual(sigma, 3.0)
        self.assertTrue(verification.accepted)

    def test_sample_balls_in_bins_llr_is_seed_reproducible(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        llr_1 = sample_balls_in_bins_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=5000,
            seed=42,
        )
        llr_2 = sample_balls_in_bins_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=5000,
            seed=42,
        )
        self.assertTrue(torch.allclose(llr_1, llr_2))

    def test_make_bnb_calibration_report_schema(self) -> None:
        verification = DeltaVerificationResult(
            delta_estimate=0.11,
            upper_confidence_bound=0.12,
            confidence_alpha=1e-4,
            accepted=True,
        )
        report = make_bnb_calibration_report(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=2.0,
            num_samples=10000,
            seed=7,
            bands=2,
            verification=verification,
            evr_confidence_alpha_total=1e-4,
            evr_num_checks=2,
            evr_per_check_alpha=5e-5,
            evr_pass_count=2,
        )
        self.assertIsInstance(report, BNBCalibrationReport)
        payload = report.to_dict()
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["bands"], 2)
        self.assertEqual(payload["num_samples"], 10000)
        self.assertTrue(payload["verification_passed"])
        self.assertEqual(payload["evr_num_checks"], 2)
        self.assertEqual(payload["evr_pass_count"], 2)
        self.assertEqual(payload["verification_contract"], "evr_union_bound_alpha_split_v1")
        self.assertGreaterEqual(payload["evr_composed_delta_upper_bound"], payload["target_delta"])

    def test_split_confidence_alpha(self) -> None:
        alpha = split_confidence_alpha(total_confidence_alpha=1e-3, num_checks=4)
        self.assertAlmostEqual(alpha, 2.5e-4)

    def test_verify_evr_confidence_split(self) -> None:
        llr = torch.full((5000,), 2.0, dtype=torch.float64)
        worst, pass_count, per_alpha = verify_evr_confidence_split(
            llr_samples_seq=[llr, llr, llr],
            epsilon=1.0,
            target_delta=0.7,
            total_confidence_alpha=0.5,
        )
        self.assertTrue(worst.accepted)
        self.assertEqual(pass_count, 3)
        self.assertAlmostEqual(per_alpha, 0.5 / 3.0)

    def test_select_evr_candidate_ladder_chooses_first_accept(self) -> None:
        def llr_samples_seq_fn(sigma: float):
            # sigma<1.0 fails (large llr -> large hockey-stick delta),
            # sigma>=1.0 passes (llr=0 -> delta=0).
            if sigma < 1.0:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(3)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(3)]

        sigma, verification, pass_count, per_alpha = select_evr_candidate_ladder(
            candidate_sigmas=[0.6, 0.8, 1.0, 1.2],
            llr_samples_seq_fn=llr_samples_seq_fn,
            epsilon=1.0,
            target_delta=0.2,
            total_confidence_alpha=0.1,
        )
        self.assertEqual(sigma, 1.0)
        self.assertTrue(verification.accepted)
        self.assertEqual(pass_count, 3)
        self.assertAlmostEqual(per_alpha, 0.1 / (4.0 * 3.0))

    def test_select_evr_candidate_ladder_requires_strictly_increasing_order(self) -> None:
        def llr_samples_seq_fn(_sigma: float):
            return [torch.zeros((128,), dtype=torch.float64)]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_evr_candidate_ladder(
                candidate_sigmas=[1.0, 0.9, 1.1],
                llr_samples_seq_fn=llr_samples_seq_fn,
                epsilon=1.0,
                target_delta=0.2,
                total_confidence_alpha=0.1,
            )

    def test_select_evr_candidate_ladder_two_sided_chooses_first_accept(self) -> None:
        def llr_samples_seq_fn_forward(sigma: float):
            if sigma < 1.0:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(3)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(3)]

        def llr_samples_seq_fn_reverse(sigma: float):
            if sigma < 1.0:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(3)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(3)]

        sigma, verification, pass_count, per_alpha = select_evr_candidate_ladder_two_sided(
            candidate_sigmas=[0.6, 0.8, 1.0, 1.2],
            llr_samples_seq_fn_forward=llr_samples_seq_fn_forward,
            llr_samples_seq_fn_reverse=llr_samples_seq_fn_reverse,
            epsilon=1.0,
            target_delta=0.2,
            total_confidence_alpha=0.1,
        )
        self.assertEqual(sigma, 1.0)
        self.assertTrue(verification.accepted)
        self.assertEqual(pass_count, 6)  # 2 directions * 3 checks
        self.assertAlmostEqual(per_alpha, 0.1 / (4.0 * 2.0 * 3.0))

    def test_select_evr_candidate_ladder_two_sided_rejects_if_one_direction_fails(self) -> None:
        def llr_samples_seq_fn_forward(sigma: float):
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(2)]

        def llr_samples_seq_fn_reverse(sigma: float):
            if sigma < 1.2:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(2)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(2)]

        sigma, verification, pass_count, per_alpha = select_evr_candidate_ladder_two_sided(
            candidate_sigmas=[1.0, 1.1, 1.2],
            llr_samples_seq_fn_forward=llr_samples_seq_fn_forward,
            llr_samples_seq_fn_reverse=llr_samples_seq_fn_reverse,
            epsilon=1.0,
            target_delta=0.2,
            total_confidence_alpha=0.3,
        )
        self.assertEqual(sigma, 1.2)
        self.assertTrue(verification.accepted)
        self.assertEqual(pass_count, 4)  # 2 directions * 2 checks at accepted candidate
        self.assertAlmostEqual(per_alpha, 0.3 / (3.0 * 2.0 * 2.0))

    def test_parse_bnb_calibration_report_rejects_unsupported_version(self) -> None:
        payload = {"version": 1}
        with self.assertRaisesRegex(ValueError, "unsupported BNB calibration report version"):
            parse_bnb_calibration_report(payload)

    def test_parse_bnb_calibration_report_v2(self) -> None:
        payload_v2 = {
            "version": 2,
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "noise_multiplier": 2.0,
            "num_samples": 1000,
            "seed": 7,
            "bands": 2,
            "delta_estimate_at_target_epsilon": 0.1,
            "delta_upper_confidence_bound": 0.11,
            "confidence_alpha": 1e-4,
            "verification_passed": False,
            "evr_confidence_alpha_total": 1e-4,
            "evr_num_checks": 3,
            "evr_per_check_alpha": 1.0 / 30000.0,
            "evr_pass_count": 2,
        }
        parsed = parse_bnb_calibration_report(payload_v2)
        self.assertEqual(parsed.version, 2)
        self.assertEqual(parsed.evr_num_checks, 3)
        self.assertEqual(parsed.evr_pass_count, 2)
        self.assertEqual(parsed.verification_contract, "evr_union_bound_alpha_split_v1")

    def test_describe_bnb_calibration_report(self) -> None:
        payload_v2 = {
            "version": 2,
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "noise_multiplier": 2.0,
            "num_samples": 1000,
            "seed": 7,
            "bands": 2,
            "delta_estimate_at_target_epsilon": 0.1,
            "delta_upper_confidence_bound": 0.11,
            "confidence_alpha": 1e-4,
            "verification_passed": False,
            "evr_confidence_alpha_total": 1e-4,
            "evr_num_checks": 3,
            "evr_per_check_alpha": 1.0 / 30000.0,
            "evr_pass_count": 2,
        }
        s = describe_bnb_calibration_report(payload_v2)
        self.assertIn("BNB calibration v2 [FAIL]", s)
        self.assertIn("checks=2/3", s)
        self.assertIn("guard_delta=", s)


if __name__ == "__main__":
    unittest.main()
