#!/usr/bin/env python3

from __future__ import annotations

import math

import torch

from opacus.accountants.analysis.bnb import (
    DeltaVerificationResult,
    GaussianMixture,
    build_b_min_sep_gaussian_mixture,
    compute_llr_samples,
    estimate_hockey_stick_delta_from_llr_samples,
    make_bnb_calibration_report,
    select_evr_candidate_ladder,
)


def _internal_sampling_prob(p0: float, b: int) -> float:
    return p0 / (1.0 - p0 * float(b - 1))


def _evr_delta(delta: float, q: float) -> float:
    return delta + q * (1.0 - delta)


def test_internal_sampling_prob_eq_one_of_p0_eq_inv_bridge() -> None:
    # Lean bridge: Mf.DP.BNB.internalSamplingProb_eq_one_of_p0_eq_inv
    b = 4
    p0 = 1.0 / float(b)
    p = _internal_sampling_prob(p0, b)
    assert abs(p - 1.0) < 1e-12


def test_evr_delta_bridge_matches_report_composed_delta() -> None:
    # Lean bridge: Mf.DP.BNB.evrDelta
    verification = DeltaVerificationResult(
        delta_estimate=0.04,
        upper_confidence_bound=0.06,
        confidence_alpha=1e-4,
        accepted=True,
    )
    report = make_bnb_calibration_report(
        target_epsilon=1.0,
        target_delta=1e-5,
        noise_multiplier=1.7,
        num_samples=4096,
        seed=11,
        bands=2,
        verification=verification,
        evr_confidence_alpha_total=2e-4,
        evr_num_checks=4,
        evr_per_check_alpha=5e-5,
        evr_pass_count=4,
    )
    expected = _evr_delta(report.target_delta, report.evr_confidence_alpha_total)
    assert math.isclose(report.evr_composed_delta_upper_bound, expected, rel_tol=0, abs_tol=1e-18)
    assert 0.0 <= report.evr_composed_delta_upper_bound <= 1.0


def test_evr_split_union_bound_bridge_matches_runtime_fields() -> None:
    # Lean bridge: evrPerCheckAlpha/runtime split + evrDelta report composition.
    alpha_total = 0.12
    num_checks = 3
    candidate_sigmas = [0.6, 0.8, 1.0, 1.2]
    n_candidates = len(candidate_sigmas)

    def llr_samples_seq_fn(_sigma: float):
        # Always-pass synthetic verifier samples.
        return [torch.zeros((2048,), dtype=torch.float64) for _ in range(num_checks)]

    _, verification, pass_count, per_check_alpha = select_evr_candidate_ladder(
        candidate_sigmas=candidate_sigmas,
        llr_samples_seq_fn=llr_samples_seq_fn,
        epsilon=1.0,
        target_delta=0.2,
        total_confidence_alpha=alpha_total,
    )

    expected_per_check = alpha_total / float(n_candidates * num_checks)
    assert math.isclose(per_check_alpha, expected_per_check, rel_tol=0, abs_tol=1e-18)
    assert math.isclose(
        float(n_candidates * num_checks) * per_check_alpha,
        alpha_total,
        rel_tol=0,
        abs_tol=1e-15,
    )

    report = make_bnb_calibration_report(
        target_epsilon=1.0,
        target_delta=0.2,
        noise_multiplier=1.2,
        num_samples=2048,
        seed=7,
        bands=2,
        verification=verification,
        evr_confidence_alpha_total=alpha_total,
        evr_num_checks=num_checks,
        evr_per_check_alpha=per_check_alpha,
        evr_pass_count=pass_count,
    )
    assert math.isclose(report.evr_per_check_alpha, expected_per_check, rel_tol=0, abs_tol=1e-18)
    expected_delta = _evr_delta(report.target_delta, report.evr_confidence_alpha_total)
    assert math.isclose(report.evr_composed_delta_upper_bound, expected_delta, rel_tol=0, abs_tol=1e-18)


def test_b_min_sep_mixture_branch_sum_decomposition_bridge() -> None:
    # Bridge to Lean branch/suffix decomposition spirit:
    # modes are per-offset sums across the cycle axis.
    c = torch.tensor(
        [
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            [4.0, 40.0, 5.0, 50.0, 6.0, 60.0],
        ],
        dtype=torch.float64,
    )
    bands = 3
    gm = build_b_min_sep_gaussian_mixture(c_matrix=c, bands=bands)
    expected_modes = torch.stack(
        [c[:, offset::bands].sum(dim=1) for offset in range(bands)],
        dim=0,
    )
    assert torch.allclose(gm.modes, expected_modes)


def test_recurrence_consistency_zero_llr_when_p_equals_q_bridge() -> None:
    # Recurrence consistency bridge: if P == Q, log-likelihood ratio is exactly 0.
    gm = GaussianMixture(
        modes=torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float64),
        probs=torch.tensor([0.3, 0.7], dtype=torch.float64),
    )
    llr = compute_llr_samples(
        up_gm=gm,
        lo_gm=gm,
        sigma=1.1,
        num_samples=2048,
        generator=torch.Generator().manual_seed(20260213),
    )
    assert torch.max(torch.abs(llr)).item() < 1e-10
    # With epsilon >= 0 and llr=0, hockey-stick delta estimate should be 0.
    delta = estimate_hockey_stick_delta_from_llr_samples(epsilon=0.5, llr_samples=llr)
    assert delta == 0.0
