#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math

import pytest

from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    bsr_fixed_batch_epsilon_upper_bound,
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
    generate_bsr_coeffs_from_sgd_workload,
)


def _toeplitz_entry(coeffs: list[float], i: int, j: int) -> float:
    if j > i:
        return 0.0
    lag = i - j
    if lag >= len(coeffs):
        return 0.0
    return float(coeffs[lag])


def _participation_objective(coeffs: list[float], steps: int, support: tuple[int, ...]) -> float:
    total_sq = 0.0
    for i in range(steps):
        row_sum = 0.0
        for j in support:
            row_sum += _toeplitz_entry(coeffs, i, j)
        total_sq += row_sum * row_sum
    return math.sqrt(total_sq)


def _is_valid_support(support: tuple[int, ...], k: int, b: int) -> bool:
    if len(support) > k:
        return False
    for x, y in zip(support, support[1:]):
        if (y - x) < b:
            return False
    return True


def _exact_sensitivity_oracle(coeffs: list[float], steps: int, k: int, b: int) -> float:
    """
    Independent finite oracle:
    max over all supports S with |S|<=k and pairwise min-separation >= b.
    """
    best = 0.0
    universe = list(range(steps))
    for r in range(0, min(k, steps) + 1):
        for support in itertools.combinations(universe, r):
            if not _is_valid_support(support, k, b):
                continue
            best = max(best, _participation_objective(coeffs, steps, support))
    return best


def test_contract_fixed_batch_sensitivity_matches_exact_oracle_small_grids() -> None:
    # BSR-C02: fixed-batch sensitivity (BSR Eq. (10), Theorem 2) matches an
    # independent finite oracle over admissible participation supports.
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=1.0,
    )
    cases = [
        (5, 1, 1),
        (5, 2, 1),
        (6, 2, 2),
        (7, 3, 2),
    ]
    for steps, k, b in cases:
        got = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        exact = _exact_sensitivity_oracle(coeffs, steps, k, b)
        assert math.isclose(got, exact, rel_tol=1e-12, abs_tol=1e-12)


def test_contract_sensitivity_monotone_in_k() -> None:
    # BSR-C03: sensitivity is nondecreasing with max_participations k.
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.1,
        weight_decay=0.9,
    )
    steps, b = 10, 2
    prev = 0.0
    for k in range(1, 6):
        cur = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        assert cur + 1e-12 >= prev
        prev = cur


def test_contract_sensitivity_antitone_in_b() -> None:
    # BSR-C03: stronger min-separation (larger b) should not increase sensitivity.
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=1.0,
    )
    steps, k = 10, 4
    prev = None
    for b in range(1, 6):
        cur = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        if prev is not None:
            assert cur <= prev + 1e-12
        prev = cur


def test_contract_kappa_monotone_and_plateaus_after_visible_horizon() -> None:
    # BSR-C04: kappa(T)=||prefix(coeffs, min(T,bands))||_2 is monotone and
    # plateaus once T exceeds coefficient horizon.
    coeffs = [1.0, 0.5, 0.25, 0.125]
    values = [
        compute_bsr_kappa_from_coeffs(coeffs=coeffs, steps=t) for t in range(1, 10)
    ]
    for a, b in zip(values, values[1:]):
        assert b + 1e-12 >= a
    assert math.isclose(values[len(coeffs) - 1], values[-1], rel_tol=0.0, abs_tol=1e-12)


def test_contract_fixed_epsilon_decreases_with_noise_multiplier() -> None:
    # BSR-C05: fixed-batch epsilon upper bound decreases as noise increases.
    eps_small_noise = bsr_fixed_batch_epsilon_upper_bound(
        noise_multiplier=0.8,
        target_delta=1e-5,
        mf_sensitivity=1.5,
    )
    eps_big_noise = bsr_fixed_batch_epsilon_upper_bound(
        noise_multiplier=1.6,
        target_delta=1e-5,
        mf_sensitivity=1.5,
    )
    assert eps_big_noise < eps_small_noise


def test_contract_fixed_epsilon_decreases_for_extreme_noise_range() -> None:
    # BSR-C05: monotone decrease should hold across a wide practical range.
    multipliers = [0.8, 1.6, 5.0, 10.0, 50.0]
    eps_values = [
        bsr_fixed_batch_epsilon_upper_bound(
            noise_multiplier=nm,
            target_delta=1e-5,
            mf_sensitivity=1.5,
        )
        for nm in multipliers
    ]
    assert all(math.isfinite(e) and e >= 0.0 for e in eps_values)
    for prev, cur in zip(eps_values, eps_values[1:]):
        assert cur < prev


def test_contract_cyclic_epsilon_decreases_with_noise_multiplier() -> None:
    # BSR-C06: cyclic epsilon upper bound decreases as noise increases.
    common = {
        "target_delta": 1e-5,
        "steps": 200,
        "sample_rate": 0.01,
        "bands": 10,
    }
    eps_small_noise = bsr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=0.8, **common
    )
    eps_big_noise = bsr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=1.6, **common
    )
    assert eps_big_noise < eps_small_noise


def test_contract_cyclic_epsilon_decreases_for_extreme_noise_range() -> None:
    # BSR-C06: monotone decrease should hold up to very large multipliers.
    common = {
        "target_delta": 1e-5,
        "steps": 200,
        "sample_rate": 0.01,
        "bands": 10,
    }
    multipliers = [0.8, 1.6, 5.0, 10.0, 50.0]
    eps_values = [
        bsr_cyclic_poisson_epsilon_upper_bound(noise_multiplier=nm, **common)
        for nm in multipliers
    ]
    assert all(math.isfinite(e) and e >= 0.0 for e in eps_values)
    for prev, cur in zip(eps_values, eps_values[1:]):
        assert cur < prev


def test_contract_fixed_epsilon_increases_with_mf_sensitivity() -> None:
    # BSR-C05: with fixed runtime noise, larger sensitivity implies larger epsilon.
    eps_small_sens = bsr_fixed_batch_epsilon_upper_bound(
        noise_multiplier=1.2,
        target_delta=1e-5,
        mf_sensitivity=1.0,
    )
    eps_big_sens = bsr_fixed_batch_epsilon_upper_bound(
        noise_multiplier=1.2,
        target_delta=1e-5,
        mf_sensitivity=2.0,
    )
    assert eps_big_sens > eps_small_sens


def test_contract_cyclic_epsilon_increases_with_sensitivity_scale() -> None:
    # BSR-C06: cyclic branch uses noise_eff = noise / scale, so larger scale
    # should increase epsilon at fixed raw noise.
    common = {
        "target_delta": 1e-5,
        "steps": 200,
        "sample_rate": 0.01,
        "bands": 10,
    }
    raw_noise = 1.2
    eps_small_scale = bsr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=raw_noise / 1.0, **common
    )
    eps_big_scale = bsr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=raw_noise / 2.0, **common
    )
    assert eps_big_scale > eps_small_scale


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"noise_multiplier": 0.0, "target_delta": 1e-5, "mf_sensitivity": 1.0}, "noise_multiplier must be > 0"),
        ({"noise_multiplier": -1.0, "target_delta": 1e-5, "mf_sensitivity": 1.0}, "noise_multiplier must be > 0"),
        ({"noise_multiplier": 1.0, "target_delta": 0.0, "mf_sensitivity": 1.0}, "target_delta must be in"),
        ({"noise_multiplier": 1.0, "target_delta": 2.0, "mf_sensitivity": 1.0}, "target_delta must be in"),
        ({"noise_multiplier": 1.0, "target_delta": 1e-5, "mf_sensitivity": 0.0}, "mf_sensitivity must be > 0"),
    ],
)
def test_contract_fixed_epsilon_rejects_invalid_parameters(kwargs: dict, err: str) -> None:
    # BSR-C05: boundary/invalid parameter contract.
    with pytest.raises(ValueError, match=err):
        bsr_fixed_batch_epsilon_upper_bound(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"noise_multiplier": 0.0, "target_delta": 1e-5, "steps": 10, "sample_rate": 0.1, "bands": 2}, "noise_multiplier must be > 0"),
        ({"noise_multiplier": 1.0, "target_delta": 0.0, "steps": 10, "sample_rate": 0.1, "bands": 2}, "target_delta must be in"),
        ({"noise_multiplier": 1.0, "target_delta": 1e-5, "steps": -1, "sample_rate": 0.1, "bands": 2}, "steps must be >= 0"),
        ({"noise_multiplier": 1.0, "target_delta": 1e-5, "steps": 10, "sample_rate": 0.0, "bands": 2}, "sample_rate must be in"),
        ({"noise_multiplier": 1.0, "target_delta": 1e-5, "steps": 10, "sample_rate": 1.1, "bands": 2}, "sample_rate must be in"),
        ({"noise_multiplier": 1.0, "target_delta": 1e-5, "steps": 10, "sample_rate": 0.1, "bands": 0}, "bands must be > 0"),
        ({"noise_multiplier": 1.0, "target_delta": 1e-5, "steps": 10, "sample_rate": 0.6, "bands": 2}, "bands \\* sample_rate"),
    ],
)
def test_contract_cyclic_epsilon_rejects_invalid_parameters(kwargs: dict, err: str) -> None:
    # BSR-C06/BSR-C07: boundary/invalid parameter contract.
    with pytest.raises(ValueError, match=err):
        bsr_cyclic_poisson_epsilon_upper_bound(**kwargs)


def test_contract_cyclic_epsilon_zero_steps_is_zero() -> None:
    # BSR-C06: zero horizon should return zero privacy cost.
    eps = bsr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=1.0,
        target_delta=1e-5,
        steps=0,
        sample_rate=0.01,
        bands=10,
    )
    assert eps == pytest.approx(0.0, rel=0.0, abs=0.0)
