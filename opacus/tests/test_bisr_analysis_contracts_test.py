#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math

import pytest

from opacus.accountants.analysis.bisr import (
    bisr_cyclic_poisson_epsilon_upper_bound,
    compute_bisr_abs_majorant_coeffs,
    compute_bisr_kappa_from_coeffs,
    compute_bisr_mf_sensitivity_upper_bound_from_coeffs,
    generate_bisr_coeffs_from_sgd_workload,
)


# Lean theorem mapping (spec -> test):
# - `Mf.DP.BISR.tildeCCoeff_zero/one/two`:
#     first coefficients match closed forms.
# - `Mf.DP.BISR.bisrInvC_entry_abs_le_bisrAbsC`:
#     entrywise abs-majorant inequality (`|c_j| <= abs(c_j)`).
# - `Mf.DP.Sensitivity.sensitivityUpperBound_bisrInvC_le_bisrAbsC`:
#     fixed-batch sensitivity bounded by abs-majorant sensitivity.
# - `Mf.DP.CyclicBandMF.bisr_finiteHorizonKappa_nonneg`:
#     cyclic `kappa(T)` is nonnegative and matches the finite-horizon prefix norm.
# - `Mf.DP.CyclicBandMF.cyclicReduction_contract`:
#     cyclic epsilon upper bound decreases as runtime noise increases.


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


def _exact_sensitivity(coeffs: list[float], steps: int, k: int, b: int) -> float:
    best = 0.0
    universe = list(range(steps))
    for r in range(0, min(k, steps) + 1):
        for support in itertools.combinations(universe, r):
            if not _is_valid_support(support, k, b):
                continue
            best = max(best, _participation_objective(coeffs, steps, support))
    return best


def test_contract_bisr_coeff_low_order_identities() -> None:
    # Lean: tildeCCoeff_zero/one/two
    alpha = 0.9
    beta = 0.4
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=3,
        momentum=beta,
        weight_decay=alpha,
    )
    assert coeffs[0] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert coeffs[1] == pytest.approx(-((alpha + beta) / 2.0), rel=0.0, abs=1e-12)
    assert coeffs[2] == pytest.approx(-(((alpha - beta) ** 2) / 8.0), rel=0.0, abs=1e-12)


def test_contract_bisr_majorant_is_abs_entrywise() -> None:
    # Lean: bisrInvC_entry_abs_le_bisrAbsC (entrywise majorant route).
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=8,
        momentum=0.3,
        weight_decay=0.95,
    )
    majorant = compute_bisr_abs_majorant_coeffs(coeffs=coeffs)
    assert len(majorant) == len(coeffs)
    for c, m in zip(coeffs, majorant):
        assert m == pytest.approx(abs(c), rel=0.0, abs=1e-12)
        assert abs(c) <= m + 1e-12


def test_contract_bisr_fixed_batch_upper_bound_dominates_exact_small_grids() -> None:
    # Lean: sensitivityUpperBound_bisrInvC_le_bisrAbsC (finite oracle sanity).
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
    )
    for steps, k, b in [(5, 1, 1), (5, 2, 1), (6, 2, 2), (7, 3, 2)]:
        exact = _exact_sensitivity(coeffs, steps, k, b)
        upper = compute_bisr_mf_sensitivity_upper_bound_from_coeffs(
            coeffs=coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        assert upper + 1e-12 >= exact


def test_contract_bisr_majorant_requires_nonnegative_decreasing_sequence() -> None:
    # Closed-form majorant path contract: reject non-monotone abs-majorant.
    with pytest.raises(ValueError, match="nonnegative decreasing"):
        compute_bisr_mf_sensitivity_upper_bound_from_coeffs(
            coeffs=[1.0, -0.1, -0.2],
            steps=5,
            max_participations=2,
            min_separation=1,
        )


def test_contract_bisr_cyclic_kappa_matches_prefix_norm() -> None:
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
    )
    expected = math.sqrt(sum(c * c for c in coeffs[:4]))
    got = compute_bisr_kappa_from_coeffs(coeffs=coeffs, steps=4)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert got >= 0.0


def test_contract_bisr_cyclic_epsilon_decreases_with_noise_multiplier() -> None:
    common = {
        "target_delta": 1e-5,
        "steps": 200,
        "sample_rate": 0.01,
        "bands": 10,
    }
    eps_small_noise = bisr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=0.8,
        **common,
    )
    eps_big_noise = bisr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=1.6,
        **common,
    )
    assert eps_big_noise < eps_small_noise


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        ({"bands": 0, "momentum": 0.1, "weight_decay": 0.9}, "bands must be >= 1"),
        ({"bands": 4, "momentum": -0.1, "weight_decay": 0.9}, "momentum must satisfy"),
        ({"bands": 4, "momentum": 1.0, "weight_decay": 0.9}, "momentum must satisfy"),
        ({"bands": 4, "momentum": 0.1, "weight_decay": -1.0}, "weight_decay must satisfy"),
    ],
)
def test_contract_bisr_coeff_generation_validation(kwargs: dict, err: str) -> None:
    with pytest.raises(ValueError, match=err):
        generate_bisr_coeffs_from_sgd_workload(**kwargs)
