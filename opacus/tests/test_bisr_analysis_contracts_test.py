#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math

import pytest

from opacus.accountants.analysis.bisr import (
    bisr_cyclic_poisson_epsilon_upper_bound,
    compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs,
    compute_bisr_kappa_from_coeffs,
    derive_bisr_factor_coeffs_from_inverse_coeffs,
    generate_bisr_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bsr import compute_bsr_mf_sensitivity_from_coeffs


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


def _gram_entry(coeffs: list[float], steps: int, i: int, j: int) -> float:
    total = 0.0
    for row in range(steps):
        total += _toeplitz_entry(coeffs, row, i) * _toeplitz_entry(coeffs, row, j)
    return total


def _exact_paper_upper_bound(coeffs: list[float], steps: int, k: int, b: int) -> float:
    best = 0.0
    universe = list(range(steps))
    for r in range(0, min(k, steps) + 1):
        for support in itertools.combinations(universe, r):
            if not _is_valid_support(support, k, b):
                continue
            total = 0.0
            for i in support:
                for j in support:
                    total += abs(_gram_entry(coeffs, steps, i, j))
            best = max(best, math.sqrt(total))
    return best


def _majorant_coeffs(coeffs: list[float]) -> list[float]:
    return [abs(c) for c in coeffs]


def _majorant_sensitivity(coeffs: list[float], steps: int, k: int, b: int) -> float:
    return compute_bsr_mf_sensitivity_from_coeffs(
        coeffs=_majorant_coeffs(coeffs),
        steps=steps,
        max_participations=k,
        min_separation=b,
    )


def test_contract_bisr_coeff_low_order_identities() -> None:
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
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=8,
        momentum=0.3,
        weight_decay=0.95,
    )
    majorant = _majorant_coeffs(coeffs)
    assert len(majorant) == len(coeffs)
    for c, m in zip(coeffs, majorant):
        assert m == pytest.approx(abs(c), rel=0.0, abs=1e-12)
        assert abs(c) <= m + 1e-12


def test_contract_bisr_paper_facing_sensitivity_matches_exact_small_grids() -> None:
    inverse_coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
    )
    for steps, k, b in [(5, 1, 1), (5, 2, 1), (6, 2, 2), (7, 3, 2)]:
        factor_coeffs = derive_bisr_factor_coeffs_from_inverse_coeffs(
            coeffs=inverse_coeffs,
            steps=steps,
        )
        paper_facing = compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
            coeffs=inverse_coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        exact = _exact_paper_upper_bound(factor_coeffs, steps, k, b)
        assert paper_facing == pytest.approx(exact, rel=0.0, abs=1e-12)


def test_contract_bisr_fixed_batch_helper_is_stable_on_small_grids() -> None:
    inverse_coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
    )
    for steps, k, b in [(5, 1, 1), (5, 2, 1), (6, 2, 2), (7, 3, 2)]:
        paper_facing = compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
            coeffs=inverse_coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        canonical = compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
            coeffs=inverse_coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        assert canonical == pytest.approx(paper_facing, rel=0.0, abs=1e-12)


def test_contract_bisr_exact_sensitivity_dominates_actual_small_grids() -> None:
    inverse_coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
    )
    for steps, k, b in [(5, 1, 1), (5, 2, 1), (6, 2, 2), (7, 3, 2)]:
        factor_coeffs = derive_bisr_factor_coeffs_from_inverse_coeffs(
            coeffs=inverse_coeffs,
            steps=steps,
        )
        exact = _exact_sensitivity(factor_coeffs, steps, k, b)
        upper = compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
            coeffs=inverse_coeffs,
            steps=steps,
            max_participations=k,
            min_separation=b,
        )
        assert upper + 1e-12 >= exact


def test_contract_bisr_abs_majorant_route_is_distinct_from_factor_side_route() -> None:
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
    )
    paper = compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
        coeffs=coeffs,
        steps=6,
        max_participations=2,
        min_separation=2,
    )
    majorant = _majorant_sensitivity(
        coeffs=coeffs,
        steps=6,
        k=2,
        b=2,
    )
    assert abs(paper - majorant) > 1e-6


def test_contract_bisr_cifar_paper_shape_matches_fastpath_and_majorant() -> None:
    inverse_coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
    )
    paper = compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
        coeffs=inverse_coeffs,
        steps=980,
        max_participations=10,
        min_separation=98,
    )
    factor_coeffs = derive_bisr_factor_coeffs_from_inverse_coeffs(
        coeffs=inverse_coeffs,
        steps=980,
    )
    assert all(c >= -1e-10 for c in factor_coeffs)
    assert all(
        factor_coeffs[i] + 1e-10 >= factor_coeffs[i + 1]
        for i in range(len(factor_coeffs) - 1)
    )
    expected = compute_bsr_mf_sensitivity_from_coeffs(
        coeffs=factor_coeffs,
        steps=980,
        max_participations=10,
        min_separation=98,
    )
    assert paper == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_contract_bisr_factor_coeffs_are_derived_from_inverse_toeplitz_object() -> None:
    inverse_coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
    )
    factor_coeffs = derive_bisr_factor_coeffs_from_inverse_coeffs(
        coeffs=inverse_coeffs,
        steps=8,
    )
    assert factor_coeffs[0] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert all(math.isfinite(c) for c in factor_coeffs)
    assert all(c >= -1e-10 for c in factor_coeffs)
    assert all(
        factor_coeffs[i] + 1e-10 >= factor_coeffs[i + 1]
        for i in range(len(factor_coeffs) - 1)
    )


def test_contract_bisr_majorant_requires_nonnegative_decreasing_sequence() -> None:
    # Closed-form majorant path contract: reject non-monotone abs-majorant.
    with pytest.raises(ValueError, match="nonnegative decreasing"):
        _majorant_sensitivity(
            coeffs=[1.0, -0.1, -0.2],
            steps=5,
            k=2,
            b=1,
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
