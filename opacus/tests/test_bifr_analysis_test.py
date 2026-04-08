from __future__ import annotations

import pytest

from opacus.accountants.analysis.bifr import (
    build_bifr_analytic_factor_family,
    compute_bifr_fixed_batch_sensitivity_from_sgd_workload,
    generate_bifr_factor_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload


def _unscaled_workload_coeff(alpha: float, beta: float, j: int) -> float:
    total = 0.0
    for i in range(j + 1):
        total += (alpha ** (j - i)) * (beta**i)
    return total


def test_bifr_frac_zero_is_identity_factor_side() -> None:
    coeffs = generate_bifr_factor_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.3,
        weight_decay=0.9,
        frac=0.0,
    )
    assert coeffs == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_bifr_frac_half_matches_bsr_square_root_slice() -> None:
    got = generate_bifr_factor_coeffs_from_sgd_workload(
        bands=6,
        momentum=0.3,
        weight_decay=0.9,
        frac=0.5,
    )
    expected = generate_bsr_coeffs_from_sgd_workload(
        bands=6,
        momentum=0.3,
        weight_decay=0.9,
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_bifr_frac_one_matches_full_workload_endpoint() -> None:
    alpha = 0.9
    beta = 0.3
    got = generate_bifr_factor_coeffs_from_sgd_workload(
        bands=6,
        momentum=beta,
        weight_decay=alpha,
        frac=1.0,
    )
    expected = [_unscaled_workload_coeff(alpha, beta, j) for j in range(6)]
    assert got == pytest.approx(expected, abs=1e-12)


def test_bifr_fixed_batch_sensitivity_supports_endpoint_with_disjoint_fallback() -> None:
    got = compute_bifr_fixed_batch_sensitivity_from_sgd_workload(
        bands=4,
        steps=980,
        max_participations=10,
        min_separation=98,
        momentum=0.9,
        weight_decay=0.9999,
        frac=1.0,
        allow_disjoint_fallback=True,
    )
    assert got > 0.0


def test_bifr_family_keeps_analysis_only_boundary_explicit() -> None:
    family = build_bifr_analytic_factor_family(
        bands=4,
        steps=980,
        momentum=0.9,
        weight_decay=0.9999,
        frac=0.5,
    )
    assert family.source == "bifr"
