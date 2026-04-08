from __future__ import annotations

import math

import pytest

from opacus.accountants.analysis.toeplitz_family import (
    InverseSideToeplitzFamily,
    ToeplitzMechanismFamily,
    compute_disjoint_toeplitz_mf_sensitivity,
)


def test_toeplitz_family_fixed_batch_sensitivity_matches_closed_form() -> None:
    family = ToeplitzMechanismFamily(
        coeffs=[1.0, 0.5, 0.25],
        steps=8,
        source="bsr",
    )
    assert family.fixed_batch_sensitivity(
        max_participations=2,
        min_separation=3,
    ) == pytest.approx(math.sqrt(2.0) * math.sqrt(1.0 + 0.25 + 0.0625), abs=1e-12)


def test_toeplitz_family_disjoint_fallback_handles_non_monotone_coeffs() -> None:
    family = ToeplitzMechanismFamily(
        coeffs=[1.0, 2.0, 3.0],
        steps=10,
        source="bifr",
    )
    assert family.fixed_batch_sensitivity(
        max_participations=3,
        min_separation=3,
        allow_disjoint_fallback=True,
    ) == pytest.approx(
        compute_disjoint_toeplitz_mf_sensitivity(
            coeffs=[1.0, 2.0, 3.0],
            steps=10,
            max_participations=3,
            min_separation=3,
        ),
        abs=1e-12,
    )


def test_inverse_side_family_uses_analytic_override_before_numeric_fallback() -> None:
    family = InverseSideToeplitzFamily(
        inv_coeffs=[1.0, -0.25],
        steps=4,
        source="bifr",
        analytic_factor_override=lambda: [1.0, 0.25, 0.0625, 0.015625],
    )
    assert family.factor_coeffs() == pytest.approx([1.0, 0.25, 0.0625, 0.015625], abs=1e-12)


def test_inverse_side_family_numeric_fallback_recovers_lower_toeplitz_inverse() -> None:
    family = InverseSideToeplitzFamily(
        inv_coeffs=[1.0, -0.25],
        steps=4,
        source="bisr",
    )
    assert family.factor_coeffs() == pytest.approx([1.0, 0.25, 0.0625, 0.015625], abs=1e-12)
