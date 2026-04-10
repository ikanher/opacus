from __future__ import annotations

import pytest
import torch

from opacus.accountants.analysis.bifr import (
    bifr_exact_factor_recurrence_spectral_radius_from_inverse_coeffs,
    build_bifr_exact_factor_family_from_sgd_workload,
    build_bifr_amplified_bnb_inputs_from_factor_coeffs,
    compute_bifr_fixed_batch_sensitivity_from_sgd_workload,
    derive_bifr_amplified_accountant_coeffs_from_factor_coeffs,
    derive_bifr_factor_coeffs_from_inverse_coeffs,
    generate_bifr_inverse_coeffs_from_sgd_workload,
    resolve_bifr_exact_factor_coeffs_for_accounting,
)
from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload
from opacus.accountants.analysis.toeplitz_family import build_lower_toeplitz_matrix_from_coeffs


def _unscaled_workload_coeff(alpha: float, beta: float, j: int) -> float:
    total = 0.0
    for i in range(j + 1):
        total += (alpha ** (j - i)) * (beta**i)
    return total


def test_bifr_frac_zero_is_identity_factor_side() -> None:
    coeffs = derive_bifr_factor_coeffs_from_inverse_coeffs(
        coeffs=generate_bifr_inverse_coeffs_from_sgd_workload(
            bands=5,
            momentum=0.3,
            weight_decay=0.9,
            frac=0.0,
        ),
        steps=5,
    )
    assert coeffs == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_bifr_frac_half_matches_bsr_square_root_slice() -> None:
    got = derive_bifr_factor_coeffs_from_inverse_coeffs(
        coeffs=generate_bifr_inverse_coeffs_from_sgd_workload(
            bands=6,
            momentum=0.3,
            weight_decay=0.9,
            frac=0.5,
        ),
        steps=6,
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
    got = derive_bifr_factor_coeffs_from_inverse_coeffs(
        coeffs=generate_bifr_inverse_coeffs_from_sgd_workload(
            bands=6,
            momentum=beta,
            weight_decay=alpha,
            frac=1.0,
        ),
        steps=6,
    )
    expected = [_unscaled_workload_coeff(alpha, beta, j) for j in range(6)]
    assert got == pytest.approx(expected, abs=1e-12)


def test_bifr_fixed_batch_sensitivity_supports_endpoint_with_disjoint_fallback() -> None:
    got = compute_bifr_fixed_batch_sensitivity_from_sgd_workload(
        bands=4,
        steps=4,
        max_participations=10,
        min_separation=98,
        momentum=0.9,
        weight_decay=0.9999,
        frac=1.0,
        allow_disjoint_fallback=True,
    )
    assert got > 0.0


def test_bifr_family_keeps_exact_finite_horizon_boundary_explicit() -> None:
    family = build_bifr_exact_factor_family_from_sgd_workload(
        bands=4,
        steps=980,
        momentum=0.9,
        weight_decay=0.9999,
        frac=0.5,
    )
    assert family.source == "bifr"


def test_resolve_bifr_exact_factor_coeffs_for_accounting_accepts_explicit_factor_state() -> None:
    coeffs, source = resolve_bifr_exact_factor_coeffs_for_accounting(coeffs=[1.0, 0.2, 0.1])
    assert coeffs == pytest.approx([1.0, 0.2, 0.1])
    assert source == "explicit_exact_factor_c_col"


def test_derive_bifr_amplified_accountant_coeffs_returns_abs_factor_column() -> None:
    coeffs = derive_bifr_amplified_accountant_coeffs_from_factor_coeffs(
        coeffs=[1.0, -0.2, 0.05]
    )
    assert coeffs == pytest.approx([1.0, 0.2, 0.05])


def test_build_bifr_amplified_bnb_inputs_returns_matrix_and_contract() -> None:
    resolved = build_bifr_amplified_bnb_inputs_from_factor_coeffs(
        coeffs=[1.0, 0.2],
        bands=2,
        horizon=6,
    )
    assert resolved["bnb_accountant_coeffs_source"] == "abs_exact_factor_c_col"
    assert resolved["bnb_bands"] == 2
    assert resolved["bnb_horizon"] == 6
    assert tuple(resolved["bnb_c_matrix"].shape) == (6, 6)
    assert resolved["bnb_c_matrix_contract"]["bands"] == 2


def test_bifr_exact_factor_recovery_matches_dense_inverse_reference_on_representative_grid() -> None:
    steps = 128
    for bands in (2, 4, 8, 16):
        for frac in (0.0, 0.25, 0.5):
            inv_coeffs = generate_bifr_inverse_coeffs_from_sgd_workload(
                bands=bands,
                momentum=0.9,
                weight_decay=0.9999,
                frac=frac,
            )
            inverse_matrix = build_lower_toeplitz_matrix_from_coeffs(
                coeffs=inv_coeffs,
                steps=steps,
            )
            dense_inverse_first_col = torch.linalg.inv(inverse_matrix)[:, 0].tolist()
            solve_first_col = derive_bifr_factor_coeffs_from_inverse_coeffs(
                coeffs=inv_coeffs,
                steps=steps,
            )
            assert solve_first_col == pytest.approx(
                dense_inverse_first_col,
                rel=1e-10,
                abs=1e-10,
            )


def test_bifr_exact_factor_recurrence_radius_detects_unstable_p2_upper_half_slice() -> None:
    inv_coeffs = generate_bifr_inverse_coeffs_from_sgd_workload(
        bands=2,
        momentum=0.9,
        weight_decay=0.9999,
        frac=0.625,
    )
    radius = bifr_exact_factor_recurrence_spectral_radius_from_inverse_coeffs(
        coeffs=inv_coeffs,
    )
    assert radius > 1.0


def test_bifr_exact_factor_recovery_rejects_structurally_unstable_long_horizon_slice() -> None:
    inv_coeffs = generate_bifr_inverse_coeffs_from_sgd_workload(
        bands=2,
        momentum=0.9,
        weight_decay=0.9999,
        frac=0.625,
    )
    with pytest.raises(ValueError, match="unstable_exact_bifr_slice"):
        derive_bifr_factor_coeffs_from_inverse_coeffs(
            coeffs=inv_coeffs,
            steps=980,
        )
