from __future__ import annotations

import numpy as np
import pytest

from opacus.accountants.analysis.blt import (
    BLTParams,
    BLTPairedParams,
    blt_coeff,
    blt_forward_coeffs_for_amplified_accounting,
    blt_materialize,
    blt_pair_from_theta_pair,
    build_blt_amplified_bnb_accountant_coeffs,
    build_blt_amplified_bnb_inputs,
)
from opacus.accountants.blt_inputs import resolve_blt_workload_mechanism_state
from opacus.mechanism_contracts import SamplingSemantics


def _dense_identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float64)


def test_blt_params_validate_rejects_invalid_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="same length"):
        BLTParams(theta=[0.5, 0.25], omega=[0.1]).validate()

    with pytest.raises(ValueError, match="1D"):
        BLTParams(theta=[[0.5], [0.25]], omega=[0.1, 0.2]).validate()

    with pytest.raises(ValueError, match="finite"):
        BLTParams(theta=[0.5, np.inf], omega=[0.1, 0.2]).validate()

    with pytest.raises(ValueError, match="finite"):
        BLTParams(theta=[0.5, 0.25], omega=[0.1, np.nan]).validate()


def test_blt_coeff_matches_closed_form_definition() -> None:
    params = BLTParams(theta=[0.75, 0.25], omega=[0.2, -0.1])
    params.validate()

    assert blt_coeff(params, 0) == pytest.approx(1.0)
    assert blt_coeff(params, 1) == pytest.approx(0.1)
    assert blt_coeff(params, 2) == pytest.approx(0.2 * 0.75 - 0.1 * 0.25)
    assert blt_coeff(params, 3) == pytest.approx(0.2 * 0.75**2 - 0.1 * 0.25**2)


def test_blt_materialize_is_lower_triangular_toeplitz() -> None:
    params = BLTParams(theta=[0.8, 0.4], omega=[0.15, -0.05])
    params.validate()

    mat = blt_materialize(params, n=6)
    assert mat.shape == (6, 6)
    assert np.allclose(np.triu(mat, k=1), 0.0)
    first_col = np.array([blt_coeff(params, lag) for lag in range(6)], dtype=np.float64)
    assert np.allclose(mat[:, 0], first_col)
    for lag in range(6):
        diag = np.diag(mat, k=-lag)
        assert np.allclose(diag, first_col[lag])


def test_blt_pair_from_theta_pair_gives_both_inverse_orders() -> None:
    pair = blt_pair_from_theta_pair(theta=[0.8, 0.3], theta_hat=[0.6, 0.1])
    pair.validate()

    forward = blt_materialize(pair.forward, n=10)
    inverse = blt_materialize(pair.inverse, n=10)

    assert np.allclose(forward @ inverse, _dense_identity(10), atol=1e-10, rtol=1e-8)
    assert np.allclose(inverse @ forward, _dense_identity(10), atol=1e-10, rtol=1e-8)


def test_blt_paired_params_validate_checks_both_sides() -> None:
    pair = BLTPairedParams(
        forward=BLTParams(theta=[0.8, 0.3], omega=[0.1, 0.2]),
        inverse=BLTParams(theta=[0.6], omega=[0.4]),
    )
    pair.validate()

    with pytest.raises(ValueError, match="same length"):
        BLTPairedParams(
            forward=BLTParams(theta=[0.8, 0.3], omega=[0.1]),
            inverse=BLTParams(theta=[0.6], omega=[0.4]),
        ).validate()


def test_blt_forward_coeffs_for_amplified_accounting_matches_forward_first_column() -> None:
    pair = blt_pair_from_theta_pair(theta=[0.8], theta_hat=[0.6])
    coeffs = blt_forward_coeffs_for_amplified_accounting(pair=pair, horizon=6)

    dense = blt_materialize(pair.forward, n=6)
    assert coeffs == pytest.approx(dense[:, 0].tolist())
    assert all(c >= 0.0 for c in coeffs)


def test_build_blt_amplified_bnb_accountant_coeffs_normalizes_forward_coeffs() -> None:
    pair = blt_pair_from_theta_pair(theta=[0.8], theta_hat=[0.6])
    coeffs, source = build_blt_amplified_bnb_accountant_coeffs(pair=pair, horizon=6)

    assert source == "normalized_forward_c_col"
    assert coeffs[0] > 0.0
    assert np.linalg.norm(np.asarray(coeffs, dtype=np.float64)) == pytest.approx(1.0)


def test_build_blt_amplified_bnb_accountant_coeffs_rejects_signed_forward_object() -> None:
    pair = BLTPairedParams(
        forward=BLTParams(theta=[0.8, 0.3], omega=[0.1, -0.5]),
        inverse=BLTParams(theta=[0.6, 0.1], omega=[0.2, 0.1]),
    )

    with pytest.raises(ValueError, match="nonnegative"):
        build_blt_amplified_bnb_accountant_coeffs(pair=pair, horizon=8)


def test_build_blt_amplified_bnb_inputs_returns_matrix_and_contract() -> None:
    pair = blt_pair_from_theta_pair(theta=[0.8], theta_hat=[0.6])
    resolved = build_blt_amplified_bnb_inputs(pair=pair, bands=4, horizon=6)

    assert resolved["bnb_accountant_coeffs_source"] == "normalized_forward_c_col"
    assert resolved["bnb_bands"] == 4
    assert resolved["bnb_horizon"] == 6
    assert resolved["bnb_c_matrix"].shape == (8, 8)
    assert resolved["bnb_c_matrix_contract"]["horizon"] == 6
    assert resolved["bnb_c_matrix_contract"]["padded_horizon"] == 8


def test_resolve_blt_workload_mechanism_state_fixed_batch_avoids_lambda_surface() -> None:
    state = resolve_blt_workload_mechanism_state(
        total_steps=8,
        dataset_size=32,
        logical_batch_size=8,
        max_grad_norm=1.0,
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
        buffers=2,
        noise_multiplier_ref=1.5,
    )

    assert state["blt_buffers"] == 2
    assert state["blt_selection_mode"] == "implicit_workload_default"
    assert state["blt_horizon"] == 8
    assert state["blt_min_separation"] == 4
    assert state["blt_max_participations"] == 2
    assert state["noise_multiplier_ref"] == pytest.approx(1.5)
    assert "forward" in state and "inverse" in state
    assert "lambda" not in state and "blt_lambda" not in state


def test_resolve_blt_workload_mechanism_state_bnb_contract_uses_sampling_semantics() -> None:
    state = resolve_blt_workload_mechanism_state(
        total_steps=8,
        dataset_size=32,
        logical_batch_size=8,
        max_grad_norm=1.0,
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 4},
        ),
        buffers=2,
        noise_multiplier_ref=1.25,
    )

    assert state["blt_buffers"] == 2
    assert state["blt_horizon"] == 8
    assert state["blt_selection_mode"] == "implicit_workload_default"
    assert state["noise_multiplier_ref"] == pytest.approx(1.25)
