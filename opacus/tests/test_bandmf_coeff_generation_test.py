from __future__ import annotations

import math

import numpy as np
import pytest

from opacus.accountants.analysis.bandmf import (
    build_bandmf_toeplitz_family_from_runtime_coeffs,
    compute_bandmf_fixed_batch_sensitivity_from_column_normalized_matrix,
    compute_bandmf_max_column_norm_from_column_normalized_matrix,
    compute_bandmf_mf_sensitivity_from_coeffs,
    compute_bandmf_objective_from_strategy,
    generate_bandmf_coeffs_from_sgd_workload,
    generate_bandmf_initial_strategy_coeffs,
    materialize_column_normalized_banded_bandmf_matrix,
    materialize_bandmf_toeplitz_matrix,
    normalize_bandmf_strategy_coeffs,
    optimize_bandmf_strategy_coeffs,
)


def test_generate_bandmf_initial_strategy_coeffs_is_normalized() -> None:
    coeffs = generate_bandmf_initial_strategy_coeffs(bands=4)
    assert math.isclose(
        float(np.linalg.norm(np.asarray(coeffs, dtype=np.float64))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert coeffs[0] > 0.0


def test_build_bandmf_toeplitz_family_preserves_shared_first_column_surface() -> None:
    family = build_bandmf_toeplitz_family_from_runtime_coeffs(
        coeffs=[2.0, 1.0, 0.5],
        steps=8,
    )
    assert family.source == "bandmf"
    assert math.isclose(
        float(np.linalg.norm(np.asarray(family.coeffs, dtype=np.float64))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_normalize_bandmf_strategy_coeffs_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="positive L2 norm"):
        normalize_bandmf_strategy_coeffs(coeffs=[0.0, 0.0])


def test_optimize_bandmf_strategy_coeffs_is_deterministic_and_non_worsening() -> None:
    init = generate_bandmf_initial_strategy_coeffs(bands=4)
    got1 = optimize_bandmf_strategy_coeffs(steps=32, bands=4, max_optimizer_steps=25)
    got2 = optimize_bandmf_strategy_coeffs(steps=32, bands=4, max_optimizer_steps=25)
    assert got1 == pytest.approx(got2, rel=0.0, abs=1e-12)

    init_obj = compute_bandmf_objective_from_strategy(coeffs=init, steps=32)
    got_obj = compute_bandmf_objective_from_strategy(coeffs=got1, steps=32)
    assert got_obj <= init_obj + 1e-12


def test_generate_bandmf_coeffs_from_sgd_workload_requires_steps() -> None:
    with pytest.raises(TypeError):
        generate_bandmf_coeffs_from_sgd_workload(  # type: ignore[call-arg]
            bands=4,
            momentum=0.9,
            weight_decay=0.9999,
        )


def test_generate_bandmf_coeffs_cifar_contract_has_expected_fixed_batch_sensitivity() -> None:
    coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        steps=980,
        max_optimizer_steps=50,
    )
    sensitivity = compute_bandmf_mf_sensitivity_from_coeffs(
        coeffs=coeffs,
        steps=980,
        max_participations=10,
        min_separation=98,
    )
    assert sensitivity == pytest.approx(math.sqrt(10.0))


def test_materialized_bandmf_toeplitz_matrix_has_truncated_boundary_columns() -> None:
    coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        steps=16,
        max_optimizer_steps=25,
    )
    matrix = materialize_bandmf_toeplitz_matrix(
        coeffs=coeffs,
        steps=16,
    )
    col_norms = np.linalg.norm(matrix, axis=0)
    assert col_norms[0] == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert col_norms[-1] < 1.0


def test_column_normalized_bandmf_matches_toeplitz_on_interior_columns() -> None:
    coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        steps=980,
        max_optimizer_steps=50,
    )
    toeplitz = materialize_bandmf_toeplitz_matrix(coeffs=coeffs, steps=980)
    normalized = materialize_column_normalized_banded_bandmf_matrix(
        coeffs=coeffs,
        steps=980,
    )
    assert normalized[:, 0] == pytest.approx(toeplitz[:, 0], rel=0.0, abs=1e-12)
    assert normalized[:, -1] != pytest.approx(toeplitz[:, -1], rel=1e-12, abs=1e-12)


def test_materialized_column_normalized_bandmf_matrix_has_unit_column_norms() -> None:
    coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        steps=16,
        max_optimizer_steps=25,
    )
    matrix = materialize_column_normalized_banded_bandmf_matrix(
        coeffs=coeffs,
        steps=16,
    )
    col_norms = np.linalg.norm(matrix, axis=0)
    assert compute_bandmf_max_column_norm_from_column_normalized_matrix(
        matrix=matrix
    ) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert np.allclose(col_norms, 1.0, rtol=0.0, atol=1e-12)


def test_column_normalized_bandmf_exact_sqrt_k_when_min_sep_ge_bands() -> None:
    coeffs = [1.0, 0.5, 0.25]
    matrix = materialize_column_normalized_banded_bandmf_matrix(coeffs=coeffs, steps=10)
    got = compute_bandmf_fixed_batch_sensitivity_from_column_normalized_matrix(
        matrix=matrix,
        max_participations=4,
        min_separation=3,
    )
    assert got == pytest.approx(math.sqrt(4.0), rel=0.0, abs=1e-12)


@pytest.mark.parametrize("steps,bands", [(32, 4), (64, 8)])
def test_generate_bandmf_coeffs_matches_jax_toeplitz_optimizer(
    steps: int, bands: int
) -> None:
    pytest.importorskip("jax_privacy")
    from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz

    got = generate_bandmf_coeffs_from_sgd_workload(
        bands=bands,
        momentum=0.9,
        weight_decay=0.9999,
        steps=steps,
        max_optimizer_steps=50,
    )
    expected = [float(x) for x in np.asarray(jax_toeplitz.optimize_banded_toeplitz(steps, bands))]
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("steps,bands", [(8, 3), (12, 4)])
def test_materialized_column_normalized_bandmf_matches_jax(
    steps: int, bands: int
) -> None:
    pytest.importorskip("jax_privacy")
    from jax_privacy.matrix_factorization.banded import ColumnNormalizedBanded

    coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=bands,
        momentum=0.9,
        weight_decay=0.9999,
        steps=steps,
        max_optimizer_steps=25,
    )
    got = materialize_column_normalized_banded_bandmf_matrix(
        coeffs=coeffs,
        steps=steps,
    )
    expected = np.asarray(
        ColumnNormalizedBanded.from_banded_toeplitz(steps, np.asarray(coeffs)).materialize()
    )
    assert got == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_jax_dropx1_candidate_has_non_equal_column_norms() -> None:
    pytest.importorskip("jax_privacy")
    from jax_privacy.matrix_factorization import dense as jax_dense

    strategy = np.asarray(
        jax_dense.optimize(20, epochs=4, bands=4, equal_norm=False, max_optimizer_steps=50)
    )
    col_norms = np.linalg.norm(strategy, axis=0)
    assert float(col_norms.max()) > float(col_norms.min()) + 1e-6
