from __future__ import annotations

import math

import numpy as np
import pytest

from opacus.accountants.analysis.bandmf import (
    compute_bandmf_mf_sensitivity_from_coeffs,
    compute_bandmf_objective_from_strategy,
    generate_bandmf_coeffs_from_sgd_workload,
    generate_bandmf_initial_strategy_coeffs,
    generate_legacy_bandmf_placeholder_coeffs_from_sgd_workload,
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


def test_generate_bandmf_coeffs_cifar_contract_materially_differs_from_legacy_placeholder() -> None:
    new_coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        steps=980,
        max_optimizer_steps=50,
    )
    legacy_coeffs = generate_legacy_bandmf_placeholder_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
    )

    assert new_coeffs != pytest.approx(legacy_coeffs, rel=1e-4, abs=1e-4)

    new_objective = compute_bandmf_objective_from_strategy(coeffs=new_coeffs, steps=980)
    legacy_objective = compute_bandmf_objective_from_strategy(
        coeffs=legacy_coeffs,
        steps=980,
    )
    assert new_objective < legacy_objective

    new_sensitivity = compute_bandmf_mf_sensitivity_from_coeffs(
        coeffs=new_coeffs,
        steps=980,
        max_participations=10,
        min_separation=98,
    )
    legacy_sensitivity = compute_bandmf_mf_sensitivity_from_coeffs(
        coeffs=legacy_coeffs,
        steps=980,
        max_participations=10,
        min_separation=98,
    )
    assert new_sensitivity == pytest.approx(math.sqrt(10.0))
    assert legacy_sensitivity == pytest.approx(math.sqrt(10.0))


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
