from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    bsr_fixed_batch_epsilon_upper_bound,
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
    generate_bsr_coeffs_from_sgd_workload,
)


pytest.importorskip("jax_privacy")


def _load_fixture() -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / "bsr_jax_golden_values.json"
    return json.loads(path.read_text())


def _coeff_sets() -> dict[str, list[float]]:
    fixture = _load_fixture()
    names = {case["coeffs"] for case in fixture["sensitivity_cases"]}
    coeffs_by_name = {}
    # Read one representative per name from stored cases by matching computed values.
    # The fixture stores values, not raw vectors, so we use stable known definitions here.
    coeffs_by_name["unit"] = [1.0]
    coeffs_by_name["short"] = [1.0, 0.5]
    coeffs_by_name["custom"] = [1.0, 0.8, 0.4, 0.1]
    coeffs_by_name["bsr_a0999_b09_p8"] = generate_bsr_coeffs_from_sgd_workload(
        bands=8,
        momentum=0.9,
        weight_decay=0.9999,
    )
    missing = names - set(coeffs_by_name.keys())
    if missing:
        raise AssertionError(f"Fixture contains unknown coeff sets: {sorted(missing)}")
    return coeffs_by_name


def test_bsr_sensitivity_matches_frozen_jax_goldens() -> None:
    fixture = _load_fixture()
    coeffs_by_name = _coeff_sets()

    for case in fixture["sensitivity_cases"]:
        got = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=coeffs_by_name[case["coeffs"]],
            steps=int(case["n"]),
            min_separation=int(case["b"]),
            max_participations=int(case["k"]),
        )
        assert math.isclose(
            float(got),
            float(case["jax_sensitivity"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ), case


def test_bsr_kappa_matches_frozen_jax_goldens() -> None:
    fixture = _load_fixture()
    coeffs_by_name = _coeff_sets()

    for case in fixture["kappa_cases"]:
        got = compute_bsr_kappa_from_coeffs(
            coeffs=coeffs_by_name[case["coeffs"]],
            steps=int(case["n"]),
        )
        assert math.isclose(
            float(got),
            float(case["jax_kappa"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ), case


def test_bsr_accounting_matches_frozen_goldens() -> None:
    fixture = _load_fixture()
    cases = {c["name"]: c for c in fixture["accounting_cases"]}

    fixed = cases["fixed_small_1"]
    fixed_eps = bsr_fixed_batch_epsilon_upper_bound(**fixed["params"])
    assert math.isclose(float(fixed_eps), float(fixed["epsilon"]), rel_tol=0.0, abs_tol=1e-12)

    cyclic = cases["cyclic_small_1"]
    cyclic_eps = bsr_cyclic_poisson_epsilon_upper_bound(**cyclic["params"])
    assert math.isclose(
        float(cyclic_eps), float(cyclic["epsilon"]), rel_tol=0.0, abs_tol=1e-12
    )

    boundary = cases["no_amplification_boundary_match"]
    cyclic_boundary = bsr_cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=boundary["params"]["noise_multiplier"],
        target_delta=boundary["params"]["target_delta"],
        steps=boundary["params"]["steps"],
        sample_rate=boundary["params"]["sample_rate"],
        bands=boundary["params"]["bands"],
    )
    fixed_boundary = bsr_fixed_batch_epsilon_upper_bound(
        noise_multiplier=boundary["params"]["noise_multiplier"],
        target_delta=boundary["params"]["target_delta"],
        mf_sensitivity=boundary["params"]["mf_sensitivity"],
    )
    assert math.isclose(
        float(cyclic_boundary),
        float(boundary["cyclic_epsilon"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(fixed_boundary),
        float(boundary["fixed_epsilon"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
