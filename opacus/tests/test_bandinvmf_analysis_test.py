from __future__ import annotations

import pytest

from opacus.accountants.analysis.bandinvmf import (
    compute_bandinvmf_objective_from_inv_coeffs,
    generate_bandinvmf_init_inv_coeffs_from_sgd_workload,
    optimize_bandinvmf_inv_coeffs_for_sgd_workload,
)
from opacus.accountants.analysis.bisr import generate_bisr_coeffs_from_sgd_workload


def test_bandinvmf_init_matches_analytic_bisr_coeffs() -> None:
    kwargs = {"bands": 4, "momentum": 0.3, "weight_decay": 0.9}
    assert generate_bandinvmf_init_inv_coeffs_from_sgd_workload(**kwargs) == pytest.approx(
        generate_bisr_coeffs_from_sgd_workload(**kwargs),
        rel=0.0,
        abs=1e-12,
    )


def test_bandinvmf_objective_is_deterministic() -> None:
    coeffs = generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.3,
        weight_decay=0.9,
    )
    kwargs = {
        "steps": 12,
        "max_participations": 3,
        "min_separation": 2,
        "momentum": 0.3,
        "weight_decay": 0.9,
    }
    first = compute_bandinvmf_objective_from_inv_coeffs(inv_coeffs=coeffs, **kwargs)
    second = compute_bandinvmf_objective_from_inv_coeffs(inv_coeffs=coeffs, **kwargs)
    assert first == pytest.approx(second, rel=0.0, abs=1e-12)


def test_bandinvmf_optimization_is_deterministic_and_nonworsening() -> None:
    init = generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.3,
        weight_decay=0.9,
    )
    kwargs = {
        "steps": 12,
        "max_participations": 3,
        "min_separation": 2,
        "momentum": 0.3,
        "weight_decay": 0.9,
    }
    init_obj = compute_bandinvmf_objective_from_inv_coeffs(inv_coeffs=init, **kwargs)
    first = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=4,
        optimizer_steps=20,
        **kwargs,
    )
    second = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=4,
        optimizer_steps=20,
        **kwargs,
    )
    assert first == pytest.approx(second, rel=0.0, abs=1e-9)
    opt_obj = compute_bandinvmf_objective_from_inv_coeffs(inv_coeffs=first, **kwargs)
    assert opt_obj <= init_obj + 1e-10


@pytest.mark.parametrize(
    ("kwargs", "err"),
    [
        (
            {"bands": 0, "momentum": 0.3, "weight_decay": 0.9},
            "bands must be >= 1",
        ),
        (
            {
                "bands": 4,
                "momentum": 0.3,
                "weight_decay": 0.9,
                "steps": 0,
                "max_participations": 2,
                "min_separation": 1,
                "optimizer_steps": 20,
            },
            "steps must be >= 1",
        ),
        (
            {
                "bands": 4,
                "momentum": 0.3,
                "weight_decay": 0.9,
                "steps": 10,
                "max_participations": 2,
                "min_separation": 1,
                "optimizer_steps": 0,
            },
            "optimizer_steps must be >= 1",
        ),
    ],
)
def test_bandinvmf_validation(kwargs: dict, err: str) -> None:
    if "steps" in kwargs:
        with pytest.raises(ValueError, match=err):
            optimize_bandinvmf_inv_coeffs_for_sgd_workload(**kwargs)
    else:
        with pytest.raises(ValueError, match=err):
            generate_bandinvmf_init_inv_coeffs_from_sgd_workload(**kwargs)
