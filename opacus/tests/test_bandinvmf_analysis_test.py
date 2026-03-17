from __future__ import annotations

import math

import pytest

from opacus.accountants.analysis.bandinvmf import (
    derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs,
    derive_bandinvmf_factor_coeffs_from_inv_coeffs,
    compute_bandinvmf_objective_from_inv_coeffs,
    compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs,
    derive_bandinvmf_runtime_coeffs_from_inv_coeffs,
    generate_bandinvmf_init_inv_coeffs_from_sgd_workload,
    optimize_bandinvmf_inv_coeffs_for_sgd_workload,
)
from opacus.accountants.analysis.bisr import (
    compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs,
    generate_bisr_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bsr import compute_bsr_mf_sensitivity_from_coeffs


def test_bandinvmf_init_matches_analytic_bisr_coeffs() -> None:
    kwargs = {"bands": 4, "momentum": 0.3, "weight_decay": 0.9}
    assert generate_bandinvmf_init_inv_coeffs_from_sgd_workload(**kwargs) == pytest.approx(
        generate_bisr_coeffs_from_sgd_workload(**kwargs),
        rel=0.0,
        abs=1e-12,
    )


def test_bandinvmf_amplified_accountant_coeffs_are_abs_factor_column() -> None:
    inv_coeffs = generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.3,
        weight_decay=0.9,
    )
    steps = 16
    factor_coeffs = derive_bandinvmf_factor_coeffs_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=steps,
    )
    accountant_coeffs = derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=steps,
    )
    assert len(accountant_coeffs) == steps
    assert accountant_coeffs == pytest.approx([abs(float(c)) for c in factor_coeffs], rel=0.0, abs=1e-12)
    assert accountant_coeffs[0] > 0.0


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


def test_bandinvmf_optimization_handles_cifar_like_search_instability() -> None:
    kwargs = {
        "bands": 4,
        "steps": 980,
        "max_participations": 10,
        "min_separation": 98,
        "momentum": 0.9,
        "weight_decay": 0.9999,
        "optimizer_steps": 20,
    }
    coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(**kwargs)
    obj = compute_bandinvmf_objective_from_inv_coeffs(
        inv_coeffs=coeffs,
        steps=kwargs["steps"],
        max_participations=kwargs["max_participations"],
        min_separation=kwargs["min_separation"],
        momentum=kwargs["momentum"],
        weight_decay=kwargs["weight_decay"],
    )
    assert all(math.isfinite(c) for c in coeffs)
    assert math.isfinite(obj)


def test_bandinvmf_fixed_batch_paper_sensitivity_matches_bisr_for_init() -> None:
    coeffs = generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
    )
    kwargs = {
        "steps": 980,
        "max_participations": 10,
        "min_separation": 98,
    }
    assert compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
        inv_coeffs=coeffs,
        **kwargs,
    ) == pytest.approx(
        compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
            coeffs=coeffs,
            **kwargs,
        ),
        rel=0.0,
        abs=1e-12,
    )


def test_bandinvmf_factor_side_fixed_batch_sensitivity_is_distinct_from_runtime_side() -> None:
    kwargs = {
        "bands": 4,
        "steps": 980,
        "max_participations": 10,
        "min_separation": 98,
        "momentum": 0.9,
        "weight_decay": 0.9999,
        "optimizer_steps": 20,
    }
    inv_coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(**kwargs)
    runtime_coeffs = derive_bandinvmf_runtime_coeffs_from_inv_coeffs(inv_coeffs=inv_coeffs)
    runtime_sensitivity = compute_bsr_mf_sensitivity_from_coeffs(
        coeffs=runtime_coeffs,
        steps=kwargs["steps"],
        max_participations=kwargs["max_participations"],
        min_separation=kwargs["min_separation"],
    )
    factor_side_sensitivity = compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=kwargs["steps"],
        max_participations=kwargs["max_participations"],
        min_separation=kwargs["min_separation"],
    )
    assert math.isfinite(runtime_sensitivity)
    assert math.isfinite(factor_side_sensitivity)
    assert factor_side_sensitivity > runtime_sensitivity


def test_bandinvmf_optimization_raises_when_final_candidate_does_not_improve(monkeypatch) -> None:
    def _flat_objective(*, inv_coeffs, steps, max_participations, min_separation, momentum, weight_decay):
        del inv_coeffs, steps, max_participations, min_separation, momentum, weight_decay
        return 1.0

    monkeypatch.setattr(
        "opacus.accountants.analysis.bandinvmf.compute_bandinvmf_objective_from_inv_coeffs",
        _flat_objective,
    )

    with pytest.raises(RuntimeError, match="did not improve over initialization"):
        optimize_bandinvmf_inv_coeffs_for_sgd_workload(
            bands=4,
            steps=12,
            max_participations=3,
            min_separation=2,
            momentum=0.3,
            weight_decay=0.9,
            optimizer_steps=1,
        )


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
