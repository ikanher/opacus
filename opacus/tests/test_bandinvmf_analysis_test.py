from __future__ import annotations

import inspect
import math

import pytest
import torch

import opacus.accountants.analysis.bandinvmf as _bandinvmf_module
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


@pytest.fixture(autouse=True)
def _clear_bandinvmf_cache_between_tests():
    _bandinvmf_module.clear_bandinvmf_optimization_cache()
    yield
    _bandinvmf_module.clear_bandinvmf_optimization_cache()


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
    runtime_coeffs = derive_bandinvmf_runtime_coeffs_from_inv_coeffs(inv_coeffs=coeffs)
    factor_coeffs = derive_bandinvmf_factor_coeffs_from_inv_coeffs(
        inv_coeffs=coeffs,
        steps=kwargs["steps"],
    )
    assert all(math.isfinite(c) for c in coeffs)
    assert all(math.isfinite(c) for c in runtime_coeffs)
    assert all(math.isfinite(c) for c in factor_coeffs)
    assert math.isfinite(obj)


def test_bandinvmf_optimization_improves_no_momentum_no_decay_fast_proxy_row() -> None:
    init = generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=0.0,
    )
    init_obj = compute_bandinvmf_objective_from_inv_coeffs(
        inv_coeffs=init,
        steps=64,
        max_participations=2,
        min_separation=6,
        momentum=0.0,
        weight_decay=0.0,
    )
    coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=4,
        steps=64,
        max_participations=2,
        min_separation=6,
        momentum=0.0,
        weight_decay=0.0,
        optimizer_steps=20,
    )
    obj = compute_bandinvmf_objective_from_inv_coeffs(
        inv_coeffs=coeffs,
        steps=64,
        max_participations=2,
        min_separation=6,
        momentum=0.0,
        weight_decay=0.0,
    )
    assert obj < init_obj - 1e-6
    assert coeffs != pytest.approx(init, rel=0.0, abs=1e-9)


def test_bandinvmf_optimization_uses_jax_style_default_and_has_no_powell_hook() -> None:
    signature = inspect.signature(optimize_bandinvmf_inv_coeffs_for_sgd_workload)

    assert signature.parameters["optimizer_steps"].default == 1000
    assert "use_powell_refinement" not in signature.parameters
    assert not hasattr(_bandinvmf_module, "_powell_refine_candidate")
    assert not hasattr(_bandinvmf_module, "scipy_optimize")


def test_bandinvmf_optimization_cache_reuses_result_and_returns_copy(monkeypatch) -> None:
    _bandinvmf_module.clear_bandinvmf_optimization_cache()
    calls = {"count": 0}

    def _fake_uncached(**kwargs):
        del kwargs
        calls["count"] += 1
        return [1.0, -0.25]

    monkeypatch.setattr(
        _bandinvmf_module,
        "_optimize_bandinvmf_inv_coeffs_for_sgd_workload_uncached",
        _fake_uncached,
    )

    try:
        first = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
            bands=2,
            steps=7,
            max_participations=2,
            min_separation=3,
            momentum=0.0,
            weight_decay=0.0,
            optimizer_steps=1,
        )
        first[1] = -99.0
        second = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
            bands=2,
            steps=7,
            max_participations=2,
            min_separation=3,
            momentum=-0.0,
            weight_decay=0.0,
            optimizer_steps=1,
        )
    finally:
        _bandinvmf_module.clear_bandinvmf_optimization_cache()

    assert calls["count"] == 1
    assert second == pytest.approx([1.0, -0.25], rel=0.0, abs=1e-12)


def test_bandinvmf_optimization_handles_pretrained_cifar100_amplified_row() -> None:
    init = generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.9,
        weight_decay=0.9999,
    )
    init_obj = compute_bandinvmf_objective_from_inv_coeffs(
        inv_coeffs=init,
        steps=784,
        max_participations=8,
        min_separation=98,
        momentum=0.9,
        weight_decay=0.9999,
    )
    kwargs = {
        "bands": 5,
        "steps": 784,
        "max_participations": 8,
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
    assert obj <= init_obj + 1e-10


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


def test_bandinvmf_fixed_batch_sensitivity_bypasses_raw_bsr_monotonicity_guard() -> None:
    inv_coeffs = [1.0, -1.2, 0.7]
    runtime_coeffs = derive_bandinvmf_runtime_coeffs_from_inv_coeffs(inv_coeffs=inv_coeffs)

    with pytest.raises(
        ValueError,
        match="closed-form Toeplitz sensitivity requires nonnegative decreasing coefficients",
    ):
        compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=runtime_coeffs,
            steps=20,
            max_participations=3,
            min_separation=4,
        )

    sensitivity = compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=20,
        max_participations=3,
        min_separation=4,
    )

    assert math.isfinite(sensitivity)
    assert sensitivity > 0.0


def test_bandinvmf_fixed_batch_sensitivity_handles_pretrained_sun397_nonamplified_row() -> None:
    inv_coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=5,
        steps=1192,
        max_participations=8,
        min_separation=149,
        momentum=0.9,
        weight_decay=0.9999,
        optimizer_steps=20,
    )
    sensitivity = compute_bandinvmf_fixed_batch_sensitivity_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=1192,
        max_participations=8,
        min_separation=149,
    )
    assert math.isfinite(sensitivity)
    assert sensitivity > 0.0


def test_bandinvmf_optimization_raises_when_final_candidate_does_not_improve(monkeypatch) -> None:
    def _flat_objective(*, inv_coeffs, steps, max_participations, min_separation, momentum, weight_decay):
        del inv_coeffs, steps, max_participations, min_separation, momentum, weight_decay
        return 1.0

    monkeypatch.setattr(
        "opacus.accountants.analysis.bandinvmf.compute_bandinvmf_objective_from_inv_coeffs",
        _flat_objective,
    )

    coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=4,
        steps=12,
        max_participations=3,
        min_separation=2,
        momentum=0.3,
        weight_decay=0.9,
        optimizer_steps=1,
    )
    assert coeffs == pytest.approx(
        generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
            bands=4,
            momentum=0.3,
            weight_decay=0.9,
        ),
        rel=0.0,
        abs=1e-12,
    )


def test_bandinvmf_optimization_recovers_best_finite_candidate_when_final_candidate_is_nonfinite(
    monkeypatch,
) -> None:
    original_compute = compute_bandinvmf_objective_from_inv_coeffs
    state = {"calls": 0}

    def _scripted_objective(*, inv_coeffs, steps, max_participations, min_separation, momentum, weight_decay):
        state["calls"] += 1
        if state["calls"] == 1:
            return 10.0
        if state["calls"] == 2:
            return 5.0
        raise ValueError("synthetic non-finite final candidate")

    class _DummyLBFGS:
        def __init__(self, params, max_iter, line_search_fn):
            del max_iter, line_search_fn
            self._params = params

        def zero_grad(self, set_to_none=True):
            del set_to_none
            for param in self._params:
                param.grad = None

        def step(self, closure):
            closure()
            with torch.no_grad():
                self._params[0].copy_(torch.tensor([-0.5, 0.0, 0.0], dtype=self._params[0].dtype))
            closure()
            with torch.no_grad():
                self._params[0].fill_(float("nan"))

    monkeypatch.setattr(
        "opacus.accountants.analysis.bandinvmf.compute_bandinvmf_objective_from_inv_coeffs",
        _scripted_objective,
    )
    monkeypatch.setattr(
        "opacus.accountants.analysis.bandinvmf.torch.optim.LBFGS",
        _DummyLBFGS,
    )

    coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=4,
        steps=12,
        max_participations=3,
        min_separation=2,
        momentum=0.3,
        weight_decay=0.9,
        optimizer_steps=2,
    )

    assert coeffs == pytest.approx(
        generate_bandinvmf_init_inv_coeffs_from_sgd_workload(
            bands=4,
            momentum=0.3,
            weight_decay=0.9,
        ),
        rel=0.0,
        abs=1e-12,
    )
    assert math.isfinite(
        original_compute(
            inv_coeffs=coeffs,
            steps=12,
            max_participations=3,
            min_separation=2,
            momentum=0.3,
            weight_decay=0.9,
        )
    )


def test_bandinvmf_runtime_coeff_derivation_converts_floating_point_failures_to_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dot = _bandinvmf_module.np.dot

    def _boom(*args, **kwargs):
        del args, kwargs
        raise FloatingPointError("synthetic overflow")

    monkeypatch.setattr(_bandinvmf_module.np, "dot", _boom)

    with pytest.raises(
        ValueError,
        match="BandInvMF Toeplitz inversion produced non-finite runtime coefficients",
    ):
        derive_bandinvmf_runtime_coeffs_from_inv_coeffs(inv_coeffs=[1.0, -0.5, 0.1])

    monkeypatch.setattr(_bandinvmf_module.np, "dot", original_dot)


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
