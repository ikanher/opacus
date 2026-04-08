import torch

import opacus.accountants.analysis.bnb as bnb_analysis_mod
import opacus.accountants.bnb as bnb_accountant_mod
from opacus import SamplingSemantics
from opacus.accountants import BNBAccountant
from opacus.accountants.analysis.bnb import (
    build_bnb_toeplitz_c_matrix_and_contract,
    estimate_balls_in_bins_epsilon_monte_carlo_optimistic,
    resolve_bnb_calibration_kwargs,
)


def test_bnb_calibration_defaults_match_monte_carlo_reference() -> None:
    cfg = resolve_bnb_calibration_kwargs()
    assert cfg["bnb_calibration_mode"] == "evr"
    assert cfg["bnb_num_samples"] == 500_000
    assert cfg["bnb_seed"] == 154
    assert cfg["bnb_max_iterations"] == 1000
    assert cfg["bnb_tolerance"] == 1e-7
    assert cfg["bnb_confidence_alpha"] == 1e-6
    assert cfg["bnb_chunk_size"] is None
    assert cfg["bnb_num_workers"] == 0


def test_overrides_take_precedence() -> None:
    cfg = resolve_bnb_calibration_kwargs(
        overrides={
            "bnb_calibration_mode": "optimistic",
            "bnb_num_samples": 1234,
            "bnb_require_evr_pass": False,
            "bnb_chunk_size": 100,
        },
    )
    assert cfg["bnb_calibration_mode"] == "optimistic"
    assert cfg["bnb_num_samples"] == 1234
    assert cfg["bnb_require_evr_pass"] is False
    assert cfg["bnb_chunk_size"] == 100


def test_evr_mode_enables_acceptance_guard_by_default() -> None:
    cfg = resolve_bnb_calibration_kwargs(
        overrides={
            "bnb_calibration_mode": "evr",
        },
    )
    assert cfg["bnb_require_evr_pass"] is True


def test_optimistic_balls_in_bins_epsilon_does_not_use_base_delta(monkeypatch) -> None:
    def _fail_base_delta(*, num_samples: int, target_delta: float) -> float:
        del num_samples, target_delta
        raise AssertionError("optimistic estimator should not query base_delta")

    def _fake_chunks(**kwargs):
        del kwargs
        return [torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)]

    monkeypatch.setattr(bnb_analysis_mod, "get_bnb_base_delta", _fail_base_delta)
    monkeypatch.setattr(bnb_analysis_mod, "sample_balls_in_bins_llr_chunks", _fake_chunks)

    epsilon = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
        coeffs=[1.0],
        cycle_length=1,
        horizon=4,
        noise_multiplier=1.0,
        target_delta=1e-5,
        num_samples=10,
    )
    assert float(epsilon) >= 0.0


def _bnb_balls_in_bins_state() -> tuple[dict, SamplingSemantics]:
    c_matrix, contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=[1.0],
        bands=1,
        horizon=4,
    )
    state = {
        "bnb_c_matrix": c_matrix,
        "bnb_c_matrix_contract": contract,
        "bnb_accountant_coeffs": [1.0],
        "bnb_bands": 1,
        "bnb_cycle_length": 1,
    }
    semantics = SamplingSemantics(
        sampling_mode="balls_in_bins",
        privacy_metadata={"bands": 1, "bins": 1},
    )
    return state, semantics


def test_bnb_accountant_get_epsilon_optimistic_uses_optimistic_estimator(monkeypatch) -> None:
    state, semantics = _bnb_balls_in_bins_state()
    state["_bnb_accounting_kwargs"] = {
        "bnb_calibration_mode": "optimistic",
        "bnb_num_samples": 123,
        "bnb_seed": 7,
        "bnb_chunk_size": None,
        "bnb_num_workers": 0,
        "bnb_backend": "auto",
        "bnb_device": None,
        "bnb_distributed_mode": "none",
        "bnb_distributed_dp_runtime": False,
    }

    accountant = BNBAccountant()
    accountant.step(noise_multiplier=1.0, sample_rate=1.0)

    def _fail_evr(**kwargs):
        del kwargs
        raise AssertionError("optimistic get_epsilon should not use EVR estimator")

    def _optimistic(**kwargs):
        assert kwargs["num_samples"] == 123
        assert kwargs["seed"] == 7
        return 1.234

    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo",
        _fail_evr,
    )
    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo_optimistic",
        _optimistic,
    )

    epsilon = accountant.get_epsilon(
        1e-5,
        mechanism_state=state,
        sampling_semantics=semantics,
    )
    assert float(epsilon) == 1.234


def test_bnb_accountant_get_epsilon_evr_uses_base_delta_estimator(monkeypatch) -> None:
    state, semantics = _bnb_balls_in_bins_state()
    state["_bnb_accounting_kwargs"] = {
        "bnb_calibration_mode": "evr",
        "bnb_num_samples": 321,
        "bnb_seed": 17,
        "bnb_chunk_size": None,
        "bnb_num_workers": 0,
        "bnb_backend": "auto",
        "bnb_device": None,
        "bnb_distributed_mode": "none",
        "bnb_distributed_dp_runtime": False,
    }

    accountant = BNBAccountant()
    accountant.step(noise_multiplier=1.0, sample_rate=1.0)

    def _evr(**kwargs):
        assert kwargs["num_samples"] == 321
        assert kwargs["seed"] == 17
        return 2.345

    def _fail_optimistic(**kwargs):
        del kwargs
        raise AssertionError("EVR get_epsilon should not use optimistic estimator")

    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo",
        _evr,
    )
    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo_optimistic",
        _fail_optimistic,
    )

    epsilon = accountant.get_epsilon(
        1e-5,
        mechanism_state=state,
        sampling_semantics=semantics,
    )
    assert float(epsilon) == 2.345


def test_bnb_accountant_get_epsilon_direct_overrides_apply_together(monkeypatch) -> None:
    state, semantics = _bnb_balls_in_bins_state()
    state["_bnb_accounting_kwargs"] = {
        "bnb_calibration_mode": "evr",
        "bnb_num_samples": 321,
        "bnb_seed": 17,
        "bnb_chunk_size": None,
        "bnb_num_workers": 0,
        "bnb_backend": "auto",
        "bnb_device": None,
        "bnb_distributed_mode": "none",
        "bnb_distributed_dp_runtime": False,
    }

    accountant = BNBAccountant()
    accountant.step(noise_multiplier=1.0, sample_rate=1.0)

    def _fail_evr(**kwargs):
        del kwargs
        raise AssertionError("direct optimistic overrides should not use EVR estimator")

    def _optimistic(**kwargs):
        assert kwargs["num_samples"] == 123
        assert kwargs["seed"] == 7
        assert kwargs["chunk_size"] == 99
        assert kwargs["num_workers"] == 3
        assert kwargs["backend"] == "cpu"
        assert kwargs["device"] == "cpu"
        assert kwargs["distributed_mode"] == "chunk_shard"
        assert kwargs["distributed_dp_runtime"] is True
        return 3.456

    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo",
        _fail_evr,
    )
    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo_optimistic",
        _optimistic,
    )

    epsilon = accountant.get_epsilon(
        1e-5,
        mechanism_state=state,
        sampling_semantics=semantics,
        bnb_calibration_mode="optimistic",
        bnb_num_samples=123,
        bnb_seed=7,
        bnb_chunk_size=99,
        bnb_num_workers=3,
        bnb_backend="cpu",
        bnb_device="cpu",
        bnb_distributed_mode="chunk_shard",
        bnb_distributed_dp_runtime=True,
    )
    assert float(epsilon) == 3.456
