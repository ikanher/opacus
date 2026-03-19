import torch

import opacus.accountants.analysis.bnb as bnb_mod
from opacus.accountants.analysis.bnb import (
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

    monkeypatch.setattr(bnb_mod, "get_bnb_base_delta", _fail_base_delta)
    monkeypatch.setattr(bnb_mod, "sample_balls_in_bins_llr_chunks", _fake_chunks)

    epsilon = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
        coeffs=[1.0],
        cycle_length=1,
        horizon=4,
        noise_multiplier=1.0,
        target_delta=1e-5,
        num_samples=10,
    )
    assert float(epsilon) >= 0.0
