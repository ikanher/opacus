from __future__ import annotations

import pytest

from opacus.accountants.blt import BLTAccountant
from opacus.mechanism_contracts import SamplingSemantics
from opacus.accountants.blt_fixed_batch import optimize_blt_fixed_batch


def test_optimize_blt_fixed_batch_returns_canonical_paired_runtime_state() -> None:
    result = optimize_blt_fixed_batch(
        target_epsilon=4.0,
        target_delta=1e-5,
        total_steps=8,
        dataset_size=32,
        logical_batch_size=8,
        max_grad_norm=1.0,
        buffers=2,
    )

    state = result.mechanism_state
    assert set(state.keys()) == {
        "forward",
        "inverse",
        "z_std",
        "blt_horizon",
        "blt_min_separation",
        "blt_max_participations",
        "noise_multiplier_ref",
    }
    assert set(state["forward"].keys()) == {"theta", "omega"}
    assert set(state["inverse"].keys()) == {"theta", "omega"}
    assert state["blt_horizon"] == 8
    assert state["blt_min_separation"] == 4
    assert state["blt_max_participations"] == 2
    assert state["z_std"] > 0.0
    assert state["noise_multiplier_ref"] > 0.0
    assert result.candidate_count >= 1

    accountant = BLTAccountant()
    accountant.history = [(state["noise_multiplier_ref"], 1.0, 8)]
    epsilon = accountant.get_epsilon(
        1e-5,
        mechanism_state=state,
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    assert epsilon <= 4.05


def test_optimize_blt_fixed_batch_rejects_unsupported_sampling_contract() -> None:
    with pytest.raises(ValueError, match="fixed-batch torch_sampler contract only"):
        optimize_blt_fixed_batch(
            target_epsilon=4.0,
            target_delta=1e-5,
            total_steps=8,
            dataset_size=32,
            logical_batch_size=8,
            max_grad_norm=1.0,
            buffers=2,
            sampling_semantics=SamplingSemantics(
                sampling_mode="poisson",
                privacy_metadata={},
            ),
        )


def test_optimize_blt_fixed_batch_is_reproducible() -> None:
    kwargs = dict(
        target_epsilon=4.0,
        target_delta=1e-5,
        total_steps=8,
        dataset_size=32,
        logical_batch_size=8,
        max_grad_norm=1.0,
        buffers=2,
    )
    result_a = optimize_blt_fixed_batch(**kwargs)
    result_b = optimize_blt_fixed_batch(**kwargs)

    assert result_a.selected_candidate_index == result_b.selected_candidate_index
    assert result_a.selected_theta == result_b.selected_theta
    assert result_a.selected_theta_hat == result_b.selected_theta_hat
    assert result_a.mechanism_state == result_b.mechanism_state
    assert result_a.score == pytest.approx(result_b.score)
