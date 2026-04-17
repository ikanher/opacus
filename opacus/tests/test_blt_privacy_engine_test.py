from __future__ import annotations

import copy
import io

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine
import opacus.accountants.bnb as bnb_accountant_mod
from opacus.accountants.analysis.blt import (
    BLTPairedParams,
    BLTParams,
    blt_pair_from_theta_pair,
)
from opacus.mechanism_contracts import SamplingSemantics
from opacus.noise_mechanisms import BufferedToeplitzNoiseMechanism
from opacus.optimizers import DistributedDPOptimizer
from opacus.accountants.blt_fixed_batch import optimize_blt_fixed_batch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import opacus.privacy_engine as pe_mod


def _loader(
    *, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260402)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def _run_pre_step(private_model: nn.Module, dp_optimizer, batch) -> None:
    x, y = batch
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    assert dp_optimizer.pre_step() is True


def _supported_fixed_batch_blt_state() -> dict:
    return {
        "theta": [0.8, 0.3],
        "theta_hat": [0.6, 0.1],
        "z_std": 0.03,
        "blt_max_participations": 2,
        "blt_min_separation": 4,
        "blt_horizon": 8,
    }


def _patch_distributed_primitives(monkeypatch, *, rank: int, world_size: int = 2) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)


def test_blt_canonical_exports_are_available() -> None:
    pair = blt_pair_from_theta_pair(theta=[0.8, 0.3], theta_hat=[0.6, 0.1])
    assert isinstance(pair, BLTPairedParams)
    assert isinstance(pair.forward, BLTParams)

    result = optimize_blt_fixed_batch(
        target_epsilon=4.0,
        target_delta=1e-5,
        total_steps=8,
        dataset_size=32,
        logical_batch_size=8,
        max_grad_norm=1.0,
        buffers=2,
    )
    assert result.mechanism_state["noise_multiplier_ref"] > 0.0


def test_make_private_builds_blt_from_decay_pair() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(20260403),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
            },
        ),
    )

    del private_model
    assert isinstance(dp_optimizer.noise_mechanism, BufferedToeplitzNoiseMechanism)
    assert pe.accountant.mechanism() == "blt_runtime_only"
    state = pe.noise_mechanism_config.mechanism_state
    assert pe.noise_mechanism_config.accounting_mode == "blt_accountant"
    assert {"forward", "inverse", "z_std"}.issubset(state.keys())
    assert state["_blt_distributed_policy"] == "ddp_flat_only"
    assert state["_blt_distributed_runtime"] is False
    assert set(state["forward"].keys()) == {"theta", "omega"}
    assert set(state["inverse"].keys()) == {"theta", "omega"}


def test_make_private_builds_blt_from_explicit_paired_params() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    _private_model, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(20260404),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "forward": {"theta": [0.8, 0.3], "omega": [0.13125, -0.03125]},
                "inverse": {"theta": [0.6, 0.1], "omega": [-0.5333333333333333, 0.03333333333333333]},
                "z_std": 0.03,
            },
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, BufferedToeplitzNoiseMechanism)
    state = pe.noise_mechanism_config.mechanism_state
    assert {"forward", "inverse", "z_std"}.issubset(state.keys())
    assert state["_blt_distributed_policy"] == "ddp_flat_only"
    assert state["_blt_distributed_runtime"] is False


def test_blt_checkpoint_roundtrip_preserves_future_sequence() -> None:
    base = nn.Linear(4, 3)
    model1 = copy.deepcopy(base)
    model2 = copy.deepcopy(base)

    def _make_private(model: nn.Module, *, seed: int):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        pe = PrivacyEngine()
        private_model, dp_optimizer, private_loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_generator=torch.Generator().manual_seed(seed),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="standard_step_accountant",
                mechanism_state={
                    "theta": [0.8, 0.3],
                    "theta_hat": [0.6, 0.1],
                    "z_std": 0.03,
                },
            ),
        )
        return private_model, dp_optimizer, private_loader, pe

    pmodel1, opt1, loader1, pe1 = _make_private(model1, seed=20260405)
    pmodel2, opt2, loader2, pe2 = _make_private(model2, seed=20260406)
    batches1 = list(loader1)
    batches2 = list(loader2)

    _run_pre_step(pmodel1, opt1, batches1[0])
    _run_pre_step(pmodel1, opt1, batches1[1])

    with io.BytesIO() as bio:
        pe1.save_checkpoint(path=bio, module=pmodel1, optimizer=opt1)
        bio.seek(0)
        pe2.load_checkpoint(path=bio, module=pmodel2, optimizer=opt2)

    _run_pre_step(pmodel1, opt1, batches1[2])
    _run_pre_step(pmodel2, opt2, batches2[2])

    grad1 = torch.cat([p.grad.reshape(-1) for p in opt1.params])
    grad2 = torch.cat([p.grad.reshape(-1) for p in opt2.params])
    assert torch.allclose(grad1, grad2, atol=1e-7, rtol=1e-6)
    assert isinstance(opt2.noise_mechanism, BufferedToeplitzNoiseMechanism)
    assert pe2.accountant.mechanism() == "blt_runtime_only"
    assert pe2.noise_mechanism_config.mechanism_state == pe1.noise_mechanism_config.mechanism_state


def test_make_private_with_epsilon_supports_default_torch_sampler_blt_contract() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=4.0,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
            },
        ),
    )

    assert pe.accountant.mechanism() == "blt"
    assert pe.noise_mechanism_config.accounting_mode == "blt_accountant"
    assert "noise_multiplier_ref" in pe.noise_mechanism_config.mechanism_state
    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))
    assert pe.get_epsilon(1e-5) > 0.0


def test_make_private_with_epsilon_supports_fixed_batch_blt() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=4.0,
        target_delta=1e-5,
        total_steps=8,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert pe.accountant.mechanism() == "blt"
    assert isinstance(dp_optimizer.noise_mechanism, BufferedToeplitzNoiseMechanism)
    assert "noise_multiplier_ref" in pe.noise_mechanism_config.mechanism_state
    assert pe.noise_mechanism_config.mechanism_state["z_std"] > 0.0

    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))
    eps = pe.get_epsilon(1e-5)
    assert eps > 0.0
    assert torch.isfinite(torch.tensor(eps))


def test_make_private_with_epsilon_supports_blt_balls_in_bins_bnb_accountant(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.25

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=4.0,
        target_delta=1e-5,
        total_steps=8,
        max_grad_norm=1.0,
        poisson_sampling=False,
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 8},
        ),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="bnb_accountant",
            mechanism_state={
                "theta": [0.8],
                "theta_hat": [0.6],
                "z_std": 0.03,
                "blt_min_separation": 4,
                "blt_horizon": 12,
            },
        ),
        bnb_num_samples=32,
    )

    assert isinstance(dp_optimizer.noise_mechanism, BufferedToeplitzNoiseMechanism)
    assert pe.accountant.mechanism() == "bnb"
    assert captured["accountant"] == "bnb"
    mechanism_state = captured["mechanism_state"]
    assert mechanism_state["blt_min_separation"] == 4
    assert mechanism_state["blt_horizon"] == 12
    assert captured["sampling_semantics"].sampling_mode == "balls_in_bins"
    final_state = pe.noise_mechanism_config.mechanism_state
    assert final_state["bnb_c_matrix"] is not None
    assert final_state["bnb_c_matrix_contract"] is not None
    assert final_state["bnb_bands"] == 4

    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))


def test_make_private_with_epsilon_rejects_unsupported_blt_coefficient_regime() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(
        ValueError,
        match="BLT target-epsilon calibration is only supported for the BLT fixed-batch or supported amplified BNB accountant contracts",
    ):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=4.0,
            target_delta=1e-5,
            total_steps=8,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="standard_step_accountant",
                mechanism_state={
                    "forward": {"theta": [0.8, 0.3], "omega": [0.1, -0.4]},
                    "inverse": {"theta": [0.6, 0.1], "omega": [-0.12, -0.28]},
                    "z_std": 0.03,
                    "blt_max_participations": 2,
                    "blt_min_separation": 4,
                    "blt_horizon": 8,
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )


def test_blt_accepts_random_allocation_accountant_mode() -> None:
    cfg = NoiseMechanismConfig(
        mechanism="blt",
        accounting_mode="random_allocation_accountant",
        mechanism_state={"theta": [0.8], "theta_hat": [0.6], "z_std": 0.03},
    )
    assert cfg.accounting_mode == "random_allocation_accountant"


def test_make_private_supports_blt_balls_in_bins_bnb_accountant(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_estimator(**kwargs):
        captured.update(kwargs)
        return 1.25

    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo_optimistic",
        _fake_estimator,
    )

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 8},
        ),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="bnb_accountant",
                mechanism_state={
                    "theta": [0.8],
                    "theta_hat": [0.6],
                    "z_std": 0.03,
                    "blt_min_separation": 4,
                    "blt_horizon": 12,
                },
            ),
        )

    assert isinstance(dp_optimizer.noise_mechanism, BufferedToeplitzNoiseMechanism)
    assert pe.accountant.mechanism() == "bnb"
    state = pe.noise_mechanism_config.mechanism_state
    assert state["bnb_accountant_coeffs_source"] == "normalized_forward_c_col"
    assert state["bnb_bands"] == 4
    assert state["bnb_cycle_length"] == 8
    assert "bnb_c_matrix" in state

    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))
    eps = pe.get_epsilon(1e-5, bnb_calibration_mode="optimistic", bnb_num_samples=32)
    assert eps == pytest.approx(1.25)
    assert captured["cycle_length"] == 8
    assert captured["horizon"] == state["bnb_c_matrix_contract"]["horizon"]


def test_make_private_rejects_blt_balls_in_bins_without_accountant_contract_inputs() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(ValueError, match="BLT balls_in_bins accounting requires"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 8},
            ),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "theta": [0.8],
                    "theta_hat": [0.6],
                    "z_std": 0.03,
                },
            ),
        )


def test_blt_pre_step_tracks_runtime_only_events_without_standard_history() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(20260407),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
            },
        ),
    )

    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))

    assert pe.accountant.mechanism() == "blt_runtime_only"
    assert len(pe.accountant) == 1
    assert getattr(pe.accountant, "history", []) == []


def test_get_epsilon_rejects_blt_runtime_only_boundary() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="BLT accountant support is unavailable for the current BLT contract",
    ):
        pe.get_epsilon(1e-5)


def test_blt_accounting_telemetry_reports_runtime_only_boundary() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
            },
        ),
    )

    telemetry = pe.get_accounting_telemetry(delta=1e-5)
    assert telemetry["mechanism"] == "blt"
    assert telemetry["accountant"] == "blt_runtime_only"
    assert telemetry["events_recorded"] == 0
    assert telemetry["accounting_supported"] is False
    assert telemetry["distributed_support"] == "ddp_flat_only"
    assert telemetry["distributed_runtime"] is False
    assert "BLT accountant support is unavailable for the current BLT contract" in telemetry["epsilon_at_target_delta_error"]


def test_supported_fixed_batch_blt_selects_fixed_batch_accountant() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        total_steps=8,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert pe.accountant.mechanism() == "blt"


def test_supported_fixed_batch_blt_get_epsilon_returns_finite() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        total_steps=8,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(20260408),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    batches = list(private_loader)
    _run_pre_step(private_model, dp_optimizer, batches[0])
    _run_pre_step(private_model, dp_optimizer, batches[1])

    eps = pe.get_epsilon(1e-5)
    assert eps > 0.0
    assert torch.isfinite(torch.tensor(eps))


def test_blt_poisson_sampling_stays_runtime_only() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        total_steps=8,
        poisson_sampling=True,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
    )

    assert pe.accountant.mechanism() == "blt_runtime_only"
    with pytest.raises(
        ValueError,
        match="BLT accountant support is unavailable for the current BLT contract",
    ):
        pe.get_epsilon(1e-5)


def test_blt_unsupported_coefficient_regime_stays_runtime_only() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        total_steps=8,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state={
                "forward": {"theta": [0.8, 0.3], "omega": [0.1, -0.4]},
                "inverse": {"theta": [0.6, 0.1], "omega": [-0.12, -0.28]},
                "z_std": 0.03,
                "blt_max_participations": 2,
                "blt_min_separation": 4,
                "blt_horizon": 8,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert pe.accountant.mechanism() == "blt_runtime_only"
    with pytest.raises(
        ValueError,
        match="BLT accountant support is unavailable for the current BLT contract",
    ):
        pe.get_epsilon(1e-5)


def test_make_private_with_epsilon_rejects_poisson_blt() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(
        ValueError,
        match="BLT target-epsilon calibration is only supported for the BLT fixed-batch or supported amplified BNB accountant contracts",
    ):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=4.0,
            target_delta=1e-5,
            total_steps=8,
            max_grad_norm=1.0,
            poisson_sampling=True,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="standard_step_accountant",
                mechanism_state=_supported_fixed_batch_blt_state(),
            ),
        )


def test_make_private_with_epsilon_fixed_batch_blt_checkpoint_roundtrip() -> None:
    base = nn.Linear(4, 3)
    model1 = copy.deepcopy(base)
    model2 = copy.deepcopy(base)

    def _make_private(model: nn.Module):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        pe = PrivacyEngine()
        private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=4.0,
            target_delta=1e-5,
            total_steps=8,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="standard_step_accountant",
                mechanism_state=_supported_fixed_batch_blt_state(),
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )
        return private_model, dp_optimizer, private_loader, pe

    pmodel1, opt1, loader1, pe1 = _make_private(model1)
    pmodel2, opt2, loader2, pe2 = _make_private(model2)
    batches1 = list(loader1)
    batches2 = list(loader2)

    _run_pre_step(pmodel1, opt1, batches1[0])
    _run_pre_step(pmodel1, opt1, batches1[1])

    with io.BytesIO() as bio:
        pe1.save_checkpoint(path=bio, module=pmodel1, optimizer=opt1)
        bio.seek(0)
        pe2.load_checkpoint(path=bio, module=pmodel2, optimizer=opt2)

    _run_pre_step(pmodel1, opt1, batches1[2])
    _run_pre_step(pmodel2, opt2, batches2[2])

    assert pe2.accountant.mechanism() == "blt"
    assert pe2.get_epsilon(1e-5) == pytest.approx(pe1.get_epsilon(1e-5), rel=0.0, abs=1e-12)


def test_fixed_batch_blt_checkpoint_roundtrip_preserves_accountant_and_epsilon() -> None:
    base = nn.Linear(4, 3)
    model1 = copy.deepcopy(base)
    model2 = copy.deepcopy(base)

    def _make_private(model: nn.Module, *, seed: int):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        pe = PrivacyEngine()
        private_model, dp_optimizer, private_loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            total_steps=8,
            poisson_sampling=False,
            noise_generator=torch.Generator().manual_seed(seed),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="standard_step_accountant",
                mechanism_state=_supported_fixed_batch_blt_state(),
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )
        return private_model, dp_optimizer, private_loader, pe

    pmodel1, opt1, loader1, pe1 = _make_private(model1, seed=20260409)
    pmodel2, opt2, loader2, pe2 = _make_private(model2, seed=20260410)
    batches1 = list(loader1)
    batches2 = list(loader2)

    _run_pre_step(pmodel1, opt1, batches1[0])
    _run_pre_step(pmodel1, opt1, batches1[1])

    with io.BytesIO() as bio:
        pe1.save_checkpoint(path=bio, module=pmodel1, optimizer=opt1)
        bio.seek(0)
        pe2.load_checkpoint(path=bio, module=pmodel2, optimizer=opt2)

    _run_pre_step(pmodel1, opt1, batches1[2])
    _run_pre_step(pmodel2, opt2, batches2[2])

    eps1 = pe1.get_epsilon(1e-5)
    eps2 = pe2.get_epsilon(1e-5)
    assert pe2.accountant.mechanism() == "blt"
    assert eps2 == pytest.approx(eps1, rel=0.0, abs=1e-12)


def test_fixed_batch_blt_accounting_telemetry_reports_accountant_backed_status() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        total_steps=8,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))

    telemetry = pe.get_accounting_telemetry(delta=1e-5)
    assert telemetry["mechanism"] == "blt"
    assert telemetry["accountant"] == "blt"
    assert telemetry["accounting_supported"] is True
    assert telemetry["epsilon_at_target_delta"] > 0.0
    assert telemetry["distributed_support"] == "ddp_flat_only"
    assert telemetry["distributed_runtime"] is False


def test_distributed_blt_supports_ddp_flat_hooks(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert isinstance(dp_optimizer, DistributedDPOptimizer)
    assert isinstance(dp_optimizer.noise_mechanism, BufferedToeplitzNoiseMechanism)
    assert private_loader.batch_size == 4
    state = pe.noise_mechanism_config.mechanism_state
    assert state["_blt_distributed_policy"] == "ddp_flat_only"
    assert state["_blt_distributed_runtime"] is True


def test_distributed_blt_rejects_explicit_runtime_mechanism(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(
        ValueError,
        match="distributed BLT requires noise_mechanism_config with mechanism='blt'",
    ):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_mechanism=BufferedToeplitzNoiseMechanism(
                pair=blt_pair_from_theta_pair(theta=[0.8, 0.3], theta_hat=[0.6, 0.1]),
                z_std=0.03,
            ),
        )


def test_distributed_blt_rejects_fsdp_early(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "FSDPModule", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(
        ValueError,
        match="blt noise mechanism is not yet supported with FSDP",
    ):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="blt",
                accounting_mode="standard_step_accountant",
                mechanism_state=_supported_fixed_batch_blt_state(),
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )


def test_distributed_blt_make_private_with_epsilon_supports_ddp_and_uses_global_logical_batch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=4.0,
        target_delta=1e-5,
        total_steps=8,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert isinstance(dp_optimizer, DistributedDPOptimizer)
    assert private_loader.batch_size == 4
    state = pe.noise_mechanism_config.mechanism_state
    assert state["_blt_distributed_runtime"] is True
    assert state["blt_min_separation"] == 4
    assert state["blt_max_participations"] == 2
    assert state["blt_horizon"] == 8
    assert state["noise_multiplier_ref"] > 0.0


def test_blt_checkpoint_load_rejects_missing_mechanism_state() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    _private_model, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(20260403),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="standard_step_accountant",
            mechanism_state=_supported_fixed_batch_blt_state(),
        ),
    )

    state = dp_optimizer.state_dict()
    del state["_dp_noise_mechanism_state"]

    with pytest.raises(ValueError, match="missing blt noise mechanism state"):
        dp_optimizer.load_state_dict(state)


def test_distributed_blt_balls_in_bins_bnb_reports_distributed_runtime(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_estimator(**kwargs):
        captured.update(kwargs)
        return 1.25

    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)
    monkeypatch.setattr(
        bnb_accountant_mod,
        "estimate_balls_in_bins_epsilon_monte_carlo_optimistic",
        _fake_estimator,
    )

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 8},
        ),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="bnb_accountant",
            mechanism_state={
                "theta": [0.8],
                "theta_hat": [0.6],
                "z_std": 0.03,
                "blt_min_separation": 4,
                "blt_horizon": 12,
            },
        ),
    )

    assert isinstance(dp_optimizer, DistributedDPOptimizer)
    state = pe.noise_mechanism_config.mechanism_state
    assert state["_blt_distributed_policy"] == "ddp_flat_only"
    assert state["_blt_distributed_runtime"] is True

    _run_pre_step(private_model, dp_optimizer, next(iter(private_loader)))
    eps = pe.get_epsilon(1e-5, bnb_calibration_mode="optimistic", bnb_num_samples=32)
    assert eps == pytest.approx(1.25)
    assert captured["distributed_dp_runtime"] is True
    assert captured["distributed_mode"] == "chunk_shard"
