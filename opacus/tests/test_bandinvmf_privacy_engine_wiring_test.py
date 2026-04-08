#!/usr/bin/env python3

import io
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.optimizers import CorrelatedNoiseMechanism, InverseBandNoiseMechanism

import opacus.privacy_engine as pe_mod


def _loader(
    *, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260310)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def _bandinvmf_supported_fixed_batch_loader() -> DataLoader:
    return _loader(n_samples=160, batch_size=8)


def test_bandinvmf_make_private_generates_deterministic_runtime_state() -> None:
    def _run_once() -> dict:
        pe = PrivacyEngine()
        model = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
        )
        _private_model, dp_optimizer, _private_loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            total_steps=16,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bandinvmf",
                accounting_mode="bsr_accountant",
                mechanism_state={"bsr_bands": 4},
            ),
        )
        assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
        return pe.noise_mechanism_config.mechanism_state

    first = _run_once()
    second = _run_once()
    assert first["bsr_min_separation"] == 4
    assert first["bsr_max_participations"] == 4
    assert first["bandinvmf_inv_coeffs"] == pytest.approx(
        second["bandinvmf_inv_coeffs"], rel=0.0, abs=1e-12
    )
    assert first["coeffs"] == pytest.approx(second["coeffs"], rel=0.0, abs=1e-12)


def test_bandinvmf_make_private_with_epsilon_fixed_batch_passes_resolved_mf_sensitivity(
    monkeypatch,
) -> None:
    captured = {}

    def _fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
    )

    _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_bandinvmf_supported_fixed_batch_loader(),
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=80,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
    assert captured["accountant"] == "bsr"
    assert float(captured["bsr_mf_sensitivity"]) > 0.0
    assert "bsr_sensitivity_scale" not in captured
    state = pe.noise_mechanism_config.mechanism_state
    assert state["bsr_min_separation"] == 20
    assert state["bsr_max_participations"] == 4
    assert float(state["bsr_mf_sensitivity"]) > 0.0
    assert "bsr_sensitivity_scale" not in state
    assert "bandinvmf_inv_coeffs" in state


def test_bandinvmf_make_private_with_epsilon_cyclic_passes_resolved_scale(monkeypatch) -> None:
    captured = {}

    def _fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
    )

    semantics = SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"bands": 4},
    )
    _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(batch_size=8, n_samples=80),
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=16,
        sampling_semantics=semantics,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
    assert float(captured["bsr_sensitivity_scale"]) > 0.0
    state = pe.noise_mechanism_config.mechanism_state
    assert float(state["bsr_sensitivity_scale"]) > 0.0


def test_bandinvmf_balls_in_bins_auto_state_uses_bins_as_min_separation() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
    )

    semantics = SamplingSemantics(
        sampling_mode="balls_in_bins",
        privacy_metadata={"bands": 4, "bins": 20},
    )
    _private_model, dp_optimizer, _private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(n_samples=160, batch_size=8),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=16,
        sampling_semantics=semantics,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bnb_accountant",
            mechanism_state={"bsr_bands": 4},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
    state = pe.noise_mechanism_config.mechanism_state
    assert state["bsr_min_separation"] == 20
    assert state["bsr_bands"] == 4
    assert "bandinvmf_inv_coeffs" in state


def test_bandinvmf_fixed_batch_accountant_get_epsilon_uses_bandinvmf_resolver() -> None:
    pe = PrivacyEngine(accountant="bsr")
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
    )

    _private_model, _dp_optimizer, _private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_bandinvmf_supported_fixed_batch_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=80,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4},
        ),
    )

    pe.accountant.step(noise_multiplier=1.0, sample_rate=8 / 160)
    epsilon = pe.get_epsilon(delta=1e-5)
    assert math.isfinite(float(epsilon))
    assert float(epsilon) > 0.0
    state = pe.noise_mechanism_config.mechanism_state
    assert "bsr_sensitivity_scale" not in state


def test_bandinvmf_checkpoint_resume_preserves_generated_runtime_state() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
    )
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=16,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4},
        ),
    )
    state_before = pe.noise_mechanism_config.mechanism_state
    batch = next(iter(private_loader))
    x, y = batch
    dp_optimizer.zero_grad()
    nn.functional.cross_entropy(private_model(x), y).backward()
    assert dp_optimizer.pre_step() is True

    restored_pe = PrivacyEngine()
    restored_model = nn.Linear(4, 3)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01
    )
    restored_model, restored_dp_optimizer, _ = restored_pe.make_private(
        module=restored_model,
        optimizer=restored_optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=16,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4},
        ),
    )

    with io.BytesIO() as bio:
        pe.save_checkpoint(path=bio, module=private_model, optimizer=dp_optimizer)
        bio.seek(0)
        restored_pe.load_checkpoint(
            path=bio, module=restored_model, optimizer=restored_dp_optimizer
        )

    state_after = restored_pe.noise_mechanism_config.mechanism_state
    assert state_after["bandinvmf_inv_coeffs"] == pytest.approx(
        state_before["bandinvmf_inv_coeffs"], rel=0.0, abs=1e-12
    )
    assert state_after["coeffs"] == pytest.approx(
        state_before["coeffs"], rel=0.0, abs=1e-12
    )
    assert isinstance(restored_dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)


def test_bandinvmf_explicit_inverse_state_builds_inverse_band_mechanism() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _private_model, dp_optimizer, _private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=16,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandinvmf",
            accounting_mode="bsr_accountant",
            mechanism_state={"bandinvmf_inv_coeffs": [1.0, -0.2], "z_std": 0.01},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
