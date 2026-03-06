#!/usr/bin/env python3

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.optimizers import CorrelatedNoiseMechanism

import opacus.privacy_engine as pe_mod


def _loader(
    *, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260306)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def test_noise_mechanism_config_bisr_requires_bsr_accountant() -> None:
    with pytest.raises(ValueError, match="bisr mechanism requires bsr_accountant"):
        NoiseMechanismConfig(
            mechanism="bisr",
            accounting_mode="standard_step_accountant",
        )


def test_make_private_builds_bisr_noise_mechanism() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bisr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, -0.5], "z_std": 0.01},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert pe.noise_mechanism_config.mechanism == "bisr"


def test_make_private_with_epsilon_bisr_fixed_batch_resolves_mf_sensitivity(monkeypatch) -> None:
    captured = {}

    def _fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

    pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(batch_size=8),
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=10,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bisr",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 8},
        ),
    )

    assert captured["accountant"] == "bsr"
    assert float(captured["bsr_mf_sensitivity"]) > 0.0
    assert pe.noise_mechanism_config.mechanism_state["coeff_source"] == "analytical_auto"
    assert float(pe.noise_mechanism_config.mechanism_state["bsr_mf_sensitivity"]) > 0.0
