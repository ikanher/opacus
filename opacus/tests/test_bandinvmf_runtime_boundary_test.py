from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.optimizers import CorrelatedNoiseMechanism, InverseBandNoiseMechanism


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


def test_bandinvmf_noise_mechanism_config_accepts_bsr_accountant() -> None:
    cfg = NoiseMechanismConfig(
        mechanism="bandinvmf",
        accounting_mode="bsr_accountant",
    )
    assert cfg.mechanism == "bandinvmf"
    assert cfg.accounting_mode == "bsr_accountant"


def test_bandinvmf_noise_mechanism_config_accepts_bnb_accountant() -> None:
    cfg = NoiseMechanismConfig(
        mechanism="bandinvmf",
        accounting_mode="bnb_accountant",
    )
    assert cfg.mechanism == "bandinvmf"
    assert cfg.accounting_mode == "bnb_accountant"


def test_bandinvmf_make_private_builds_inverse_band_noise_mechanism() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

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
    state = pe.noise_mechanism_config.mechanism_state
    assert state["_noise_mechanism"] == "bandinvmf"
    assert "bandinvmf_inv_coeffs" in state
    assert "coeffs" in state


def test_bandinvmf_is_not_silently_substituted_by_bisr() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

    pe.make_private(
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

    assert pe.noise_mechanism_config.mechanism == "bandinvmf"


def test_bandinvmf_legacy_coeffs_keep_correlated_compatibility() -> None:
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
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
