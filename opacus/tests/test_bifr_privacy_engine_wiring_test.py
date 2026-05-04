#!/usr/bin/env python3

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.accountants.analysis.bifr import (
    generate_bifr_inverse_coeffs_from_sgd_workload,
    resolve_bifr_exact_factor_coeffs_for_accounting,
)
from opacus.optimizers import InverseBandNoiseMechanism

import opacus.privacy_engine as pe_mod


def _loader(
    *, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260504)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def test_make_private_with_epsilon_bifr_auto_state_uses_inverse_band_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pe_mod, "get_noise_multiplier", lambda **kwargs: 1.0)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.2, weight_decay=0.9
    )

    _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(batch_size=8),
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=12,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bifr",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4, "bifr_frac": 0.25},
        ),
    )

    state = pe.noise_mechanism_config.mechanism_state
    expected_inv = generate_bifr_inverse_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.2,
        weight_decay=0.9,
        frac=0.25,
    )
    expected_factor, _source = resolve_bifr_exact_factor_coeffs_for_accounting(
        inverse_coeffs=expected_inv,
        steps=12,
    )

    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, InverseBandNoiseMechanism)
    assert state["bifr_inv_coeffs"] == pytest.approx(expected_inv, rel=0.0, abs=1e-12)
    assert state["coeffs"] == pytest.approx(expected_factor, rel=0.0, abs=1e-12)
    assert mechanism.inverse_coeffs == pytest.approx(expected_inv, rel=0.0, abs=1e-12)
    assert mechanism.coeffs == pytest.approx(expected_inv, rel=0.0, abs=1e-12)

    # The runtime cache follows the inverse-side bandwidth, not the exact
    # finite-horizon factor column retained for accounting.
    assert len(state["bifr_inv_coeffs"]) == 4
    assert len(state["coeffs"]) == 12
    assert mechanism.bandwidth == 4
    assert mechanism.max_state_depth == 3
    assert mechanism.max_state_depth < len(state["coeffs"]) - 1


def test_make_private_rejects_bifr_coeffs_only_state() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="coeffs`-only states"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bifr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, 0.3], "z_std": 0.01},
            ),
        )
