#!/usr/bin/env python3

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.accountants.analysis.bifr import (
    resolve_bifr_exact_factor_coeffs_for_accounting,
)
from opacus.accountants.bifr import (
    ensure_bifr_exact_runtime_coeffs,
    resolve_bifr_mf_sensitivity_for_fixed_batch,
)
from opacus.mf.bifr_family import BIFRFamily
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


@pytest.mark.parametrize("momentum", [0.0, 0.95])
@pytest.mark.parametrize("frac", [0.85, 0.95])
def test_make_private_with_epsilon_bifr_auto_state_uses_inverse_band_runtime(
    monkeypatch, momentum, frac
) -> None:
    monkeypatch.setattr(pe_mod, "get_noise_multiplier", lambda **kwargs: 1.0)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=momentum, weight_decay=0.0
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
        total_steps=440,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bifr",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 4, "bifr_frac": frac},
        ),
    )

    state = pe.noise_mechanism_config.mechanism_state
    # Independent prefix-sum binomial coefficients, without momentum workload.
    expected_inv = [
        1.0,
        -frac,
        frac * (frac - 1) / 2,
        -frac * (frac - 1) * (frac - 2) / 6,
    ]
    expected_factor, _source = resolve_bifr_exact_factor_coeffs_for_accounting(
        inverse_coeffs=expected_inv,
        steps=440,
    )

    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, InverseBandNoiseMechanism)
    assert state["bifr_inv_coeffs"] == pytest.approx(expected_inv, rel=0.0, abs=1e-12)
    assert state["coeffs"] == pytest.approx(expected_factor, rel=0.0, abs=1e-12)
    assert mechanism.inverse_coeffs == pytest.approx(expected_inv, rel=0.0, abs=1e-12)
    assert mechanism.coeffs == pytest.approx(expected_inv, rel=0.0, abs=1e-12)
    assert dp_optimizer.original_optimizer is optimizer
    assert optimizer.param_groups[0]["momentum"] == momentum

    # The runtime cache follows the inverse-side bandwidth, not the exact
    # finite-horizon factor column retained for accounting.
    assert len(state["bifr_inv_coeffs"]) == 4
    assert len(state["coeffs"]) == 440
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


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("frac", [0.85, 0.95])
def test_bifr_balls_in_bins_coefficients_ignore_optimizer_momentum(prepared, frac):
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.95)
    state = {"bsr_bands": 4, "bifr_frac": frac}
    if prepared:
        state = ensure_bifr_exact_runtime_coeffs(
            mechanism_config=NoiseMechanismConfig(
                mechanism="bifr", accounting_mode="bnb_accountant", mechanism_state=state
            ),
            optimizer=optimizer,
            sampling_semantics=None,
            kwargs={"total_steps": 440},
        ).mechanism_state

    expected_inv = [
        1.0,
        -frac,
        frac * (frac - 1) / 2,
        -frac * (frac - 1) * (frac - 2) / 6,
    ]
    # Dense inversion independently checks the column used by the accountant.
    inverse_matrix = torch.eye(440, dtype=torch.float64)
    for lag, coefficient in enumerate(expected_inv[1:], start=1):
        inverse_matrix += torch.diag(
            torch.full((440 - lag,), coefficient, dtype=torch.float64),
            diagonal=-lag,
        )
    expected_factor = torch.linalg.inv(inverse_matrix)[:, 0]
    resolved = BIFRFamily().resolve_balls_in_bins(
        mechanism_state=state,
        context={
            "optimizer": optimizer,
            "total_steps": 440,
            "metadata": {"bins": 44},
            "kwargs": {"momentum": 0.95, "weight_decay": 0.0},
        },
    )
    assert resolved["coeffs"] == pytest.approx(expected_factor.tolist())
    assert resolved["bnb_accountant_coeffs"] == pytest.approx(expected_factor.abs().tolist())
    if prepared:
        assert resolved["bifr_inv_coeffs"] == pytest.approx(expected_inv)
    assert optimizer.param_groups[0]["momentum"] == 0.95


def test_bifr_fixed_batch_fallback_ignores_momentum():
    sensitivities = [
        resolve_bifr_mf_sensitivity_for_fixed_batch(
            mechanism_state={"bsr_bands": 4, "bifr_frac": 0.95},
            sampling_semantics=None,
            steps=440,
            sample_rate=1 / 44,
            kwargs={"momentum": momentum, "weight_decay": 0.0},
        )
        for momentum in (0.0, 0.95)
    ]
    assert sensitivities[1] == pytest.approx(sensitivities[0])
