#!/usr/bin/env python3

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.optimizers import CorrelatedNoiseMechanism, InverseBandNoiseMechanism

import opacus.privacy_engine as pe_mod


def _loader(*, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8) -> DataLoader:
    gen = torch.Generator().manual_seed(20260306)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def _resolve_kwargs_once(*, total_steps: int | None, epochs: float | None) -> dict:
    captured = {}

    def _fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)
    try:
        pe = PrivacyEngine()
        model = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=0.01,
        )
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8, n_samples=32),
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            total_steps=total_steps,
            epochs=epochs,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bisr",
                accounting_mode="bsr_accountant",
                mechanism_state={"bsr_bands": 8},
            ),
        )
    finally:
        monkeypatch.undo()
    return captured


def _cyclic_semantics(*, bands: int = 8) -> SamplingSemantics:
    return SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"bands": bands},
    )


def test_contract_bisr_fixed_runtime_calibration_tracks_extreme_noise_range(monkeypatch) -> None:
    target_to_noise = {8.0: 0.8, 4.0: 1.6, 1.0: 5.0, 0.5: 10.0, 0.1: 50.0}
    captured = []

    def _fake_get_noise_multiplier(*, target_epsilon, **kwargs):
        captured.append(dict(kwargs, target_epsilon=target_epsilon))
        return float(target_to_noise[target_epsilon])

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    z_stds = []
    for target_epsilon in [8.0, 4.0, 1.0, 0.5, 0.1]:
        pe = PrivacyEngine()
        model = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=0.01,
        )
        _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8, n_samples=32),
            target_epsilon=target_epsilon,
            target_delta=1e-5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            total_steps=64,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bisr",
                accounting_mode="bsr_accountant",
                mechanism_state={"bsr_bands": 8},
            ),
        )
        assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
        z_stds.append(float(dp_optimizer.noise_mechanism.z_std))

    assert len(captured) == 5
    assert all(float(call["bsr_mf_sensitivity"]) > 0.0 for call in captured)
    for prev, cur in zip(z_stds, z_stds[1:]):
        assert cur > prev


def test_contract_bisr_make_private_with_epsilon_rejects_steps_below_bands() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

    with pytest.raises(ValueError, match="requires steps >= bands"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8, n_samples=32),
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            total_steps=4,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bisr",
                accounting_mode="bsr_accountant",
                mechanism_state={"bsr_bands": 8},
            ),
        )


def test_contract_bisr_make_private_with_epsilon_rejects_epochs_below_bands() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

    with pytest.raises(ValueError, match="requires steps >= bands"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8, n_samples=32),
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            epochs=1.0,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bisr",
                accounting_mode="bsr_accountant",
                mechanism_state={"bsr_bands": 8},
            ),
        )


def test_contract_bisr_make_private_with_epsilon_calibration_kwargs_are_deterministic() -> None:
    run_a = _resolve_kwargs_once(total_steps=8, epochs=None)
    run_b = _resolve_kwargs_once(total_steps=8, epochs=None)
    assert copy.deepcopy(run_a) == copy.deepcopy(run_b)
    assert run_a["accountant"] == "bsr"
    assert float(run_a["bsr_mf_sensitivity"]) > 0.0


def test_contract_bisr_make_private_with_epsilon_total_steps_epochs_parity() -> None:
    # len(loader)=4, so epochs=2.0 corresponds to total_steps=8.
    by_steps = _resolve_kwargs_once(total_steps=8, epochs=None)
    by_epochs = _resolve_kwargs_once(total_steps=None, epochs=2.0)

    keys = ["accountant", "sample_rate", "steps", "bsr_mf_sensitivity"]
    for key in keys:
        assert by_steps[key] == pytest.approx(by_epochs[key], rel=0.0, abs=1e-12)


def test_contract_bisr_make_private_with_epsilon_accepts_cyclic_poisson(monkeypatch) -> None:
    captured = {}

    def _fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

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
        total_steps=64,
        sampling_semantics=_cyclic_semantics(bands=8),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bisr",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 8},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, InverseBandNoiseMechanism)
    assert float(captured["bsr_sensitivity_scale"]) > 0.0
    assert captured.get("bsr_mf_sensitivity") is None
    assert float(pe.noise_mechanism_config.mechanism_state["bsr_sensitivity_scale"]) > 0.0


def test_contract_bisr_make_private_accepts_explicit_cyclic_state() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _private_model, dp_optimizer, _private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(batch_size=8, n_samples=80),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=64,
        sampling_semantics=_cyclic_semantics(bands=8),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bisr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0, -0.5, -0.125],
                "z_std": 0.01,
                "bsr_bands": 8,
                "bsr_sensitivity_scale": 1.0,
            },
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)


def test_contract_bisr_cyclic_rejects_missing_bands_metadata(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "get_noise_multiplier", lambda **kwargs: 1.0)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

    with pytest.raises(ValueError, match="requires privacy_metadata\\['bands'\\]"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8, n_samples=32),
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            total_steps=64,
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={},
            ),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bisr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, -0.5, -0.125]},
            ),
        )


def test_contract_bisr_cyclic_rejects_too_small_partition(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "get_noise_multiplier", lambda **kwargs: 1.0)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)

    with pytest.raises(ValueError, match="batch_size must be <= partition size"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8, n_samples=80),
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            total_steps=64,
            sampling_semantics=_cyclic_semantics(bands=11),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bisr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, -0.5, -0.125], "bsr_bands": 11},
            ),
        )
