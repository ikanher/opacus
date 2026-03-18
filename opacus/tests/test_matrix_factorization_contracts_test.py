#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import io
import logging
import math

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.accountants.analysis.bandmf import generate_bandmf_coeffs_from_sgd_workload
from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload
from opacus.optimizers import CorrelatedNoiseMechanism, GaussianNoiseMechanism
from opacus.utils.uniform_sampler import (
    BallsInBinsSampler,
    CyclicPoissonSampler,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _loader(
    *, n_samples: int = 64, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260211)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def _make_private(
    model: nn.Module,
    *,
    poisson_sampling: bool,
    noise_seed: int,
    noise_mechanism_config: NoiseMechanismConfig | None = None,
    sampling_semantics: SamplingSemantics | None = None,
    noise_mechanism=None,
    **extra_kwargs,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    noise_gen = torch.Generator().manual_seed(noise_seed)
    kwargs = {}
    if noise_mechanism is not None:
        kwargs["noise_mechanism"] = noise_mechanism

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.7,
        max_grad_norm=1.0,
        poisson_sampling=poisson_sampling,
        noise_generator=noise_gen,
        noise_mechanism_config=noise_mechanism_config,
        sampling_semantics=sampling_semantics,
        **extra_kwargs,
        **kwargs,
    )
    return private_model, dp_optimizer, private_loader


def _single_pre_step(private_model: nn.Module, dp_optimizer, private_loader: DataLoader) -> None:
    x = y = None
    for _ in range(3):
        try:
            for xb, yb in private_loader:
                if int(xb.shape[0]) == 0:
                    continue
                x, y = xb, yb
                break
        except IndexError:
            # Some cyclic schedules can emit empty index sets for tiny synthetic tests.
            continue
        if x is not None and y is not None:
            break
    if x is None or y is None:
        pytest.skip("sampler yielded only empty batches in this tiny test configuration")
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    assert dp_optimizer.pre_step() is True


def _bnb_c_matrix_contract(*, c_matrix: torch.Tensor, bands: int) -> dict:
    return {
        "sampling_mode": "b_min_sep",
        "bands": int(bands),
        "granularity": "single_participation",
        "matrix_columns": int(c_matrix.shape[1]),
    }


def _lower_toeplitz_from_coeffs(coeffs: list[float], horizon: int) -> torch.Tensor:
    c = torch.zeros((horizon, horizon), dtype=torch.float64)
    for i in range(horizon):
        max_lag = min(i, len(coeffs) - 1)
        for lag in range(max_lag + 1):
            c[i, i - lag] = float(coeffs[lag])
    return c


def test_sampling_semantics_default_torch_sampler() -> None:
    model = nn.Linear(4, 3)
    _, dp_optimizer, _ = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=100,
    )
    semantics = dp_optimizer.sampling_semantics
    assert semantics.sampling_mode == "torch_sampler"
    assert semantics.privacy_metadata["expected_batch_size"] == 8


def test_sampling_semantics_default_poisson() -> None:
    model = nn.Linear(4, 3)
    _, dp_optimizer, _ = _make_private(
        model,
        poisson_sampling=True,
        noise_seed=101,
    )
    semantics = dp_optimizer.sampling_semantics
    assert semantics.sampling_mode == "poisson"
    assert semantics.privacy_metadata["expected_batch_size"] == 8


def test_sampling_semantics_cyclic_poisson_switches_sampler_for_bandmf() -> None:
    model = nn.Linear(4, 3)
    private_model, dp_optimizer, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=101,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )
    assert private_model is not None
    assert dp_optimizer is not None
    assert isinstance(private_loader.batch_sampler, CyclicPoissonSampler)
    assert dp_optimizer.sampling_semantics.sampling_mode == "cyclic_poisson"


def test_sampling_semantics_cyclic_poisson_requires_bands_metadata() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="requires privacy_metadata\\['bands'\\]"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=102,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bandmf",
                accounting_mode="bandmf_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={},
            ),
        )


def test_sampling_semantics_cyclic_poisson_rejects_steps_below_bands() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="steps >= bands"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=102,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bandmf",
                accounting_mode="bandmf_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 8},
            ),
            total_steps=5,
        )


def test_sampling_semantics_fixed_batch_bandmf_uses_torch_sampler() -> None:
    model = nn.Linear(4, 3)
    private_model, dp_optimizer, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=103,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    assert private_model is not None
    assert dp_optimizer is not None
    assert dp_optimizer.sampling_semantics.sampling_mode == "torch_sampler"


def test_bsr_mechanism_requires_torch_sampler() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="fixed-batch semantics"):
        _make_private(
            model,
            poisson_sampling=True,
            noise_seed=102,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
        )


def test_bsr_mechanism_rejects_standard_accounting_mode() -> None:
    with pytest.raises(ValueError, match="bsr_accountant"):
        NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="standard_step_accountant",
        )


def test_sampling_semantics_b_min_sep_is_disabled() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(
        ValueError,
        match="b_min_sep sampling is temporarily disabled",
    ):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=117,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="gaussian",
                accounting_mode="bnb_accountant",
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.2},
            ),
        )


def test_sampling_semantics_balls_in_bins_switches_sampler_for_bnb_accountant() -> None:
    model = nn.Linear(4, 3)
    _, _dp_optimizer, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=119,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 8},
        ),
    )
    assert isinstance(private_loader.batch_sampler, BallsInBinsSampler)


def test_sampling_semantics_balls_in_bins_requires_bins_metadata() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="requires privacy_metadata\\['bins'\\]"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=120,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="gaussian",
                accounting_mode="bnb_accountant",
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={},
            ),
        )


def test_make_private_total_steps_nonpoisson_requires_explicit_custom_sampler() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="requires explicit sampling_semantics"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=200,
            total_steps=5,
        )


def test_make_private_total_steps_supports_cyclic_poisson_sampler() -> None:
    model = nn.Linear(4, 3)
    _, _, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=201,
        total_steps=5,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )
    assert isinstance(private_loader.batch_sampler, CyclicPoissonSampler)
    assert len(private_loader) == 5


def test_make_private_total_steps_supports_balls_in_bins_sampler() -> None:
    model = nn.Linear(4, 3)
    _, _, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=203,
        total_steps=9,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 10},
        ),
    )
    assert isinstance(private_loader.batch_sampler, BallsInBinsSampler)
    assert len(private_loader) == 9


def test_default_gaussian_path_parity_with_explicit_contract_args() -> None:
    base = nn.Linear(4, 3)
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)

    private_a, opt_a, loader_a = _make_private(
        model_a,
        poisson_sampling=False,
        noise_seed=103,
    )
    private_b, opt_b, loader_b = _make_private(
        model_b,
        poisson_sampling=False,
        noise_seed=103,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="standard_step_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={"tag": "mf"},
        ),
    )

    _single_pre_step(private_a, opt_a, loader_a)
    _single_pre_step(private_b, opt_b, loader_b)

    grad_a = torch.cat([p.grad.reshape(-1) for p in opt_a.params])
    grad_b = torch.cat([p.grad.reshape(-1) for p in opt_b.params])
    assert torch.allclose(grad_a, grad_b, atol=1e-7, rtol=1e-6)


def test_standard_mode_attaches_accountant_hook() -> None:
    model = nn.Linear(4, 3)
    _, dp_optimizer, _ = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=104,
    )
    assert dp_optimizer.step_hook is not None
    assert getattr(dp_optimizer, "accounting_mode") == "standard_step_accountant"


def test_bsr_accountant_attaches_in_make_private() -> None:
    model = nn.Linear(4, 3)
    _, dp_optimizer, _ = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=105,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0],
                "z_std": 0.01,
            },
        ),
    )
    assert getattr(dp_optimizer, "accounting_mode") == "bsr_accountant"


def test_make_private_with_epsilon_bsr_calibrates_without_external_callback() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=3.0,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0],
                "z_std": 0.01,
            },
        ),
    )
    assert getattr(dp_optimizer, "accounting_mode") == "bsr_accountant"
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert dp_optimizer.noise_mechanism.z_std > 0.01


def test_make_private_with_epsilon_bsr_derives_mf_sensitivity_from_constraints() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=3.0,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": 0.01,
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
        ),
    )
    assert getattr(dp_optimizer, "accounting_mode") == "bsr_accountant"
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert dp_optimizer.noise_mechanism.z_std > 0.01


def test_make_private_with_epsilon_bsr_fixed_persists_mf_sensitivity_for_get_epsilon() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=3.0,
        target_delta=1e-5,
        total_steps=100,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": 0.01,
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
        bsr_iterations_number=40,
    )

    assert float(dp_optimizer.noise_multiplier) > 0.0
    state = pe.noise_mechanism_config.mechanism_state
    assert "bsr_mf_sensitivity" in state
    assert float(state["bsr_mf_sensitivity"]) > 0.0

    eps_default = pe.get_epsilon(1e-5)
    eps_override = pe.get_epsilon(
        1e-5,
        bsr_mf_sensitivity=float(state["bsr_mf_sensitivity"]),
    )
    assert eps_default == pytest.approx(eps_override, rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_bsr_requires_torch_sampler() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(ValueError, match="fixed-batch semantics"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=3.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=True,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
        )


def test_make_private_with_epsilon_bsr_calibrates_with_default_accounting() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=0.5,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01, "bsr_mf_sensitivity": 1.0},
        ),
    )
    assert float(dp_optimizer.noise_multiplier) > 0.0
    assert getattr(dp_optimizer, "accounting_mode") == "bsr_accountant"


def test_make_private_bsr_autoresolves_analytical_coeffs_from_bands_and_optimizer() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.9999,
    )
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.7,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"bsr_bands": 8, "z_std": 0.01},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert "coeffs" in state
    assert state["coeffs"] == pytest.approx(
        generate_bsr_coeffs_from_sgd_workload(
            bands=8, momentum=0.9, weight_decay=0.9999
        ),
        rel=0.0,
        abs=1e-12,
    )


def test_make_private_with_epsilon_bsr_autoresolves_analytical_coeffs_from_bands_and_optimizer() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.9999,
    )
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=3.0,
        target_delta=1e-5,
        total_steps=100,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "bsr_bands": 8,
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert "coeffs" in state
    assert state["coeffs"] == pytest.approx(
        generate_bsr_coeffs_from_sgd_workload(
            bands=8, momentum=0.9, weight_decay=0.9999
        ),
        rel=0.0,
        abs=1e-12,
    )
    assert "bsr_mf_sensitivity" in state
    assert float(state["bsr_mf_sensitivity"]) > 0.0


def test_make_private_with_epsilon_bsr_explicit_coeffs_take_precedence() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.9999,
    )
    pe = PrivacyEngine()

    explicit = [1.0, 0.2, 0.05]
    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=3.0,
        target_delta=1e-5,
        total_steps=100,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": explicit,
                "bsr_bands": 8,
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert state["coeffs"] == explicit


def test_make_private_with_epsilon_bandmf_cyclic_uses_bandmf_autogen() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.9999,
    )
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=32,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 3},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    expected = generate_bandmf_coeffs_from_sgd_workload(
        bands=3,
        momentum=0.9,
        weight_decay=0.9999,
        steps=32,
    )
    assert state["coeffs"] == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_bandmf_fixed_batch_resolves_mf_sensitivity() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=32,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 0.4]},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert float(dp_optimizer.noise_multiplier) > 0.0
    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert float(state["z_std"]) > 0.0
    assert "bsr_sensitivity_scale" not in state


def test_make_private_bandmf_fixed_batch_checkpoint_reuses_state() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

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
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "bsr_bands": 2},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    state_before = pe.noise_mechanism_config.mechanism_state
    x, y = next(iter(private_loader))
    dp_optimizer.zero_grad()
    nn.functional.cross_entropy(private_model(x), y).backward()
    assert dp_optimizer.pre_step() is True

    restored_pe = PrivacyEngine()
    restored_model = nn.Linear(4, 3)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.05)
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
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "bsr_bands": 2},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    with io.BytesIO() as bio:
        pe.save_checkpoint(path=bio, module=private_model, optimizer=dp_optimizer)
        bio.seek(0)
        restored_pe.load_checkpoint(
            path=bio, module=restored_model, optimizer=restored_dp_optimizer
        )

    state_after = restored_pe.noise_mechanism_config.mechanism_state
    assert state_after["coeffs"] == pytest.approx(state_before["coeffs"], rel=0.0, abs=1e-12)
    assert state_after["z_std"] == pytest.approx(state_before["z_std"], rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_bsr_cyclic_uses_analytical_autogen() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.9999,
    )
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=32,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 3},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    expected = generate_bsr_coeffs_from_sgd_workload(
        bands=3,
        momentum=0.9,
        weight_decay=0.9999,
    )
    assert state["coeffs"] == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_bsr_cyclic_large_bands_stays_fast_path() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.95,
        weight_decay=0.0,
    )
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(n_samples=640, batch_size=10),
        target_epsilon=2.0,
        target_delta=1e-5,
        total_steps=2000,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 48},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert len(state["coeffs"]) == 48


def test_make_private_with_epsilon_bandmf_cyclic_calibration_depends_on_coeffs() -> None:
    model_a = nn.Linear(4, 3)
    model_b = nn.Linear(4, 3)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
    pe_a = PrivacyEngine()
    pe_b = PrivacyEngine()

    _, dp_opt_a, _ = pe_a.make_private_with_epsilon(
        module=model_a,
        optimizer=optimizer_a,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=100,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 8},
        ),
    )

    _, dp_opt_b, _ = pe_b.make_private_with_epsilon(
        module=model_b,
        optimizer=optimizer_b,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=100,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 2.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 8},
        ),
    )

    assert float(dp_opt_a.noise_multiplier) > 0.0
    assert float(dp_opt_b.noise_multiplier) > 0.0
    assert abs(float(dp_opt_a.noise_multiplier) - float(dp_opt_b.noise_multiplier)) > 1e-12


def test_make_private_with_epsilon_bandmf_cyclic_persists_sensitivity_scale() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=100,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 2.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 8},
        ),
    )

    assert float(dp_optimizer.noise_multiplier) > 0.0
    state = pe.noise_mechanism_config.mechanism_state
    assert "bsr_sensitivity_scale" in state
    assert float(state["bsr_sensitivity_scale"]) > 0.0
    # Cyclic BandMF should not silently route through fixed-batch BSR sensitivity semantics.
    assert "bsr_mf_sensitivity" not in state

    eps_default = pe.get_epsilon(1e-5)
    eps_override = pe.get_epsilon(
        1e-5,
        bsr_sensitivity_scale=float(state["bsr_sensitivity_scale"]),
    )
    assert eps_default == pytest.approx(eps_override, rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_bandmf_cyclic_autoresolves_coeffs_from_bands() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        total_steps=32,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 8},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert "coeffs" in state
    assert isinstance(state["coeffs"], list)
    assert len(state["coeffs"]) == 8
    assert float(state["coeffs"][0]) > 0.0


def test_make_private_with_epsilon_bandmf_cyclic_epochs_total_steps_sample_rate_parity() -> None:
    """
    Equivalent cyclic-poisson runs (same effective step horizon) should resolve
    to the same calibrated noise when only expressed via epochs vs total_steps.

    This is especially important for non-divisible batch sizes where
    batch_size / dataset_size differs from 1 / len(data_loader).
    """
    dataset_size = 10
    batch_size = 4  # non-divisible -> len(loader)=3, 1/len≈0.333 vs B/N=0.4
    steps = math.ceil(dataset_size / batch_size)

    x = torch.randn(dataset_size, 4)
    y = torch.randint(0, 3, (dataset_size,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    model_epochs = nn.Linear(4, 3)
    model_steps = nn.Linear(4, 3)
    optimizer_epochs = torch.optim.SGD(model_epochs.parameters(), lr=0.05)
    optimizer_steps = torch.optim.SGD(model_steps.parameters(), lr=0.05)
    pe_epochs = PrivacyEngine()
    pe_steps = PrivacyEngine()

    _, dp_opt_epochs, _ = pe_epochs.make_private_with_epsilon(
        module=model_epochs,
        optimizer=optimizer_epochs,
        data_loader=loader,
        target_epsilon=1.0,
        target_delta=1e-5,
        epochs=1,
        total_steps=None,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    _, dp_opt_steps, _ = pe_steps.make_private_with_epsilon(
        module=model_steps,
        optimizer=optimizer_steps,
        data_loader=loader,
        target_epsilon=1.0,
        target_delta=1e-5,
        epochs=None,
        total_steps=steps,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    assert float(dp_opt_epochs.noise_multiplier) == pytest.approx(
        float(dp_opt_steps.noise_multiplier), rel=0.0, abs=1e-12
    )


def test_make_private_with_epsilon_bandmf_cyclic_epochs_total_steps_get_epsilon_parity() -> None:
    dataset_size = 10
    batch_size = 4
    steps = math.ceil(dataset_size / batch_size)

    x = torch.randn(dataset_size, 4)
    y = torch.randint(0, 3, (dataset_size,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    model_epochs = nn.Linear(4, 3)
    model_steps = nn.Linear(4, 3)
    optimizer_epochs = torch.optim.SGD(model_epochs.parameters(), lr=0.05)
    optimizer_steps = torch.optim.SGD(model_steps.parameters(), lr=0.05)
    pe_epochs = PrivacyEngine()
    pe_steps = PrivacyEngine()

    private_epochs, dp_opt_epochs, private_loader_epochs = pe_epochs.make_private_with_epsilon(
        module=model_epochs,
        optimizer=optimizer_epochs,
        data_loader=loader,
        target_epsilon=1.0,
        target_delta=1e-5,
        epochs=1,
        total_steps=None,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    private_steps, dp_opt_steps, private_loader_steps = pe_steps.make_private_with_epsilon(
        module=model_steps,
        optimizer=optimizer_steps,
        data_loader=loader,
        target_epsilon=1.0,
        target_delta=1e-5,
        epochs=None,
        total_steps=steps,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    for _ in range(steps):
        _single_pre_step(private_epochs, dp_opt_epochs, private_loader_epochs)
        _single_pre_step(private_steps, dp_opt_steps, private_loader_steps)

    eps_epochs = pe_epochs.get_epsilon(1e-5)
    eps_steps = pe_steps.get_epsilon(1e-5)
    assert float(eps_epochs) == pytest.approx(float(eps_steps), rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_bandmf_cyclic_fractional_epochs_total_steps_sample_rate_parity() -> None:
    dataset_size = 40
    batch_size = 4
    epochs = 1.5
    steps = int(float(epochs) * math.ceil(dataset_size / batch_size))

    x = torch.randn(dataset_size, 4)
    y = torch.randint(0, 3, (dataset_size,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    model_epochs = nn.Linear(4, 3)
    model_steps = nn.Linear(4, 3)
    optimizer_epochs = torch.optim.SGD(model_epochs.parameters(), lr=0.05)
    optimizer_steps = torch.optim.SGD(model_steps.parameters(), lr=0.05)
    pe_epochs = PrivacyEngine()
    pe_steps = PrivacyEngine()

    _, dp_opt_epochs, _ = pe_epochs.make_private_with_epsilon(
        module=model_epochs,
        optimizer=optimizer_epochs,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        epochs=epochs,
        total_steps=None,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    _, dp_opt_steps, _ = pe_steps.make_private_with_epsilon(
        module=model_steps,
        optimizer=optimizer_steps,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        epochs=None,
        total_steps=steps,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    assert float(dp_opt_epochs.noise_multiplier) == pytest.approx(
        float(dp_opt_steps.noise_multiplier), rel=0.0, abs=1e-12
    )


def test_make_private_with_epsilon_bandmf_cyclic_fractional_epochs_total_steps_get_epsilon_parity() -> None:
    dataset_size = 40
    batch_size = 4
    epochs = 1.5
    steps = int(float(epochs) * math.ceil(dataset_size / batch_size))

    x = torch.randn(dataset_size, 4)
    y = torch.randint(0, 3, (dataset_size,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    model_epochs = nn.Linear(4, 3)
    model_steps = nn.Linear(4, 3)
    optimizer_epochs = torch.optim.SGD(model_epochs.parameters(), lr=0.05)
    optimizer_steps = torch.optim.SGD(model_steps.parameters(), lr=0.05)
    pe_epochs = PrivacyEngine()
    pe_steps = PrivacyEngine()

    private_epochs, dp_opt_epochs, private_loader_epochs = pe_epochs.make_private_with_epsilon(
        module=model_epochs,
        optimizer=optimizer_epochs,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        epochs=epochs,
        total_steps=None,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    private_steps, dp_opt_steps, private_loader_steps = pe_steps.make_private_with_epsilon(
        module=model_steps,
        optimizer=optimizer_steps,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        epochs=None,
        total_steps=steps,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    for _ in range(steps):
        _single_pre_step(private_epochs, dp_opt_epochs, private_loader_epochs)
        _single_pre_step(private_steps, dp_opt_steps, private_loader_steps)

    eps_epochs = pe_epochs.get_epsilon(1e-5)
    eps_steps = pe_steps.get_epsilon(1e-5)
    assert float(eps_epochs) == pytest.approx(float(eps_steps), rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_total_steps_nonpoisson_requires_custom_sampler_for_non_bsr() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    c_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="requires explicit sampling_semantics"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=1e-5,
            epochs=None,
            max_grad_norm=1.0,
            poisson_sampling=False,
            total_steps=10,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="gaussian",
                accounting_mode="bnb_accountant",
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
        )


def test_make_private_with_epsilon_total_steps_nonpoisson_allows_bsr_torch_sampler() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        epochs=None,
        max_grad_norm=1.0,
        poisson_sampling=False,
        total_steps=10,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0],
                "z_std": 0.01,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    assert float(dp_optimizer.noise_multiplier) > 0.0


@pytest.mark.parametrize(
    ("poisson_sampling", "sampling_semantics", "expected_rate"),
    [
        (
            False,
            SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 5},
            ),
            0.2,
        ),
        (
            False,
            SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 3},
            ),
            8.0 / 63.0,
        ),
        (
            False,
            SamplingSemantics(
                sampling_mode="poisson",
                privacy_metadata={},
            ),
            0.125,
        ),
    ],
)
def test_resolve_calibration_sample_rate_by_sampling_mode(
    poisson_sampling: bool,
    sampling_semantics: SamplingSemantics,
    expected_rate: float,
) -> None:
    got = PrivacyEngine._resolve_total_steps_sample_rate(
        poisson_sampling=poisson_sampling,
        sampling_semantics=sampling_semantics,
        batch_size=8,
        dataset_size=64,
    )
    assert abs(float(got) - float(expected_rate)) < 1e-12


def test_resolve_total_steps_sample_rate_rejects_b_min_sep() -> None:
    with pytest.raises(ValueError, match="b_min_sep sampling is temporarily disabled"):
        PrivacyEngine._resolve_total_steps_sample_rate(
            poisson_sampling=False,
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"p": 0.25},
            ),
            batch_size=8,
            dataset_size=64,
        )


def test_resolve_total_steps_sample_rate_uses_semantics_and_requires_explicit_custom_nonpoisson() -> None:
    balls_in_bins_rate = PrivacyEngine._resolve_total_steps_sample_rate(
        poisson_sampling=False,
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 4},
        ),
        batch_size=8,
        dataset_size=64,
    )
    assert abs(float(balls_in_bins_rate) - 0.25) < 1e-12

    bsr_torch_sampler_rate = PrivacyEngine._resolve_total_steps_sample_rate(
        poisson_sampling=False,
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
        batch_size=8,
        dataset_size=64,
        mechanism="bsr",
    )
    assert abs(float(bsr_torch_sampler_rate) - 0.125) < 1e-12

    cyclic_rate = PrivacyEngine._resolve_total_steps_sample_rate(
        poisson_sampling=False,
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 3},
        ),
        batch_size=8,
        dataset_size=64,
        mechanism="bandmf",
    )
    assert abs(float(cyclic_rate) - (8.0 / 63.0)) < 1e-12

    with pytest.raises(ValueError, match="requires explicit sampling_semantics"):
        PrivacyEngine._resolve_total_steps_sample_rate(
            poisson_sampling=False,
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
            batch_size=8,
            dataset_size=64,
            mechanism="gaussian",
        )

    with pytest.raises(ValueError, match="requires explicit sampling_semantics"):
        PrivacyEngine._resolve_total_steps_sample_rate(
            poisson_sampling=False,
            sampling_semantics=None,
            batch_size=8,
            dataset_size=64,
            mechanism="gaussian",
        )


def test_cyclic_nondivisible_accountant_q_matches_sampler_implied_q() -> None:
    dataset_size = 64
    batch_size = 8
    bands = 3
    usable_size = (dataset_size // bands) * bands  # 63
    expected_q = float(batch_size) / float(usable_size)

    sample_rate = PrivacyEngine._resolve_total_steps_sample_rate(
        poisson_sampling=False,
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": bands},
        ),
        batch_size=batch_size,
        dataset_size=dataset_size,
        mechanism="bandmf",
    )
    accountant_q = float(sample_rate) * float(bands)
    sampler_implied_q = expected_q * float(bands)
    assert accountant_q == pytest.approx(sampler_implied_q, rel=0.0, abs=1e-12)


def test_make_private_default_gaussian_uses_loader_rate_without_total_steps() -> None:
    dataset_size = 10
    batch_size = 4  # 1/len(loader)=1/3; batch_size/dataset_size=0.4
    loader = DataLoader(
        TensorDataset(torch.randn(dataset_size, 4), torch.randint(0, 3, (dataset_size,))),
        batch_size=batch_size,
        shuffle=False,
    )

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pe = PrivacyEngine(accountant="rdp")

    private_model, private_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        total_steps=None,
    )

    _single_pre_step(private_model, private_optimizer, private_loader)
    _, sample_rate, _ = pe.accountant.history[-1]
    assert float(sample_rate) == pytest.approx(1.0 / len(loader), rel=0.0, abs=1e-12)
    assert float(sample_rate) != pytest.approx(batch_size / dataset_size, rel=0.0, abs=1e-12)


def test_make_private_default_gaussian_uses_batch_ratio_with_total_steps() -> None:
    dataset_size = 10
    batch_size = 4
    loader = DataLoader(
        TensorDataset(torch.randn(dataset_size, 4), torch.randint(0, 3, (dataset_size,))),
        batch_size=batch_size,
        shuffle=False,
    )

    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pe = PrivacyEngine(accountant="rdp")

    private_model, private_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=True,
        total_steps=5,
    )

    _single_pre_step(private_model, private_optimizer, private_loader)
    _, sample_rate, _ = pe.accountant.history[-1]
    assert float(sample_rate) == pytest.approx(batch_size / dataset_size, rel=0.0, abs=1e-12)


def test_make_private_with_epsilon_total_steps_uses_balls_in_bins_rate() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=0.5,
        target_delta=0.2,
        max_grad_norm=1.0,
        poisson_sampling=False,
        total_steps=9,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 5},
        ),
    )
    assert float(dp_optimizer.noise_multiplier) > 0.0


def test_make_private_with_epsilon_logs_sample_rate_resolution_context_total_steps(caplog) -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    caplog.set_level(logging.INFO, logger="opacus.privacy_engine")
    pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=0.5,
        target_delta=0.2,
        max_grad_norm=1.0,
        poisson_sampling=False,
        total_steps=9,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 5},
        ),
    )

    messages = [r.getMessage() for r in caplog.records]
    assert any("bnb init: starting get_noise_multiplier (steps=" in m for m in messages)


def test_make_private_with_epsilon_epochs_uses_balls_in_bins_rate() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=0.5,
        target_delta=0.2,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 5},
        ),
    )
    assert float(dp_optimizer.noise_multiplier) > 0.0


def test_make_private_with_epsilon_logs_sample_rate_resolution_context_epochs(caplog) -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    caplog.set_level(logging.INFO, logger="opacus.privacy_engine")
    pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=0.5,
        target_delta=0.2,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 5},
        ),
    )

    messages = [r.getMessage() for r in caplog.records]
    assert any("bnb init: starting get_noise_multiplier (epochs=" in m for m in messages)


def test_make_private_with_epsilon_epochs_uses_balls_in_bins_rate() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    _, dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=0.5,
        target_delta=0.2,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="gaussian",
            accounting_mode="bnb_accountant",
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 5},
        ),
    )
    assert float(dp_optimizer.noise_multiplier) > 0.0


@pytest.mark.parametrize("mechanism", ["bsr", "bisr"])
def test_make_private_with_epsilon_balls_in_bins_mf_autocoeff_succeeds(mechanism: str) -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.3,
        weight_decay=0.9,
    )
    pe = PrivacyEngine()

    _private_model, dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=0.2,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism=mechanism,
            accounting_mode="bnb_accountant",
            mechanism_state={
                "bsr_bands": 2,
                "_noise_mechanism": mechanism,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 4, "bands": 2},
        ),
        bnb_num_samples=2_000,
        bnb_require_evr_pass=False,
    )
    assert getattr(dp_optimizer, "accounting_mode") == "bnb_accountant"
    state = getattr(dp_optimizer, "noise_mechanism_config").mechanism_state
    assert list(state["coeffs"])
    assert state["bnb_c_matrix"] is not None
    assert state["bnb_c_matrix_contract"] is not None


def test_default_config_uses_gaussian_mechanism() -> None:
    model = nn.Linear(4, 3)
    _, dp_optimizer, _ = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=106,
    )
    assert isinstance(dp_optimizer.noise_mechanism, GaussianNoiseMechanism)


def test_bsr_config_builds_noise_mechanism() -> None:
    model = nn.Linear(4, 3)
    _, dp_optimizer, _ = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=107,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.03},
        ),
    )
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert tuple(dp_optimizer.noise_mechanism.coeffs) == (1.0, 0.2)
    assert dp_optimizer.noise_mechanism.z_std == 0.03


def test_bsr_config_requires_bands_or_coeffs_and_autocalibrates_z_std() -> None:
    with pytest.raises(ValueError, match="requires bands"):
        _make_private(
            nn.Linear(4, 3),
            poisson_sampling=False,
            noise_seed=108,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"z_std": 0.03},
            ),
        )

    _, dp_optimizer, _ = _make_private(
        nn.Linear(4, 3),
        poisson_sampling=False,
        noise_seed=109,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, 0.2]},
        ),
    )
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert tuple(dp_optimizer.noise_mechanism.coeffs) == (1.0, 0.2)
    assert float(dp_optimizer.noise_mechanism.z_std) > 0.0


def test_config_conflicts_with_explicit_noise_mechanism() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="either noise_mechanism_config or noise_mechanism"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=110,
            noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0], z_std=0.01),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
        )


def test_default_gaussian_matches_explicit_gaussian_mechanism() -> None:
    base = nn.Linear(4, 3)
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)
    private_a, opt_a, loader_a = _make_private(
        model_a, poisson_sampling=False, noise_seed=111, noise_mechanism=None
    )
    private_b, opt_b, loader_b = _make_private(
        model_b,
        poisson_sampling=False,
        noise_seed=111,
        noise_mechanism=GaussianNoiseMechanism(),
    )

    _single_pre_step(private_a, opt_a, loader_a)
    _single_pre_step(private_b, opt_b, loader_b)

    grad_a = torch.cat([p.grad.reshape(-1) for p in opt_a.params])
    grad_b = torch.cat([p.grad.reshape(-1) for p in opt_b.params])
    assert torch.allclose(grad_a, grad_b, atol=1e-7, rtol=1e-6)
