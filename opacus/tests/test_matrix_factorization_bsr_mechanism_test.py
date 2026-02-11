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

import copy

import pytest
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.optimizers import CorrelatedNoiseMechanism
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _build_loader(
    *, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
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
    noise_seed: int,
    noise_multiplier: float,
    max_grad_norm: float,
    noise_mechanism,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    privacy_engine = PrivacyEngine()
    noise_gen = torch.Generator().manual_seed(noise_seed)
    private_model, dp_optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_build_loader(),
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=noise_gen,
        noise_mechanism=noise_mechanism,
    )
    return private_model, dp_optimizer, private_loader


def _one_pre_step(private_model: nn.Module, dp_optimizer, batch) -> None:
    x, y = batch
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    assert dp_optimizer.pre_step() is True


def test_identity_coeff_matches_gaussian_one_step() -> None:
    base = nn.Linear(4, 3)
    model_gauss = copy.deepcopy(base)
    model_corr = copy.deepcopy(base)
    noise_multiplier = 0.6
    max_grad_norm = 1.0
    batch_size = 8

    private_g, opt_g, loader_g = _make_private(
        model_gauss,
        noise_seed=41,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        noise_mechanism=None,
    )
    private_c, opt_c, loader_c = _make_private(
        model_corr,
        noise_seed=41,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        noise_mechanism=CorrelatedNoiseMechanism(
            coeffs=[1.0],
            z_std=noise_multiplier * max_grad_norm / batch_size,
        ),
    )

    _one_pre_step(private_g, opt_g, next(iter(loader_g)))
    _one_pre_step(private_c, opt_c, next(iter(loader_c)))

    grad_g = torch.cat([p.grad.reshape(-1) for p in opt_g.params])
    grad_c = torch.cat([p.grad.reshape(-1) for p in opt_c.params])
    assert torch.allclose(grad_g, grad_c, atol=1e-7, rtol=1e-6)


def test_history_depth_bounded_by_bandwidth() -> None:
    model = nn.Linear(4, 3)
    mechanism = CorrelatedNoiseMechanism(coeffs=[1.0, 0.3, -0.15, 0.05], z_std=0.04)
    private_model, dp_optimizer, private_loader = _make_private(
        model,
        noise_seed=42,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        noise_mechanism=mechanism,
    )
    assert mechanism.max_state_depth == 3

    for batch in private_loader:
        _one_pre_step(private_model, dp_optimizer, batch)
        assert mechanism.state_depth <= mechanism.max_state_depth

    assert mechanism.state_depth == mechanism.max_state_depth


def test_dtype_preservation_double() -> None:
    model = nn.Linear(4, 3).double()
    mechanism = CorrelatedNoiseMechanism(coeffs=[1.0, 0.2], z_std=0.03)
    private_model, dp_optimizer, private_loader = _make_private(
        model,
        noise_seed=43,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        noise_mechanism=mechanism,
    )
    x, y = next(iter(private_loader))
    _one_pre_step(private_model, dp_optimizer, (x.double(), y))

    for p in dp_optimizer.params:
        assert p.grad is not None
        assert p.grad.dtype == torch.float64


def test_deterministic_replay_with_seed() -> None:
    coeffs = [1.2, 0.35, -0.1]
    z_std = 0.02
    model1 = nn.Linear(4, 3)
    model2 = copy.deepcopy(model1)

    mech1 = CorrelatedNoiseMechanism(coeffs=coeffs, z_std=z_std)
    mech2 = CorrelatedNoiseMechanism(coeffs=coeffs, z_std=z_std)
    private1, opt1, loader1 = _make_private(
        model1,
        noise_seed=44,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        noise_mechanism=mech1,
    )
    private2, opt2, loader2 = _make_private(
        model2,
        noise_seed=44,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        noise_mechanism=mech2,
    )

    u_seq_1 = []
    u_seq_2 = []
    for batch1, batch2 in zip(loader1, loader2):
        _one_pre_step(private1, opt1, batch1)
        _one_pre_step(private2, opt2, batch2)
        assert mech1.last_flat_u is not None
        assert mech2.last_flat_u is not None
        u_seq_1.append(mech1.last_flat_u.detach().clone())
        u_seq_2.append(mech2.last_flat_u.detach().clone())

    assert len(u_seq_1) == len(u_seq_2)
    for a, b in zip(u_seq_1, u_seq_2):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-8)


def test_toeplitz_recurrence_matches_lean_ldtoep_contract() -> None:
    coeffs = [1.1, 0.3, -0.2, 0.05]
    mechanism = CorrelatedNoiseMechanism(coeffs=coeffs, z_std=0.05)
    model = nn.Linear(4, 3)
    private_model, dp_optimizer, private_loader = _make_private(
        model,
        noise_seed=45,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        noise_mechanism=mechanism,
    )

    z_rows = []
    u_rows = []
    for batch in private_loader:
        _one_pre_step(private_model, dp_optimizer, batch)
        assert mechanism.last_flat_z is not None
        assert mechanism.last_flat_u is not None
        z_rows.append(mechanism.last_flat_z.detach().clone())
        u_rows.append(mechanism.last_flat_u.detach().clone())

    z = torch.stack(z_rows, dim=0)
    u = torch.stack(u_rows, dim=0)
    for t in range(z.shape[0]):
        rhs = coeffs[0] * u[t]
        max_lag = min(t, len(coeffs) - 1)
        for lag in range(1, max_lag + 1):
            rhs = rhs + coeffs[lag] * u[t - lag]
        assert torch.allclose(rhs, z[t], atol=1e-7, rtol=1e-6)


def test_rejects_non_finite_or_tiny_coefficients() -> None:
    with pytest.raises(ValueError, match="finite"):
        CorrelatedNoiseMechanism(coeffs=[1.0, float("nan")], z_std=0.01)

    with pytest.raises(ValueError, match="finite"):
        CorrelatedNoiseMechanism(coeffs=[1.0, float("inf")], z_std=0.01)

    with pytest.raises(ValueError, match=r"coeffs\[0\]"):
        CorrelatedNoiseMechanism(coeffs=[1e-15, 0.1], z_std=0.01)


def test_rejects_non_finite_z_std() -> None:
    with pytest.raises(ValueError, match="finite"):
        CorrelatedNoiseMechanism(coeffs=[1.0], z_std=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        CorrelatedNoiseMechanism(coeffs=[1.0], z_std=float("inf"))
