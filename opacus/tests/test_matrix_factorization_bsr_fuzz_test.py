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

import io
import math

import hypothesis.strategies as st
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.optimizers import CorrelatedNoiseMechanism
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Testing approach borrowed from the jax_privacy matrix-factorization suite:
# use randomized/property tests to validate algebraic invariants and stateful
# mechanism behavior across broad parameter combinations.


def _loader(
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
    model: nn.Module, *, noise_seed: int, mechanism_state: dict
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    noise_gen = torch.Generator().manual_seed(noise_seed)
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=noise_gen,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state=mechanism_state,
        ),
    )
    return private_model, dp_optimizer, private_loader, pe


def _run_pre_step(private_model: nn.Module, dp_optimizer, batch) -> None:
    x, y = batch
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    assert dp_optimizer.pre_step() is True


def _valid_coeffs(raw: list[float], c0: float) -> list[float]:
    out = [float(c0)]
    out.extend(float(x) for x in raw)
    return out


@settings(deadline=None, max_examples=25)
@given(
    bandwidth=st.integers(min_value=1, max_value=5),
    c0=st.floats(min_value=0.25, max_value=2.0, allow_nan=False, allow_infinity=False),
    tail=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=4,
    ),
    z_std=st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
)
def test_fuzz_toeplitz_recurrence_invariant(
    bandwidth: int, c0: float, tail: list[float], z_std: float
) -> None:
    coeffs = _valid_coeffs(tail[: max(0, bandwidth - 1)], c0)
    model = nn.Linear(4, 3)
    private_model, dp_optimizer, private_loader, _ = _make_private(
        model, noise_seed=123, mechanism_state={"coeffs": coeffs, "z_std": float(z_std)}
    )
    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, CorrelatedNoiseMechanism)

    z_rows = []
    u_rows = []
    for batch in private_loader:
        _run_pre_step(private_model, dp_optimizer, batch)
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
        assert torch.allclose(rhs, z[t], atol=1e-6, rtol=1e-5)


@settings(deadline=None, max_examples=15)
@given(
    c0=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    c1=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    c2=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    z_std=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_fuzz_checkpoint_replay_equivalence(c0: float, c1: float, c2: float, z_std: float) -> None:
    coeffs = [float(c0), float(c1), float(c2)]
    mech_state = {"coeffs": coeffs, "z_std": float(z_std)}
    base = nn.Linear(4, 3)
    model1 = nn.Linear(4, 3)
    model2 = nn.Linear(4, 3)
    model1.load_state_dict(base.state_dict())
    model2.load_state_dict(base.state_dict())

    pmodel1, opt1, loader1, pe1 = _make_private(
        model1, noise_seed=55, mechanism_state=mech_state
    )
    pmodel2, opt2, loader2, pe2 = _make_private(
        model2, noise_seed=999, mechanism_state=mech_state
    )
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


@settings(deadline=None, max_examples=20)
@given(
    bad_coeff=st.one_of(
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
        st.floats(min_value=-1e-16, max_value=1e-16, allow_nan=False, allow_infinity=False),
    ),
    bad_z_std=st.one_of(
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
    ),
)
def test_fuzz_validation_rejects_invalid_inputs(bad_coeff: float, bad_z_std: float) -> None:
    if (not math.isfinite(bad_coeff)) or bad_coeff <= 1e-12:
        with pytest.raises(ValueError):
            CorrelatedNoiseMechanism(coeffs=[bad_coeff], z_std=0.01)

    with pytest.raises(ValueError):
        CorrelatedNoiseMechanism(coeffs=[1.0], z_std=bad_z_std)
