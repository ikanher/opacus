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

import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _build_loader(
    *, n_samples: int = 128, in_dim: int = 4, n_classes: int = 3, batch_size: int = 16
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260212)
    x = torch.randn(n_samples, in_dim, generator=gen)
    w = torch.tensor(
        [
            [1.1, -0.3, 0.7],
            [-0.4, 0.9, -0.2],
            [0.2, 0.1, 0.6],
            [0.3, -0.7, 0.8],
        ],
        dtype=torch.float32,
    )
    logits = x @ w
    y = torch.argmax(logits, dim=1).to(torch.long)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def test_bsr_dp_training_smoke_loop() -> None:
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.Tanh(),
        nn.Linear(16, 3),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = _build_loader()
    pe = PrivacyEngine()

    noise_multiplier = 0.7
    max_grad_norm = 1.0
    batch_size = loader.batch_size
    assert batch_size is not None

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(7),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": noise_multiplier * max_grad_norm / float(batch_size),
            },
        ),
    )

    initial = [p.detach().clone() for p in private_model.parameters() if p.requires_grad]
    losses = []

    for _epoch in range(2):
        for xb, yb in private_loader:
            dp_optimizer.zero_grad()
            logits = private_model(xb)
            loss = F.cross_entropy(logits, yb)
            assert torch.isfinite(loss)
            loss.backward()
            dp_optimizer.step()
            losses.append(float(loss.detach()))

    assert losses
    assert all(torch.isfinite(torch.tensor(losses)))

    final = [p.detach() for p in private_model.parameters() if p.requires_grad]
    total_change = sum((f - i).abs().sum().item() for i, f in zip(initial, final))
    assert total_change > 0.0

    eps = pe.get_epsilon(
        delta=1e-5,
        epsilon_fn=lambda **kwargs: (
            kwargs["steps"] * kwargs["sample_rate"]
        ) / max(kwargs["noise_multiplier"], 1e-9),
    )
    assert eps > 0.0


def _run_bnb_training_smoke(*, sampling_semantics: SamplingSemantics) -> None:
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = _build_loader()
    pe = PrivacyEngine()

    noise_multiplier = 0.6
    max_grad_norm = 1.0
    batch_size = loader.batch_size
    assert batch_size is not None

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(17),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": noise_multiplier * max_grad_norm / float(batch_size),
            },
        ),
        sampling_semantics=sampling_semantics,
    )

    initial = [p.detach().clone() for p in private_model.parameters() if p.requires_grad]

    seen_losses = []
    for _epoch in range(1):
        for xb, yb in private_loader:
            dp_optimizer.zero_grad()
            logits = private_model(xb)
            loss = F.cross_entropy(logits, yb)
            assert torch.isfinite(loss)
            loss.backward()
            dp_optimizer.step()
            seen_losses.append(float(loss.detach()))

    assert seen_losses
    final = [p.detach() for p in private_model.parameters() if p.requires_grad]
    total_change = sum((f - i).abs().sum().item() for i, f in zip(initial, final))
    assert total_change > 0.0


def test_bnb_b_min_sep_training_smoke_loop() -> None:
    _run_bnb_training_smoke(
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.2},
        )
    )


def test_bnb_b_min_sep_training_smoke_loop_alt() -> None:
    _run_bnb_training_smoke(
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 3, "p": 0.25},
        ),
    )
