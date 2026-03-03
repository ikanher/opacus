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

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.accountants.analysis.bnb import build_bnb_toeplitz_c_matrix_and_contract
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
                "mf_sensitivity": 1.0,
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

    eps = pe.get_epsilon(delta=1e-5)
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
    with pytest.raises(ValueError, match="b_min_sep sampling is temporarily disabled"):
        _run_bnb_training_smoke(
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.2},
            )
        )


def test_bnb_b_min_sep_training_smoke_loop_alt() -> None:
    with pytest.raises(ValueError, match="b_min_sep sampling is temporarily disabled"):
        _run_bnb_training_smoke(
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 3, "p": 0.25},
            ),
        )


def test_bandmf_cyclic_poisson_training_smoke_loop() -> None:
    model = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = _build_loader()
    pe = PrivacyEngine()

    batch_size = loader.batch_size
    assert batch_size is not None
    noise_multiplier = 0.7
    max_grad_norm = 1.0

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(23),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": noise_multiplier * max_grad_norm / float(batch_size),
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )

    x, y = next(iter(private_loader))
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    dp_optimizer.step()
    assert torch.isfinite(loss)


def test_bnb_balls_in_bins_training_smoke_loop() -> None:
    _run_bnb_training_smoke(
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 4},
        )
    )


def test_target_epsilon_sampler_paths_smoke() -> None:
    cases = [
        (
            "bsr",
            SamplingSemantics(sampling_mode="torch_sampler", privacy_metadata={}),
            {"coeffs": [1.0, 0.2]},
        ),
        (
            "bandmf",
            SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 2},
            ),
            {"coeffs": [1.0, 0.2]},
        ),
        (
            "bnb",
            SamplingSemantics(
                sampling_mode="balls_in_bins",
                privacy_metadata={"bins": 4, "bands": 2},
            ),
            {
                "coeffs": [1.0, 0.2],
                "bands": 2,
                "c_matrix": build_bnb_toeplitz_c_matrix_and_contract(
                    coeffs=[1.0, 0.2],
                    bands=2,
                    horizon=8,
                )[0],
                "c_matrix_contract": build_bnb_toeplitz_c_matrix_and_contract(
                    coeffs=[1.0, 0.2],
                    bands=2,
                    horizon=8,
                )[1],
            },
        ),
    ]

    for mechanism, semantics, state in cases:
        model = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        loader = _build_loader()
        pe = PrivacyEngine()

        private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=1.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_generator=torch.Generator().manual_seed(31),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism=mechanism,
                accounting_mode=f"{mechanism}_accountant",
                mechanism_state=state,
            ),
            sampling_semantics=semantics,
            bnb_require_evr_pass=False,
        )

        xb, yb = next(iter(private_loader))
        dp_optimizer.zero_grad()
        loss = F.cross_entropy(private_model(xb), yb)
        loss.backward()
        dp_optimizer.step()
        assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "mechanism,sampling_semantics,coeffs,seed,epochs",
    [
        ("bsr", SamplingSemantics(sampling_mode="torch_sampler", privacy_metadata={}), [1.0, 0.2], 41, 2),
        ("bsr", SamplingSemantics(sampling_mode="torch_sampler", privacy_metadata={}), [1.0, 0.4, 0.1], 43, 2),
        (
            "bandmf",
            SamplingSemantics(sampling_mode="cyclic_poisson", privacy_metadata={"bands": 2}),
            [1.0, 0.2],
            47,
            2,
        ),
        (
            "bandmf",
            SamplingSemantics(sampling_mode="cyclic_poisson", privacy_metadata={"bands": 3}),
            [1.0, 0.3, 0.1],
            53,
            2,
        ),
    ],
)
def test_matrix_factorization_short_stability_no_nans_across_representative_settings(
    mechanism: str,
    sampling_semantics: SamplingSemantics,
    coeffs: list[float],
    seed: int,
    epochs: int,
) -> None:
    model = nn.Sequential(nn.Linear(4, 20), nn.Tanh(), nn.Linear(20, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loader = _build_loader()
    pe = PrivacyEngine()

    noise_multiplier = 0.65
    max_grad_norm = 1.0
    batch_size = loader.batch_size
    assert batch_size is not None

    mechanism_state = {
        "coeffs": coeffs,
        "z_std": noise_multiplier * max_grad_norm / float(batch_size),
    }
    if sampling_semantics.sampling_mode == "torch_sampler":
        mechanism_state["mf_sensitivity"] = 1.0

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(seed),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism=mechanism,
            accounting_mode=f"{mechanism}_accountant",
            mechanism_state=mechanism_state,
        ),
        sampling_semantics=sampling_semantics,
    )

    losses = []
    for _ in range(epochs):
        for xb, yb in private_loader:
            dp_optimizer.zero_grad()
            logits = private_model(xb)
            loss = F.cross_entropy(logits, yb)
            assert torch.isfinite(loss)
            loss.backward()
            for p in private_model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
            dp_optimizer.step()
            losses.append(float(loss.detach()))

    assert losses
    assert torch.isfinite(torch.tensor(losses)).all()
