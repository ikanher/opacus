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
import io

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.optimizers import CorrelatedNoiseMechanism
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def test_bsr_resume_matches_uninterrupted_next_step() -> None:
    mech_state = {"coeffs": [1.1, 0.3, -0.2], "z_std": 0.03}
    base = nn.Linear(4, 3)
    model1 = copy.deepcopy(base)
    model2 = copy.deepcopy(base)

    pmodel1, opt1, loader1, pe1 = _make_private(
        model1, noise_seed=123, mechanism_state=mech_state
    )
    pmodel2, opt2, loader2, pe2 = _make_private(
        model2, noise_seed=999, mechanism_state=mech_state
    )
    batches1 = list(loader1)
    batches2 = list(loader2)
    assert len(batches1) >= 3
    assert len(batches2) >= 3

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

    mech1 = opt1.noise_mechanism
    mech2 = opt2.noise_mechanism
    assert isinstance(mech1, CorrelatedNoiseMechanism)
    assert isinstance(mech2, CorrelatedNoiseMechanism)
    assert mech1.last_flat_u is not None
    assert mech2.last_flat_u is not None
    assert torch.allclose(mech1.last_flat_u, mech2.last_flat_u, atol=1e-7, rtol=1e-6)


def test_missing_bsr_mechanism_state_fails_loudly() -> None:
    mech_state = {"coeffs": [1.0, 0.2], "z_std": 0.02}
    model1 = nn.Linear(4, 3)
    pmodel1, opt1, loader1, _ = _make_private(
        model1, noise_seed=200, mechanism_state=mech_state
    )
    _run_pre_step(pmodel1, opt1, next(iter(loader1)))
    state = opt1.state_dict()
    state.pop("_dp_noise_mechanism_state", None)

    model2 = nn.Linear(4, 3)
    _, opt2, _, _ = _make_private(model2, noise_seed=201, mechanism_state=mech_state)
    with pytest.raises(ValueError, match="missing bsr noise mechanism state"):
        opt2.load_state_dict(state)


def test_checkpoint_load_preserves_bnb_report_and_sampling_metadata() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    private_model, dp_optimizer, _loader_, _ = _make_private(
        model,
        noise_seed=300,
        mechanism_state={"coeffs": [1.0], "z_std": 0.01},
    )

    # Inject a v2 BNB report into saved mechanism config payload.
    checkpoint_dict: dict = {}
    with io.BytesIO() as bio:
        pe.noise_mechanism_config = NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={
                "coeffs": [1.0],
                "z_std": 0.01,
                "_bnb_calibration_report": {
                    "version": 2,
                    "target_epsilon": 1.0,
                    "target_delta": 1e-5,
                    "noise_multiplier": 2.0,
                    "num_samples": 1000,
                    "seed": 7,
                    "bands": 2,
                    "delta_estimate_at_target_epsilon": 0.1,
                    "delta_upper_confidence_bound": 0.11,
                    "confidence_alpha": 1e-4,
                    "verification_passed": True,
                    "evr_confidence_alpha_total": 1e-4,
                    "evr_num_checks": 1,
                    "evr_per_check_alpha": 1e-4,
                    "evr_pass_count": 1,
                    "verification_contract": "evr_union_bound_alpha_split_v1",
                    "evr_composed_delta_upper_bound": 1e-5 + 1e-4 * (1 - 1e-5),
                },
            },
        )
        pe.sampling_semantics = SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bands": 2},
        )
        pe.save_checkpoint(
            path=bio,
            module=private_model,
            optimizer=dp_optimizer,
            checkpoint_dict=checkpoint_dict,
        )
        bio.seek(0)
        loaded = pe.load_checkpoint(path=bio, module=private_model, optimizer=dp_optimizer)

    report = pe.noise_mechanism_config.mechanism_state["_bnb_calibration_report"]
    assert report["version"] == 2
    assert report["evr_num_checks"] == 1
    assert "bnb_calibration_summary" in loaded
    assert pe.sampling_semantics.sampling_mode == "balls_in_bins"
    parsed = pe.get_bnb_calibration_report()
    assert parsed is not None
    summary = pe.get_bnb_calibration_summary()
    assert isinstance(summary, str)
    assert "BNB calibration v2" in summary
    status = pe.get_bnb_calibration_status()
    assert status is not None
    assert status.report.version == 2
    assert status.sampling_mode == "balls_in_bins"
