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

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.optimizers import CorrelatedNoiseMechanism, GaussianNoiseMechanism
from opacus.utils.uniform_sampler import (
    BMinSepSampler,
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
    x, y = next(iter(private_loader))
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


def test_sampling_semantics_cyclic_poisson_switches_sampler_for_bsr() -> None:
    model = nn.Linear(4, 3)
    private_model, dp_optimizer, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=101,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
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
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={},
            ),
        )


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


def test_bnb_mechanism_rejects_standard_accounting_mode() -> None:
    with pytest.raises(ValueError, match="bnb_accountant"):
        NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="standard_step_accountant",
        )


def test_non_bnb_mechanism_rejects_bnb_sampling_modes() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(
        ValueError,
        match="b_min_sep sampling is supported only for mechanism='bnb'",
    ):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=114,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="gaussian",
                accounting_mode="standard_step_accountant",
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={},
            ),
        )

    with pytest.raises(
        ValueError,
        match="b_min_sep sampling is supported only for mechanism='bnb'",
    ):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=115,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={},
            ),
            
        )


def test_bnb_mechanism_requires_explicit_sampling_semantics() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="requires explicit sampling_semantics"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=118,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
            sampling_semantics=None,
        )


def test_sampling_semantics_b_min_sep_switches_sampler_for_bnb() -> None:
    model = nn.Linear(4, 3)
    _, _dp_optimizer, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=116,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.1},
        ),
    )
    assert isinstance(private_loader.batch_sampler, BMinSepSampler)


def test_sampling_semantics_b_min_sep_switches_sampler_for_bnb_alt() -> None:
    model = nn.Linear(4, 3)
    _, _dp_optimizer, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=117,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.2},
        ),
    )
    assert isinstance(private_loader.batch_sampler, BMinSepSampler)


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
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
    )
    assert isinstance(private_loader.batch_sampler, CyclicPoissonSampler)
    assert len(private_loader) == 5


def test_make_private_total_steps_supports_b_min_sep_sampler() -> None:
    model = nn.Linear(4, 3)
    _, _, private_loader = _make_private(
        model,
        poisson_sampling=False,
        noise_seed=202,
        total_steps=7,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.2},
        ),
    )
    assert isinstance(private_loader.batch_sampler, BMinSepSampler)
    assert len(private_loader) == 7


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
                "mf_sensitivity": 1.0,
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
                "mf_sensitivity": 1.0,
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
                "max_participations": 1,
                "min_separation": 1,
            },
        ),
    )
    assert getattr(dp_optimizer, "accounting_mode") == "bsr_accountant"
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert dp_optimizer.noise_mechanism.z_std > 0.01


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
            epsilon_fn=lambda **_: 1.0,
        )


def test_make_private_with_epsilon_bsr_calibrates_with_callback() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    calls: list[dict[str, float | int | str]] = []

    def epsilon_fn(
        *,
        noise_multiplier: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        mechanism: str,
        **kwargs,
    ) -> float:
        calls.append(
            {
                "noise_multiplier": float(noise_multiplier),
                "target_delta": float(target_delta),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
                "mechanism": mechanism,
            }
        )
        # Monotone decreasing synthetic epsilon model for calibration tests.
        return 1.0 / float(noise_multiplier)

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
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        epsilon_fn=epsilon_fn,
    )
    assert 1.9 <= float(dp_optimizer.noise_multiplier) <= 2.1
    assert calls
    assert calls[-1]["mechanism"] == "bsr"
    assert getattr(dp_optimizer, "accounting_mode") == "bsr_accountant"


def test_make_private_with_epsilon_total_steps_nonpoisson_requires_custom_sampler() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

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
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01, "mf_sensitivity": 1.0},
            ),
            epsilon_fn=lambda **_: 1.0,
        )


def test_make_private_with_epsilon_total_steps_uses_b_min_sep_rate() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    calls: list[dict[str, float | int | str]] = []

    def epsilon_fn(
        *,
        noise_multiplier: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        mechanism: str,
        **kwargs,
    ) -> float:
        calls.append(
            {
                "noise_multiplier": float(noise_multiplier),
                "target_delta": float(target_delta),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
                "mechanism": mechanism,
            }
        )
        return 1.0 / float(noise_multiplier)

    _, _dp_optimizer, _ = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        poisson_sampling=False,
        total_steps=9,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.2, "bands": 1},
        ),
        epsilon_fn=epsilon_fn,
    )

    assert calls
    assert calls[-1]["mechanism"] == "bnb"
    assert calls[-1]["steps"] == 9
    assert abs(float(calls[-1]["sample_rate"]) - 0.2) < 1e-12


def test_make_private_with_epsilon_bnb_calibrates_with_callback() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    calls: list[dict[str, float | int | str]] = []

    def epsilon_fn(
        *,
        noise_multiplier: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        mechanism: str,
        **kwargs,
    ) -> float:
        calls.append(
            {
                "noise_multiplier": float(noise_multiplier),
                "target_delta": float(target_delta),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
                "mechanism": mechanism,
            }
        )
        return 1.0 / float(noise_multiplier)

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
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.1},
        ),
        epsilon_fn=epsilon_fn,
    )
    assert 1.9 <= float(dp_optimizer.noise_multiplier) <= 2.1
    assert calls
    assert calls[-1]["mechanism"] == "bnb"
    assert getattr(dp_optimizer, "accounting_mode") == "bnb_accountant"
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert dp_optimizer.noise_mechanism.z_std > 0.01


def test_make_private_with_epsilon_bnb_requires_mc_inputs_without_callback() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()

    with pytest.raises(ValueError, match="requires b_min_sep inputs"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=3.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.01},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
        )


def test_make_private_with_epsilon_bnb_noncallback_b_min_sep_succeeds() -> None:
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
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": 0.01,
                "c_matrix": c_matrix,
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
        ),
            
        bnb_num_samples=10_000,
        bnb_seed=7,
        bnb_confidence_alpha=0.5,
        bnb_evr_num_checks=4,
        bnb_require_evr_pass=False,
    )

    assert getattr(dp_optimizer, "accounting_mode") == "bnb_accountant"
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert dp_optimizer.noise_mechanism.z_std > 0.0
    report = pe.noise_mechanism_config.mechanism_state.get("_bnb_calibration_report")
    assert isinstance(report, dict)
    assert report["version"] == 2
    assert report["bands"] == 2
    assert report["num_samples"] == 10_000
    assert report["evr_num_checks"] == 8  # 2 directions * bnb_evr_num_checks
    assert 0 <= report["evr_pass_count"] <= report["evr_num_checks"]
    assert report["verification_contract"] == "evr_union_bound_alpha_split_v1"
    assert report["evr_composed_delta_upper_bound"] >= report["target_delta"]
    parsed = pe.get_bnb_calibration_report()
    assert parsed is not None
    summary = pe.get_bnb_calibration_summary()
    assert isinstance(summary, str)
    assert "BNB calibration v2" in summary
    status = pe.get_bnb_calibration_status()
    assert status is not None
    assert status.mechanism == "bnb"
    assert status.accounting_mode == "bnb_accountant"
    assert status.sampling_mode == "b_min_sep"
    assert status.report.version == 2
    status_dict = status.to_dict()
    assert status_dict["report"]["version"] == 2


def test_make_private_with_epsilon_bnb_noncallback_requires_b_min_sep_mode() -> None:
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

    with pytest.raises(ValueError, match="requires sampling_semantics in \\{'b_min_sep'\\}"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0, 0.2],
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
            bnb_num_samples=2_000,
        )


def test_make_private_with_epsilon_bnb_noncallback_rejects_bands_coeffs_mismatch() -> None:
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

    with pytest.raises(ValueError, match="bands.*len\\(coeffs\\)"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0],  # mismatch with bands=2
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
            bnb_num_samples=2_000,
            bnb_seed=11,
        )


def test_make_private_with_epsilon_bnb_noncallback_rejects_metadata_bands_mismatch() -> None:
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

    with pytest.raises(ValueError, match="privacy_metadata\\['bands'\\].*accounting bands"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0, 0.2],
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 3},
            ),
            
            bnb_num_samples=2_000,
            bnb_seed=11,
            bnb_bands=2,
        )


def test_make_private_with_epsilon_bnb_noncallback_rejects_contract_mismatch() -> None:
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
    bad_contract = _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2)
    bad_contract["matrix_columns"] = 3

    with pytest.raises(ValueError, match="c_matrix_contract\\['matrix_columns'\\]"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0, 0.2],
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": bad_contract,
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
            bnb_num_samples=2_000,
            bnb_seed=11,
        )


def test_make_private_with_epsilon_bnb_noncallback_rejects_toeplitz_derivation_mismatch() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pe = PrivacyEngine()
    coeffs = [1.0, 0.2]
    c_matrix = _lower_toeplitz_from_coeffs(coeffs, horizon=4)
    c_matrix[3, 2] += 0.3
    contract = _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2)
    contract["derivation"] = "lower_toeplitz_from_coeffs"
    contract["horizon"] = 4
    contract["atol"] = 1e-12

    with pytest.raises(ValueError, match="lower_toeplitz_from_coeffs derivation"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=4),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": coeffs,
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": contract,
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
            bnb_num_samples=2_000,
            bnb_seed=11,
        )


def test_make_private_with_epsilon_bnb_noncallback_rejects_unordered_candidate_ladder() -> None:
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

    with pytest.raises(ValueError, match="strictly increasing"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0, 0.2],
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
            bnb_num_samples=2_000,
            bnb_seed=11,
            bnb_candidate_sigmas=[1.0, 0.8, 1.2],
        )


def test_make_private_with_epsilon_bnb_noncallback_rejects_below_calibrated_sigma() -> None:
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

    with pytest.raises(ValueError, match="must not include values below"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=0.2,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0, 0.2],
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
            bnb_num_samples=2_000,
            bnb_seed=11,
            bnb_candidate_sigmas=[0.5, 1.0, 1.2],
        )


def test_make_private_with_epsilon_bnb_noncallback_fails_on_evr_reject() -> None:
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

    with pytest.raises(ValueError, match="verification guard failed"):
        pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=1.0,
            target_delta=1e-6,
            epochs=1,
            max_grad_norm=1.0,
            poisson_sampling=False,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={
                    "coeffs": [1.0, 0.2],
                    "z_std": 0.01,
                    "c_matrix": c_matrix,
                    "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
                },
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
            ),
            
            bnb_num_samples=100,
            bnb_seed=11,
            bnb_confidence_alpha=1e-6,
            bnb_evr_num_checks=3,
        )


def test_make_private_with_epsilon_bnb_noncallback_allows_evr_reject_when_disabled() -> None:
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

    _private_model, _dp_optimizer, _private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=1.0,
        target_delta=1e-6,
        epochs=1,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bnb",
            accounting_mode="bnb_accountant",
            mechanism_state={
                "coeffs": [1.0, 0.2],
                "z_std": 0.01,
                "c_matrix": c_matrix,
                "c_matrix_contract": _bnb_c_matrix_contract(c_matrix=c_matrix, bands=2),
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="b_min_sep",
            privacy_metadata={"b": 2, "p": 0.1, "bands": 2},
        ),
            
        bnb_num_samples=100,
        bnb_seed=11,
        bnb_confidence_alpha=1e-6,
        bnb_evr_num_checks=3,
        bnb_require_evr_pass=False,
    )
    report = pe.noise_mechanism_config.mechanism_state.get("_bnb_calibration_report")
    assert isinstance(report, dict)
    assert report["verification_passed"] is False


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


def test_bsr_config_requires_coeffs_and_z_std() -> None:
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="coeffs"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=108,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"z_std": 0.03},
            ),
        )

    with pytest.raises(ValueError, match="z_std"):
        _make_private(
            model,
            poisson_sampling=False,
            noise_seed=109,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, 0.2]},
            ),
        )


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


def test_default_mechanism_state_dict_round_trip_and_legacy_load() -> None:
    model_1 = nn.Linear(4, 3)
    private_1, opt_1, loader_1 = _make_private(
        model_1, poisson_sampling=False, noise_seed=112, noise_mechanism=None
    )
    _single_pre_step(private_1, opt_1, loader_1)
    state = opt_1.state_dict()

    assert "_dp_noise_mechanism_name" in state
    assert "_dp_noise_mechanism_state" in state
    assert state["_dp_noise_mechanism_name"] == "GaussianNoiseMechanism"
    assert state["_dp_noise_mechanism_state"] == {}

    model_2 = nn.Linear(4, 3)
    _, opt_2, _ = _make_private(
        model_2, poisson_sampling=False, noise_seed=113, noise_mechanism=None
    )
    opt_2.load_state_dict(state)
    loaded = opt_2.state_dict()
    assert loaded["_dp_noise_mechanism_name"] == "GaussianNoiseMechanism"
    assert loaded["_dp_noise_mechanism_state"] == {}

    legacy_state = {
        k: v for k, v in state.items() if not k.startswith("_dp_noise_mechanism_")
    }
    opt_2.load_state_dict(legacy_state)
