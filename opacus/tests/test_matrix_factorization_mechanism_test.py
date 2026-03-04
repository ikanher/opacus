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
from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.optimizers import CorrelatedNoiseMechanism, DistributedDPOptimizer
from opacus.mechanism_contracts import SamplingSemantics
from opacus.utils.uniform_sampler import (
    DistributedBMinSepSampler,
    DistributedCyclicPoissonSampler,
    DistributedUniformWithReplacementSampler,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import opacus.privacy_engine as pe_mod


def _loader(
    *, n_samples: int = 16, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8
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


def _index_loader(*, n_samples: int = 64, batch_size: int = 8) -> DataLoader:
    x = torch.arange(n_samples, dtype=torch.int64)
    y = torch.zeros(n_samples, dtype=torch.int64)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def _patch_distributed_primitives(monkeypatch, *, rank: int, world_size: int = 2) -> None:
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)


def _one_step(private_model: nn.Module, dp_optimizer, private_loader: DataLoader) -> None:
    x, y = next(iter(private_loader))
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    dp_optimizer.step()


def _take_first_index_batches(loader: DataLoader, num_batches: int) -> list[list[int]]:
    out: list[list[int]] = []
    for i, (x, _y) in enumerate(loader):
        if i >= num_batches:
            break
        out.append([int(v) for v in x.tolist()])
    return out


def _set_zero_grad_samples(dp_optimizer, *, batch_size: int) -> None:
    for p in dp_optimizer.params:
        p.grad_sample = torch.zeros(
            (batch_size,) + tuple(p.shape),
            dtype=p.dtype,
            device=p.device,
        )


def test_noise_mechanism_config_bsr_requires_bsr_accountant() -> None:
    with pytest.raises(ValueError, match="bsr mechanism requires bsr_accountant"):
        NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="standard_step_accountant",
        )


def test_noise_mechanism_config_bandmf_requires_bandmf_accountant() -> None:
    with pytest.raises(ValueError, match="bandmf mechanism requires bandmf_accountant"):
        NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="standard_step_accountant",
        )


def test_noise_mechanism_config_rejects_correlated_alias() -> None:
    with pytest.raises(ValueError, match="mechanism must be one of"):
        NoiseMechanismConfig(
            mechanism="correlated",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.1},
        )


def test_make_private_builds_bsr_noise_mechanism() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
    )

    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)


def test_make_private_rejects_correlated_alias() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="mechanism must be one of"):
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
                mechanism="correlated",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
            ),
        )


def test_make_private_bsr_requires_torch_sampler() -> None:
    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="bsr mechanism requires fixed-batch semantics"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=True,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
            ),
        )


def test_distributed_bsr_supported_for_flat_hooks(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
    )

    assert isinstance(dp_optimizer, DistributedDPOptimizer)
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)


def test_distributed_torch_sampler_shards_batches_across_ranks(monkeypatch) -> None:
    world_size = 2
    semantics = SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={},
    )

    _patch_distributed_primitives(monkeypatch, rank=0, world_size=world_size)
    pe0 = PrivacyEngine()
    loader0 = pe0._prepare_data_loader(
        _index_loader(n_samples=64, batch_size=8),
        poisson_sampling=False,
        distributed=True,
        sampling_semantics=semantics,
        total_steps=8,
    )

    _patch_distributed_primitives(monkeypatch, rank=1, world_size=world_size)
    pe1 = PrivacyEngine()
    loader1 = pe1._prepare_data_loader(
        _index_loader(n_samples=64, batch_size=8),
        poisson_sampling=False,
        distributed=True,
        sampling_semantics=semantics,
        total_steps=8,
    )

    b0 = _take_first_index_batches(loader0, num_batches=3)
    b1 = _take_first_index_batches(loader1, num_batches=3)
    assert b0 != b1
    assert len(set(b0[0]).intersection(set(b1[0]))) == 0
    assert len(set(b0[0]).union(set(b1[0]))) == 16


def test_distributed_cyclic_poisson_shards_batches_across_ranks(monkeypatch) -> None:
    world_size = 2
    semantics = SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"bands": 4},
    )

    _patch_distributed_primitives(monkeypatch, rank=0, world_size=world_size)
    pe0 = PrivacyEngine()
    loader0 = pe0._prepare_data_loader(
        _index_loader(n_samples=64, batch_size=8),
        poisson_sampling=False,
        distributed=True,
        sampling_semantics=semantics,
        total_steps=8,
    )

    _patch_distributed_primitives(monkeypatch, rank=1, world_size=world_size)
    pe1 = PrivacyEngine()
    loader1 = pe1._prepare_data_loader(
        _index_loader(n_samples=64, batch_size=8),
        poisson_sampling=False,
        distributed=True,
        sampling_semantics=semantics,
        total_steps=8,
    )

    assert isinstance(loader0.batch_sampler, DistributedCyclicPoissonSampler)
    assert isinstance(loader1.batch_sampler, DistributedCyclicPoissonSampler)

    b0 = _take_first_index_batches(loader0, num_batches=3)
    b1 = _take_first_index_batches(loader1, num_batches=3)
    assert b0 != b1
    for left, right in zip(b0, b1):
        assert len(set(left).intersection(set(right))) == 0


def test_distributed_poisson_shards_batches_across_ranks(monkeypatch) -> None:
    world_size = 2

    _patch_distributed_primitives(monkeypatch, rank=0, world_size=world_size)
    pe0 = PrivacyEngine()
    loader0 = pe0._prepare_data_loader(
        _index_loader(n_samples=64, batch_size=32),
        poisson_sampling=True,
        distributed=True,
        sampling_semantics=None,
        total_steps=8,
    )

    _patch_distributed_primitives(monkeypatch, rank=1, world_size=world_size)
    pe1 = PrivacyEngine()
    loader1 = pe1._prepare_data_loader(
        _index_loader(n_samples=64, batch_size=32),
        poisson_sampling=True,
        distributed=True,
        sampling_semantics=None,
        total_steps=8,
    )

    assert isinstance(loader0.batch_sampler, DistributedUniformWithReplacementSampler)
    assert isinstance(loader1.batch_sampler, DistributedUniformWithReplacementSampler)

    b0 = _take_first_index_batches(loader0, num_batches=3)
    b1 = _take_first_index_batches(loader1, num_batches=3)
    assert b0 != b1
    for left, right in zip(b0, b1):
        assert len(set(left).intersection(set(right))) == 0


def test_distributed_bandmf_supports_cyclic_poisson_sampling(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bandmf",
            accounting_mode="bandmf_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
    )

    assert isinstance(dp_optimizer, DistributedDPOptimizer)
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)
    assert isinstance(private_loader.batch_sampler, DistributedCyclicPoissonSampler)


def test_distributed_dpoptimizer_rank0_noise_and_global_mean_semantics(
    monkeypatch,
) -> None:
    world_size = 2

    def _make_optimizer(*, rank: int, seed: int):
        _patch_distributed_primitives(monkeypatch, rank=rank, world_size=world_size)
        model = nn.Linear(2, 1, bias=False)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_optimizer = DistributedDPOptimizer(
            base_optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=4,
            loss_reduction="mean",
            generator=torch.Generator().manual_seed(seed),
        )
        param = dp_optimizer.params[0]
        return dp_optimizer, param

    # Distinct per-rank clipped sums make the reduction semantics observable.
    dp0, p0 = _make_optimizer(rank=0, seed=1337)
    dp1, p1 = _make_optimizer(rank=1, seed=1337)
    p0.summed_grad = torch.tensor([[1.0, -3.0]], dtype=p0.dtype)
    p1.summed_grad = torch.tensor([[-2.0, 5.0]], dtype=p1.dtype)

    dp0.add_noise()
    dp1.add_noise()

    # Noise is injected only on rank 0.
    assert not torch.allclose(p0.grad, p0.summed_grad)
    assert torch.allclose(p1.grad, p1.summed_grad)

    g0_pre_reduce = p0.grad.detach().clone()
    g1_pre_reduce = p1.grad.detach().clone()
    reduced_sum = g0_pre_reduce + g1_pre_reduce

    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None: tensor.copy_(reduced_sum),
    )
    dp0.reduce_gradients()
    dp1.reduce_gradients()

    expected_global_mean = reduced_sum / world_size
    assert torch.allclose(p0.grad, expected_global_mean, rtol=0.0, atol=1e-7)
    assert torch.allclose(p1.grad, expected_global_mean, rtol=0.0, atol=1e-7)


def test_distributed_bnb_rejects_b_min_sep_sampling(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="b_min_sep sampling is temporarily disabled"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.25},
            ),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={"coeffs": [1.0, 0.3], "z_std": 0.01, "bands": 2},
            ),
        )


def test_distributed_bnb_rejects_b_min_sep_sampling_alt(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="b_min_sep sampling is temporarily disabled"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"b": 2, "p": 0.25},
            ),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bnb",
                accounting_mode="bnb_accountant",
                mechanism_state={"coeffs": [1.0, 0.3], "z_std": 0.01, "bands": 2},
            ),
        )


def test_distributed_bsr_rejects_non_flat_clipping(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match=r"got clipping='per_layer'"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="per_layer",
            grad_sample_mode="hooks",
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="bsr",
                accounting_mode="bsr_accountant",
                mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
            ),
        )


def test_distributed_bsr_save_rejects_nonzero_rank(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=1, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    private_model, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
    )

    with io.BytesIO() as bio:
        with pytest.raises(
            ValueError,
            match="distributed correlated-noise checkpoint save is supported only on rank 0",
        ):
            pe.save_checkpoint(path=bio, module=private_model, optimizer=dp_optimizer)


def test_distributed_bsr_load_rejects_nonzero_saved_rank(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0, 0.2], "z_std": 0.01},
        ),
    )

    state = dp_optimizer.state_dict()
    state["_dp_distributed_saved_rank"] = 1
    with pytest.raises(
        ValueError,
        match="distributed correlated-noise checkpoint must be saved on rank 0",
    ):
        dp_optimizer.load_state_dict(state)


def test_distributed_bsr_supported_for_flat_ew(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="ew",
        noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0], z_std=0.01),
    )

    assert isinstance(dp_optimizer, DistributedDPOptimizer)
    assert isinstance(dp_optimizer.noise_mechanism, CorrelatedNoiseMechanism)


def test_distributed_bsr_rejects_unsupported_grad_sample_mode(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="grad_sample_mode"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="ghost",
            noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0], z_std=0.01),
        )

    with pytest.raises(ValueError, match=r"got grad_sample_mode='ghost'"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="ghost",
            noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0], z_std=0.01),
        )


def test_distributed_bsr_rejects_fsdp(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "FSDPModule", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="not yet supported with FSDP"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0], z_std=0.01),
        )


def test_distributed_bsr_one_step_rank0_smoke(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0, 0.2], z_std=0.01),
    )

    _one_step(private_model, dp_optimizer, private_loader)
    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, CorrelatedNoiseMechanism)
    assert mechanism.steps_with_noise == 1
    assert mechanism.last_flat_u is not None
    assert torch.isfinite(mechanism.last_flat_u).all()


def test_distributed_bsr_one_step_rank_nonzero_smoke(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=1, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0, 0.2], z_std=0.01),
    )

    _one_step(private_model, dp_optimizer, private_loader)
    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, CorrelatedNoiseMechanism)
    assert mechanism.steps_with_noise == 0
    assert mechanism.last_flat_u is None


def test_distributed_bsr_rank0_checkpoint_resume_parity(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    base = nn.Linear(4, 3)
    model1 = copy.deepcopy(base)
    model2 = copy.deepcopy(base)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.05)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.05)
    pe1 = PrivacyEngine()
    pe2 = PrivacyEngine()
    noise_gen1 = torch.Generator().manual_seed(123)
    noise_gen2 = torch.Generator().manual_seed(999)

    private_model1, dp_optimizer1, private_loader1 = pe1.make_private(
        module=model1,
        optimizer=optimizer1,
        data_loader=_loader(n_samples=32, batch_size=8),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_generator=noise_gen1,
        noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.1, 0.3, -0.2], z_std=0.03),
    )
    private_model2, dp_optimizer2, private_loader2 = pe2.make_private(
        module=model2,
        optimizer=optimizer2,
        data_loader=_loader(n_samples=32, batch_size=8),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_generator=noise_gen2,
        noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.1, 0.3, -0.2], z_std=0.03),
    )

    batches1 = list(private_loader1)
    batches2 = list(private_loader2)
    for i in [0, 1]:
        x1, y1 = batches1[i]
        dp_optimizer1.zero_grad()
        F.cross_entropy(private_model1(x1), y1).backward()
        dp_optimizer1.step()

    with io.BytesIO() as bio:
        pe1.save_checkpoint(path=bio, module=private_model1, optimizer=dp_optimizer1)
        bio.seek(0)
        pe2.load_checkpoint(path=bio, module=private_model2, optimizer=dp_optimizer2)

    x1, y1 = batches1[0]
    x2, y2 = batches2[0]
    dp_optimizer1.zero_grad()
    dp_optimizer2.zero_grad()
    F.cross_entropy(private_model1(x1), y1).backward()
    F.cross_entropy(private_model2(x2), y2).backward()
    dp_optimizer1.step()
    dp_optimizer2.step()

    grad1 = torch.cat([p.grad.reshape(-1) for p in dp_optimizer1.params])
    grad2 = torch.cat([p.grad.reshape(-1) for p in dp_optimizer2.params])
    assert torch.allclose(grad1, grad2, atol=1e-7, rtol=1e-6)


def test_distributed_bsr_global_noise_scale_rank0_only(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)
    _patch_distributed_primitives(monkeypatch, rank=0, world_size=2)

    pe = PrivacyEngine()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    noise_gen = torch.Generator().manual_seed(1729)
    _, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(batch_size=8),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        noise_generator=noise_gen,
        noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0], z_std=0.4),
    )

    dp_optimizer.zero_grad()
    _set_zero_grad_samples(dp_optimizer, batch_size=8)
    dp_optimizer.step()

    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, CorrelatedNoiseMechanism)
    assert mechanism.last_flat_u is not None
    observed = torch.cat([p.grad.reshape(-1) for p in dp_optimizer.params])
    expected = mechanism.last_flat_u / (
        float(dp_optimizer.expected_batch_size) * float(dp_optimizer.world_size)
    )
    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-6)


def test_distributed_bsr_history_progression_and_bounds(monkeypatch) -> None:
    monkeypatch.setattr(pe_mod, "DDP", nn.Linear)

    def _run_steps_for_rank(rank: int) -> tuple[int, int, int]:
        _patch_distributed_primitives(monkeypatch, rank=rank, world_size=2)
        pe = PrivacyEngine()
        model = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        noise_gen = torch.Generator().manual_seed(1234)
        _, dp_optimizer, _ = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(batch_size=8),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_generator=noise_gen,
            noise_mechanism=CorrelatedNoiseMechanism(
                coeffs=[1.0, 0.4, -0.1], z_std=0.2
            ),
        )
        mechanism = dp_optimizer.noise_mechanism
        assert isinstance(mechanism, CorrelatedNoiseMechanism)
        for _ in range(5):
            dp_optimizer.zero_grad()
            _set_zero_grad_samples(dp_optimizer, batch_size=8)
            dp_optimizer.step()
        return mechanism.steps_with_noise, mechanism.state_depth, mechanism.max_state_depth

    steps_rank0, depth_rank0, max_depth_rank0 = _run_steps_for_rank(0)
    steps_rank1, depth_rank1, max_depth_rank1 = _run_steps_for_rank(1)
    assert steps_rank0 == 5
    assert depth_rank0 == max_depth_rank0 == 2
    assert steps_rank1 == 0
    assert depth_rank1 == 0
    assert max_depth_rank1 == 2
