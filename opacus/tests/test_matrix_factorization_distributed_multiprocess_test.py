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
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.mechanism_contracts import SamplingSemantics
from opacus.optimizers import CorrelatedNoiseMechanism
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset


RUN_DISTRIBUTED = os.getenv("OPACUS_RUN_DISTRIBUTED_TESTS") == "1"


def _supports_local_gloo_process_group() -> bool:
    if not dist.is_available():
        return False

    with tempfile.NamedTemporaryFile(delete=False) as sync:
        init_method = f"file://{sync.name}"

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=0,
            world_size=1,
        )
        return True
    except Exception:
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


HAS_LOCAL_GLOO = _supports_local_gloo_process_group()


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


def _set_rank_env(rank: int, world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")


def _run_step(model: nn.Module, dp_optimizer, batch) -> torch.Tensor:
    x, y = batch
    dp_optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    dp_optimizer.step()
    return torch.cat([p.grad.reshape(-1).detach().cpu() for p in dp_optimizer.params])


def _worker_one_step_smoke(
    rank: int,
    world_size: int,
    init_method: str,
    results_dir: str,
) -> None:
    _set_rank_env(rank, world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DDP(nn.Linear(4, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        pe = PrivacyEngine()
        noise_gen = torch.Generator().manual_seed(123)
        loader = _loader()

        private_model, dp_optimizer, private_loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_generator=noise_gen,
            noise_mechanism=CorrelatedNoiseMechanism(coeffs=[1.0, 0.25], z_std=0.05),
        )

        grad = _run_step(private_model, dp_optimizer, next(iter(private_loader)))
        finite = bool(torch.isfinite(grad).all())
        torch.save({"rank": rank, "ok": finite}, Path(results_dir) / f"result_{rank}.pt")
    finally:
        dist.destroy_process_group()


def _worker_resume_parity(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_path: str,
    results_dir: str,
) -> None:
    _set_rank_env(rank, world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        base = nn.Linear(4, 3)
        ref_model = DDP(copy.deepcopy(base))
        resume_model = DDP(copy.deepcopy(base))
        load_model = DDP(copy.deepcopy(base))

        ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.05)
        resume_opt = torch.optim.SGD(resume_model.parameters(), lr=0.05)
        load_opt = torch.optim.SGD(load_model.parameters(), lr=0.05)

        pe_ref = PrivacyEngine()
        pe_resume = PrivacyEngine()
        pe_load = PrivacyEngine()

        loader_ref = _loader()
        loader_resume = _loader()
        loader_load = _loader()
        batches_ref = list(loader_ref)
        batches_resume = list(loader_resume)
        batches_load = list(loader_load)

        ref_gen = torch.Generator().manual_seed(321)
        resume_gen = torch.Generator().manual_seed(321)
        load_gen = torch.Generator().manual_seed(999)

        ref_model, ref_dp_opt, _ = pe_ref.make_private(
            module=ref_model,
            optimizer=ref_opt,
            data_loader=loader_ref,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_generator=ref_gen,
            noise_mechanism=CorrelatedNoiseMechanism(
                coeffs=[1.1, 0.3, -0.2], z_std=0.03
            ),
        )
        resume_model, resume_dp_opt, _ = pe_resume.make_private(
            module=resume_model,
            optimizer=resume_opt,
            data_loader=loader_resume,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_generator=resume_gen,
            noise_mechanism=CorrelatedNoiseMechanism(
                coeffs=[1.1, 0.3, -0.2], z_std=0.03
            ),
        )
        load_model, load_dp_opt, _ = pe_load.make_private(
            module=load_model,
            optimizer=load_opt,
            data_loader=loader_load,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_generator=load_gen,
            noise_mechanism=CorrelatedNoiseMechanism(
                coeffs=[1.1, 0.3, -0.2], z_std=0.03
            ),
        )

        for i in (0, 1):
            _run_step(ref_model, ref_dp_opt, batches_ref[i])
            _run_step(resume_model, resume_dp_opt, batches_resume[i])

        if rank == 0:
            pe_resume.save_checkpoint(
                path=checkpoint_path,
                module=resume_model,
                optimizer=resume_dp_opt,
            )
        dist.barrier()
        pe_load.load_checkpoint(
            path=checkpoint_path, module=load_model, optimizer=load_dp_opt
        )

        ref_grad = _run_step(ref_model, ref_dp_opt, batches_ref[2])
        load_grad = _run_step(load_model, load_dp_opt, batches_load[2])
        torch.save(
            {
                "rank": rank,
                "ok": bool(torch.allclose(ref_grad, load_grad, atol=1e-7, rtol=1e-6)),
            },
            Path(results_dir) / f"result_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _worker_resume_parity_bnb(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_path: str,
    results_dir: str,
) -> None:
    _set_rank_env(rank, world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        base = nn.Linear(4, 3)
        ref_model = DDP(copy.deepcopy(base))
        resume_model = DDP(copy.deepcopy(base))
        load_model = DDP(copy.deepcopy(base))

        ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.05)
        resume_opt = torch.optim.SGD(resume_model.parameters(), lr=0.05)
        load_opt = torch.optim.SGD(load_model.parameters(), lr=0.05)

        pe_ref = PrivacyEngine()
        pe_resume = PrivacyEngine()
        pe_load = PrivacyEngine()

        loader_ref = _loader()
        loader_resume = _loader()
        loader_load = _loader()
        batches_ref = list(loader_ref)
        batches_resume = list(loader_resume)
        batches_load = list(loader_load)

        ref_gen = torch.Generator().manual_seed(621)
        resume_gen = torch.Generator().manual_seed(621)
        load_gen = torch.Generator().manual_seed(999)

        common_kwargs = dict(
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"bands": 2},
            ),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="gaussian",
                accounting_mode="bnb_accountant",
            ),
        )

        ref_model, ref_dp_opt, _ = pe_ref.make_private(
            module=ref_model,
            optimizer=ref_opt,
            data_loader=loader_ref,
            noise_generator=ref_gen,
            **common_kwargs,
        )
        resume_model, resume_dp_opt, _ = pe_resume.make_private(
            module=resume_model,
            optimizer=resume_opt,
            data_loader=loader_resume,
            noise_generator=resume_gen,
            **common_kwargs,
        )
        load_model, load_dp_opt, _ = pe_load.make_private(
            module=load_model,
            optimizer=load_opt,
            data_loader=loader_load,
            noise_generator=load_gen,
            **common_kwargs,
        )

        for i in (0, 1):
            _run_step(ref_model, ref_dp_opt, batches_ref[i])
            _run_step(resume_model, resume_dp_opt, batches_resume[i])

        if rank == 0:
            pe_resume.save_checkpoint(
                path=checkpoint_path,
                module=resume_model,
                optimizer=resume_dp_opt,
            )
        dist.barrier()
        pe_load.load_checkpoint(
            path=checkpoint_path, module=load_model, optimizer=load_dp_opt
        )

        ref_grad = _run_step(ref_model, ref_dp_opt, batches_ref[2])
        load_grad = _run_step(load_model, load_dp_opt, batches_load[2])
        torch.save(
            {
                "rank": rank,
                "ok": bool(torch.allclose(ref_grad, load_grad, atol=1e-7, rtol=1e-6)),
            },
            Path(results_dir) / f"result_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _worker_one_step_bnb_smoke(
    rank: int,
    world_size: int,
    init_method: str,
    results_dir: str,
) -> None:
    _set_rank_env(rank, world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DDP(nn.Linear(4, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        pe = PrivacyEngine()
        noise_gen = torch.Generator().manual_seed(777)
        loader = _loader()

        private_model, dp_optimizer, private_loader = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            grad_sample_mode="hooks",
            noise_generator=noise_gen,
            sampling_semantics=SamplingSemantics(
                sampling_mode="b_min_sep",
                privacy_metadata={"bands": 2},
            ),
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism="gaussian",
                accounting_mode="bnb_accountant",
            ),
        )

        grad = _run_step(private_model, dp_optimizer, next(iter(private_loader)))
        finite = bool(torch.isfinite(grad).all())
        torch.save({"rank": rank, "ok": finite}, Path(results_dir) / f"result_{rank}.pt")
    finally:
        dist.destroy_process_group()


def _run_two_rank_workers(worker, *, checkpoint_path: str | None = None) -> list[tuple[int, bool]]:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.NamedTemporaryFile(delete=False) as sync:
            init_method = f"file://{sync.name}"

        args = (
            (2, init_method, tmpdir)
            if checkpoint_path is None
            else (2, init_method, checkpoint_path, tmpdir)
        )
        procs = [ctx.Process(target=worker, args=(rank, *args)) for rank in range(2)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=60)
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"distributed worker failed with exit code {proc.exitcode}"
                )

        results = []
        for rank in range(2):
            payload = torch.load(Path(tmpdir) / f"result_{rank}.pt")
            results.append((int(payload["rank"]), bool(payload["ok"])))

        results.sort(key=lambda x: x[0])
        return results


@pytest.mark.skipif(
    not RUN_DISTRIBUTED or not HAS_LOCAL_GLOO,
    reason=(
        "requires OPACUS_RUN_DISTRIBUTED_TESTS=1 and local gloo process-group support"
    ),
)
def test_distributed_bsr_two_rank_smoke() -> None:
    results = _run_two_rank_workers(_worker_one_step_smoke)
    assert results == [(0, True), (1, True)]


@pytest.mark.skipif(
    not RUN_DISTRIBUTED or not HAS_LOCAL_GLOO,
    reason=(
        "requires OPACUS_RUN_DISTRIBUTED_TESTS=1 and local gloo process-group support"
    ),
)
def test_distributed_bnb_two_rank_smoke() -> None:
    results = _run_two_rank_workers(_worker_one_step_bnb_smoke)
    assert results == [(0, True), (1, True)]


@pytest.mark.skipif(
    not RUN_DISTRIBUTED or not HAS_LOCAL_GLOO,
    reason=(
        "requires OPACUS_RUN_DISTRIBUTED_TESTS=1 and local gloo process-group support"
    ),
)
def test_distributed_bsr_two_rank_rank0_resume_parity() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as ckpt:
        checkpoint_path = ckpt.name

    results = _run_two_rank_workers(
        _worker_resume_parity,
        checkpoint_path=checkpoint_path,
    )
    assert results == [(0, True), (1, True)]


@pytest.mark.skipif(
    not RUN_DISTRIBUTED or not HAS_LOCAL_GLOO,
    reason=(
        "requires OPACUS_RUN_DISTRIBUTED_TESTS=1 and local gloo process-group support"
    ),
)
def test_distributed_bnb_two_rank_rank0_resume_parity() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as ckpt:
        checkpoint_path = ckpt.name

    results = _run_two_rank_workers(
        _worker_resume_parity_bnb,
        checkpoint_path=checkpoint_path,
    )
    assert results == [(0, True), (1, True)]
