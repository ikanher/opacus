"""Real process groups; no mocked reductions or rank ownership."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus import NoiseMechanismConfig, PrivacyEngine
from opacus.optimizers import DistributedDPOptimizer
from opacus.tests.test_inverse_noise_regeneration_test import (
    assert_tree_equal,
    coefficients,
    mechanism,
    optimizer,
    update,
)


def _build_worker_optimizer(family, p, device, seed, fault):
    if family == "explicit":
        noiser = mechanism(p)
    else:
        key = {
            "bifr": "bifr_inv_coeffs",
            "bisr": "bisr_inv_coeffs",
            "bandinvmf": "bandinvmf_inv_coeffs",
        }[family]
        noiser = PrivacyEngine._build_noise_mechanism_from_config(
            NoiseMechanismConfig(
                mechanism=family,
                accounting_mode="bsr_accountant",
                mechanism_state={key: coefficients(p), "z_std": 0.2},
            )
        )
    if fault == "zero_noise":
        noiser.z_std = 0.0
    param = torch.nn.Parameter(torch.zeros(3, device=device))
    opt = DistributedDPOptimizer(
        optimizer=torch.optim.SGD([param], lr=0.05, momentum=0.9),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
        generator=torch.Generator(device=device).manual_seed(seed),
        noise_mechanism=noiser,
    )
    return param, opt, noiser


def _assert_reference_update(actual, ref, device, step):
    matches = torch.ones((), dtype=torch.int32, device=device)
    if ref is not None:
        try:
            torch.testing.assert_close(actual, update(ref), rtol=1e-6, atol=1e-7)
        except AssertionError:
            matches.zero_()
    # All workers fail at the same boundary: otherwise a peer's Gloo
    # disconnect can hide the numerical assertion in the parent.
    dist.all_reduce(matches, op=dist.ReduceOp.MIN)
    assert matches.item() == 1, f"numerical reference mismatch at step {step}"


def _worker(rank, world, rendezvous, out_dir, family, backend, p, fault=None):
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    try:

        param, opt, noiser = _build_worker_optimizer(family, p, device, 31, fault)
        # Rank 0 owns the noise: use that same accelerator for the global-batch
        # reference even when the checking worker is on another accelerator.
        ref = (
            optimizer(p, device="cuda:0" if backend == "nccl" else "cpu")
            if rank == 0
            else None
        )
        resumed_nonowner_depth = None
        for step in range(p + 4):
            actual = update(opt, (1, 1))
            _assert_reference_update(actual, ref, device, step)
            # All worker parameters must agree after each nonzero-noise release.
            gathered = [torch.empty_like(param) for _ in range(world)]
            dist.all_gather(gathered, param.detach())
            for other in gathered:
                torch.testing.assert_close(other, param)
            if step == p:
                saved = [copy.deepcopy(opt.state_dict()) if rank == 0 else None]
                dist.broadcast_object_list(saved, src=0)
                saved_params = param.detach().clone()
                param, opt, noiser = _build_worker_optimizer(
                    family, p, device, 999, fault
                )
                # Diverge history/moments and stream before restoring, then set
                # the separately checkpointed model parameter back to its value.
                update(opt, (1, 1))
                param.data.copy_(saved_params)
                if fault != "discard_checkpoint":
                    opt.load_state_dict(saved[0])
                resumed_nonowner_depth = noiser.state_depth if rank else 0
        # Finish collectives before checking missing behavior, so failures cannot hang peers.
        bad = copy.deepcopy(opt.state_dict())
        bad["_dp_distributed_saved_rank"] = 0
        bad["_dp_distributed_world_size"] = world + 1
        before_bad = copy.deepcopy(opt.state_dict())
        rejected = False
        try:
            opt.load_state_dict(bad)
        except ValueError:
            rejected = True
        if rejected:
            assert_tree_equal(opt.state_dict(), before_bad)
        torch.save(
            {
                "post_resume_nonowner_depth": resumed_nonowner_depth,
                "world_size_rejected": rejected,
                "state_depth": noiser.state_depth,
            },
            f"{out_dir}/rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _run(tmp_path, family, backend="gloo", p=4, fault=None):
    mp.spawn(
        _worker,
        args=(2, f"file://{tmp_path}/sync", str(tmp_path), family, backend, p, fault),
        nprocs=2,
        join=True,
    )
    records = [
        torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)
    ]
    violations = []
    if records[1]["post_resume_nonowner_depth"] != 0:
        violations.append("checkpoint restore copied rank-0 history onto rank 1")
    if not all(r["world_size_rejected"] for r in records):
        violations.append("changed world size was accepted")
    assert not violations, violations


@pytest.mark.parametrize("family", ["explicit", "bifr", "bisr", "bandinvmf"])
def test_real_gloo_inverse_resume_ownership_and_scaling(tmp_path, family):
    _run(tmp_path, family)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two real accelerators"
)
def test_real_multi_gpu_inverse_replay_resume(tmp_path):
    _run(tmp_path, "bifr", "nccl", 64)


@pytest.mark.parametrize("fault", ["discard_checkpoint", "zero_noise"])
def test_distributed_numerical_oracle_rejects_broken_implementations(tmp_path, fault):
    # A meta-regression: the worker must fail its numerical assertion before
    # metadata checks when restoration is discarded or noise is missing.
    with pytest.raises(mp.ProcessRaisedException, match="numerical reference mismatch"):
        _run(tmp_path, "explicit", fault=fault)
