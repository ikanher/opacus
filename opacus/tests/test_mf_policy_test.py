from __future__ import annotations

import pytest
import torch

from opacus.mf.optimizer_utils import resolve_uniform_sgd_workload_from_optimizer


def test_resolve_uniform_sgd_workload_from_optimizer_reads_uniform_groups() -> None:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight], "momentum": 0.9, "weight_decay": 0.01},
            {"params": [model.bias], "momentum": 0.9, "weight_decay": 0.01},
        ],
        lr=0.1,
    )

    momentum, weight_decay = resolve_uniform_sgd_workload_from_optimizer(
        optimizer=optimizer
    )

    assert momentum == pytest.approx(0.9)
    assert weight_decay == pytest.approx(0.01)


def test_resolve_uniform_sgd_workload_from_optimizer_rejects_mixed_momentum() -> None:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight], "momentum": 0.9, "weight_decay": 0.01},
            {"params": [model.bias], "momentum": 0.8, "weight_decay": 0.01},
        ],
        lr=0.1,
    )

    with pytest.raises(ValueError, match="uniform optimizer momentum"):
        resolve_uniform_sgd_workload_from_optimizer(optimizer=optimizer)
