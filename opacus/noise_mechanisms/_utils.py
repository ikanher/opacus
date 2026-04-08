from __future__ import annotations

from typing import List, Union

import torch
from torch.distributed.tensor import DTensor


def mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]) -> None:
    """
    Mark tensors that have already been consumed by a DP optimizer step.
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def check_processed_flag_tensor(x: torch.Tensor) -> None:
    """
    Reject reuse of tensors that have already been consumed by a DP step.
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]) -> None:
    """
    Check one tensor or a list of tensors for the processed marker.
    """

    if isinstance(obj, torch.Tensor):
        check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            check_processed_flag_tensor(x)


def generate_noise(
    std: float,
    reference: Union[torch.Tensor, DTensor],
    generator=None,
    secure_mode: bool = False,
) -> Union[torch.Tensor, DTensor]:
    """
    Generate Gaussian noise matching the shape/device of `reference`.
    """

    zeros = torch.zeros(reference.shape, device=reference.device, dtype=reference.dtype)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )
        total = zeros
        for _ in range(4):
            total += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return total / 2

    return torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
        generator=generator,
        dtype=reference.dtype,
    )
