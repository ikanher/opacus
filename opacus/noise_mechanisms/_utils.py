"""
Shared runtime helpers for noise mechanisms.

These helpers own three pieces of runtime plumbing shared by both iid and
correlated mechanisms:
- processed-flag bookkeeping on `summed_grad` buffers
- reuse detection across optimizer steps
- Gaussian sampling aligned to a reference tensor's shape/device/dtype

They are implementation-contract helpers only; they do not encode accountant
claims.
"""

from __future__ import annotations

from typing import List, Union

import torch
from torch.distributed.tensor import DTensor


def mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]) -> None:
    """
    Mark tensors that have already been consumed by a DP optimizer step.

    Noise mechanisms call this after they have converted `summed_grad` into the
    step-local `grad` release. Reusing the same summed buffer without a
    `zero_grad()` would invalidate the runtime privacy contract, so subsequent
    checks reject that reuse explicitly.
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def check_processed_flag_tensor(x: torch.Tensor) -> None:
    """
    Reject reuse of a tensor already consumed by a DP step.
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
    Generate Gaussian noise matching the shape/device/dtype of `reference`.

    Args:
        std: Target marginal standard deviation for each entry.
        reference: Tensor whose shape, device, and dtype define the output
            layout.
        generator: Optional PyTorch random generator.
        secure_mode: Whether to use the hardened Opacus sampling path instead of
            a single `torch.normal` call.

    Returns:
        A tensor with the same shape/device/dtype as `reference`.

    Notes:
        - `std == 0` returns an all-zero tensor.
        - The secure-mode path preserves the same target variance but avoids the
          plain single-call sampling route. This is an implementation-contract
          hardening detail, not a new accountant surface.
    """

    zeros = torch.zeros(reference.shape, device=reference.device, dtype=reference.dtype)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        # Follow the hardened Opacus sampling route: discard one scalar draw,
        # then combine four iid draws so the returned tensor keeps variance
        # `std^2` without using the plain single-call path.
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
