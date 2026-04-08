from __future__ import annotations

"""
BLT runtime noising mechanism.

This module owns the streamed BLT runtime release mechanism used during
centralized training. It consumes a canonical BLT parameter pair and emits the
correlated per-step noise sequence online. It does not own accountant-side
coefficient algebra or calibration search.
"""

from typing import Any, List, Mapping, Optional

import torch

from opacus.accountants.analysis.blt import BLTPairedParams

from ._utils import check_processed_flag, generate_noise, mark_as_processed
from .base import NoiseMechanism, NoiseMechanismOptimizer


def _pair_signature(pair: BLTPairedParams) -> tuple[tuple[float, ...], ...]:
    forward = pair.forward.canonicalized()
    inverse = pair.inverse.canonicalized()
    return (
        tuple(float(x) for x in forward.theta_array()),
        tuple(float(x) for x in forward.omega_array()),
        tuple(float(x) for x in inverse.theta_array()),
        tuple(float(x) for x in inverse.omega_array()),
    )


class BufferedToeplitzNoiseMechanism(NoiseMechanism):
    """
    Streamed BLT runtime mechanism.

    The runtime is parameterized by the canonical BLT pair
    `(theta, omega, theta_hat, omega_hat)` packaged as `BLTPairedParams`. The
    per-step runtime scalar `z_std` is the BLT runtime noise level. This should
    not be confused with amplified BNB accounting surfaces, which work from a
    normalized forward `c_col` instead.
    """

    def __init__(self, *, pair: BLTPairedParams, z_std: float):
        if float(z_std) < 0.0:
            raise ValueError("z_std must be >= 0")

        self.pair = pair.canonicalized()
        self.pair.validate()
        self.z_std = float(z_std)

        forward = self.pair.forward.canonicalized()
        self._theta = tuple(float(x) for x in forward.theta_array())
        self._omega = tuple(float(x) for x in forward.omega_array())
        self._buffers: list[torch.Tensor] = []
        self.last_flat_z: Optional[torch.Tensor] = None
        self.last_flat_u: Optional[torch.Tensor] = None
        self.steps_with_noise: int = 0

    @property
    def num_buffers(self) -> int:
        return len(self._theta)

    def reset_state(self) -> None:
        self._buffers = []
        self.last_flat_z = None
        self.last_flat_u = None
        self.steps_with_noise = 0

    def _pre_scale_noise_std(self, optimizer: NoiseMechanismOptimizer) -> float:
        if optimizer.loss_reduction == "sum":
            return self.z_std

        assert optimizer.expected_batch_size is not None
        return (
            self.z_std
            * float(optimizer.expected_batch_size)
            * float(optimizer.accumulated_iterations)
        )

    def _flatten_generated_z(
        self, optimizer: NoiseMechanismOptimizer, std: float
    ) -> tuple[List[tuple[torch.Tensor, torch.Size, int]], torch.Tensor]:
        specs: List[tuple[torch.Tensor, torch.Size, int]] = []
        chunks: List[torch.Tensor] = []

        for p in optimizer.params:
            assert p.summed_grad is not None
            check_processed_flag(p.summed_grad)
            noise = generate_noise(
                std=std,
                reference=p.summed_grad,
                generator=optimizer.generator,
                secure_mode=optimizer.secure_mode,
            )
            specs.append((p, p.shape, p.numel()))
            chunks.append(noise.reshape(-1))

        if not chunks:
            return specs, torch.zeros((0,), dtype=torch.float32)

        return specs, torch.cat(chunks, dim=0)

    def _flatten_summed_grads(
        self, specs: List[tuple[torch.Tensor, torch.Size, int]]
    ) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for p, _, _ in specs:
            assert p.summed_grad is not None
            chunks.append(p.summed_grad.reshape(-1))
        if not chunks:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def _assign_noised_grads(
        self,
        specs: List[tuple[torch.Tensor, torch.Size, int]],
        *,
        summed_flat: torch.Tensor,
        u_flat: torch.Tensor,
    ) -> None:
        offset = 0
        for p, shape, numel in specs:
            next_offset = offset + numel
            summed_chunk = summed_flat[offset:next_offset].reshape(shape)
            noise_chunk = u_flat[offset:next_offset].reshape(shape)
            p.grad = (summed_chunk + noise_chunk).view_as(p)
            mark_as_processed(p.summed_grad)
            offset = next_offset

    def _ensure_buffer_shape(self, reference: torch.Tensor) -> None:
        if self.num_buffers == 0:
            return
        if self._buffers:
            if len(self._buffers) != self.num_buffers:
                raise ValueError("invalid BLT buffer state")
            for buf in self._buffers:
                if buf.shape != reference.shape:
                    raise ValueError("BLT buffer shape mismatch")
                if buf.dtype != reference.dtype:
                    raise ValueError("BLT buffer dtype mismatch")
                if buf.device != reference.device:
                    raise ValueError("BLT buffer device mismatch")
            return
        self._buffers = [torch.zeros_like(reference) for _ in range(self.num_buffers)]

    def _read_state(self) -> torch.Tensor:
        assert self._buffers
        result = self._buffers[0] * self._omega[0]
        for omega_i, buf_i in zip(self._omega[1:], self._buffers[1:]):
            result = result + omega_i * buf_i
        return result

    def _apply_inverse_stream(self, z_flat: torch.Tensor) -> torch.Tensor:
        self._ensure_buffer_shape(z_flat)
        if self.num_buffers == 0:
            u_flat = z_flat.detach().clone()
        else:
            u_flat = z_flat - self._read_state()
            self._buffers = [
                theta_i * buf_i + u_flat
                for theta_i, buf_i in zip(self._theta, self._buffers)
            ]

        self.last_flat_z = z_flat.detach().clone()
        self.last_flat_u = u_flat.detach().clone()
        self.steps_with_noise += 1
        return u_flat

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        std = self._pre_scale_noise_std(optimizer)
        specs, z_flat = self._flatten_generated_z(optimizer, std=std)
        if not specs:
            return
        summed_flat = self._flatten_summed_grads(specs)
        u_flat = self._apply_inverse_stream(z_flat)
        self._assign_noised_grads(specs, summed_flat=summed_flat, u_flat=u_flat)

    def state_dict(self) -> Mapping[str, Any]:
        """Serialize the runtime-only BLT recurrence state for checkpointing."""
        return {
            "pair": {
                "forward": {
                    "theta": list(_pair_signature(self.pair)[0]),
                    "omega": list(_pair_signature(self.pair)[1]),
                },
                "inverse": {
                    "theta": list(_pair_signature(self.pair)[2]),
                    "omega": list(_pair_signature(self.pair)[3]),
                },
            },
            "z_std": self.z_std,
            "buffers": [buf.detach().clone() for buf in self._buffers],
            "steps_with_noise": self.steps_with_noise,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load a previously serialized runtime-only BLT recurrence state."""
        pair_state = state_dict.get("pair")
        z_std = float(state_dict.get("z_std", self.z_std))
        if z_std != self.z_std:
            raise ValueError("cannot load state with mismatched z_std")
        if pair_state is None:
            raise ValueError("missing BLT pair in mechanism state")

        state_signature = (
            tuple(float(x) for x in pair_state["forward"]["theta"]),
            tuple(float(x) for x in pair_state["forward"]["omega"]),
            tuple(float(x) for x in pair_state["inverse"]["theta"]),
            tuple(float(x) for x in pair_state["inverse"]["omega"]),
        )
        if state_signature != _pair_signature(self.pair):
            raise ValueError("cannot load state with mismatched BLT pair")

        self._buffers = [
            torch.as_tensor(buf).detach().clone()
            for buf in state_dict.get("buffers", [])
        ]
        self.steps_with_noise = int(state_dict.get("steps_with_noise", 0))
        self.last_flat_z = None
        self.last_flat_u = None


__all__ = ["BufferedToeplitzNoiseMechanism"]
