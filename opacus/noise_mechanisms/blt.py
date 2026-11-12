"""
BLT runtime noising mechanism.

This module owns the streamed BLT runtime release mechanism used during
centralized training. It consumes a canonical BLT parameter pair and emits the
correlated per-step noise sequence online. It does not own accountant-side
coefficient algebra or calibration search.

Traceability:
- BLT runtime parameter pairs and streaming practice follow BLT Practice
  (McMahan et al., 2024)
- the paired forward/inverse parameterization reused here is provided by the
  accountant-side BLT helpers in `opacus.accountants.analysis.blt`

Mapping type: implementation-contract. This module realizes the runtime BLT
release recurrence; accountant guarantees remain outside this file.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from opacus.accountants.analysis.blt import BLTPairedParams

from ._utils import check_processed_flag, generate_noise, mark_as_processed
from .base import NoiseMechanism, NoiseMechanismOptimizer


def _pair_signature(pair: BLTPairedParams) -> tuple[tuple[float, ...], ...]:
    """Return a stable numeric signature for BLT pair-equality checks in checkpoints."""
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

    Source: BLT Practice (McMahan et al., 2024) for the buffered
    lower-triangular Toeplitz runtime family.

    Mapping type: implementation-contract. The recurrence is exact for the
    provided BLT pair, while privacy interpretation remains accountant-side.

    `record_noise_diagnostics` defaults to false. Enable it explicitly to retain
    copies of the latest raw and correlated noise in last_flat_z and last_flat_u
    for debugging. This adds two full noise-vector copies and their per-step
    copy work; it does not change the noise or recurrence. Required recurrence
    buffers are counted separately by noise_state_inventory().
    """

    def __init__(
        self,
        *,
        pair: BLTPairedParams,
        z_std: float,
        record_noise_diagnostics: bool = False,
    ):
        if float(z_std) < 0.0:
            raise ValueError("z_std must be >= 0")

        self.record_noise_diagnostics = bool(record_noise_diagnostics)
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
        """Number of streamed BLT state buffers required by the forward side."""
        return len(self._theta)

    def noise_state_inventory(self):
        """Return how many bytes BLT buffers and diagnostic tensors retain.

        Returns category -> memory location -> integer byte count. History counts
        recurrence buffers; diagnostics counts optional last_flat_z/last_flat_u
        copies; scratch is empty. Shared backing allocations are counted once,
        including the whole allocation for a view. Excludes within-step
        temporaries and model/gradient memory. Does not clone or synchronize.
        """
        from ._utils import storage_inventory

        return storage_inventory(self._buffers, [self.last_flat_z, self.last_flat_u])

    def reset_state(self) -> None:
        """Drop the buffered BLT recurrence state and cached debug tensors."""
        self._buffers = []
        self.last_flat_z = None
        self.last_flat_u = None
        self.steps_with_noise = 0

    def _pre_scale_noise_std(self, optimizer: NoiseMechanismOptimizer) -> float:
        """
        Resolve the per-step iid draw scale consumed by the BLT recurrence.

        Mean-style reductions are rescaled back to the summed-gradient regime so
        the runtime recurrence acts on the same release object the optimizer is
        about to update.
        """
        if optimizer.loss_reduction == "sum":
            return self.z_std

        assert optimizer.expected_batch_size is not None
        return (
            self.z_std
            * float(optimizer.expected_batch_size)
            * float(optimizer.accumulated_iterations)
        )

    def _ensure_buffer_shape(self, reference: torch.Tensor) -> None:
        """
        Materialize or validate the BLT state buffers for the current flat shape.

        The runtime stores one buffer per forward-side `theta` entry. Checkpoint
        restores therefore need explicit shape/dtype/device validation before
        the recurrence can continue.
        """
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

        # First runtime step (or reset): materialize one zero buffer per BLT
        # factor so later steps can update the streamed state in place.
        self._buffers = [torch.zeros_like(reference) for _ in range(self.num_buffers)]

    def _advance_inverse_stream(self, u_flat, block_sizes):
        """Cancel old state before updating any buffer, using block-sized scratch.

        The owned input starts as iid noise and becomes the correlated release.
        Multiplication and addition stay separate to preserve their rounding.
        """
        self._ensure_buffer_shape(u_flat)
        self.last_flat_z = (
            u_flat.detach().clone() if self.record_noise_diagnostics else None
        )
        if self._buffers:
            offset = 0

            for size in block_sizes:
                end = offset + size
                weighted = self._buffers[0][offset:end] * self._omega[0]

                for omega, buf in zip(self._omega[1:], self._buffers[1:]):
                    block = omega * buf[offset:end]
                    weighted.add_(block)
                    del block

                u_flat[offset:end].sub_(weighted)

                offset = end

                # Make sure temproary memory is freed
                del weighted

            # Every old buffer has contributed to the complete release above.
            # Reuse its storage instead of retaining old and replacement sets.
            for theta, buf in zip(self._theta, self._buffers):
                buf.mul_(theta)
                buf.add_(u_flat)

        self.last_flat_u = (
            u_flat.detach().clone() if self.record_noise_diagnostics else None
        )
        self.steps_with_noise += 1

        return u_flat

    def _apply_inverse_stream(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Apply the recurrence without modifying the caller's iid draw."""
        return self._advance_inverse_stream(z_flat.detach().clone(), [z_flat.numel()])

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        """Add BLT noise with one full release and parameter-sized workspace."""
        params = list(optimizer.params)
        if not params:
            return

        for p in params:
            assert p.summed_grad is not None
            check_processed_flag(p.summed_grad)

        dtype = params[0].summed_grad.dtype
        for p in params[1:]:
            dtype = torch.promote_types(dtype, p.summed_grad.dtype)

        u_flat = torch.empty(
            sum(p.numel() for p in params), device=params[0].device, dtype=dtype
        )

        std = self._pre_scale_noise_std(optimizer)
        offset = 0
        for p in params:
            block = generate_noise(
                std=std,
                reference=p.summed_grad,
                generator=optimizer.generator,
                secure_mode=optimizer.secure_mode,
            )
            u_flat[offset : offset + p.numel()].copy_(block.reshape(-1))
            offset += p.numel()

            # Make sure temporary memory is freed
            del block

        self._advance_inverse_stream(u_flat, [p.numel() for p in params])
        offset = 0

        for p in params:
            noise = u_flat[offset : offset + p.numel()].view_as(p)
            p.grad = (p.summed_grad + noise).view_as(p)
            mark_as_processed(p.summed_grad)
            offset += p.numel()

    def state_dict(self) -> Mapping[str, Any]:
        """
        Serialize the runtime-only BLT recurrence state for checkpointing.

        The payload records the canonical BLT pair signature, the runtime noise
        scale, and the current streamed buffers. It is not an accountant state.
        """
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
        """
        Load a previously serialized runtime-only BLT recurrence state.

        The restore validates both `z_std` and the canonical BLT pair so that a
        checkpoint cannot silently cross wires between incompatible BLT runs.
        """
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

        # The buffers are the only mutable runtime state; the pair itself is
        # treated as immutable configuration and must match exactly.
        self._buffers = [
            torch.as_tensor(buf).detach().clone()
            for buf in state_dict.get("buffers", [])
        ]
        self.steps_with_noise = int(state_dict.get("steps_with_noise", 0))
        self.last_flat_z = None
        self.last_flat_u = None


__all__ = ["BufferedToeplitzNoiseMechanism"]
