from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Deque, List, Mapping, Optional, Sequence

import torch

from ._utils import check_processed_flag, generate_noise, mark_as_processed
from .base import NoiseMechanism, NoiseMechanismOptimizer


MIN_C0 = 1e-12


class CorrelatedNoiseMechanism(NoiseMechanism):
    """
    Correlated-noise mechanism with lower-triangular Toeplitz solve.
    """

    def __init__(
        self,
        *,
        coeffs: Sequence[float],
        z_std: float,
        debug_non_finite: bool = False,
    ):
        if not math.isfinite(z_std) or z_std < 0.0:
            raise ValueError("z_std must be finite and >= 0")

        if len(coeffs) == 0:
            raise ValueError("coeffs must be non-empty")

        self.coeffs = tuple(float(c) for c in coeffs)
        if not all(math.isfinite(c) for c in self.coeffs):
            raise ValueError("all coefficients must be finite")

        if self.coeffs[0] <= MIN_C0:
            raise ValueError(f"coeffs[0] must be > {MIN_C0:g}")

        self.z_std = float(z_std)
        self._history: Deque[torch.Tensor] = deque(
            maxlen=max(0, len(self.coeffs) - 1)
        )
        self.last_flat_z: Optional[torch.Tensor] = None
        self.last_flat_u: Optional[torch.Tensor] = None
        self.steps_with_noise: int = 0
        self.debug_non_finite: bool = bool(debug_non_finite) or (
            os.getenv("OPACUS_BSR_DEBUG_FINITE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.first_non_finite_event: Optional[dict[str, Any]] = None

    @property
    def bandwidth(self) -> int:
        return len(self.coeffs)

    @property
    def c0(self) -> float:
        return self.coeffs[0]

    @property
    def state_depth(self) -> int:
        return len(self._history)

    @property
    def max_state_depth(self) -> int:
        return max(0, self.bandwidth - 1)

    def reset_state(self) -> None:
        self._history = deque(maxlen=self.max_state_depth)
        self.last_flat_z = None
        self.last_flat_u = None
        self.steps_with_noise = 0
        self.first_non_finite_event = None

    def _describe_tensor(self, *, name: str, tensor: torch.Tensor, step: int) -> str:
        numel = int(tensor.numel())
        finite_mask = torch.isfinite(tensor)
        n_non_finite = int((~finite_mask).sum().item())
        message = (
            f"non-finite correlated-noise tensor detected: {name} at step={step}; "
            f"numel={numel}, non_finite={n_non_finite}"
        )
        if not self.debug_non_finite:
            return message

        finite_values = tensor[finite_mask]
        if finite_values.numel() == 0:
            return f"{message}, finite_min=nan, finite_max=nan, finite_l2=nan"

        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        finite_l2 = float(torch.linalg.vector_norm(finite_values).item())
        return (
            f"{message}, finite_min={finite_min:.6g}, "
            f"finite_max={finite_max:.6g}, finite_l2={finite_l2:.6g}"
        )

    def _assert_finite(self, *, name: str, tensor: torch.Tensor, step: int) -> None:
        if torch.isfinite(tensor).all():
            return

        if self.first_non_finite_event is None:
            self.first_non_finite_event = {
                "step": int(step),
                "tensor": str(name),
                "numel": int(tensor.numel()),
                "non_finite": int((~torch.isfinite(tensor)).sum().item()),
            }
        raise ValueError(self._describe_tensor(name=name, tensor=tensor, step=step))

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

    def _solve_correlated_noise(self, z_flat: torch.Tensor) -> torch.Tensor:
        self._assert_finite(
            name="z_flat",
            tensor=z_flat,
            step=self.steps_with_noise,
        )
        rhs = z_flat
        max_lag = min(len(self._history), self.bandwidth - 1)
        for lag in range(1, max_lag + 1):
            rhs = rhs - self.coeffs[lag] * self._history[lag - 1]

        u_flat = rhs / self.c0
        self._assert_finite(
            name="u_flat",
            tensor=u_flat,
            step=self.steps_with_noise,
        )
        if self._history.maxlen:
            self._history.appendleft(u_flat.detach().clone())

        return u_flat

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

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        std = self._pre_scale_noise_std(optimizer)
        specs, z_flat = self._flatten_generated_z(optimizer, std=std)

        if not specs:
            return

        summed_flat = self._flatten_summed_grads(specs)
        self._assert_finite(
            name="summed_flat",
            tensor=summed_flat,
            step=self.steps_with_noise,
        )
        u_flat = self._solve_correlated_noise(z_flat)

        self._assign_noised_grads(specs, summed_flat=summed_flat, u_flat=u_flat)

        self.last_flat_z = z_flat.detach().clone()
        self.last_flat_u = u_flat.detach().clone()
        self.steps_with_noise += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "coeffs": self.coeffs,
            "z_std": self.z_std,
            "history": [h.detach().clone() for h in self._history],
            "steps_with_noise": self.steps_with_noise,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        coeffs = tuple(float(c) for c in state_dict.get("coeffs", self.coeffs))
        z_std = float(state_dict.get("z_std", self.z_std))
        history = state_dict.get("history", [])
        steps_with_noise = int(state_dict.get("steps_with_noise", 0))

        if coeffs != self.coeffs:
            raise ValueError("cannot load state with mismatched Toeplitz coefficients")
        if z_std != self.z_std:
            raise ValueError("cannot load state with mismatched z_std")

        self._history = deque(maxlen=self.max_state_depth)
        for h in history:
            self._history.append(torch.as_tensor(h).detach().clone())

        self.steps_with_noise = steps_with_noise
        self.last_flat_z = None
        self.last_flat_u = None


class InverseBandNoiseMechanism(CorrelatedNoiseMechanism):
    """
    Direct inverse-band runtime used by canonical BISR / BandInvMF paths.
    """

    def __init__(
        self,
        *,
        inverse_coeffs: Sequence[float],
        z_std: float,
        debug_non_finite: bool = False,
    ):
        super().__init__(
            coeffs=inverse_coeffs,
            z_std=z_std,
            debug_non_finite=debug_non_finite,
        )

    @property
    def inverse_coeffs(self) -> tuple[float, ...]:
        return self.coeffs

    def _solve_correlated_noise(self, z_flat: torch.Tensor) -> torch.Tensor:
        self._assert_finite(
            name="z_flat",
            tensor=z_flat,
            step=self.steps_with_noise,
        )
        u_flat = self.inverse_coeffs[0] * z_flat
        max_lag = min(len(self._history), self.bandwidth - 1)
        for lag in range(1, max_lag + 1):
            u_flat = u_flat + self.inverse_coeffs[lag] * self._history[lag - 1]

        self._assert_finite(
            name="u_flat",
            tensor=u_flat,
            step=self.steps_with_noise,
        )
        if self._history.maxlen:
            self._history.appendleft(z_flat.detach().clone())

        return u_flat

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "inverse_coeffs": self.inverse_coeffs,
            "coeffs": self.inverse_coeffs,
            "z_std": self.z_std,
            "history": [h.detach().clone() for h in self._history],
            "steps_with_noise": self.steps_with_noise,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        coeffs = tuple(
            float(c)
            for c in state_dict.get(
                "inverse_coeffs",
                state_dict.get("coeffs", self.inverse_coeffs),
            )
        )
        z_std = float(state_dict.get("z_std", self.z_std))
        history = state_dict.get("history", [])
        steps_with_noise = int(state_dict.get("steps_with_noise", 0))

        if coeffs != self.inverse_coeffs:
            raise ValueError("cannot load state with mismatched inverse-band coefficients")
        if z_std != self.z_std:
            raise ValueError("cannot load state with mismatched z_std")

        self._history = deque(maxlen=self.max_state_depth)
        for h in history:
            self._history.append(torch.as_tensor(h).detach().clone())

        self.steps_with_noise = steps_with_noise
        self.last_flat_z = None
        self.last_flat_u = None
