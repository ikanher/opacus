"""
Generic streamed Toeplitz correlated-noise mechanisms.

This module implements the runtime noisers for matrix-factorization families
whose public noising object can be represented by a short lower-triangular
Toeplitz first column. The core runtime idea is the gradient-space recurrence

    `C u = z`

where `z` is iid Gaussian and `u` is the correlated noise actually added to the
clipped summed gradient stream.

Traceability:
- factor-side Toeplitz MF runtime surfaces follow BSR (Kalinin and Lampert,
  2024) and Scaling BandMF (McKenna, 2025)
- inverse-side streamed recurrence follows BISR (Kalinin et al., 2026)

All classes here are implementation-contract surfaces. They realize the
streaming recurrence used during training, but they do not themselves express a
privacy bound.
"""

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
    Streamed factor-side Toeplitz correlated-noise mechanism.

    The public runtime object is the lower-triangular Toeplitz first column
    `coeffs = [c_0, ..., c_{p-1}]`. At each step the mechanism draws iid
    Gaussian `z_t` with per-step scale `z_std`, then solves the forward
    substitution recurrence induced by `C u = z` online. The resulting `u_t` is
    added to the clipped summed gradient.

    Source: BSR (Kalinin and Lampert, 2024, Section 3.2, Equation (10)) for
    the Toeplitz-factor-side runtime view; Scaling BandMF (McKenna, 2025) for
    the scalable Toeplitz MF engineering setting.

    Mapping type: implementation-contract. The runtime recurrence is exact for
    the configured Toeplitz matrix `C`, but accountant semantics live outside
    this module.
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
        """Number of retained Toeplitz lags, including the diagonal coefficient."""
        return len(self.coeffs)

    @property
    def c0(self) -> float:
        """Leading Toeplitz coefficient `c_0`, which must remain strictly positive."""
        return self.coeffs[0]

    @property
    def state_depth(self) -> int:
        """Current number of cached past states held by the streamed recurrence."""
        return len(self._history)

    @property
    def max_state_depth(self) -> int:
        """Maximum cached history length required by the configured bandwidth."""
        return max(0, self.bandwidth - 1)

    def reset_state(self) -> None:
        """Drop the streamed recurrence history and cached debug tensors."""
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
        """
        Resolve the per-step iid draw scale seen by the streamed recurrence.

        When gradients were accumulated under mean-style reduction, the runtime
        recurrence must be fed the corresponding summed-gradient scale. This is
        a runtime normalization rule, not an accountant statement.
        """
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
        """
        Solve one Toeplitz forward-substitution step for `u_t`.

        If `coeffs = [c_0, c_1, ..., c_{p-1}]`, this computes

            `u_t = (z_t - sum_{i>=1} c_i u_{t-i}) / c_0`

        using only the cached recent history. The solve is exact for the
        streamed runtime representation of `C u = z`.
        """
        self._assert_finite(
            name="z_flat",
            tensor=z_flat,
            step=self.steps_with_noise,
        )
        rhs = z_flat
        max_lag = min(len(self._history), self.bandwidth - 1)
        for lag in range(1, max_lag + 1):
            # Subtract the already-solved lag contributions before dividing by
            # `c_0`; this is the streamed forward-substitution step.
            rhs = rhs - self.coeffs[lag] * self._history[lag - 1]

        u_flat = rhs / self.c0
        self._assert_finite(
            name="u_flat",
            tensor=u_flat,
            step=self.steps_with_noise,
        )
        if self._history.maxlen:
            # Cache the solved `u_t` because future factor-side steps depend on
            # past correlated noise values rather than past iid draws.
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
            # Slice the flattened release back into parameter-shaped chunks so
            # the optimizer sees the same layout it would under iid Gaussian noise.
            summed_chunk = summed_flat[offset:next_offset].reshape(shape)
            noise_chunk = u_flat[offset:next_offset].reshape(shape)
            p.grad = (summed_chunk + noise_chunk).view_as(p)

            mark_as_processed(p.summed_grad)
            offset = next_offset

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        """Add one correlated-noise step to the optimizer's clipped summed gradients."""
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
        """Serialize runtime-only Toeplitz recurrence state for checkpointing."""
        return {
            "coeffs": self.coeffs,
            "z_std": self.z_std,
            "history": [h.detach().clone() for h in self._history],
            "steps_with_noise": self.steps_with_noise,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore runtime-only Toeplitz recurrence state from a checkpoint payload."""
        coeffs = tuple(float(c) for c in state_dict.get("coeffs", self.coeffs))
        z_std = float(state_dict.get("z_std", self.z_std))
        history = state_dict.get("history", [])
        steps_with_noise = int(state_dict.get("steps_with_noise", 0))

        if coeffs != self.coeffs:
            raise ValueError("cannot load state with mismatched Toeplitz coefficients")

        if z_std != self.z_std:
            raise ValueError("cannot load state with mismatched z_std")

        # Restore only runtime recurrence state; accountant-side payloads should
        # never be threaded through this mechanism checkpoint.
        self._history = deque(maxlen=self.max_state_depth)
        for h in history:
            self._history.append(torch.as_tensor(h).detach().clone())

        self.steps_with_noise = steps_with_noise
        self.last_flat_z = None
        self.last_flat_u = None


class InverseBandNoiseMechanism(CorrelatedNoiseMechanism):
    """
    Streamed inverse-side banded runtime used by BISR / BandInvMF-style paths.

    Here the public runtime object is the inverse-side coefficient list
    `inverse_coeffs`. Instead of solving `C u = z` by forward substitution, the
    runtime directly applies the inverse-side lag recurrence

        `u_t = sum_i (C^{-1})_i z_{t-i}`

    online. This is the inverse-family implementation contract used by
    BISR (Kalinin et al., 2026) and BandInvMF-style runtime surfaces.
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
        """Inverse-side lag coefficients defining the streamed recurrence."""
        return self.coeffs

    def _solve_correlated_noise(self, z_flat: torch.Tensor) -> torch.Tensor:
        """
        Apply one inverse-side streamed recurrence step.

        This computes the current correlated-noise vector directly from the
        current iid draw and the cached past iid draws. The cached state is
        therefore `z`-history rather than `u`-history.
        """
        self._assert_finite(
            name="z_flat",
            tensor=z_flat,
            step=self.steps_with_noise,
        )
        u_flat = self.inverse_coeffs[0] * z_flat
        max_lag = min(len(self._history), self.bandwidth - 1)
        for lag in range(1, max_lag + 1):
            # In the inverse-side runtime the cached state is past iid draws,
            # because the current release is formed directly from `C^{-1} z`.
            u_flat = u_flat + self.inverse_coeffs[lag] * self._history[lag - 1]

        self._assert_finite(
            name="u_flat",
            tensor=u_flat,
            step=self.steps_with_noise,
        )
        if self._history.maxlen:
            # Cache `z_t`, not `u_t`: future inverse-side steps reuse past iid
            # draws under the explicit `C^{-1}` lag recurrence.
            self._history.appendleft(z_flat.detach().clone())

        return u_flat

    def state_dict(self) -> Mapping[str, Any]:
        """Serialize runtime-only inverse-side recurrence state for checkpointing."""
        return {
            "inverse_coeffs": self.inverse_coeffs,
            "coeffs": self.inverse_coeffs,
            "z_std": self.z_std,
            "history": [h.detach().clone() for h in self._history],
            "steps_with_noise": self.steps_with_noise,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore runtime-only inverse-side recurrence state from checkpoint data."""
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
