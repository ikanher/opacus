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

import copy
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
        self._history: Deque[torch.Tensor] = deque(maxlen=max(0, len(self.coeffs) - 1))
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

    def noise_state_inventory(self):
        """Return how many bytes historical noise and diagnostic tensors retain.

        Returns category → memory location → integer byte count. History counts
        cached correlated noise vectors; diagnostics counts last_flat_z and
        last_flat_u copies; scratch is empty. Shared backing allocations are
        counted once, including the whole allocation for a view. Excludes
        within-step temporaries and model/gradient memory. Does not clone or
        synchronize devices.
        """
        from ._utils import storage_inventory

        return storage_inventory(self._history, [self.last_flat_z, self.last_flat_u])

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
            p.grad = (summed_chunk + noise_chunk).to(dtype=p.dtype).view_as(p)

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
    Apply the finite inverse filter for BIFR, BISR and BandInvMF.

    Here the public runtime object is the inverse-side coefficient list
    `inverse_coeffs`. Instead of solving `C u = z` by forward substitution, the
    runtime directly applies the inverse-side lag recurrence

        `u_t = sum_i (C^{-1})_i z_{t-i}`

    online. This is the inverse-family implementation contract used by
    BISR (Kalinin et al., 2026 draft), Algorithm 1, “Differentially Private SGD
    with Banded Inverse Matrix Factorization”, and BIFR (Kalinin et al., 2026
    draft, ICLR version), Algorithm 1, “Differentially Private Model Training
    with γ-BIFR”, and Section 3's “Noise Regeneration” paragraph.

    Mapping type: implementation-contract. For supplied inverse coefficients
    a_j, this evaluates u_t = ∑ a_j z_{t-j}, with missing startup lags zero.
    It does not derive coefficients or certify a privacy bound. Exactness is
    algebraic; floating-point releases are compared with dtype tolerances.

    Args:
        inverse_coeffs: Finite first column of C⁻¹; its length p is bandwidth,
            including the current draw. Coefficients must be finite, with
            a_0 > MIN_C0. At most p-1 historical descriptors/vectors are kept.
        z_std: Finite nonnegative iid noise standard deviation in the caller's
            reduction convention, not a dimensionless DP noise multiplier.
            Mean reduction multiplies it by expected_batch_size and
            accumulated_iterations before drawing on the summed-gradient scale.
        debug_non_finite: Include tensor summaries in non-finite diagnostics.
        noise_history_mode: "regenerate" (default) keeps PRNG snapshots and
            original draw scales; "stored" keeps full historical iid vectors.
        record_noise_diagnostics: Retain last_flat_z and last_flat_u copies
            when true; defaults to false. Their storage is inventoried separately.

    One device per noiser is required. Layout/order must stay fixed until reset;
    secure replay is rejected. Ordinary CPU replay is validated locally;
    CUDA/ROCm replay has separate accelerator acceptance tests pending execution.
    """

    def __init__(
        self,
        *,
        inverse_coeffs: Sequence[float],
        z_std: float,
        debug_non_finite: bool = False,
        noise_history_mode: str = "regenerate",
        record_noise_diagnostics: bool = False,
    ):
        super().__init__(
            coeffs=inverse_coeffs,
            z_std=z_std,
            debug_non_finite=debug_non_finite,
        )

        if noise_history_mode not in ("regenerate", "stored"):
            raise ValueError("noise_history_mode must be regenerate or stored")

        self.noise_history_mode = noise_history_mode
        self.record_noise_diagnostics = bool(record_noise_diagnostics)
        self._replay_history = deque(maxlen=self.max_state_depth)
        self._layout = None
        self._parameter_ids = None
        self._live_generator = None
        self._default_generator = False

    @property
    def state_depth(self):
        """Number of retained prior draws, in [0, p-1], excluding the fresh draw."""
        history = (
            self._replay_history
            if self.noise_history_mode == "regenerate"
            else self._history
        )
        return len(history)

    def reset_state(self):
        """Clear history, layout and counters without rewinding any generator."""
        super().reset_state()

        self._replay_history = deque(maxlen=self.max_state_depth)
        self._layout = self._parameter_ids = None
        self._live_generator = None
        self._default_generator = False

    @staticmethod
    def _draw(params, std, generator, secure_mode=False):
        # Parameter-wise shapes and order affect RNG consumption; one flat draw
        # is not an interchangeable replay of these calls. Zero scale consumes no RNG.
        return torch.cat(
            [
                generate_noise(
                    std=std,
                    reference=p.summed_grad,
                    generator=generator,
                    secure_mode=secure_mode,
                ).reshape(-1)
                for p in params
            ]
        )

    def _validate_parameters(self, params):
        """Check layout and gradient availability before any RNG advancement."""
        layout = tuple((tuple(p.shape), str(p.dtype), str(p.device)) for p in params)
        identities = tuple(id(p) for p in params)
        if len({entry[2] for entry in layout}) != 1:
            raise ValueError("inverse noise requires a single-device parameter layout")

        if self._layout is not None and (
            layout != self._layout
            or (self._parameter_ids is not None and identities != self._parameter_ids)
        ):
            raise ValueError("inverse noise parameter layout changed with live state")

        for p in params:
            if p.summed_grad is None:
                raise ValueError("missing summed gradient")
            check_processed_flag(p.summed_grad)

        return layout, identities

    def add_noise(self, optimizer):
        """Apply the finite inverse filter using PRNG replay or stored draws.

        Adaptation of BIFR (Kalinin et al., 2026 draft, ICLR version), Algorithm 1,
        “Differentially Private Model Training with γ-BIFR”. Consumes
        optimizer.summed_grad and writes optimizer.grad.
        The paper replays sequentially before drawing fresh noise; here a private
        stream restores each prior snapshot after the one fresh draw. This also
        handles external draws on a shared default stream and historical scales.
        Replay leaves that live stream untouched. Skipped microbatches must be
        filtered by the optimizer before calling this method.
        The finite filter is unchanged; no accountant guarantee is added.
        """
        params = list(optimizer.params)
        if not params:
            return

        if optimizer.secure_mode and self.noise_history_mode == "regenerate":
            raise ValueError(
                "secure generator replay is not supported; use stored history mode"
            )

        layout, identities = self._validate_parameters(params)

        device = params[0].device
        generator = optimizer.generator
        default = generator is None
        if default:
            generator = (
                torch.default_generator
                if device.type == "cpu"
                else torch.cuda.default_generators[device.index or 0]
            )

        if not isinstance(generator, torch.Generator) or generator.device != device:
            raise ValueError("unsupported noise generator/device for inverse replay")

        std = self._pre_scale_noise_std(optimizer)
        if not math.isfinite(std) or std < 0:
            raise ValueError("noise scale must be finite and nonnegative")

        # Keep snapshots only for history; the opaque live generator is borrowed.
        before = (
            generator.get_state() if self.noise_history_mode == "regenerate" else None
        )
        if self.noise_history_mode == "stored":
            z_flat = self._draw(params, std, generator, optimizer.secure_mode)
            self._assert_finite(
                name="z_flat", tensor=z_flat, step=self.steps_with_noise
            )
            u_flat = self._solve_correlated_noise(z_flat)
            summed_flat = torch.cat([p.summed_grad.reshape(-1) for p in params])
            self._assert_finite(
                name="summed_flat", tensor=summed_flat, step=self.steps_with_noise
            )
            self._assert_finite(
                name="u_flat", tensor=u_flat, step=self.steps_with_noise
            )
            self._assign_noised_grads(
                [(p, p.shape, p.numel()) for p in params],
                summed_flat=summed_flat,
                u_flat=u_flat,
            )
        else:
            z_flat, u_flat = self._regenerate_noise(params, std, generator)
            if self.max_state_depth:
                self._replay_history.appendleft((before, std))
            offset = 0
            for p in params:
                noise = u_flat[offset : offset + p.numel()].view_as(p)
                self._assert_finite(
                    name="summed_flat", tensor=p.summed_grad, step=self.steps_with_noise
                )
                self._assert_finite(
                    name="u_flat", tensor=noise, step=self.steps_with_noise
                )
                p.grad = (p.summed_grad + noise).to(dtype=p.dtype)
                mark_as_processed(p.summed_grad)
                offset += p.numel()

        self._layout, self._parameter_ids = layout, identities
        self._live_generator, self._default_generator = generator, default
        self.last_flat_z = (
            z_flat.detach().clone() if self.record_noise_diagnostics else None
        )
        self.last_flat_u = (
            u_flat.detach().clone() if self.record_noise_diagnostics else None
        )
        self.steps_with_noise += 1

    def _regenerate_noise(self, params, std, generator):
        """Replay lag-major into one accumulator without full-vector draw copies.

        Parameter-shaped calls and their order must match the original stream.
        Keep multiplication separate from addition to preserve its rounding.
        Temporary payload is one full accumulator plus parameter-sized sampling
        and conversion workspace; diagnostic raw-noise capture is opt-in.
        """
        dtype = params[0].dtype
        for p in params[1:]:
            dtype = torch.promote_types(dtype, p.dtype)

        u_flat = torch.empty(
            sum(p.numel() for p in params), device=params[0].device, dtype=dtype
        )

        z_flat = torch.empty_like(u_flat) if self.record_noise_diagnostics else None

        def accumulate(scale, stream, coefficient, fresh=False):
            offset = 0
            for p in params:
                block = generate_noise(
                    std=scale, reference=p.summed_grad, generator=stream
                ).to(dtype=dtype)

                if fresh:
                    self._assert_finite(
                        name="z_flat", tensor=block, step=self.steps_with_noise
                    )
                    if z_flat is not None:
                        z_flat[offset : offset + p.numel()].copy_(block.reshape(-1))

                block.mul_(coefficient)
                target = u_flat[offset : offset + p.numel()]

                if fresh:
                    target.copy_(block.reshape(-1))
                else:
                    target.add_(block.reshape(-1))

                offset += p.numel()

                # Make sure temporary memory allocations are freed
                del block

        accumulate(std, generator, self.inverse_coeffs[0], fresh=True)

        # Draw fresh once, then restore each historical snapshot privately.
        # A complete parameter traversal per lag preserves RNG consumption.
        replay = torch.Generator(device=params[0].device)
        for lag, (state, historical_std) in enumerate(self._replay_history, 1):
            replay.set_state(state)
            accumulate(historical_std, replay, self.inverse_coeffs[lag])

        return z_flat, u_flat

    def validate_state_dict(self, state):
        """Preflight mode, coefficients, layout and replay RNG compatibility.

        Uses an isolated probe generator, without restoring the live stream.
        DPOptimizer additionally checks current parameters and RNG ownership.
        Legacy vector history is accepted only in explicit stored mode.
        This is compatibility checking, not a general untrusted-payload validator.
        """
        version = state.get("schema_version")
        if version is None and self.noise_history_mode != "stored":
            raise ValueError("legacy vector history requires explicit stored mode")
        if version is not None and version != 1:
            raise ValueError("incompatible inverse replay schema version")
        if state.get("noise_history_mode", "stored") != self.noise_history_mode:
            raise ValueError("incompatible noise history mode")

        coeffs = tuple(
            state.get("inverse_coeffs", state.get("coeffs", self.inverse_coeffs))
        )
        if (
            coeffs != self.inverse_coeffs
            or state.get("z_std", self.z_std) != self.z_std
        ):
            raise ValueError("incompatible inverse coefficients or noise scale")

        layout = state.get("layout_signature")
        if layout is not None:
            if not isinstance(layout, (tuple, list)) or not all(
                isinstance(x, (tuple, list)) and len(x) == 3 for x in layout
            ):
                raise ValueError("incompatible parameter layout signature")
            layout = tuple(tuple(x) for x in layout)
            if self._layout is not None and layout != self._layout:
                raise ValueError("incompatible parameter layout")

        history = (
            state.get("replay_history", [])
            if self.noise_history_mode == "regenerate"
            else state.get("history", [])
        )
        if len(history) > self.max_state_depth:
            raise ValueError("incompatible history length")

        if self.noise_history_mode == "regenerate":
            self._validate_replay_history(state, history, layout)

    @staticmethod
    def _validate_replay_history(state, history, layout):
        """Check saved draw scales and RNG states with an isolated generator."""
        if history and not layout:
            raise ValueError("replay history requires layout")

        device = torch.device(layout[0][2]) if layout else torch.device("cpu")
        probe = torch.Generator(device=device)
        try:
            for rng, std in history:
                if not math.isfinite(std) or std < 0:
                    raise ValueError("invalid historical scale")
                probe.set_state(rng.cpu())
            if state.get("default_generator_state") is not None:
                probe.set_state(state["default_generator_state"].cpu())
        except (TypeError, RuntimeError) as exc:
            raise ValueError("incompatible replay generator state/backend") from exc

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

    def state_dict(self):
        """Copy versioned history, layout and completed-noise-step count.

        Also saves default-stream continuation when used. Dedicated-generator
        continuation belongs to DPOptimizer.state_dict; a mechanism-only save
        requires the caller to checkpoint that generator separately. RNG state
        is private training state and must not be written to telemetry.
        """
        state = {
            "schema_version": 1,
            "noise_history_mode": self.noise_history_mode,
            "inverse_coeffs": self.inverse_coeffs,
            "coeffs": self.coeffs,
            "z_std": self.z_std,
            "steps_with_noise": self.steps_with_noise,
            "layout_signature": self._layout,
        }
        if self.noise_history_mode == "regenerate":
            state["replay_history"] = [
                (rng.clone(), std) for rng, std in self._replay_history
            ]
        else:
            state["history"] = [h.detach().clone() for h in self._history]
        if self._default_generator and self._live_generator is not None:
            state["default_generator_state"] = self._live_generator.get_state()
        return state

    def load_state_dict(self, state):
        """Restore compatible history and, if present, the device's default RNG.

        This can rewind global randomness. Dedicated RNG continuation is restored
        by DPOptimizer. Cross-device/version bitwise portability is not promised.
        """
        self.validate_state_dict(state)
        self.reset_state()
        self._layout = copy.deepcopy(state.get("layout_signature"))
        self.steps_with_noise = int(state.get("steps_with_noise", 0))

        if self.noise_history_mode == "regenerate":
            self._replay_history.extend(
                (rng.cpu().clone(), float(std))
                for rng, std in state.get("replay_history", [])
            )
        else:
            self._history.extend(h.detach().clone() for h in state.get("history", []))

        default = state.get("default_generator_state")
        if default is not None:
            device = torch.device(self._layout[0][2])
            generator = (
                torch.default_generator
                if device.type == "cpu"
                else torch.cuda.default_generators[device.index or 0]
            )

            generator.set_state(default.cpu())
            self._live_generator, self._default_generator = generator, True

    def noise_state_inventory(self):
        """Return how many bytes replay history and diagnostic tensors retain.

        Returns category → memory location → integer byte count. History counts
        saved RNG-state tensors in regenerate mode, or past iid noise vectors
        in stored mode; diagnostics counts optional last_flat_z/last_flat_u
        copies; scratch is empty. Locations are actual storage locations, so
        RNG snapshots kept on CPU are reported under cpu even for GPU training.

        Shared backing allocations are counted once, including the whole
        allocation for a view. Excludes within-step temporaries, model/gradient
        memory, Python objects and opaque live generators. Reads metadata only:
        no tensor copies, noise regeneration or device synchronization.
        """
        from ._utils import storage_inventory

        history = (
            [rng for rng, _ in self._replay_history]
            if self.noise_history_mode == "regenerate"
            else list(self._history)
        )

        return storage_inventory(history, [self.last_flat_z, self.last_flat_u])
