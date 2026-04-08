"""Public accountant wrapper for repeated random allocation.

This module exposes the public `accountant="random_allocation"` family in
Opacus. It adapts the generic accountant interface to the repeated
`k`-out-of-`t` random-allocation analysis package under
`opacus.accountants.analysis.random_allocation`.

This wrapper is intentionally limited to the repeated-accounting route. It does
not expose the later fixed-bin balls-in-bins bridge, which lives under the
analysis package and is used by dedicated report/calibration tooling instead of
the generic accountant registry.

Paper short names:

- `RA`  — "Efficient Random Allocation for DP Accounting" (Feldman and
  Shenfeld, 2025)
- `PLD` — "Random Allocation PLD Accounting" (Feldman and Shenfeld, 2025)
"""

from __future__ import annotations

from typing import Any

from opacus.accountants.analysis.random_allocation import (
    estimate_epsilon_random_allocation,
    resolve_random_allocation_accountant_inputs,
    resolve_random_allocation_gaussian_runtime_config,
)

from .accountant import IAccountant


class RandomAllocationAccountant(IAccountant):
    """Opacus accountant adapter for repeated random allocation.

    The wrapper records a constant-noise history, validates the required
    `sampling_mode="k_out_of_t"` semantics, resolves the repeated
    random-allocation analysis inputs, and returns the dominating `ε(δ)` query
    from the analysis layer.

    This class only supports the repeated route. It does not evaluate the
    fixed-bin bridge.
    """

    def __init__(self):
        super().__init__()

    def step(self, *, noise_multiplier: float, sample_rate: float):
        """Record one optimizer step under constant-noise accounting.

        Consecutive steps with identical `noise_multiplier` and `sample_rate`
        are coalesced into one history segment.
        """
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))
        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def __len__(self) -> int:
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        """Return the public accountant registry name."""
        return "random_allocation"

    def get_epsilon(
        self,
        delta: float,
        *,
        mechanism_state: Any = None,
        sampling_semantics=None,
        **kwargs,
    ) -> float:
        """Return the dominating repeated random-allocation `ε(δ)` query.

        Args:
            delta: Target privacy failure probability `δ`.
            mechanism_state: Optional mechanism metadata used to resolve the
                coefficient vector and mechanism family.
            sampling_semantics: Required Opacus sampling contract. This wrapper
                expects `sampling_mode="k_out_of_t"` plus `privacy_metadata`
                containing `num_steps` and `num_selected`.
            **kwargs: Optional runtime overrides forwarded to
                `resolve_random_allocation_gaussian_runtime_config`.

        Raises:
            ValueError: If the history contains mixed noise/sample-rate
                segments, the sampling semantics are incompatible, or the
                recorded step count is not aligned with the repeated sampler
                epochs.
        """
        if not self.history:
            return 0.0

        noise_multiplier, sample_rate, total_steps = self.history[0]
        for nm_i, sr_i, steps_i in self.history:
            if nm_i != noise_multiplier or sr_i != sample_rate:
                raise ValueError(
                    "random_allocation accountant currently expects constant noise_multiplier and sample_rate across steps"
                )

            total_steps = steps_i

        if (
            sampling_semantics is None
            or sampling_semantics.sampling_mode != "k_out_of_t"
        ):
            raise ValueError(
                "random_allocation accountant requires sampling_semantics with sampling_mode='k_out_of_t'"
            )

        metadata = sampling_semantics.privacy_metadata
        num_steps = int(metadata.get("num_steps"))
        num_selected = int(metadata.get("num_selected"))
        if total_steps % num_steps != 0:
            raise ValueError(
                f"random_allocation accountant currently requires completed sampler epochs; got recorded_steps={int(total_steps)}, num_steps={int(num_steps)}"
            )

        horizon = int(total_steps)
        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        persisted_kwargs = state.get("_random_allocation_accounting_kwargs", {})
        inputs = resolve_random_allocation_accountant_inputs(
            mechanism=str(state.get("mechanism", state.get("name", "gaussian"))),
            mechanism_state=state,
            sampling_semantics=sampling_semantics,
            kwargs={
                "num_selected": num_selected,
                "bnb_horizon": horizon,
            },
            noise_multiplier=float(noise_multiplier),
        )
        runtime_cfg = resolve_random_allocation_gaussian_runtime_config(
            target_delta=float(delta),
            runtime_policy=kwargs.get(
                "random_allocation_runtime_policy",
                persisted_kwargs.get(
                    "random_allocation_runtime_policy",
                    "strict_exact_package",
                ),
            ),
            loss_discretization=kwargs.get(
                "random_allocation_loss_discretization",
                persisted_kwargs.get("random_allocation_loss_discretization"),
            ),
            tail_truncation=kwargs.get(
                "random_allocation_tail_truncation",
                persisted_kwargs.get("random_allocation_tail_truncation"),
            ),
            max_grid_fft=kwargs.get(
                "random_allocation_max_grid_fft",
                persisted_kwargs.get("random_allocation_max_grid_fft"),
            ),
            max_grid_mult=kwargs.get(
                "random_allocation_max_grid_mult",
                persisted_kwargs.get("random_allocation_max_grid_mult"),
            ),
            convolution_method=kwargs.get(
                "random_allocation_convolution_method",
                persisted_kwargs.get("random_allocation_convolution_method"),
            ),
        )

        return float(
            estimate_epsilon_random_allocation(
                inputs=inputs,
                target_delta=float(delta),
                runtime_config=runtime_cfg,
            )
        )
