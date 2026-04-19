"""
Fixed-batch BLT accountant.

This module owns the current accountant-backed BLT privacy path in Opacus:
fixed-batch accounting under `sampling_mode='torch_sampler'`. It combines the
canonical BLT parameter pair with a fixed-batch max-loss term and reduces the
result to the existing Gaussian fixed-batch accountant surface.

Source: BLT Practice (McMahan et al., 2024) for the buffered Toeplitz runtime
family; BSR (Kalinin and Lampert, 2024) for the reduced fixed-batch Gaussian
accountant expression reused here.

Claim-type notes:
- the BLT parameter/input resolution is an implementation contract
- `compute_blt_fixed_batch_max_loss` is an accountant-side loss surrogate on
  the current fixed-batch contract
- this module does not claim amplified BLT accounting
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math

from opacus.accountants.analysis.blt import BLTPairedParams, blt_coeffs
from opacus.accountants.analysis.bsr import bsr_fixed_batch_epsilon_upper_bound
from opacus.accountants.blt_inputs import resolve_blt_fixed_batch_accountant_inputs

from .accountant import IAccountant


def _is_nonnegative_decreasing(coeffs: list[float], *, atol: float = 1e-12) -> bool:
    """Return whether the finite-horizon forward BLT column is nonnegative decreasing."""
    if len(coeffs) == 0:
        return False
    if coeffs[0] < -atol:
        return False
    prev = coeffs[0]
    for c in coeffs[1:]:
        if c < -atol:
            return False
        if c > prev + atol:
            return False
        prev = c
    return True


def compute_blt_fixed_batch_max_loss(
    *,
    pair: BLTPairedParams,
    horizon: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compute the fixed-batch BLT max-loss surrogate on the current horizon.

    The loss is the product of:
    - the fixed-batch factor-side sensitivity induced by the forward BLT column
    - the prefix RMSE term induced by the inverse BLT column

    Returns:
        A positive finite surrogate consumed by the current fixed-batch BLT
        accountant reduction.

    Mapping type: accountant-side implementation contract.
    """
    if horizon < 1:
        raise ValueError("blt_horizon must be >= 1")
    if max_participations < 1:
        raise ValueError("blt_max_participations must be >= 1")
    if min_separation < 1:
        raise ValueError("blt_min_separation must be >= 1")

    forward_coeffs = [float(x) for x in blt_coeffs(pair.forward, int(horizon))]
    if not _is_nonnegative_decreasing(forward_coeffs):
        raise ValueError(
            "BLT fixed-batch accounting requires forward coefficients to be nonnegative decreasing on the accounting horizon"
        )

    # Fixed-batch support only allows as many participations as fit into the
    # finite horizon under the declared minimum separation.
    k_eff = min(int(max_participations), ((int(horizon) - 1) // int(min_separation)) + 1)
    total_sq = 0.0
    for i in range(int(horizon)):
        j_max = min(k_eff - 1, i // int(min_separation))
        row_sum = 0.0
        for j in range(j_max + 1):
            # Sum the left-packed fixed-batch row contributions that are still
            # visible on the finite horizon.
            lag = i - j * int(min_separation)
            if lag < len(forward_coeffs):
                row_sum += forward_coeffs[lag]
        total_sq += row_sum * row_sum
    sensitivity = math.sqrt(total_sq)

    inverse_coeffs = [float(x) for x in blt_coeffs(pair.inverse, int(horizon))]
    prefix = 0.0
    err_sq = 0.0
    for c in inverse_coeffs:
        # The inverse-side prefix sums give the BLT prefix-error term on the
        # same finite horizon.
        prefix += c
        err_sq += prefix * prefix
    maxerr = math.sqrt(err_sq)

    loss = float(maxerr) * float(sensitivity)
    if not math.isfinite(loss) or loss <= 0.0:
        raise ValueError("derived BLT fixed-batch max loss must be finite and > 0")
    return float(loss)


class BLTAccountant(IAccountant):
    """
    Fixed-batch BLT accountant.

    This module owns the first BLT accountant contract currently exposed in
    Opacus: fixed-batch accounting under `sampling_mode='torch_sampler'`. It
    does not own the amplified BNB bridge; that accountant-input boundary lives
    in `opacus.accountants.blt_inputs` and `opacus.accountants.analysis.blt`.
    """

    def __init__(self):
        """Initialize an empty fixed-batch BLT accountant history."""
        super().__init__()
        self.last_contract = None

    @staticmethod
    def _resolve_fixed_batch_contract_inputs(
        *,
        state: dict,
        metadata: dict,
        kwargs: dict,
        total_steps: int,
    ) -> tuple[BLTPairedParams, float, int, int, int]:
        """Resolve the canonical fixed-batch BLT accountant tuple from state/metadata."""
        return resolve_blt_fixed_batch_accountant_inputs(
            runtime_state=state,
            metadata=metadata,
            kwargs=kwargs,
            total_steps=int(total_steps),
        )

    def step(self, *, noise_multiplier: float, sample_rate: float):
        """Record one BLT fixed-batch event, coalescing consecutive identical entries."""
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

    def get_epsilon(
        self,
        delta: float,
        *,
        mechanism_state=None,
        sampling_semantics=None,
        **kwargs,
    ) -> float:
        """
        Return the fixed-batch BLT epsilon upper bound for the current history.

        Contract:
        - fixed-batch only
        - BLT parameter surface comes from `theta`, `theta_hat`, `omega`,
          `omega_hat` via the canonical input resolver
        - `noise_multiplier_ref` is the accountant-side Gaussian reference sigma
          used with the derived BLT max-loss term

        Claim type: implementation contract backed by the current fixed-batch
        BLT accountant path, not a generic amplified BLT statement.
        """
        if not self.history:
            return 0.0

        total_steps = 0
        sample_rate = self.history[0][1]
        for _nm_i, sr_i, steps_i in self.history:
            if sr_i != sample_rate:
                raise ValueError(
                    "blt accountant currently expects constant sample_rate across fixed-batch steps"
                )
            total_steps += int(steps_i)

        metadata = (
            sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
        )
        sampling_mode = (
            sampling_semantics.sampling_mode if sampling_semantics is not None else None
        )
        if sampling_mode != "torch_sampler":
            raise ValueError(
                "BLT accountant currently supports fixed-batch semantics only "
                "(sampling_mode='torch_sampler')"
            )

        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        pair, noise_multiplier_ref, max_participations, min_separation, horizon = (
            self._resolve_fixed_batch_contract_inputs(
                state=state,
                metadata=metadata,
                kwargs=kwargs,
                total_steps=int(total_steps),
            )
        )
        max_loss = compute_blt_fixed_batch_max_loss(
            pair=pair,
            horizon=int(horizon),
            max_participations=int(max_participations),
            min_separation=int(min_separation),
        )
        self.last_contract = {
            "mechanism": "blt",
            "accounting_mode": "blt",
            "accounting_contract": "fixed_batch",
            "sampling_mode": "torch_sampler",
            "global_steps": int(total_steps),
            "sample_rate": float(sample_rate),
            "horizon": int(horizon),
            "max_participations": int(max_participations),
            "min_separation": int(min_separation),
            "blt_max_loss": float(max_loss),
            "noise_multiplier_ref": float(noise_multiplier_ref),
            "accountant_backend": "fixed_batch_prv",
        }
        return float(
            bsr_fixed_batch_epsilon_upper_bound(
                noise_multiplier=float(noise_multiplier_ref),
                target_delta=float(delta),
                mf_sensitivity=float(max_loss),
            )
        )

    @classmethod
    def mechanism(cls) -> str:
        """Return the accountant mechanism tag used by the privacy-engine router."""
        return "blt"

    def __len__(self):
        """Return the number of coalesced BLT history segments."""
        return len(self.history)
