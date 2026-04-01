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

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from opacus.accountants import create_accountant


MAX_SIGMA = 1e6
MAX_NOISE_SEARCH_BINARY_STEPS = 256
MIN_NOISE_SEARCH_SIGMA_INTERVAL = 1e-12


def _timing_enabled() -> bool:
    return os.getenv("DEBUG_TIMING", "").strip().lower() not in ("", "0", "false", "no")


def _debug_timing(message: str) -> None:
    if _timing_enabled():
        print(f"[opacus.get_noise_multiplier] [timing] {message}", flush=True)


@contextmanager
def _timed(label: str):
    if not _timing_enabled():
        yield
        return
    start = time.perf_counter()
    _debug_timing(f"start {label}")
    try:
        yield
    finally:
        _debug_timing(f"done {label} elapsed={time.perf_counter() - start:.3f}s")


@dataclass
class NoiseSearchConvergenceError(ValueError):
    accountant: str
    target_epsilon: float
    target_delta: float
    last_finite_sigma: float | None
    last_finite_epsilon: float | None
    last_nonfinite_sigma: float | None
    iterations: int

    def __str__(self) -> str:
        return (
            "Noise search did not converge for accountant="
            f"{self.accountant!r} target_epsilon={self.target_epsilon} "
            f"target_delta={self.target_delta}; last_finite_sigma={self.last_finite_sigma} "
            f"last_finite_epsilon={self.last_finite_epsilon} "
            f"last_nonfinite_sigma={self.last_nonfinite_sigma} iterations={self.iterations}"
        )


def _resolve_random_allocation_debug_context(*, sample_rate: float, steps: int, kwargs: dict) -> str:
    state = kwargs.get("mechanism_state") if isinstance(kwargs.get("mechanism_state"), dict) else {}
    semantics = kwargs.get("sampling_semantics")
    metadata = semantics.privacy_metadata if semantics is not None and hasattr(semantics, "privacy_metadata") else {}
    mechanism = str(state.get("mechanism", state.get("name", "gaussian")))
    coeffs = kwargs.get(
        "random_allocation_accountant_coeffs",
        state.get("random_allocation_accountant_coeffs", state.get("coeffs")),
    )
    coeff_norm = None
    if isinstance(coeffs, (list, tuple)) and coeffs:
        coeff_norm = math.sqrt(sum(float(c) * float(c) for c in coeffs))
    num_steps = metadata.get("num_steps")
    num_selected = metadata.get("num_selected")
    num_epochs = None
    reduced_num_steps_per_round = None
    reduced_num_rounds = None
    if num_steps is not None:
        num_steps = int(num_steps)
    if num_selected is not None:
        num_selected = int(num_selected)
    if num_steps and num_selected and steps % num_steps == 0:
        num_epochs = int(steps // num_steps)
        reduced_num_steps_per_round = int(num_steps // num_selected)
        reduced_num_rounds = int(num_selected * num_epochs)
    return (
        f"mechanism={mechanism} sample_rate={sample_rate:.12g} steps={steps} "
        f"num_steps={num_steps!r} num_selected={num_selected!r} num_epochs={num_epochs!r} "
        f"reduced_num_steps_per_round={reduced_num_steps_per_round!r} "
        f"reduced_num_rounds={reduced_num_rounds!r} coeff_norm={coeff_norm!r}"
    )


def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    search_kwargs = dict(kwargs)
    with _timed(f"accountant={accountant} target_epsilon={target_epsilon} target_delta={target_delta} steps={steps}"):
        eps_high = float("inf")
        accountant_name = str(accountant)
        accountant = create_accountant(mechanism=accountant_name)
        last_finite_sigma: float | None = None
        last_finite_epsilon: float | None = None
        last_nonfinite_sigma: float | None = None
        iterations = 0

        sigma_low, sigma_high = 0, 10
        expand_iter = 0
        if accountant_name == "random_allocation":
            _debug_timing(
                "random_allocation search_context "
                + _resolve_random_allocation_debug_context(
                    sample_rate=float(sample_rate),
                    steps=int(steps),
                    kwargs=search_kwargs,
                )
            )
        while eps_high > target_epsilon:
            sigma_high = 2 * sigma_high
            accountant.history = [(sigma_high, sample_rate, steps)]
            with _timed(f"expand_iter={expand_iter} sigma={sigma_high:.12g}"):
                eps_high = accountant.get_epsilon(delta=target_delta, **search_kwargs)
            _debug_timing(
                f"expand_iter={expand_iter} sigma={sigma_high:.12g} eps={eps_high!r}"
            )
            if math.isfinite(eps_high):
                last_finite_sigma = float(sigma_high)
                last_finite_epsilon = float(eps_high)
            else:
                last_nonfinite_sigma = float(sigma_high)
            if sigma_high > MAX_SIGMA:
                raise ValueError("The privacy budget is too low.")
            expand_iter += 1

        while target_epsilon - eps_high > epsilon_tolerance:
            if iterations >= MAX_NOISE_SEARCH_BINARY_STEPS:
                raise NoiseSearchConvergenceError(
                    accountant=accountant_name,
                    target_epsilon=float(target_epsilon),
                    target_delta=float(target_delta),
                    last_finite_sigma=last_finite_sigma,
                    last_finite_epsilon=last_finite_epsilon,
                    last_nonfinite_sigma=last_nonfinite_sigma,
                    iterations=iterations,
                )
            if sigma_high - sigma_low <= MIN_NOISE_SEARCH_SIGMA_INTERVAL:
                raise NoiseSearchConvergenceError(
                    accountant=accountant_name,
                    target_epsilon=float(target_epsilon),
                    target_delta=float(target_delta),
                    last_finite_sigma=last_finite_sigma,
                    last_finite_epsilon=last_finite_epsilon,
                    last_nonfinite_sigma=last_nonfinite_sigma,
                    iterations=iterations,
                )
            sigma = (sigma_low + sigma_high) / 2
            if sigma <= sigma_low or sigma >= sigma_high:
                raise NoiseSearchConvergenceError(
                    accountant=accountant_name,
                    target_epsilon=float(target_epsilon),
                    target_delta=float(target_delta),
                    last_finite_sigma=last_finite_sigma,
                    last_finite_epsilon=last_finite_epsilon,
                    last_nonfinite_sigma=last_nonfinite_sigma,
                    iterations=iterations,
                )
            accountant.history = [(sigma, sample_rate, steps)]
            with _timed(
                f"binary_iter={iterations} sigma={sigma:.12g} bracket=[{sigma_low:.12g},{sigma_high:.12g}]"
            ):
                eps = accountant.get_epsilon(delta=target_delta, **search_kwargs)
            _debug_timing(
                f"binary_iter={iterations} sigma={sigma:.12g} eps={eps!r} "
                f"last_finite_sigma={last_finite_sigma!r} last_nonfinite_sigma={last_nonfinite_sigma!r}"
            )
            iterations += 1

            if math.isfinite(eps) and eps < target_epsilon:
                sigma_high = sigma
                eps_high = eps
                last_finite_sigma = float(sigma)
                last_finite_epsilon = float(eps)
            else:
                sigma_low = sigma
                if not math.isfinite(eps):
                    last_nonfinite_sigma = float(sigma)

        return sigma_high
