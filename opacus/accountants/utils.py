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

from opacus.accountants.registry import create_accountant


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


def _random_allocation_runtime_policy(kwargs: dict) -> str:
    return str(kwargs.get("random_allocation_runtime_policy", "strict_exact_package"))


def _should_bootstrap_random_allocation(kwargs: dict) -> bool:
    return not bool(kwargs.get("_random_allocation_disable_bootstrap", False))


def _refined_random_allocation_kwargs(kwargs: dict, refinement_round: int) -> dict:
    refined = dict(kwargs)
    base_loss = float(refined.get("random_allocation_loss_discretization", 5e-2))
    refined["random_allocation_loss_discretization"] = base_loss / (2 ** refinement_round)
    return refined


def _random_allocation_interpolated_sigma(
    *,
    sigma_low: float,
    eps_low: float | None,
    sigma_high: float,
    eps_high: float,
    target_epsilon: float,
) -> float | None:
    if (
        eps_low is None
        or not math.isfinite(float(eps_low))
        or not math.isfinite(float(eps_high))
        or sigma_low <= 0.0
        or sigma_high <= sigma_low
        or float(eps_low) <= float(eps_high)
    ):
        return None

    denominator = float(eps_low) - float(eps_high)
    if denominator <= 0.0:
        return None

    frac = (float(target_epsilon) - float(eps_high)) / denominator
    frac = min(0.95, max(0.05, frac))
    x_low = math.log(float(sigma_low))
    x_high = math.log(float(sigma_high))
    sigma = math.exp(x_high + frac * (x_low - x_high))
    min_sigma = float(sigma_low) + 0.1 * (float(sigma_high) - float(sigma_low))
    max_sigma = float(sigma_high) - 0.1 * (float(sigma_high) - float(sigma_low))
    if not math.isfinite(sigma):
        return None
    return min(max(sigma, min_sigma), max_sigma)


def _random_allocation_downward_extrapolated_sigma(
    *,
    sigma_high: float,
    eps_high: float,
    target_epsilon: float,
) -> float | None:
    if (
        not math.isfinite(float(sigma_high))
        or not math.isfinite(float(eps_high))
        or float(sigma_high) <= 0.0
        or float(eps_high) <= 0.0
        or float(target_epsilon) <= float(eps_high)
    ):
        return None
    ratio = float(eps_high) / float(target_epsilon)
    scale = min(0.8, max(0.05, ratio))
    sigma = float(sigma_high) * scale
    if not math.isfinite(sigma) or sigma <= 0.0 or sigma >= float(sigma_high):
        return None
    return sigma


def _bootstrap_random_allocation_sigma(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    epsilon_tolerance: float,
    kwargs: dict,
) -> float | None:
    bootstrap_kwargs = dict(kwargs)
    bootstrap_kwargs["_random_allocation_disable_bootstrap"] = True
    bootstrap_kwargs["random_allocation_runtime_policy"] = (
        "repository_fast_random_allocation"
    )
    bootstrap_kwargs.pop("_progress_label", None)
    bootstrap_kwargs.pop("_progress_printer", None)
    if "random_allocation_loss_discretization" not in bootstrap_kwargs:
        bootstrap_kwargs["random_allocation_loss_discretization"] = 5e-2

    try:
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="random_allocation",
            epsilon_tolerance=max(float(epsilon_tolerance) * 4.0, 0.25),
            **bootstrap_kwargs,
        )
    except (NoiseSearchConvergenceError, ValueError):
        return None

    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return None

    return sigma


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
    if accountant == "random_allocation" and "random_allocation_runtime_policy" not in search_kwargs:
        search_kwargs["random_allocation_runtime_policy"] = "efficient_staged_grid"
    if (
        accountant == "random_allocation"
        and search_kwargs.get("random_allocation_runtime_policy") == "efficient_staged_grid"
        and "random_allocation_loss_discretization" not in search_kwargs
    ):
        search_kwargs["random_allocation_loss_discretization"] = 2e-1
    progress_label = search_kwargs.pop("_progress_label", None)
    progress_printer = search_kwargs.pop("_progress_printer", None)
    ra_runtime_policy = _random_allocation_runtime_policy(search_kwargs)
    ra_bootstrap_enabled = (
        accountant == "random_allocation"
        and ra_runtime_policy == "efficient_staged_grid"
        and _should_bootstrap_random_allocation(search_kwargs)
    )

    def _emit_progress(message: str) -> None:
        if progress_printer is not None:
            progress_printer(message)

    with _timed(f"accountant={accountant} target_epsilon={target_epsilon} target_delta={target_delta} steps={steps}"):
        eps_high = float("inf")
        accountant_name = str(accountant)
        accountant = create_accountant(mechanism=accountant_name)
        last_finite_sigma: float | None = None
        last_finite_epsilon: float | None = None
        last_nonfinite_sigma: float | None = None
        iterations = 0

        def _epsilon_for_sigma(sigma_value: float, kwargs_override: dict | None = None):
            epsilon_kwargs = dict(search_kwargs if kwargs_override is None else kwargs_override)
            if accountant_name == "blt":
                epsilon_kwargs["noise_multiplier_ref"] = float(sigma_value)
            accountant.history = [(float(sigma_value), sample_rate, steps)]
            return accountant.get_epsilon(delta=target_delta, **epsilon_kwargs)

        sigma_low, sigma_high = 0.0, 10.0
        eps_low: float | None = None
        bootstrap_sigma: float | None = None
        if ra_bootstrap_enabled:
            bootstrap_sigma = _bootstrap_random_allocation_sigma(
                target_epsilon=float(target_epsilon),
                target_delta=float(target_delta),
                sample_rate=float(sample_rate),
                steps=int(steps),
                epsilon_tolerance=float(epsilon_tolerance),
                kwargs=search_kwargs,
            )
            if bootstrap_sigma is not None:
                sigma_high = float(bootstrap_sigma)
                if progress_label is not None:
                    _emit_progress(
                        f"{progress_label} phase=bootstrap_seed sigma={sigma_high:.12g} "
                        "runtime_policy=repository_fast_random_allocation"
                    )
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
        with _timed(f"expand_iter={expand_iter} sigma={sigma_high:.12g}"):
            eps_high = _epsilon_for_sigma(sigma_high)
        if progress_label is not None:
            _emit_progress(
                f"{progress_label} phase=expand_high iter={expand_iter} "
                f"sigma={sigma_high:.12g} epsilon={eps_high!r}"
            )
        if math.isfinite(eps_high):
            last_finite_sigma = float(sigma_high)
            last_finite_epsilon = float(eps_high)
        else:
            last_nonfinite_sigma = float(sigma_high)
        expand_iter += 1

        bootstrap_bracket_scale = 1.25 if bootstrap_sigma is not None else 2.0
        if (
            accountant_name == "random_allocation"
            and ra_runtime_policy == "efficient_staged_grid"
            and bootstrap_sigma is not None
            and math.isfinite(eps_high)
            and eps_high <= target_epsilon
        ):
            sigma_low = max(
                float(bootstrap_sigma) / bootstrap_bracket_scale,
                MIN_NOISE_SEARCH_SIGMA_INTERVAL,
            )
            bootstrap_low_iter = 0
            while sigma_low > MIN_NOISE_SEARCH_SIGMA_INTERVAL:
                with _timed(
                    f"bootstrap_low_iter={bootstrap_low_iter} sigma={sigma_low:.12g}"
                ):
                    eps_low_candidate = _epsilon_for_sigma(sigma_low)
                if progress_label is not None:
                    _emit_progress(
                        f"{progress_label} phase=bootstrap_low iter={bootstrap_low_iter} "
                        f"sigma={sigma_low:.12g} epsilon={eps_low_candidate!r}"
                    )
                iterations += 1
                if math.isfinite(eps_low_candidate) and eps_low_candidate >= target_epsilon:
                    eps_low = float(eps_low_candidate)
                    last_finite_sigma = float(sigma_low)
                    last_finite_epsilon = float(eps_low_candidate)
                    break
                if not math.isfinite(eps_low_candidate):
                    last_nonfinite_sigma = float(sigma_low)
                    break
                sigma_high = sigma_low
                eps_high = float(eps_low_candidate)
                last_finite_sigma = float(sigma_low)
                last_finite_epsilon = float(eps_low_candidate)
                sigma_low = max(
                    _random_allocation_downward_extrapolated_sigma(
                        sigma_high=float(sigma_low),
                        eps_high=float(eps_low_candidate),
                        target_epsilon=float(target_epsilon),
                    )
                    or (sigma_low / bootstrap_bracket_scale),
                    MIN_NOISE_SEARCH_SIGMA_INTERVAL,
                )
                bootstrap_low_iter += 1

        while eps_high > target_epsilon:
            sigma_low = sigma_high
            eps_low = float(eps_high) if math.isfinite(eps_high) else eps_low
            sigma_high = bootstrap_bracket_scale * sigma_high
            with _timed(f"expand_iter={expand_iter} sigma={sigma_high:.12g}"):
                eps_high = _epsilon_for_sigma(sigma_high)
            if progress_label is not None:
                _emit_progress(
                    f"{progress_label} phase=expand_high iter={expand_iter} "
                    f"sigma={sigma_high:.12g} epsilon={eps_high!r}"
                )
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
            sigma: float
            if (
                accountant_name == "random_allocation"
                and ra_runtime_policy == "efficient_staged_grid"
            ):
                if sigma_low <= 0.0:
                    sigma = _random_allocation_downward_extrapolated_sigma(
                        sigma_high=float(sigma_high),
                        eps_high=float(eps_high),
                        target_epsilon=float(target_epsilon),
                    ) or ((sigma_low + sigma_high) / 2.0)
                else:
                    sigma = _random_allocation_interpolated_sigma(
                        sigma_low=float(sigma_low),
                        eps_low=eps_low,
                        sigma_high=float(sigma_high),
                        eps_high=float(eps_high),
                        target_epsilon=float(target_epsilon),
                    ) or ((sigma_low + sigma_high) / 2.0)
            else:
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
            with _timed(
                f"binary_iter={iterations} sigma={sigma:.12g} bracket=[{sigma_low:.12g},{sigma_high:.12g}]"
            ):
                eps = _epsilon_for_sigma(sigma)
                if (
                    accountant_name == "random_allocation"
                    and ra_runtime_policy == "efficient_staged_grid"
                    and sigma_low > 0.0
                    and math.isfinite(eps)
                    and abs(float(target_epsilon) - float(eps)) <= max(float(epsilon_tolerance) * 4.0, 0.5)
                ):
                    for refinement_round in (1,):
                        refined_kwargs = _refined_random_allocation_kwargs(
                            search_kwargs, refinement_round
                        )
                        refined_eps = _epsilon_for_sigma(sigma, refined_kwargs)
                        if progress_label is not None:
                            _emit_progress(
                                f"{progress_label} phase=refine_mid iter={iterations} "
                                f"refinement_round={refinement_round} sigma={sigma:.12g} "
                                f"epsilon={refined_eps!r} "
                                f"loss_discretization={refined_kwargs['random_allocation_loss_discretization']:.12g}"
                            )
                        if math.isfinite(refined_eps):
                            eps = float(refined_eps)
            if progress_label is not None:
                _emit_progress(
                    f"{progress_label} phase=binary_mid iter={iterations} "
                    f"sigma={sigma:.12g} epsilon={eps!r} "
                    f"bracket=[{sigma_low:.12g},{sigma_high:.12g}]"
                )
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
                if math.isfinite(eps):
                    eps_low = float(eps)
                if not math.isfinite(eps):
                    last_nonfinite_sigma = float(sigma)

        if progress_label is not None:
            _emit_progress(
                f"{progress_label} converged sigma={sigma_high:.12g} "
                f"final_bracket=[{sigma_low:.12g},{sigma_high:.12g}]"
            )
        return sigma_high
