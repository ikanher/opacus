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
from typing import Iterable

from opacus.accountants.analysis import rdp as rdp_analysis


def _resolve_rdp_orders(
    rdp_orders: Iterable[float] | None,
) -> list[float]:
    if rdp_orders is not None:
        return list(rdp_orders)

    # Keep BSR defaults aligned with Opacus' canonical RDP accountant defaults.
    from opacus.accountants.rdp import RDPAccountant

    return list(RDPAccountant.DEFAULT_ALPHAS)


def _validate_k_b_steps(*, steps: int, max_participations: int, min_separation: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    if max_participations < 1:
        raise ValueError("max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")


def _is_nonnegative_decreasing(coeffs: list[float], *, atol: float = 1e-12) -> bool:
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


def compute_bsr_mf_sensitivity_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
    require_nonnegative_decreasing: bool = True,
) -> float:
    """
    Computes MF sensitivity for lower-triangular Toeplitz BSR coefficients.

    For nonnegative decreasing coefficients, this is the closed-form objective
    used in our Lean development and reference implementation.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    _validate_k_b_steps(
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
    )

    if require_nonnegative_decreasing and not _is_nonnegative_decreasing(coeff_list):
        raise ValueError(
            "BSR closed-form sensitivity requires nonnegative decreasing coefficients"
        )

    k_eff = min(max_participations, (steps - 1) // min_separation + 1)
    total_sq = 0.0
    for i in range(steps):
        j_max = min(k_eff - 1, i // min_separation)
        row_sum = 0.0

        for j in range(j_max + 1):
            lag = i - j * min_separation
            if lag < len(coeff_list):
                row_sum += coeff_list[lag]

        total_sq += row_sum * row_sum

    return math.sqrt(total_sq)


def calibrate_bsr_z_std(
    *,
    noise_multiplier_ref: float,
    max_grad_norm: float,
    denominator: float,
) -> float:
    """
    Fixed-batch BSR runtime mapping (proof-aligned):
    z_std = noise_multiplier_ref * max_grad_norm / denominator
    """
    if noise_multiplier_ref <= 0.0:
        raise ValueError("noise_multiplier_ref must be > 0")

    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be > 0")

    if denominator <= 0.0:
        raise ValueError("denominator must be > 0")

    return float(noise_multiplier_ref) * float(max_grad_norm) / float(denominator)


def bsr_fixed_batch_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    mf_sensitivity: float,
    rdp_orders: Iterable[float] | None = None,
) -> float:
    """
    Fixed-batch BSR epsilon via a single Gaussian mechanism.

    For Gaussian noise with effective multiplier
      sigma_eff = noise_multiplier / mf_sensitivity,
    we delegate RDP + (epsilon, delta) conversion to Opacus' canonical
    `analysis.rdp` implementation.
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if target_delta <= 0.0 or target_delta > 1.0:
        raise ValueError("target_delta must be in (0, 1]")

    if mf_sensitivity <= 0.0:
        raise ValueError("mf_sensitivity must be > 0")

    sigma_eff = float(noise_multiplier) / float(mf_sensitivity)

    orders = _resolve_rdp_orders(rdp_orders)

    rdp_values = rdp_analysis.compute_rdp(
        q=1.0,
        noise_multiplier=sigma_eff,
        steps=1,
        orders=orders,
    )
    eps, _ = rdp_analysis.get_privacy_spent(
        orders=orders,
        rdp=rdp_values,
        delta=float(target_delta),
    )
    return float(eps)


def bsr_cyclic_poisson_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    steps: int,
    sample_rate: float,
    bands: int,
    rdp_orders: Iterable[float] | None = None,
) -> float:
    """
    Cyclic-poisson BSR accounting via sampled-Gaussian composition.

    Reduction used:
    - per-cycle sampled Gaussian with q = bands * sample_rate;
    - compose over ceil(steps / bands) cycles.

    Reference:
    "(Amplified) Banded Matrix Factorization: A unified approach to private
    training" (Choquette-Choo et al., 2023)
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if target_delta <= 0.0 or target_delta > 1.0:
        raise ValueError("target_delta must be in (0, 1]")
    
    if steps < 0:
        raise ValueError("steps must be >= 0")

    if sample_rate <= 0.0 or sample_rate > 1.0:
        raise ValueError("sample_rate must be in (0, 1]")

    if bands <= 0:
        raise ValueError("bands must be > 0")

    if steps == 0:
        return 0.0

    q = float(sample_rate) * float(bands)
    if q <= 0.0 or q > 1.0:
        raise ValueError(
            "cyclic_poisson requires bands * sample_rate in (0, 1]; "
            f"got {q}"
        )

    composed_cycles = int(math.ceil(float(steps) / float(bands)))
    orders = _resolve_rdp_orders(rdp_orders)

    rdp_values = rdp_analysis.compute_rdp(
        q=q,
        noise_multiplier=float(noise_multiplier),
        steps=composed_cycles,
        orders=orders,
    )
    eps, _ = rdp_analysis.get_privacy_spent(
        orders=orders,
        rdp=rdp_values,
        delta=float(target_delta),
    )
    return float(eps)
