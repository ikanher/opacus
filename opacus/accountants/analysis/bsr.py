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
from typing import Iterable, Literal

# Opacus MF paper-comment convention for this file:
# - Short tags: `BSR (Kalinin and Lampert, 2024)`, `BandMF (Choquette-Choo et al., 2023)`.

# ---------------------------------------------------------------------------
# Shared Toeplitz-family surface — all family-agnostic operations delegate here.
# The BSR-prefixed public names below are kept for backward compatibility.
# ---------------------------------------------------------------------------
from opacus.accountants.analysis.toeplitz_family import (
    ToeplitzMechanismFamily,
    compute_toeplitz_mf_sensitivity,
    compute_toeplitz_kappa,
    calibrate_z_std,
    resolve_fixed_batch_gaussian_contract,
    resolve_cyclic_gaussian_contract,
    fixed_batch_epsilon_upper_bound,
    cyclic_poisson_epsilon_upper_bound,
)


# ---------------------------------------------------------------------------
# BSR-local: coefficient generation from SGD workload
# ---------------------------------------------------------------------------

def generate_bsr_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
    atol: float = 1e-12,
) -> list[float]:
    """
    Generate band-limited BSR Toeplitz coefficients for the SGD workload.

    This helper translates optimizer hyperparameters into the Toeplitz sequence
    used by the BSR mechanism matrix ``C``. In practice, this is the bridge
    from runtime training knobs (momentum/weight decay) to the matrix family
    that determines both privacy sensitivity and correlated-noise structure.

    The returned list is ``[c_0, ..., c_{bands-1}]`` for a lower-triangular
    Toeplitz matrix. The implementation follows the BSR construction used for
    SGD-type workloads and handles the closed-form ``alpha == beta`` case
    separately for numerical stability.

    Source: BSR (Kalinin and Lampert, 2024), Section 3.1, Equation (9), Theorem 1.

    Mapping:
    - alpha := weight_decay (multiplicative), with weight_decay == 0 mapped to alpha = 1
    - beta := momentum
    """
    if bands < 1:
        raise ValueError("bands must be >= 1")

    beta = float(momentum)
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)

    if not (0.0 <= beta < 1.0):
        raise ValueError("momentum must satisfy 0 <= momentum < 1")

    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            "weight_decay must satisfy 0 < weight_decay <= 1 for BSR coefficient generation "
            "(or be exactly 0 to represent no weight decay)"
        )

    if beta > alpha + atol:
        raise ValueError("BSR generation requires momentum <= effective weight decay")

    if abs(alpha - beta) <= atol:
        return [alpha**j for j in range(bands)]

    r = [0.0] * bands
    r[0] = 1.0
    for i in range(1, bands):
        r[i] = r[i - 1] * ((2.0 * i - 1.0) / (2.0 * i))

    coeffs = [0.0] * bands
    for j in range(bands):
        s = 0.0
        for i in range(j + 1):
            s += (alpha ** (j - i)) * r[j - i] * r[i] * (beta**i)

        coeffs[j] = s

    return [
        0.0 if (c < 0.0 and math.isclose(c, 0.0, abs_tol=atol)) else c
        for c in coeffs
    ]


def build_bsr_family_from_sgd_workload(
    *,
    bands: int,
    steps: int,
    momentum: float,
    weight_decay: float,
    atol: float = 1e-12,
) -> ToeplitzMechanismFamily:
    return ToeplitzMechanismFamily(
        coeffs=generate_bsr_coeffs_from_sgd_workload(
            bands=bands,
            momentum=momentum,
            weight_decay=weight_decay,
            atol=atol,
        ),
        steps=int(steps),
        source="bsr",
    )


# ---------------------------------------------------------------------------
# Backward-compatible public names — thin wrappers around the shared surface
# ---------------------------------------------------------------------------

def compute_bsr_mf_sensitivity_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compute fixed-batch MF sensitivity from BSR Toeplitz coefficients.

    Math:
    ``S_{k,b}(C;T) = (sum_i (sum_j c_{i-jb})^2)^{1/2}``, where
    ``T=steps``, ``k=max_participations``, ``b=min_separation``.

    Source: BSR (Kalinin and Lampert, 2024), Section 3.2, Equation (10), Theorem 2.
    """
    return compute_toeplitz_mf_sensitivity(
        coeffs=coeffs,
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
    )


def compute_bsr_kappa_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    Compute finite-horizon ``kappa(T) = max_i ||C e_i||_2`` for Toeplitz ``C``.

    Source: BSR (Kalinin and Lampert, 2024), Section 3.2 and Equation (10),
        finite-horizon Toeplitz column-norm scale from BSR coeff parameterization.
    """
    return compute_toeplitz_kappa(
        coeffs=coeffs,
        steps=steps,
    )


def calibrate_bsr_z_std(
    *,
    noise_multiplier_ref: float,
    max_grad_norm: float,
    denominator: float,
) -> float:
    """Map proof-scale noise to runtime correlated-noise standard deviation."""
    return calibrate_z_std(
        noise_multiplier_ref=noise_multiplier_ref,
        max_grad_norm=max_grad_norm,
        denominator=denominator,
    )


def resolve_bsr_fixed_batch_gaussian_contract(
    *,
    noise_multiplier: float,
    mf_sensitivity: float,
) -> dict[str, float | int]:
    """Resolve the reduced single-event Gaussian contract for fixed-batch MF."""
    return resolve_fixed_batch_gaussian_contract(
        noise_multiplier=noise_multiplier,
        mf_sensitivity=mf_sensitivity,
    )


def resolve_bsr_cyclic_gaussian_contract(
    *,
    noise_multiplier: float,
    steps: int,
    sample_rate: float,
    bands: int,
) -> dict[str, float | int]:
    """Resolve the reduced sampled-Gaussian contract for cyclic MF accounting."""
    return resolve_cyclic_gaussian_contract(
        noise_multiplier=noise_multiplier,
        steps=steps,
        sample_rate=sample_rate,
        bands=bands,
    )


def bsr_fixed_batch_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    mf_sensitivity: float,
    accountant: Literal["prv", "rdp"] = "prv",
    rdp_orders: Iterable[float] | None = None,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> float:
    """
    Upper-bound ``epsilon`` for fixed-batch BSR via reduced Gaussian accounting.

    After reducing the mechanism to an effective Gaussian release with
    ``sigma_eff = noise_multiplier / mf_sensitivity``, we evaluate the reduced
    single-event contract with the requested accountant backend.
    """
    return fixed_batch_epsilon_upper_bound(
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        mf_sensitivity=mf_sensitivity,
        accountant=accountant,
        rdp_orders=rdp_orders,
        eps_error=eps_error,
        delta_error=delta_error,
    )


def bsr_cyclic_poisson_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    steps: int,
    sample_rate: float,
    bands: int,
    accountant: Literal["prv", "rdp"] = "prv",
    rdp_orders: Iterable[float] | None = None,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> float:
    """
    Upper-bound ``epsilon`` for cyclic-poisson BSR/BandMF-style composition.

    The cyclic participation contract is converted to a sampled Gaussian
    composition problem where each cycle has effective sampling probability
    ``q = b*p`` and the number of composed steps is ``ceil(T / b)``.
    We then evaluate the reduced contract with the requested accountant backend.

    Derived from BandMF cyclic amplification reduction:

        (q_eff = b*p, T_eff = ceil(T/b)),

    then evaluated with standard sampled-Gaussian RDP conversion."
    """
    return cyclic_poisson_epsilon_upper_bound(
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        steps=steps,
        sample_rate=sample_rate,
        bands=bands,
        accountant=accountant,
        rdp_orders=rdp_orders,
        eps_error=eps_error,
        delta_error=delta_error,
    )
