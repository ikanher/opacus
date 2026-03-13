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

"""
BISR analysis helpers.

BISR is the analytic inverse-band path from
`paper/banded-inverse-square-root`: for SGD workloads with momentum and weight
decay, it provides closed-form coefficients for a lower-triangular Toeplitz
inverse correlation factor. The fixed-batch privacy surface is described in the
paper through the separated-participation quantity

    sens_{k,b}(C) <= max_{pi in Π_{k,b}} sqrt(sum_{i,j in pi} |(C^T C)[i,j]|).

The implementation below exposes a canonical runtime-facing fixed-batch
helper that takes inverse-side runtime coefficients, derives the finite-horizon
factor-side object `C^p`, and evaluates Equation `sens_k_b` on that factor-side
Toeplitz matrix.

Source: Back to Square Roots: An Optimal Bound On The Matrix Factorization
Error For Multiepoch Differentially Private SGD (Kalinin et al., 2025)
"""

import math
from typing import Iterable

import torch

# These are reusable from the BSR implementation
from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
)


def _validate_workload_params(*, bands: int, momentum: float, weight_decay: float) -> None:
    if bands < 1:
        raise ValueError("bands must be >= 1")

    if not (0.0 <= float(momentum) < 1.0):
        raise ValueError("momentum must satisfy 0 <= momentum < 1")

    if not (0.0 < float(weight_decay) <= 1.0):
        raise ValueError(
            "weight_decay must satisfy 0 < weight_decay <= 1 for BISR coefficient generation "
            "(or be exactly 0 to represent no weight decay)"
        )


def _generate_tilde_r_coeffs(bands: int) -> list[float]:
    coeffs = [0.0] * bands
    coeffs[0] = 1.0
    for k in range(bands - 1):
        num = float(k) - 0.5
        den = float(k + 1)
        coeffs[k + 1] = (num / den) * coeffs[k]

    return coeffs


def generate_bisr_coeffs_from_sgd_workload(
    *,
    bands: int,
    momentum: float,
    weight_decay: float,
) -> list[float]:
    """
    Generate band-limited BISR inverse-square-root Toeplitz coefficients.

    The construction follows the band-limited inverse-square-root Toeplitz
    factorization from Kalinin et al. Let ``alpha`` denote the SGD
    weight-decay factor and ``beta`` the momentum. First form the truncated
    inverse-square-root kernel

        r_0 = 1,
        r_{k+1} = ((k - 1/2) / (k + 1)) r_k,

    and then convolve the ``alpha``- and ``beta``-weighted copies of this
    kernel to obtain the Toeplitz coefficients ``c_j``.
    """
    alpha = 1.0 if float(weight_decay) == 0.0 else float(weight_decay)
    beta = float(momentum)
    _validate_workload_params(bands=bands, momentum=beta, weight_decay=alpha)

    r = _generate_tilde_r_coeffs(bands)
    coeffs = [0.0] * bands
    for j in range(bands):
        s = 0.0
        for i in range(j + 1):
            s += r[j - i] * (alpha ** (j - i)) * r[i] * (beta**i)
        coeffs[j] = s
    return coeffs


def _build_lower_toeplitz_matrix_from_coeffs(
    *,
    coeffs: list[float],
    steps: int,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    matrix = torch.zeros((steps, steps), dtype=torch.float64)
    max_lag = len(coeffs) - 1
    for row in range(steps):
        for lag in range(min(row, max_lag) + 1):
            matrix[row, row - lag] = float(coeffs[lag])

    return matrix


def derive_bisr_factor_coeffs_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Derive finite-horizon factor-side Toeplitz coefficients from inverse-side runtime coefficients.

    BISR runtime generation exposes the banded inverse-side coefficients of
    ``(C^p)^{-1}``, matching the paper's Algorithm 1. The fixed-batch privacy
    quantity, however, is written in terms of the factor-side matrix ``C^p``.
    This helper constructs the finite-horizon lower-triangular Toeplitz inverse
    matrix, inverts it in float64, and returns the first-column coefficients of
    the resulting factor-side lower-triangular Toeplitz matrix.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    if int(steps) < 1:
        raise ValueError("steps must be >= 1")

    inverse_matrix = _build_lower_toeplitz_matrix_from_coeffs(
        coeffs=coeff_list,
        steps=int(steps),
    )
    factor_matrix = torch.linalg.inv(inverse_matrix)
    factor_coeffs = [float(factor_matrix[row, 0]) for row in range(int(steps))]

    if not all(math.isfinite(c) for c in factor_coeffs):
        raise ValueError("derived factor coefficients must be finite")

    return factor_coeffs


def compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Evaluate the fixed-batch BISR paper sensitivity from inverse-side runtime coefficients.

    The BISR paper states the fixed-batch non-amplified calibration in terms of

        sens_{k,b}(C) <= max_{pi in Π_{k,b}} sqrt(sum_{i,j in pi} |(C^T C)[i,j]|).

    The public BISR runtime coefficients correspond to the inverse-side object
    ``(C^p)^{-1}`` from the paper's Algorithm 1. This helper derives the
    finite-horizon factor-side coefficients for ``C^p`` and then evaluates the
    separated-participation quantity from Equation `sens_k_b` on that factor-side
    Toeplitz object. The resulting factor coefficients are nonnegative and
    decreasing in the paper's regime, so the existing BSR closed form applies
    directly.
    """
    factor_coeffs = derive_bisr_factor_coeffs_from_inverse_coeffs(
        coeffs=coeffs,
        steps=steps,
    )
    return float(
        compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=factor_coeffs,
            steps=steps,
            max_participations=max_participations,
            min_separation=min_separation,
        )
    )


def derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Derive a non-negative accountant-side first column for amplified BISR.

    Runtime BISR is defined by signed inverse-side coefficients of ``(C^p)^{-1}``.
    The amplified balls-in-bins accountant and the JAX Monte Carlo oracle,
    however, require a non-negative accountant-side ``c_col``. We therefore
    derive the finite-horizon factor-side coefficients for ``C^p`` and return
    the first column of ``|C^p|``.
    """
    factor_coeffs = derive_bisr_factor_coeffs_from_inverse_coeffs(
        coeffs=coeffs,
        steps=steps,
    )
    accountant_coeffs = [abs(float(c)) for c in factor_coeffs]
    if not all(math.isfinite(c) for c in accountant_coeffs):
        raise ValueError("derived amplified BISR accountant coefficients must be finite")

    if accountant_coeffs[0] <= 0.0:
        raise ValueError("derived amplified BISR accountant coefficients must be positive")

    return accountant_coeffs


def compute_bisr_kappa_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    Compute the cyclic finite-horizon BISR normalization factor ``kappa(T)``.

    In the cyclic reduction used for correlated sampling, ``kappa(T)`` is the
    finite-horizon normalization term induced by the Toeplitz factor over a
    run of ``T`` logical steps. BISR uses the same finite-horizon Toeplitz
    column-norm computation as BSR once the analytic coefficient sequence is
    fixed.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    return float(
        compute_bsr_kappa_from_coeffs(
            coeffs=coeff_list,
            steps=steps,
        )
    )


def bisr_cyclic_poisson_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    steps: int,
    sample_rate: float,
    bands: int,
    rdp_orders: Iterable[float] | None = None,
) -> float:
    """
    Upper-bound cyclic-poisson ``epsilon`` for BISR after cyclic reduction.

    This uses the same cyclic reduction surface as BSR: first convert the
    correlated cyclic process induced by the analytic BISR coefficients to an
    equivalent finite-horizon sampled-Gaussian problem, then evaluate the
    standard upper bound on ``epsilon`` at the requested ``delta``.
    """
    return float(
        bsr_cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            steps=steps,
            sample_rate=sample_rate,
            bands=bands,
            rdp_orders=rdp_orders,
        )
    )
