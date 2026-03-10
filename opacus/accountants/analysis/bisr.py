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

The implementation below exposes that paper-facing contract directly. Internally
the current closed-form upper bound is realized through the absolute-majorant
route: replace the signed BISR Toeplitz coefficients by their entrywise
absolute values and evaluate the existing nonnegative-decreasing BSR sensitivity
formula on that majorant.

Source: Back to Square Roots: An Optimal Bound On The Matrix Factorization
Error For Multiepoch Differentially Private SGD (Kalinin et al., 2025)
"""

import math
from typing import Iterable

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


def compute_bisr_abs_majorant_coeffs(
    *,
    coeffs: Iterable[float],
) -> list[float]:
    """
    Compute entrywise absolute Toeplitz majorant coefficients.

    This is the coefficient-wise absolute-value majorant used in the paper's
    sensitivity bound: if ``C`` is the BISR Toeplitz factor, the majorant
    factor ``C_abs`` is obtained by replacing each coefficient ``c_j`` with
    ``|c_j|`` so that entrywise bounds can be reduced to the nonnegative BSR
    closed form.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    return [abs(c) for c in coeff_list]


def _compute_bisr_sensitivity_upper_bound_via_abs_majorant(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Internal absolute-majorant realization of the BISR fixed-batch bound.

    This helper is kept as a regression oracle for the paper-facing BISR
    sensitivity surface. The intended production contract is
    `compute_bisr_separated_participation_sensitivity_upper_bound_from_coeffs`,
    not the proof-oriented majorant route itself.
    """
    abs_coeffs = compute_bisr_abs_majorant_coeffs(coeffs=coeffs)
    return float(
        compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=abs_coeffs,
            steps=steps,
            max_participations=max_participations,
            min_separation=min_separation,
        )
    )


def compute_bisr_separated_participation_sensitivity_upper_bound_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Upper-bound fixed-batch BISR sensitivity under separated participation.

    This is the paper-facing fixed-batch BISR sensitivity contract. Given the
    analytic inverse-band BISR Toeplitz coefficients, it returns an upper bound
    on the separated-participation sensitivity for horizon `steps`, at most
    `max_participations` participations, and minimum separation
    `min_separation`.

    The current implementation realizes the paper bound through the validated
    absolute-majorant route:
    1. form the entrywise absolute majorant of the signed BISR Toeplitz factor,
    2. apply the closed-form nonnegative-decreasing Toeplitz sensitivity
       formula already used for BSR.
    """
    return _compute_bisr_sensitivity_upper_bound_via_abs_majorant(
        coeffs=coeffs,
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
    )


def compute_bisr_mf_sensitivity_upper_bound_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compatibility wrapper for the BISR fixed-batch sensitivity upper bound.

    This legacy helper name is retained during the BISR paper-parity refactor.
    New code should prefer
    `compute_bisr_separated_participation_sensitivity_upper_bound_from_coeffs`.
    """
    return compute_bisr_separated_participation_sensitivity_upper_bound_from_coeffs(
        coeffs=coeffs,
        steps=steps,
        max_participations=max_participations,
        min_separation=min_separation,
    )


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
