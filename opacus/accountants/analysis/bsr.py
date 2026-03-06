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
import warnings
from typing import Iterable, Literal

# Opacus MF paper-comment convention for this file:
# - Short tags: `BSR (Kalinin and Lampert, 2024)`, `BandMF (Choquette-Choo et al., 2023)`.


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


def _resolve_rdp_orders(
    rdp_orders: Iterable[float] | None,
) -> list[float]:
    if rdp_orders is not None:
        return list(rdp_orders)

    # Keep BSR defaults aligned with Opacus' canonical RDP accountant defaults.
    # XXX: What if we are passed custom alphas?
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
) -> float:
    """
    Compute fixed-batch MF sensitivity from BSR Toeplitz coefficients.

    In the fixed-batch BSR, accounting needs one scalar sensitivity term that
    captures how much the encoded stream can change under the ``(k, b)``
    participation contract. This function computes that term directly from
    Toeplitz coefficients and a finite horizon ``steps``.

    Math:
    ``S_{k,b}(C;T) = (sum_i (sum_j c_{i-jb})^2)^{1/2}``, where
    ``T=steps``, ``k=max_participations``, ``b=min_separation``.

    Source: BSR (Kalinin and Lampert, 2024), Section 3.2, Equation (10), Theorem 2.
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

    if not _is_nonnegative_decreasing(coeff_list):
        raise ValueError(
            "BSR closed-form sensitivity requires nonnegative decreasing coefficients"
        )

    # `k`: max participations; `b`: min-separation in the participation set family.
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


def compute_bsr_kappa_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    Compute finite-horizon ``kappa(T) = max_i ||C e_i||_2`` for Toeplitz ``C``.

    The cyclic accounting branch normalizes runtime noise by ``kappa(T)``,
    which measures the largest column norm visible over the current horizon.
    For lower-triangular Toeplitz matrices this reduces to the l2 norm of the
    visible prefix of coefficients, so we can compute it without constructing
    the full matrix.

    Math:
    ``κ(T) = (Σ_{t=0}^{min(T,b)-1} c_t^2)^{1/2}``, where ``c_t`` are Toeplitz
    coefficients and ``b`` is the coefficient truncation width.

    Source: BSR (Kalinin and Lampert, 2024), Section 3.2 and Equation (10),
        finite-horizon Toeplitz column-norm scale from BSR coeff parameterization.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    if steps < 1:
        raise ValueError("steps must be >= 1")

    visible = min(len(coeff_list), int(steps))
    return math.sqrt(sum(c * c for c in coeff_list[:visible]))


def calibrate_bsr_z_std(
    *,
    noise_multiplier_ref: float,
    max_grad_norm: float,
    denominator: float,
) -> float:
    """
    Map proof-scale noise to runtime correlated-noise standard deviation.

    This keeps the implementation aligned with the fixed-batch accounting
    convention: the runtime Gaussian scale for the correlated noise process is
    a simple rescaling of the reference multiplier by clipping and denominator
    terms used by the training loop.

    - noise_multiplier_ref is the DP accountant-scale multiplier.
    - max_grad_norm is clipping norm.
    - denominator is chosen by loss reduction:
      - 1 for "sum"
      - expected_batch_size for "mean"

    """
    if noise_multiplier_ref <= 0.0:
        raise ValueError("noise_multiplier_ref must be > 0")

    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be > 0")

    if denominator <= 0.0:
        raise ValueError("denominator must be > 0")

    # `sigma` (runtime correlated noise std) := `z_std`.
    return float(noise_multiplier_ref) * float(max_grad_norm) / float(denominator)


def resolve_bsr_fixed_batch_gaussian_contract(
    *,
    noise_multiplier: float,
    mf_sensitivity: float,
) -> dict[str, float | int]:
    """
    Resolve the reduced single-event Gaussian contract for fixed-batch MF.

    The fixed-batch path reduces the whole MF mechanism to one effective
    Gaussian release with sensitivity absorbed into ``sigma_eff``.
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if mf_sensitivity <= 0.0:
        raise ValueError("mf_sensitivity must be > 0")

    sigma_eff = float(noise_multiplier) / float(mf_sensitivity)
    return {
        "effective_noise_multiplier": float(sigma_eff),
        "sample_rate": 1.0,
        "steps": 1,
        "mf_sensitivity": float(mf_sensitivity),
    }


def resolve_bsr_cyclic_gaussian_contract(
    *,
    noise_multiplier: float,
    steps: int,
    sample_rate: float,
    bands: int,
) -> dict[str, float | int]:
    """
    Resolve the reduced sampled-Gaussian contract for cyclic MF accounting.
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if steps < 0:
        raise ValueError("steps must be >= 0")

    if sample_rate <= 0.0 or sample_rate > 1.0:
        raise ValueError("sample_rate must be in (0, 1]")

    if bands <= 0:
        raise ValueError("bands must be > 0")

    if steps == 0:
        return {
            "effective_noise_multiplier": float(noise_multiplier),
            "sample_rate": float(sample_rate) * float(bands),
            "steps": 0,
            "bands": int(bands),
        }

    q = float(sample_rate) * float(bands)
    if q <= 0.0 or q > 1.0:
        raise ValueError(
            "cyclic_poisson requires bands * sample_rate in (0, 1]; "
            f"got {q}"
        )

    composed_cycles = int(math.ceil(float(steps) / float(bands)))
    return {
        "effective_noise_multiplier": float(noise_multiplier),
        "sample_rate": float(q),
        "steps": int(composed_cycles),
        "bands": int(bands),
        "global_steps": int(steps),
    }


def _evaluate_reduced_gaussian_contract(
    *,
    contract: dict[str, float | int],
    target_delta: float,
    accountant: Literal["prv", "rdp"] = "prv",
    rdp_orders: Iterable[float] | None = None,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> float:
    if target_delta <= 0.0 or target_delta > 1.0:
        raise ValueError("target_delta must be in (0, 1]")

    noise_multiplier = float(contract["effective_noise_multiplier"])
    sample_rate = float(contract["sample_rate"])
    steps = int(contract["steps"])

    if steps == 0:
        return 0.0

    if accountant == "rdp":
        from opacus.accountants.rdp import RDPAccountant

        rdp_accountant = RDPAccountant()
        rdp_accountant.history = [(noise_multiplier, sample_rate, steps)]
        return float(
            rdp_accountant.get_epsilon(
                delta=float(target_delta),
                alphas=_resolve_rdp_orders(rdp_orders),
            )
        )

    if accountant == "prv":
        from opacus.accountants.prv import PRVAccountant

        prv_accountant = PRVAccountant()
        prv_accountant.history = [(noise_multiplier, sample_rate, steps)]
        with warnings.catch_warnings():
            # q=1 is a valid single-Gaussian boundary case for the reduced MF
            # contract, but the subsampling PRV formulas hit harmless log(0)
            # branches while simplifying to that limit.
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"opacus\.accountants\.analysis\.prv\.prvs",
            )
            return float(
                prv_accountant.get_epsilon(
                    delta=float(target_delta),
                    eps_error=float(eps_error),
                    delta_error=delta_error,
                )
            )

    raise ValueError(f"Unexpected accountant backend: {accountant}")


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
    contract = resolve_bsr_fixed_batch_gaussian_contract(
        noise_multiplier=noise_multiplier,
        mf_sensitivity=mf_sensitivity,
    )
    return float(
        _evaluate_reduced_gaussian_contract(
            contract=contract,
            target_delta=target_delta,
            accountant=accountant,
            rdp_orders=rdp_orders,
            eps_error=eps_error,
            delta_error=delta_error,
        )
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
    ``q = b·p`` and the number of composed steps is ``⌈T / b⌉``.
    We then evaluate the reduced contract with the requested accountant backend.

    This is the accounting reduction used by the amplified BandMF-style path.

    Derived from BandMF cyclic amplification reduction:

        (q_eff = b·p, T_eff = ceil(T/b)),

    then evaluated with standard sampled-Gaussian RDP conversion.”
    """
    contract = resolve_bsr_cyclic_gaussian_contract(
        noise_multiplier=noise_multiplier,
        steps=steps,
        sample_rate=sample_rate,
        bands=bands,
    )
    return float(
        _evaluate_reduced_gaussian_contract(
            contract=contract,
            target_delta=target_delta,
            accountant=accountant,
            rdp_orders=rdp_orders,
            eps_error=eps_error,
            delta_error=delta_error,
        )
    )
