"""
Shared Toeplitz-family analysis surface for lower-triangular Toeplitz MF mechanisms.

This module owns the coefficient-driven operations that are common across
BSR, BISR, BIFR, BandMF, and BandInvMF:

- fixed-batch sensitivity (closed-form, requires nonneg-decreasing coefficients),
- cyclic normalization scale (kappa),
- reduced Gaussian contract resolution (fixed-batch and cyclic),
- epsilon upper bounds via reduced Gaussian accounting,
- noise calibration,
- inverse-side factor-recovery (numerical fallback + analytic override).

Family-local operations (coefficient generation, column normalization, optimization)
stay in the respective family modules.

BLT is explicitly excluded: its fixed-batch accountant consumes a paired
forward/inverse max-loss object, not a plain Toeplitz first-column family.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Callable, Iterable, Literal

import torch


ToeplitzFamilySource = Literal["bsr", "bisr", "bifr", "bandmf", "bandinvmf"]


# ---------------------------------------------------------------------------
# Coefficient predicates
# ---------------------------------------------------------------------------

def is_nonnegative_decreasing(coeffs: list[float], *, atol: float = 1e-12) -> bool:
    """Check whether a coefficient sequence is nonneg and monotone non-increasing."""
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


# ---------------------------------------------------------------------------
# Sensitivity and kappa
# ---------------------------------------------------------------------------

def _validate_k_b_steps(*, steps: int, max_participations: int, min_separation: int) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    if max_participations < 1:
        raise ValueError("max_participations must be >= 1")

    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")


def compute_toeplitz_mf_sensitivity(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """
    Compute fixed-batch MF sensitivity from Toeplitz coefficients.

    Math:
    ``S_{k,b}(C;T) = (sum_i (sum_j c_{i-jb})^2)^{1/2}``, where
    ``T=steps``, ``k=max_participations``, ``b=min_separation``.

    Requires nonnegative decreasing coefficients for the closed-form path.

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

    if not is_nonnegative_decreasing(coeff_list):
        raise ValueError(
            "closed-form Toeplitz sensitivity requires nonnegative decreasing coefficients"
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


def compute_disjoint_toeplitz_mf_sensitivity(
    *,
    coeffs: Iterable[float],
    steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    """Closed form for the disjoint-column Toeplitz fixed-batch regime."""
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

    visible_width = min(len(coeff_list), int(steps))
    if int(min_separation) < visible_width:
        raise ValueError(
            "disjoint Toeplitz sensitivity requires min_separation >= visible Toeplitz bandwidth"
        )

    k_eff = min(int(max_participations), (int(steps) - 1) // int(min_separation) + 1)
    coeff_norm = math.sqrt(sum(c * c for c in coeff_list[:visible_width]))
    return math.sqrt(float(k_eff)) * coeff_norm


def compute_toeplitz_kappa(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    Compute finite-horizon ``kappa(T) = max_i ||C e_i||_2`` for Toeplitz ``C``.

    For lower-triangular Toeplitz matrices this is the l2 norm of the visible
    prefix of coefficients.

    Math:
    ``kappa(T) = (sum_{t=0}^{min(T,b)-1} c_t^2)^{1/2}``
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


def compute_prefix_workload_frobenius_sq_from_matrix(
    *,
    c_matrix: torch.Tensor,
) -> float:
    """
    Compute ``||A C^{-1}||_F^2`` for the prefix workload ``A``.

    ``A`` is the lower-triangular all-ones matrix of the same horizon as
    ``C``. This is the shared matrix quantity behind both the normalized paper
    RMSE and the legacy unnormalized Frobenius-based diagnostic.
    """
    matrix = torch.as_tensor(c_matrix, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("c_matrix must be square")

    steps = int(matrix.shape[0])
    if steps < 1:
        raise ValueError("c_matrix must have positive size")

    a = torch.tril(torch.ones((steps, steps), dtype=torch.float64))
    ac_inv = torch.linalg.solve_triangular(
        matrix.T,
        a.T,
        upper=True,
        unitriangular=False,
    ).T
    return float(torch.sum(ac_inv * ac_inv).item())


def compute_prefix_workload_frobenius_sq_from_inverse_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
) -> float:
    """
    Compute ``||A C^{-1}||_F^2`` for the prefix workload directly from the
    Toeplitz coefficients of ``C^{-1}``.

    For the prefix workload, the Toeplitz coefficients of ``A C^{-1}`` are the
    prefix sums of the noising / inverse coefficients.
    """
    coeff_list = [float(c) for c in inv_coeffs]
    if len(coeff_list) == 0:
        raise ValueError("inv_coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("inv_coeffs must be finite")

    if steps < 1:
        raise ValueError("steps must be >= 1")

    running = 0.0
    fro_sq = 0.0
    visible = min(len(coeff_list), int(steps))
    for idx in range(int(steps)):
        if idx < visible:
            running += coeff_list[idx]
        fro_sq += float(int(steps) - idx) * running * running

    return float(fro_sq)


def compute_prefix_workload_normalized_mse_from_matrix(
    *,
    c_matrix: torch.Tensor,
    noise_multiplier: float,
) -> float:
    """
    Compute the normalized paper MSE for the prefix workload.

    This is ``sigma^2 * ||A C^{-1}||_F^2 / n`` where ``n`` is the horizon.
    """
    sigma = float(noise_multiplier)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("noise_multiplier must be finite and >= 0")

    matrix = torch.as_tensor(c_matrix, dtype=torch.float64)
    steps = int(matrix.shape[0])
    fro_sq = compute_prefix_workload_frobenius_sq_from_matrix(c_matrix=matrix)
    return float((sigma * sigma) * fro_sq / float(steps))


def compute_prefix_workload_normalized_mse_from_inverse_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
    noise_multiplier: float,
) -> float:
    """
    Compute the normalized paper MSE for the prefix workload directly from the
    Toeplitz coefficients of ``C^{-1}``.
    """
    sigma = float(noise_multiplier)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("noise_multiplier must be finite and >= 0")

    fro_sq = compute_prefix_workload_frobenius_sq_from_inverse_coeffs(
        inv_coeffs=inv_coeffs,
        steps=int(steps),
    )
    return float((sigma * sigma) * fro_sq / float(int(steps)))


def compute_prefix_workload_normalized_rmse_from_matrix(
    *,
    c_matrix: torch.Tensor,
    noise_multiplier: float,
) -> float:
    """Compute the normalized paper RMSE for the prefix workload."""
    mse = compute_prefix_workload_normalized_mse_from_matrix(
        c_matrix=c_matrix,
        noise_multiplier=noise_multiplier,
    )
    return float(math.sqrt(max(mse, 0.0)))


def compute_prefix_workload_normalized_rmse_from_inverse_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
    noise_multiplier: float,
) -> float:
    """Compute the normalized paper RMSE from inverse / noising coefficients."""
    mse = compute_prefix_workload_normalized_mse_from_inverse_coeffs(
        inv_coeffs=inv_coeffs,
        steps=int(steps),
        noise_multiplier=noise_multiplier,
    )
    return float(math.sqrt(max(mse, 0.0)))


compute_prefix_workload_paper_mse_from_matrix = compute_prefix_workload_normalized_mse_from_matrix
compute_prefix_workload_paper_mse_from_inverse_coeffs = compute_prefix_workload_normalized_mse_from_inverse_coeffs
compute_prefix_workload_paper_rmse_from_matrix = compute_prefix_workload_normalized_rmse_from_matrix
compute_prefix_workload_paper_rmse_from_inverse_coeffs = compute_prefix_workload_normalized_rmse_from_inverse_coeffs


# ---------------------------------------------------------------------------
# Noise calibration
# ---------------------------------------------------------------------------

def calibrate_z_std(
    *,
    noise_multiplier_ref: float,
    max_grad_norm: float,
    denominator: float,
) -> float:
    """
    Map proof-scale noise to runtime correlated-noise standard deviation.

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

    return float(noise_multiplier_ref) * float(max_grad_norm) / float(denominator)


# ---------------------------------------------------------------------------
# Reduced Gaussian contracts
# ---------------------------------------------------------------------------

def resolve_fixed_batch_gaussian_contract(
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


def resolve_cyclic_gaussian_contract(
    *,
    noise_multiplier: float,
    steps: int,
    sample_rate: float,
    bands: int,
) -> dict[str, float | int]:
    """Resolve the reduced sampled-Gaussian contract for cyclic MF accounting."""
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


def _resolve_rdp_orders(
    rdp_orders: Iterable[float] | None,
) -> list[float]:
    if rdp_orders is not None:
        return list(rdp_orders)

    from opacus.accountants.rdp import RDPAccountant

    return list(RDPAccountant.DEFAULT_ALPHAS)


def evaluate_reduced_gaussian_contract(
    *,
    contract: dict[str, float | int],
    target_delta: float,
    accountant: Literal["prv", "rdp"] = "prv",
    rdp_orders: Iterable[float] | None = None,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> float:
    """Evaluate epsilon from a reduced Gaussian contract via RDP or PRV."""
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


# ---------------------------------------------------------------------------
# Epsilon upper bounds
# ---------------------------------------------------------------------------

def fixed_batch_epsilon_upper_bound(
    *,
    noise_multiplier: float,
    target_delta: float,
    mf_sensitivity: float,
    accountant: Literal["prv", "rdp"] = "prv",
    rdp_orders: Iterable[float] | None = None,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> float:
    """Upper-bound epsilon for fixed-batch MF via reduced Gaussian accounting."""
    contract = resolve_fixed_batch_gaussian_contract(
        noise_multiplier=noise_multiplier,
        mf_sensitivity=mf_sensitivity,
    )
    return float(
        evaluate_reduced_gaussian_contract(
            contract=contract,
            target_delta=target_delta,
            accountant=accountant,
            rdp_orders=rdp_orders,
            eps_error=eps_error,
            delta_error=delta_error,
        )
    )


def cyclic_poisson_epsilon_upper_bound(
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
    """Upper-bound epsilon for cyclic-poisson MF composition."""
    contract = resolve_cyclic_gaussian_contract(
        noise_multiplier=noise_multiplier,
        steps=steps,
        sample_rate=sample_rate,
        bands=bands,
    )
    return float(
        evaluate_reduced_gaussian_contract(
            contract=contract,
            target_delta=target_delta,
            accountant=accountant,
            rdp_orders=rdp_orders,
            eps_error=eps_error,
            delta_error=delta_error,
        )
    )


# ---------------------------------------------------------------------------
# Inverse-side factor recovery
# ---------------------------------------------------------------------------

def _build_lower_toeplitz_matrix(
    *,
    coeffs: list[float],
    steps: int,
) -> torch.Tensor:
    """Build a lower-triangular Toeplitz matrix from first-column coefficients."""
    if steps < 1:
        raise ValueError("steps must be >= 1")

    matrix = torch.zeros((steps, steps), dtype=torch.float64)
    max_lag = len(coeffs) - 1
    for row in range(steps):
        for lag in range(min(row, max_lag) + 1):
            matrix[row, row - lag] = float(coeffs[lag])

    return matrix


def build_lower_toeplitz_matrix_from_coeffs(
    *,
    coeffs: Iterable[float],
    steps: int,
) -> torch.Tensor:
    """Public wrapper for building a lower-triangular Toeplitz matrix."""
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    return _build_lower_toeplitz_matrix(
        coeffs=coeff_list,
        steps=int(steps),
    )


def recover_factor_coeffs_numerical(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
) -> list[float]:
    """
    Recover factor-side Toeplitz coefficients by solving for the first column
    of the inverse-side lower-triangular Toeplitz matrix.

    We only need ``L^{-1} e_0``, not the full dense inverse ``L^{-1}``. A
    triangular solve is both cheaper and numerically better behaved than
    materializing the entire inverse.
    """
    coeff_list = [float(c) for c in inv_coeffs]
    if len(coeff_list) == 0:
        raise ValueError("inv_coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("inv_coeffs must be finite")

    if int(steps) < 1:
        raise ValueError("steps must be >= 1")

    inverse_matrix = _build_lower_toeplitz_matrix(
        coeffs=coeff_list,
        steps=int(steps),
    )
    rhs = torch.zeros((int(steps), 1), dtype=torch.float64)
    rhs[0, 0] = 1.0
    factor_column = torch.linalg.solve_triangular(
        inverse_matrix,
        rhs,
        upper=False,
        unitriangular=False,
    )
    factor_coeffs = [float(factor_column[row, 0]) for row in range(int(steps))]

    if not all(math.isfinite(c) for c in factor_coeffs):
        raise ValueError("derived factor coefficients must be finite")

    return factor_coeffs


def recover_factor_coeffs(
    *,
    inv_coeffs: Iterable[float],
    steps: int,
    analytic_override: Callable[[], list[float]] | None = None,
) -> list[float]:
    """
    Recover factor-side Toeplitz coefficients from inverse-side coefficients.

    If ``analytic_override`` is provided, it is called instead of numerical
    inversion. Otherwise falls back to ``recover_factor_coeffs_numerical``.

    The analytic override is a zero-argument callable that returns the
    factor-side coefficient list directly. The caller is responsible for
    closing over whatever parameters are needed (frac, momentum, etc.).
    """
    if analytic_override is not None:
        return analytic_override()

    return recover_factor_coeffs_numerical(
        inv_coeffs=inv_coeffs,
        steps=steps,
    )


@dataclass(frozen=True)
class ToeplitzMechanismFamily:
    """Shared analysis surface for lower-triangular Toeplitz coefficient families."""

    coeffs: list[float]
    steps: int
    source: ToeplitzFamilySource

    def fixed_batch_sensitivity(
        self,
        *,
        max_participations: int,
        min_separation: int,
        allow_disjoint_fallback: bool = False,
    ) -> float:
        try:
            return compute_toeplitz_mf_sensitivity(
                coeffs=self.coeffs,
                steps=self.steps,
                max_participations=max_participations,
                min_separation=min_separation,
            )
        except ValueError as exc:
            if (
                not allow_disjoint_fallback
                or "nonnegative decreasing coefficients" not in str(exc)
            ):
                raise
            return compute_disjoint_toeplitz_mf_sensitivity(
                coeffs=self.coeffs,
                steps=self.steps,
                max_participations=max_participations,
                min_separation=min_separation,
            )

    def kappa(self) -> float:
        return compute_toeplitz_kappa(
            coeffs=self.coeffs,
            steps=self.steps,
        )

    def fixed_batch_epsilon(
        self,
        *,
        noise_multiplier: float,
        target_delta: float,
        max_participations: int,
        min_separation: int,
        accountant: Literal["prv", "rdp"] = "prv",
        rdp_orders: Iterable[float] | None = None,
        eps_error: float = 0.01,
        delta_error: float | None = None,
        allow_disjoint_fallback: bool = False,
    ) -> float:
        sensitivity = self.fixed_batch_sensitivity(
            max_participations=max_participations,
            min_separation=min_separation,
            allow_disjoint_fallback=allow_disjoint_fallback,
        )
        return fixed_batch_epsilon_upper_bound(
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            mf_sensitivity=sensitivity,
            accountant=accountant,
            rdp_orders=rdp_orders,
            eps_error=eps_error,
            delta_error=delta_error,
        )

    def cyclic_poisson_epsilon(
        self,
        *,
        noise_multiplier: float,
        target_delta: float,
        sample_rate: float,
        bands: int,
        accountant: Literal["prv", "rdp"] = "prv",
        rdp_orders: Iterable[float] | None = None,
        eps_error: float = 0.01,
        delta_error: float | None = None,
    ) -> float:
        return cyclic_poisson_epsilon_upper_bound(
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            steps=self.steps,
            sample_rate=sample_rate,
            bands=bands,
            accountant=accountant,
            rdp_orders=rdp_orders,
            eps_error=eps_error,
            delta_error=delta_error,
        )


@dataclass(frozen=True)
class InverseSideToeplitzFamily:
    """Shared inverse-side route with analytic override and numerical fallback."""

    inv_coeffs: list[float]
    steps: int
    source: ToeplitzFamilySource
    analytic_factor_override: Callable[[], list[float]] | None = None

    def factor_coeffs(self) -> list[float]:
        return recover_factor_coeffs(
            inv_coeffs=self.inv_coeffs,
            steps=self.steps,
            analytic_override=self.analytic_factor_override,
        )

    def factor_family(
        self,
        *,
        source: ToeplitzFamilySource | None = None,
    ) -> ToeplitzMechanismFamily:
        return ToeplitzMechanismFamily(
            coeffs=self.factor_coeffs(),
            steps=self.steps,
            source=self.source if source is None else source,
        )
