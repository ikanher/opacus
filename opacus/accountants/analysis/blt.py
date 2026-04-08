from __future__ import annotations

"""
BLT accountant-side parameter algebra and finite-horizon coefficient helpers.

This module owns the BLT math shared by:
- fixed-batch BLT accounting
- amplified BLT `balls_in_bins` / `b_min_sep` accountant inputs
- report surfaces that need finite-horizon forward coefficients
- runtime and optimization layers that consume canonical BLT parameter pairs

It does not own the streamed runtime mechanism itself. That layer lives in
`opacus.noise_mechanisms.blt`. This file owns the accountant-facing coefficient
objects and the finite-horizon Toeplitz materialization that downstream runtime
or report layers may reuse.

Paper lineage:
- BLT follows the buffered lower-triangular Toeplitz construction used by the
  current structured correlated-noise implementations in this repository and in
  `jax_privacy`
- Balls-and-Bins (Chua et al., 2024) for the normalized forward-`c_col`
  accountant-object convention used by the amplified bridge

Claim-type notes:
- `BLTParams`, `BLTPairedParams`, and the coefficient builders are
  implementation-contract surfaces
- the amplified BNB helpers build an accountant-side bridge object, not a full
  amplified BLT accounting statement by themselves
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch

import numpy as np

from opacus.accountants.analysis.bnb import (
    build_bnb_toeplitz_c_matrix_and_contract,
    normalize_bnb_accountant_coeffs,
)


_THETA_GAP_TOL = 1e-12


def _as_float64_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validate_theta(theta: np.ndarray) -> None:
    if theta.size == 0:
        return
    if not np.all(theta > 0.0):
        raise ValueError("theta values must be > 0")
    if not np.all(theta <= 1.0):
        raise ValueError("theta values must be <= 1")
    if theta.size > 1:
        diffs = np.abs(theta[:, None] - theta[None, :])
        diffs[np.diag_indices_from(diffs)] = np.inf
        if float(np.min(diffs)) <= _THETA_GAP_TOL:
            raise ValueError("theta values must be distinct")


def _canonicalize(theta: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if theta.size == 0:
        return theta.copy(), omega.copy()
    order = np.argsort(-theta, kind="stable")
    return theta[order], omega[order]


@dataclass(frozen=True)
class BLTParams:
    """
    One side of the BLT parameter surface `(theta, omega)`.

    `theta` is the decay family and `omega` is the output-scale family for the
    same side. Together they determine the lower-triangular Toeplitz first
    column `c_col` through the finite-lag recurrence implemented by
    :func:`blt_coeff`.
    """

    theta: Sequence[float] | np.ndarray
    omega: Sequence[float] | np.ndarray

    def theta_array(self) -> np.ndarray:
        return _as_float64_vector("theta", self.theta)

    def omega_array(self) -> np.ndarray:
        return _as_float64_vector("omega", self.omega)

    def validate(self) -> None:
        theta = self.theta_array()
        omega = self.omega_array()
        if theta.shape != omega.shape:
            raise ValueError("theta and omega must have the same length")
        _validate_theta(theta)

    def canonicalized(self) -> "BLTParams":
        self.validate()
        theta, omega = _canonicalize(self.theta_array(), self.omega_array())
        return BLTParams(theta=theta, omega=omega)


@dataclass(frozen=True)
class BLTPairedParams:
    """
    Canonical forward/inverse BLT parameter package.

    The forward side determines the runtime/accountant forward Toeplitz
    coefficients. The inverse side determines the streamed inverse recurrence
    used by the runtime noiser. This paired surface is the real BLT API surface;
    report-script `lambda` values are only local sweep labels that resolve to a
    concrete pair.
    """

    forward: BLTParams
    inverse: BLTParams

    def validate(self) -> None:
        self.forward.validate()
        self.inverse.validate()

    def canonicalized(self) -> "BLTPairedParams":
        return BLTPairedParams(
            forward=self.forward.canonicalized(),
            inverse=self.inverse.canonicalized(),
        )


def blt_coeff(params: BLTParams, lag: int) -> float:
    """
    Return the BLT Toeplitz coefficient at lag `lag`.

    This is the finite-horizon first-column coefficient used to build the
    lower-triangular Toeplitz strategy matrix `C`. For amplified BLT BNB
    accounting, the forward-side coefficient list is later normalized into the
    accountant `c_col` bridge object.
    """

    if lag < 0:
        raise ValueError("lag must be >= 0")
    params = params.canonicalized()
    if lag == 0:
        return 1.0
    theta = params.theta_array()
    omega = params.omega_array()
    if theta.size == 0:
        return 0.0
    return float(np.sum(omega * np.power(theta, lag - 1), dtype=np.float64))


def blt_coeffs(params: BLTParams, n: int) -> np.ndarray:
    """Return the first `n` BLT Toeplitz first-column coefficients."""

    if n < 0:
        raise ValueError("n must be >= 0")
    return np.array([blt_coeff(params, lag) for lag in range(n)], dtype=np.float64)


def blt_materialize(params: BLTParams, *, n: int) -> np.ndarray:
    """
    Materialize the finite-horizon lower-triangular Toeplitz BLT matrix.

    This is a finite-horizon implementation helper for reporting and accountant
    bridge construction. It is not the streamed runtime representation used by
    `BufferedToeplitzNoiseMechanism`.
    """

    if n < 0:
        raise ValueError("n must be >= 0")
    coeffs = blt_coeffs(params, n)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = coeffs[i - j]
    return mat


def calc_output_scale(
    theta: Sequence[float] | np.ndarray,
    theta_hat: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Resolve the output-scale family for one side of a BLT pair.

    Given the forward decay family and its paired opposite-side decay family,
    this computes the BLT-side `omega` vector used by the implementation
    contract. This is part of the canonical BLT parameter resolution, not a
    report-only helper.
    """

    theta_arr = _as_float64_vector("theta", theta)
    theta_hat_arr = _as_float64_vector("theta_hat", theta_hat)
    if theta_arr.shape != theta_hat_arr.shape:
        raise ValueError("theta and theta_hat must have the same length")
    _validate_theta(theta_arr)
    _validate_theta(theta_hat_arr)
    if theta_arr.size == 0:
        return np.zeros((0,), dtype=np.float64)

    numerators = np.prod(theta_arr[:, None] - theta_hat_arr[None, :], axis=1)
    denominators = theta_arr[:, None] - theta_arr[None, :]
    denominators[np.diag_indices_from(denominators)] = 1.0
    denom_prod = np.prod(denominators, axis=1)
    omega = numerators / denom_prod
    if not np.all(np.isfinite(omega)):
        raise ValueError("pairing output scales must be finite")
    return omega.astype(np.float64, copy=False)


def blt_pair_from_theta_pair(
    *,
    theta: Sequence[float] | np.ndarray,
    theta_hat: Sequence[float] | np.ndarray,
) -> BLTPairedParams:
    """
    Build the canonical BLT forward/inverse pair from a decay pair.

    This is the main bridge from report- or search-facing `theta` /
    `theta_hat` inputs into the full BLT parameter surface used by both runtime
    and accounting. It is also the point where a report-script `lambda` label is
    resolved into a real BLT pair.
    """

    theta_arr = _as_float64_vector("theta", theta)
    theta_hat_arr = _as_float64_vector("theta_hat", theta_hat)
    if theta_arr.shape != theta_hat_arr.shape:
        raise ValueError("theta and theta_hat must have the same length")
    forward = BLTParams(theta=theta_arr, omega=calc_output_scale(theta_arr, theta_hat_arr))
    inverse = BLTParams(theta=theta_hat_arr, omega=calc_output_scale(theta_hat_arr, theta_arr))
    return BLTPairedParams(forward=forward, inverse=inverse).canonicalized()


def blt_forward_coeffs_for_amplified_accounting(
    *,
    pair: BLTPairedParams,
    horizon: int,
) -> list[float]:
    """
    Return the finite-horizon forward BLT first column for amplified accounting.

    This is an accountant-side bridge object: the forward BLT `c_col` truncated
    to the accounting horizon. Downstream BNB code may normalize or package it
    further, but this function does not claim a full amplified BLT privacy
    theorem by itself.
    """

    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")
    canonical_pair = pair.canonicalized()
    canonical_pair.validate()
    return [
        float(c)
        for c in blt_coeffs(canonical_pair.forward, int(horizon))
    ]


def build_blt_amplified_bnb_accountant_coeffs(
    *,
    pair: BLTPairedParams,
    horizon: int,
) -> tuple[list[float], str]:
    """
    Build normalized non-negative accountant coefficients for amplified BLT BNB accounting.

    The returned coefficients are the accountant-facing normalized forward
    `c_col`. This is the direct bridge to the current `balls_in_bins` /
    `b_min_sep` Monte Carlo consumers. Claim type: implementation-contract
    bridge, not a closed amplified BLT accounting result on its own.
    """

    forward_coeffs = blt_forward_coeffs_for_amplified_accounting(
        pair=pair,
        horizon=int(horizon),
    )
    normalized = normalize_bnb_accountant_coeffs(coeffs=forward_coeffs)
    return normalized, "normalized_forward_c_col"


def build_blt_amplified_bnb_inputs(
    *,
    pair: BLTPairedParams,
    bands: int,
    horizon: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """
    Build the accountant-side BLT BNB coefficient surface, Toeplitz matrix, and contract.

    This packages the normalized forward `c_col`, the explicit Toeplitz `C`
    matrix derived from it, and the metadata contract consumed by the current
    BNB accountant path. The output is suitable for direct `balls_in_bins` /
    `b_min_sep` accounting and report verification.
    """

    if int(bands) < 1:
        raise ValueError("bands must be >= 1")
    coeffs, coeff_source = build_blt_amplified_bnb_accountant_coeffs(
        pair=pair,
        horizon=int(horizon),
    )
    c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=coeffs,
        bands=int(bands),
        horizon=int(horizon),
        dtype=dtype,
        device=device,
        atol=float(atol),
    )
    return {
        "bnb_accountant_coeffs": [float(c) for c in coeffs],
        "bnb_accountant_coeffs_source": str(coeff_source),
        "bnb_c_matrix": c_matrix,
        "bnb_c_matrix_contract": c_matrix_contract,
        "bnb_bands": int(bands),
        "bnb_horizon": int(horizon),
    }


__all__ = [
    "BLTParams",
    "BLTPairedParams",
    "blt_coeff",
    "blt_coeffs",
    "blt_forward_coeffs_for_amplified_accounting",
    "blt_materialize",
    "calc_output_scale",
    "build_blt_amplified_bnb_accountant_coeffs",
    "build_blt_amplified_bnb_inputs",
    "blt_pair_from_theta_pair",
]
