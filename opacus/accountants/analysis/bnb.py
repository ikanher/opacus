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
Balls-in-bins and BMinSep Monte Carlo accounting helpers.

This module owns the analysis-side sampling and verification utilities used by
the amplified correlated-accounting path. It separates two certification
contracts:

- the original one-shot balls-in-bins / BMinSep Monte Carlo contract from
  Balls-and-Bins (Chua et al., 2024), where one fixed candidate mechanism is
  certified at one fixed ``epsilon``; and
- the EVR-style candidate-ladder contract from EVR (Wang et al., 2023), where
  confidence is split across multiple checked candidates through a `base_delta ->
  overall_delta` mapping.

Paper lineage:
- Balls-and-Bins (Chua et al., 2024) defines the privacy-loss sampling model
  and the one-shot Monte Carlo upper-bound contract;
- EVR (Wang et al., 2023) defines the EVR-style confidence-splitting and
  feasibility logic.
- BMinSep (Dong and Ganesh, 2025)

Implementation lineage:
- the balls-in-bins sample generation, privacy-loss chunking, and EVR-style
  feasibility probing in this module are closely adapted from
  `google-deepmind/jax_privacy`.
- original implementation based on example code from Dong et al. of
  BMinSep (Dong and Ganesh, 2025)

This module is analysis-only. Accountant history and runtime/sampler
orchestration live in `opacus.opacus.accountants.bnb`.
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Any, Dict, Sequence

import numpy as np
import scipy
import torch
import torch.distributed as dist
from torch.distributions import Bernoulli, kl_divergence

BNB_VERIFICATION_CONTRACT = "evr_union_bound_alpha_split_v1"

_BNB_CALIBRATION_DEFAULTS: Dict[str, Any] = {
    # `evr` keeps the existing guarded Monte Carlo surface; `optimistic`
    # disables the EVR acceptance guard while keeping the same sampler.
    "bnb_calibration_mode": "evr",
    # These are directly from the example script.
    "bnb_num_samples": 500_000,
    "bnb_seed": 154,
    "bnb_reduce_dimensionality": False,
    "bnb_confidence_alpha": 1e-6,
    "bnb_require_evr_pass": False,
    "bnb_tolerance": 1e-7,
    "bnb_max_iterations": 1000,
    "bnb_chunk_size": None,
    "bnb_num_workers": 0,
    "bnb_backend": "auto",
    "bnb_device": None,
    "bnb_distributed_mode": "none",
    "bnb_distributed_dp_runtime": False,
}


def normalize_bnb_accountant_coeffs(
    *,
    coeffs: Sequence[float],
) -> list[float]:
    """
    Normalize a non-negative Toeplitz first column for amplified BNB accounting.

    The amplified accountant works with an accountant-side Toeplitz first
    column whose norm is separated from the public noise multiplier. This
    helper normalizes a non-negative Toeplitz column to unit L2 norm while
    leaving runtime mechanism coefficients unchanged.

    Implementation lineage:
    - mirrors the normalized accountant-side `c_col` contract used by
      `google-deepmind/jax_privacy` for amplified balls-in-bins parity.
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    if any(c < 0.0 for c in coeff_list):
        raise ValueError("coeffs must be nonnegative")

    norm_sq = sum(c * c for c in coeff_list)
    if norm_sq <= 0.0:
        raise ValueError("coeffs must have positive L2 norm")

    norm = math.sqrt(norm_sq)

    return [c / norm for c in coeff_list]


def resolve_bnb_calibration_kwargs(
    *,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved = dict(_BNB_CALIBRATION_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if key in _BNB_CALIBRATION_DEFAULTS and value is not None:
                resolved[key] = value

    mode = str(resolved["bnb_calibration_mode"])
    if mode not in {"evr", "optimistic"}:
        raise ValueError("bnb_calibration_mode must be one of {'evr', 'optimistic'}")

    # Current production semantics: `optimistic` skips the EVR acceptance guard.
    if mode == "optimistic":
        resolved["bnb_require_evr_pass"] = False
    elif overrides is None or overrides.get("bnb_require_evr_pass") is None:
        resolved["bnb_require_evr_pass"] = True

    return resolved


@dataclass(frozen=True)
class GaussianMixture:
    """
    Finite Gaussian mixture with shared isotropic Gaussian noise scale.

    This container is the core distribution object for BMinSep Monte Carlo
    accounting: each component corresponds to a participation pattern-induced
    mean shift, and all components share the same isotropic ``sigma``.
    The sampled privacy loss is then computed between two such mixtures.

    Source:
    - Balls-and-Bins (Chua et al., 2024), privacy-loss sampling under
      participation pattern-induced Gaussian means.
    """

    modes: torch.Tensor  # [k, d]
    probs: torch.Tensor  # [k]

    def __post_init__(self) -> None:
        if self.modes.ndim != 2:
            raise ValueError("modes must have shape [k, d]")

        if self.probs.ndim != 1:
            raise ValueError("probs must have shape [k]")

        if self.modes.shape[0] != self.probs.shape[0]:
            raise ValueError("modes/probs component mismatch")

        if self.modes.shape[0] == 0:
            raise ValueError("mixture must have at least one component")

        if torch.any(self.probs < 0):
            raise ValueError("probs must be nonnegative")

        s = torch.sum(self.probs)
        if not torch.isfinite(s):
            raise ValueError("probs must be finite")

        if abs(float(s) - 1.0) > 1e-6:
            raise ValueError("probs must sum to 1")


@dataclass(frozen=True)
class DeltaVerificationResult:
    delta_estimate: float
    upper_confidence_bound: float
    confidence_alpha: float
    accepted: bool


@dataclass(frozen=True)
class SingleVerificationResult:
    """
    One-shot balls-in-bins verification result for a fixed mechanism and epsilon.

    This mirrors the original balls-in-bins Monte Carlo contract: estimate the
    hockey-stick divergence for one fixed candidate mechanism, then upper-bound
    the true ``delta(epsilon)`` with a Bernoulli-KL inversion on the empirical
    mean. No EVR/base-delta split or candidate-ladder composition is involved.

    Source:
    - Balls-and-Bins (Chua et al., 2024), one-shot single-mechanism
      verification.
    """

    delta_estimate: float
    upper_confidence_bound: float
    error_probability: float
    accepted: bool


@dataclass(frozen=True)
class BNBCalibrationReport:
    """
    Stable, versioned calibration payload for BNB Monte Carlo runs.

    The goal is reproducible diagnostics: a reviewer should be able to
    reconstruct what was calibrated, with which confidence split, and why
    acceptance passed or failed, without re-reading logs.

    This report is specific to the EVR-style candidate-ladder calibration path.
    """

    version: int
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    num_samples: int
    seed: int
    bands: int
    delta_estimate_at_target_epsilon: float
    delta_upper_confidence_bound: float
    confidence_alpha: float
    verification_passed: bool
    evr_confidence_alpha_total: float
    evr_num_checks: int
    evr_per_check_alpha: float
    evr_pass_count: int
    verification_contract: str = BNB_VERIFICATION_CONTRACT
    evr_composed_delta_upper_bound: float = 1.0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "version": int(self.version),
            "target_epsilon": float(self.target_epsilon),
            "target_delta": float(self.target_delta),
            "noise_multiplier": float(self.noise_multiplier),
            "num_samples": int(self.num_samples),
            "seed": int(self.seed),
            "bands": int(self.bands),
            "delta_estimate_at_target_epsilon": float(
                self.delta_estimate_at_target_epsilon
            ),
            "delta_upper_confidence_bound": float(self.delta_upper_confidence_bound),
            "confidence_alpha": float(self.confidence_alpha),
            "verification_passed": bool(self.verification_passed),
            "evr_confidence_alpha_total": float(self.evr_confidence_alpha_total),
            "evr_num_checks": int(self.evr_num_checks),
            "evr_per_check_alpha": float(self.evr_per_check_alpha),
            "evr_pass_count": int(self.evr_pass_count),
            "verification_contract": str(self.verification_contract),
            "evr_composed_delta_upper_bound": float(self.evr_composed_delta_upper_bound),
        }


@dataclass(frozen=True)
class BNBCalibrationStatus:
    """
    Structured runtime status wrapper around a calibration report.

    This is a lightweight object used for surfaces that need both machine-
    readable report fields and a concise human summary string.
    """

    mechanism: str
    accounting_mode: str
    sampling_mode: str | None
    report: BNBCalibrationReport
    summary: str

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "accounting_mode": self.accounting_mode,
            "sampling_mode": self.sampling_mode,
            "report": self.report.to_dict(),
            "summary": self.summary,
        }


def build_lower_toeplitz_c_matrix_from_coeffs(
    *,
    coeffs: Sequence[float],
    horizon: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a finite lower-triangular Toeplitz ``C`` from coefficient lags.

    BMinSep accounting consumes an explicit finite-horizon matrix. This helper
    materializes that matrix from the lag sequence so downstream calibration and
    consistency checks can operate on a concrete tensor.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5 (matrix-mechanism setup around Equation (2)).
    """
    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be non-empty")

    if not all(math.isfinite(c) for c in coeff_list):
        raise ValueError("coeffs must be finite")

    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")

    h = int(horizon)
    c_matrix = torch.zeros((h, h), dtype=dtype, device=device)
    for i in range(h):

        max_lag = min(i, len(coeff_list) - 1)
        for lag in range(max_lag + 1):
            c_matrix[i, i - lag] = coeff_list[lag]

    return c_matrix


def make_bnb_toeplitz_c_matrix_contract(
    *,
    c_matrix: torch.Tensor,
    bands: int,
    horizon: int | None = None,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """
    Build metadata proving how a BNB ``C`` matrix was derived.

    The contract is attached to runtime state so the accountant can verify that
    the matrix/sampler pair still matches the expected BMinSep derivation.
    """
    if c_matrix.ndim != 2:
        raise ValueError("c_matrix must have shape [d, m]")

    if int(bands) < 1:
        raise ValueError("bands must be >= 1")

    if float(atol) <= 0.0:
        raise ValueError("atol must be > 0")

    h = int(horizon) if horizon is not None else int(c_matrix.shape[1])
    if h < 1:
        raise ValueError("horizon must be >= 1")

    return {
        "sampling_mode": "b_min_sep",
        "bands": int(bands),
        "granularity": "single_participation",
        "matrix_columns": int(c_matrix.shape[1]),
        "derivation": "lower_toeplitz_from_coeffs",
        "horizon": int(h),
        "atol": float(atol),
    }


def build_bnb_toeplitz_c_matrix_and_contract(
    *,
    coeffs: Sequence[float],
    bands: int,
    horizon: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    atol: float = 1e-9,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Convenience API that returns both Toeplitz matrix and validation contract.

    Use this when callers need a one-shot, self-describing BNB state payload.
    """
    h = int(horizon)
    b = int(bands)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    if b < 1:
        raise ValueError("bands must be >= 1")

    padded_horizon = int(math.ceil(float(h) / float(b)) * b)
    c_matrix = build_lower_toeplitz_c_matrix_from_coeffs(
        coeffs=coeffs,
        horizon=int(padded_horizon),
        dtype=dtype,
        device=device,
    )
    contract = make_bnb_toeplitz_c_matrix_contract(
        c_matrix=c_matrix,
        bands=b,
        horizon=h,
        atol=float(atol),
    )
    if padded_horizon != h:
        contract["derivation"] = "lower_toeplitz_from_coeffs_right_padded"
        contract["padded_horizon"] = int(padded_horizon)
        contract["padding_columns"] = int(padded_horizon - h)
    return c_matrix, contract


def validate_bnb_c_matrix_contract(
    *,
    c_matrix: torch.Tensor,
    coeffs: Sequence[float],
    bands: int,
    c_matrix_contract: dict[str, Any],
) -> None:
    """
    Validate that runtime BNB matrix metadata still matches implementation assumptions.

    This guard prevents silent drift between sampler/accountant wiring and the
    matrix-construction path that calibration depended on. When derivation data
    is present, the check is numeric and exact up to ``atol``.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5 and Theorem 5.1 (consistency of recursion/accounting contract).
    """
    if not isinstance(c_matrix_contract, dict):
        raise ValueError("bnb consistency check requires c_matrix_contract to be a dict")
    if c_matrix_contract.get("sampling_mode") != "b_min_sep":
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract['sampling_mode'] "
            "must be 'b_min_sep'"
        )

    contract_bands = c_matrix_contract.get("bands")
    if contract_bands is None or int(contract_bands) != int(bands):
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract['bands'] "
            f"({contract_bands}) != accounting bands ({int(bands)})"
        )

    if c_matrix_contract.get("granularity") != "single_participation":
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract['granularity'] "
            "must be 'single_participation'"
        )

    matrix_columns = c_matrix_contract.get("matrix_columns")
    if matrix_columns is None or int(matrix_columns) != int(c_matrix.shape[1]):
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract['matrix_columns'] "
            f"({matrix_columns}) must equal c_matrix.shape[1] ({int(c_matrix.shape[1])})"
        )

    derivation = c_matrix_contract.get("derivation")
    if derivation is None:
        return
    if derivation not in (
        "lower_toeplitz_from_coeffs",
        "lower_toeplitz_from_coeffs_right_padded",
    ):
        raise ValueError(
            "bnb consistency check failed: unsupported c_matrix_contract['derivation']; "
            "supported values: {'lower_toeplitz_from_coeffs', 'lower_toeplitz_from_coeffs_right_padded'}"
        )

    horizon = int(c_matrix_contract.get("horizon", c_matrix.shape[1]))
    padded_horizon = int(c_matrix_contract.get("padded_horizon", horizon))
    padding_columns = int(c_matrix_contract.get("padding_columns", padded_horizon - horizon))
    if padded_horizon < horizon:
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract['padded_horizon'] "
            "must be >= c_matrix_contract['horizon']"
        )
    if padded_horizon - horizon != padding_columns:
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract padding metadata is inconsistent"
        )
    if derivation == "lower_toeplitz_from_coeffs" and padding_columns != 0:
        raise ValueError(
            "bnb consistency check failed: unpadded derivation cannot declare padding_columns"
        )
    if int(c_matrix.shape[0]) != padded_horizon or int(c_matrix.shape[1]) != padded_horizon:
        raise ValueError(
            "bnb consistency check failed: lower_toeplitz_from_coeffs requires "
            f"square c_matrix with shape [{padded_horizon}, {padded_horizon}]"
        )

    expected = build_lower_toeplitz_c_matrix_from_coeffs(
        coeffs=coeffs,
        horizon=int(padded_horizon),
        dtype=torch.float64,
        device=c_matrix.device,
    )

    atol = float(c_matrix_contract.get("atol", 1e-9))
    max_abs_diff = float(torch.max(torch.abs(c_matrix.to(dtype=torch.float64) - expected)))
    if max_abs_diff > atol:
        raise ValueError(
            "bnb consistency check failed: c_matrix does not match "
            "lower_toeplitz_from_coeffs derivation "
            f"(max_abs_diff={max_abs_diff:.3g}, atol={atol:.3g})"
        )


def make_bnb_calibration_report(
    *,
    target_epsilon: float,
    target_delta: float,
    noise_multiplier: float,
    num_samples: int,
    seed: int,
    bands: int,
    verification: DeltaVerificationResult,
    evr_confidence_alpha_total: float,
    evr_num_checks: int,
    evr_per_check_alpha: float,
    evr_pass_count: int,
) -> BNBCalibrationReport:
    composed_delta_upper = min(
        1.0,
        float(target_delta)
        + float(evr_confidence_alpha_total) * (1.0 - float(target_delta)),
    )
    return BNBCalibrationReport(
        version=2,
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        noise_multiplier=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        bands=int(bands),
        delta_estimate_at_target_epsilon=float(verification.delta_estimate),
        delta_upper_confidence_bound=float(verification.upper_confidence_bound),
        confidence_alpha=float(verification.confidence_alpha),
        verification_passed=bool(verification.accepted),
        evr_confidence_alpha_total=float(evr_confidence_alpha_total),
        evr_num_checks=int(evr_num_checks),
        evr_per_check_alpha=float(evr_per_check_alpha),
        evr_pass_count=int(evr_pass_count),
        verification_contract=BNB_VERIFICATION_CONTRACT,
        evr_composed_delta_upper_bound=float(composed_delta_upper),
    )


def parse_bnb_calibration_report(payload: dict) -> BNBCalibrationReport:
    """
    Parse serialized BNB calibration payloads for supported schema versions.

    Parsing is strict so report consumers fail fast on schema drift.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    version = int(payload["version"])
    if version == 2:
        return BNBCalibrationReport(
            version=2,
            target_epsilon=float(payload["target_epsilon"]),
            target_delta=float(payload["target_delta"]),
            noise_multiplier=float(payload["noise_multiplier"]),
            num_samples=int(payload["num_samples"]),
            seed=int(payload["seed"]),
            bands=int(payload["bands"]),
            delta_estimate_at_target_epsilon=float(
                payload["delta_estimate_at_target_epsilon"]
            ),
            delta_upper_confidence_bound=float(payload["delta_upper_confidence_bound"]),
            confidence_alpha=float(payload["confidence_alpha"]),
            verification_passed=bool(payload["verification_passed"]),
            evr_confidence_alpha_total=float(payload["evr_confidence_alpha_total"]),
            evr_num_checks=int(payload["evr_num_checks"]),
            evr_per_check_alpha=float(payload["evr_per_check_alpha"]),
            evr_pass_count=int(payload["evr_pass_count"]),
            verification_contract=str(
                payload.get("verification_contract", BNB_VERIFICATION_CONTRACT)
            ),
            evr_composed_delta_upper_bound=float(
                payload.get(
                    "evr_composed_delta_upper_bound",
                    min(
                        1.0,
                        float(payload["target_delta"])
                        + float(payload["evr_confidence_alpha_total"])
                        * (1.0 - float(payload["target_delta"])),
                    ),
                )
            ),
        )

    raise ValueError(f"unsupported BNB calibration report version: {version}")


def describe_bnb_calibration_report(
    payload: dict | BNBCalibrationReport,
) -> str:
    """
    Produce a compact single-line summary for logs and debugging dashboards.
    """
    report = payload if isinstance(payload, BNBCalibrationReport) else parse_bnb_calibration_report(payload)
    status = "PASS" if report.verification_passed else "FAIL"

    return (
        f"BNB calibration v{report.version} [{status}] "
        f"eps={report.target_epsilon:.6g} delta={report.target_delta:.6g} "
        f"sigma={report.noise_multiplier:.6g} "
        f"ucb_delta={report.delta_upper_confidence_bound:.6g} "
        f"guard_delta={report.evr_composed_delta_upper_bound:.6g} "
        f"checks={report.evr_pass_count}/{report.evr_num_checks} "
        f"alpha_total={report.evr_confidence_alpha_total:.6g}"
    )


def split_confidence_alpha(
    *,
    total_confidence_alpha: float,
    num_checks: int,
) -> float:
    """
    Split a total EVR confidence budget across repeated checks.

    BNB often evaluates multiple Monte Carlo checks (and sometimes multiple
    sigma candidates). This helper uses a conservative union-bound split so
    each check gets ``alpha_total / num_checks`` and the overall failure budget
    remains controlled.

    Math: ``α_i = α_total / n_checks``.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Appendix A, Theorem A.1 (EVR-style confidence allocation).
    """
    if total_confidence_alpha <= 0.0 or total_confidence_alpha >= 1.0:
        raise ValueError("total_confidence_alpha must be in (0, 1)")

    if num_checks <= 0:
        raise ValueError("num_checks must be > 0")

    # `alpha` budget split: alpha_i = alpha_total / num_checks.
    return float(total_confidence_alpha) / float(num_checks)


def verify_evr_confidence_split(
    *,
    llr_samples_seq: Sequence[torch.Tensor],
    epsilon: float,
    target_delta: float,
    total_confidence_alpha: float,
) -> tuple[DeltaVerificationResult, int, float]:
    """
    Run EVR verification over multiple LLR batches with a shared alpha budget.

    Each batch is checked independently with Hoeffding bounds, then this
    function returns the worst upper confidence bound plus how many checks
    passed. The caller can then decide whether to accept only all-pass runs.

    Returns:
    ``(worst_case_verification, pass_count, per_check_alpha)``.
    """
    if len(llr_samples_seq) == 0:
        raise ValueError("llr_samples_seq must be non-empty")

    per_check_alpha = split_confidence_alpha(
        total_confidence_alpha=total_confidence_alpha,
        num_checks=len(llr_samples_seq),
    )
    # `delta` target is passed through as `target_delta` in each EVR check.

    checks: list[DeltaVerificationResult] = []
    for llr in llr_samples_seq:
        checks.append(
            verify_hockey_stick_delta_hoeffding(
                epsilon=epsilon,
                llr_samples=llr,
                target_delta=target_delta,
                confidence_alpha=per_check_alpha,
            )
        )

    pass_count = sum(1 for c in checks if c.accepted)
    worst = max(checks, key=lambda c: c.upper_confidence_bound)

    return (
        DeltaVerificationResult(
            delta_estimate=worst.delta_estimate,
            upper_confidence_bound=worst.upper_confidence_bound,
            confidence_alpha=per_check_alpha,
            accepted=pass_count == len(checks),
        ),
        int(pass_count),
        float(per_check_alpha),
    )


def select_evr_candidate_ladder(
    *,
    candidate_sigmas: Sequence[float],
    llr_samples_seq_fn,
    epsilon: float,
    target_delta: float,
    total_confidence_alpha: float,
) -> tuple[float, DeltaVerificationResult, int, float]:
    """
    Scan an increasing sigma ladder and pick the first EVR-accepted candidate.

    This is a pragmatic calibration strategy: test a small ordered grid of
    noise scales, spend confidence budget across candidates, and stop as soon
    as the EVR upper bound satisfies the target delta.

    If none pass, return diagnostics for the largest candidate.
    """
    if len(candidate_sigmas) == 0:
        raise ValueError("candidate_sigmas must be non-empty")

    cleaned = [float(s) for s in candidate_sigmas]
    if any(s <= 0.0 for s in cleaned):
        raise ValueError("candidate_sigmas must be > 0")

    if any(cleaned[i] >= cleaned[i + 1] for i in range(len(cleaned) - 1)):
        raise ValueError("candidate_sigmas must be strictly increasing")

    chosen_sigma = cleaned[-1]
    chosen_verification = None
    chosen_pass_count = 0
    chosen_per_alpha = 0.0
    # Ladder confidence split: each sigma candidate receives alpha_total / |ladder|.
    per_candidate_alpha = float(total_confidence_alpha) / float(len(cleaned))

    for sigma in cleaned:
        verification, pass_count, per_alpha = verify_evr_confidence_split(
            llr_samples_seq=llr_samples_seq_fn(float(sigma)),
            epsilon=float(epsilon),
            target_delta=float(target_delta),
            total_confidence_alpha=per_candidate_alpha,
        )
        chosen_sigma = float(sigma)
        chosen_verification = verification
        chosen_pass_count = int(pass_count)
        chosen_per_alpha = float(per_alpha)

        if verification.accepted:
            break

    assert chosen_verification is not None

    return chosen_sigma, chosen_verification, chosen_pass_count, chosen_per_alpha


def select_evr_candidate_ladder_two_sided(
    *,
    candidate_sigmas: Sequence[float],
    llr_samples_seq_fn_forward,
    llr_samples_seq_fn_reverse,
    epsilon: float,
    target_delta: float,
    total_confidence_alpha: float,
) -> tuple[float, DeltaVerificationResult, int, float]:
    """
    Two-sided EVR ladder where each sigma must pass both DP directions.

    For ``(epsilon, delta)`` guarantees we may need both ``P||Q`` and ``Q||P``
    checks. This function splits confidence per candidate and per direction,
    then accepts only when both directions pass.
    """
    if len(candidate_sigmas) == 0:
        raise ValueError("candidate_sigmas must be non-empty")

    cleaned = [float(s) for s in candidate_sigmas]
    if any(s <= 0.0 for s in cleaned):
        raise ValueError("candidate_sigmas must be > 0")

    if any(cleaned[i] >= cleaned[i + 1] for i in range(len(cleaned) - 1)):
        raise ValueError("candidate_sigmas must be strictly increasing")

    if total_confidence_alpha <= 0.0 or total_confidence_alpha >= 1.0:
        raise ValueError("total_confidence_alpha must be in (0, 1)")

    chosen_sigma = cleaned[-1]
    chosen_verification = None
    chosen_pass_count = 0
    chosen_per_alpha = 0.0
    per_candidate_alpha = float(total_confidence_alpha) / float(len(cleaned))
    per_direction_alpha = per_candidate_alpha / 2.0

    for sigma in cleaned:
        f_verification, f_pass_count, f_per_alpha = verify_evr_confidence_split(
            llr_samples_seq=llr_samples_seq_fn_forward(float(sigma)),
            epsilon=float(epsilon),
            target_delta=float(target_delta),
            total_confidence_alpha=per_direction_alpha,
        )
        r_verification, r_pass_count, r_per_alpha = verify_evr_confidence_split(
            llr_samples_seq=llr_samples_seq_fn_reverse(float(sigma)),
            epsilon=float(epsilon),
            target_delta=float(target_delta),
            total_confidence_alpha=per_direction_alpha,
        )
        if abs(float(f_per_alpha) - float(r_per_alpha)) > 1e-18:
            raise ValueError("forward/reverse per-check alpha mismatch")

        chosen_sigma = float(sigma)
        chosen_pass_count = int(f_pass_count + r_pass_count)
        chosen_per_alpha = float(f_per_alpha)
        chosen_verification = DeltaVerificationResult(
            delta_estimate=max(
                float(f_verification.delta_estimate),
                float(r_verification.delta_estimate),
            ),
            upper_confidence_bound=max(
                float(f_verification.upper_confidence_bound),
                float(r_verification.upper_confidence_bound),
            ),
            confidence_alpha=float(f_per_alpha),
            accepted=bool(f_verification.accepted and r_verification.accepted),
        )
        if chosen_verification.accepted:
            break

    assert chosen_verification is not None

    return chosen_sigma, chosen_verification, chosen_pass_count, chosen_per_alpha


def build_b_min_sep_gaussian_mixture(
    *,
    c_matrix: torch.Tensor,
    bands: int | None = None,
    cycle_length: int | None = None,
    reduce_dimensionality: bool = False,
) -> GaussianMixture:
    """
    Build the BMinSep Gaussian mixture induced by matrix ``C``.

    Under cycle-aligned participation such as balls-in-bins, one sample's
    contribution induces a finite set of possible mean shifts in the matrix
    mechanism output. For cycle length ``b``, the positive distribution is a
    uniform mixture over the ``b`` possible starting offsets. Each mode is the
    sum of the columns of ``C`` whose indices share the same residue modulo
    ``b``.

    This matches the current JAX balls-in-bins Monte Carlo construction, where
    the sensitive example is assigned one cycle offset uniformly at random and
    then participates in every iteration with that offset.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5, Equations (2)-(4).
    """
    if c_matrix.ndim != 2:
        raise ValueError("c_matrix must have shape [d, m]")

    grouping = (
        int(cycle_length)
        if cycle_length is not None
        else (int(bands) if bands is not None else None)
    )
    if grouping is None:
        raise ValueError("cycle_length or bands must be provided")
    if grouping <= 0:
        raise ValueError("cycle_length must be > 0")

    d, _m = c_matrix.shape
    modes = torch.stack(
        [c_matrix[:, offset::grouping].sum(dim=1) for offset in range(grouping)],
        dim=0,
    )  # [grouping, d]

    if reduce_dimensionality:
        # Keep the same behavior class as notebook-style dimensionality reduction.
        modes = torch.linalg.qr(modes.T, mode="r").R.T

    modes = modes.contiguous()  # [b, d']
    probs = torch.full((grouping,), 1.0 / float(grouping), dtype=modes.dtype)

    return GaussianMixture(modes=modes, probs=probs)


def _mixture_logpdf(points: torch.Tensor, gm: GaussianMixture, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    if points.ndim != 2:
        raise ValueError("points must have shape [n, d]")

    if points.shape[1] != gm.modes.shape[1]:
        raise ValueError("points dimension must match mixture modes")

    sigma_sq = sigma * sigma

    # Compute squared distances without materializing [n, k, d]:
    # ||x - m||^2 = ||x||^2 + ||m||^2 - 2 x m^T
    points_sq = torch.sum(points * points, dim=1, keepdim=True)  # [n, 1]
    modes_sq = torch.sum(gm.modes * gm.modes, dim=1).unsqueeze(0)  # [1, k]
    cross = points @ gm.modes.T  # [n, k]
    sq_dist = points_sq + modes_sq - 2.0 * cross
    sq_dist = torch.clamp(sq_dist, min=0.0)

    component_log_probs = torch.log(gm.probs)[None, :] - 0.5 * sq_dist / sigma_sq

    return torch.logsumexp(component_log_probs, dim=1)


def generate_mixture_samples(
    *,
    gm: GaussianMixture,
    sigma: float,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    device = gm.modes.device
    dtype = gm.modes.dtype
    component_ids = torch.multinomial(
        gm.probs.to(device=device, dtype=torch.float64),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    ).to(device=device)

    means = gm.modes[component_ids]
    noise = torch.randn(
        num_samples,
        gm.modes.shape[1],
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return component_ids, means + sigma * noise


def _validate_bnb_chunking(
    *,
    num_samples: int,
    chunk_size: int | None,
    num_workers: int,
) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be > 0 when provided")

    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")


def _resolve_bnb_backend_and_device(
    *,
    backend: str,
    device: str | torch.device | None,
) -> tuple[str, torch.device]:
    resolved_backend = str(backend).lower()
    if resolved_backend not in ("auto", "cpu", "cuda"):
        raise ValueError("bnb_backend must be one of {'auto', 'cpu', 'cuda'}")

    resolved_device: torch.device | None = None
    if device is not None:
        resolved_device = torch.device(device)

    if resolved_backend == "auto":
        if resolved_device is not None:
            if resolved_device.type == "cuda" and torch.cuda.is_available():
                return "cuda", resolved_device

            return "cpu", torch.device("cpu")

        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")

        return "cpu", torch.device("cpu")

    if resolved_backend == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("bnb_backend='cuda' requires CUDA to be available")
        if resolved_device is None:
            resolved_device = torch.device("cuda")
        elif resolved_device.type != "cuda":
            raise ValueError("bnb_device must be a CUDA device when bnb_backend='cuda'")

        return "cuda", resolved_device

    return "cpu", torch.device("cpu")


def _make_bnb_broadcast_result_tensor(
    value: float,
    *,
    backend: str,
    device: str | torch.device | None,
) -> torch.Tensor:
    resolved_backend, resolved_device = _resolve_bnb_backend_and_device(
        backend=backend,
        device=device,
    )
    tensor_device = resolved_device if resolved_backend == "cuda" else torch.device("cpu")

    return torch.tensor(float(value), dtype=torch.float64, device=tensor_device)


def _resolve_bnb_distributed_mode(
    *,
    distributed_mode: str | None,
    distributed_dp_runtime: bool,
) -> tuple[str, bool]:
    if distributed_mode is None:
        requested_mode = "chunk_shard" if distributed_dp_runtime else "none"
        auto_selected = bool(distributed_dp_runtime)
    else:
        requested_mode = str(distributed_mode)
        auto_selected = False

    if requested_mode not in ("none", "chunk_shard"):
        raise ValueError("bnb_distributed_mode must be one of {'none', 'chunk_shard'}")

    return requested_mode, auto_selected


def _assign_bnb_chunk_specs_to_shard(
    *,
    specs: list[tuple[int, int]],
    rank: int,
    world_size: int,
) -> list[tuple[int, int]]:
    if rank < 0:
        raise ValueError("rank must be >= 0")

    if world_size <= 0:
        raise ValueError("world_size must be > 0")

    if rank >= world_size:
        raise ValueError("rank must be < world_size")

    return [spec for index, spec in enumerate(specs) if index % world_size == rank]


def _reduce_bnb_llr_chunks_to_coordinator(
    *,
    local_chunks: list[torch.Tensor],
    distributed_mode: str,
) -> list[torch.Tensor]:
    if distributed_mode != "chunk_shard":
        return local_chunks

    if not dist.is_available() or not dist.is_initialized():
        raise ValueError(
            "bnb_distributed_mode='chunk_shard' requires torch.distributed to be initialized"
        )

    world_size = dist.get_world_size()
    gathered: list[list[torch.Tensor] | None] = [None for _ in range(world_size)]
    payload = [chunk.detach().cpu() for chunk in local_chunks]
    dist.all_gather_object(gathered, payload)
    rank = dist.get_rank()
    if rank != 0:
        return []

    reduced: list[torch.Tensor] = []
    for shard_chunks in gathered:
        if shard_chunks:
            reduced.extend([chunk.to(dtype=torch.float64) for chunk in shard_chunks])

    return reduced


def _derive_bnb_chunk_specs(
    *,
    num_samples: int,
    seed: int,
    chunk_size: int | None,
) -> list[tuple[int, int]]:
    if chunk_size is None or chunk_size >= num_samples:
        return [(int(num_samples), int(seed))]

    specs: list[tuple[int, int]] = []
    remaining = int(num_samples)
    chunk_index = 0

    while remaining > 0:
        current = min(int(chunk_size), remaining)
        specs.append((current, int(seed + chunk_index)))
        remaining -= current
        chunk_index += 1

    return specs


def compute_llr_samples(
    *,
    up_gm: GaussianMixture,
    lo_gm: GaussianMixture,
    sigma: float,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if up_gm.modes.shape[1] != lo_gm.modes.shape[1]:
        raise ValueError("up_gm and lo_gm must have the same dimension")

    _, points = generate_mixture_samples(
        gm=up_gm,
        sigma=sigma,
        num_samples=num_samples,
        generator=generator,
    )

    return _mixture_logpdf(points, up_gm, sigma) - _mixture_logpdf(points, lo_gm, sigma)


def _hoeffding_bound(num_samples: int, tau: float, delta: float) -> float:
    if tau < 1.0:
        raise ValueError("tau must be >= 1")

    q = torch.tensor(float(delta), dtype=torch.float64)
    p = torch.tensor(float(tau) * float(delta), dtype=torch.float64)
    kl = kl_divergence(Bernoulli(probs=q), Bernoulli(probs=p))

    return float(torch.exp(-float(num_samples) * kl))


def get_bnb_overall_delta(
    *,
    num_samples: int,
    base_delta: float,
) -> float:
    if base_delta <= 0.0 or base_delta > 1.0:
        raise ValueError("base_delta must be in (0, 1]")

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    def _overall_delta_from_tau(tau: float) -> float:
        q = _hoeffding_bound(int(num_samples), float(tau), float(base_delta))
        return float(tau) * float(base_delta) + q * (1.0 - float(tau) * float(base_delta))

    best_tau = scipy.optimize.minimize_scalar(
        _overall_delta_from_tau,
        bounds=(1.0, 1.0 / float(base_delta)),
        method="bounded",
    ).x

    return min(_overall_delta_from_tau(float(best_tau)), 1.0)


def get_bnb_base_delta(
    *,
    num_samples: int,
    target_delta: float,
) -> float:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if target_delta < 0.0 or target_delta > 1.0:
        raise ValueError("target_delta must be in [0, 1]")

    tol = 1e-4 * float(target_delta)
    base_delta = scipy.optimize.minimize_scalar(
        lambda d: abs(get_bnb_overall_delta(num_samples=int(num_samples), base_delta=float(d)) - float(target_delta)),
        bounds=(0.0, float(target_delta)),
        method="bounded",
        options={"xatol": tol},
    ).x

    if get_bnb_overall_delta(num_samples=int(num_samples), base_delta=float(base_delta)) < float(target_delta):
        return float(base_delta)

    conservative_base_delta = float(base_delta) - tol
    if conservative_base_delta > 0.0 and get_bnb_overall_delta(
        num_samples=int(num_samples), base_delta=conservative_base_delta
    ) < float(target_delta):
        return conservative_base_delta

    raise ValueError("Failed to find a valid base_delta. num_samples may be too small.")


def _build_balls_in_bins_modes_matrix(
    *,
    coeffs: Sequence[float],
    cycle_length: int,
    horizon: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:

    def _convolve_full_1d(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            lhs.numel() + rhs.numel() - 1, dtype=torch.float64, device=lhs.device
        )
        for idx in range(lhs.numel()):
            out[idx : idx + rhs.numel()] += lhs[idx] * rhs

        return out

    def _toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        rows = int(c.numel())
        cols = int(r.numel())
        out = torch.empty((rows, cols), dtype=torch.float64, device=c.device)
        for i in range(rows):
            for j in range(cols):
                out[i, j] = r[j - i] if j >= i else c[i - j]

        return out

    coeff_list = [float(c) for c in coeffs]
    if len(coeff_list) == 0:
        raise ValueError("coeffs must be a non-empty 1-D sequence")

    if any(c < 0.0 for c in coeff_list):
        raise ValueError("coeffs must be nonnegative")

    if cycle_length <= 0:
        raise ValueError("cycle_length must be positive")

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if len(coeff_list) > horizon:
        coeff_list = coeff_list[:horizon]

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    coeff_t = torch.tensor(coeff_list, dtype=torch.float64, device=resolved_device)
    x_t = (torch.arange(horizon, device=resolved_device) % int(cycle_length) == 0).to(dtype=torch.float64)
    first_mode = _convolve_full_1d(
        coeff_t,
        x_t[: horizon - coeff_t.numel() + 1],
    )

    if len(coeff_list) > 1:
        bot_block = _toeplitz(
            coeff_t[:-1],
            torch.zeros(coeff_t.numel() - 1, dtype=torch.float64, device=resolved_device),
        )
        bot_prod = torch.mv(bot_block, x_t[-coeff_t.numel() + 1 :])
        first_mode[-coeff_t.numel() + 1 :] += bot_prod

    elementary_vector = torch.zeros(int(cycle_length), dtype=torch.float64, device=resolved_device)
    elementary_vector[0] = coeff_t[0]

    return _toeplitz(elementary_vector, first_mode)


def _generate_balls_in_bins_samples_chunk(
    *,
    coeffs: Sequence[float],
    cycle_length: int,
    horizon: int,
    sigma: float,
    num_samples: int,
    seed: int,
    positive_sample: bool,
    modes_matrix: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if modes_matrix is None:
        modes_matrix = _build_balls_in_bins_modes_matrix(
            coeffs=coeffs,
            cycle_length=cycle_length,
            horizon=horizon,
            device=device,
        )

    resolved_device = (
        torch.device(device)
        if device is not None
        else modes_matrix.device
    )
    generator = torch.Generator(device=resolved_device).manual_seed(int(seed))
    if positive_sample:
        starting_indices = torch.randint(
            low=0,
            high=int(cycle_length),
            size=(int(num_samples),),
            generator=generator,
            device=resolved_device,
        )
        means = modes_matrix[starting_indices]
    else:
        means = torch.zeros(
            (int(num_samples), int(horizon)),
            dtype=torch.float64,
            device=resolved_device,
        )

    noise = torch.randn(
        int(num_samples),
        int(horizon),
        generator=generator,
        device=resolved_device,
        dtype=torch.float64,
    ) * float(sigma)

    return means + noise


def _compute_balls_in_bins_privacy_loss_chunk(
    *,
    samples: torch.Tensor,
    sigma: float,
    modes_matrix: torch.Tensor,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    if samples.ndim != 2:
        raise ValueError("samples must have shape [n, horizon]")

    dot_products = torch.matmul(samples, modes_matrix.T)
    squared_mode_norms = torch.sum(modes_matrix * modes_matrix, dim=1)
    per_mode_privacy_loss = (2.0 * dot_products - squared_mode_norms.unsqueeze(0)) / (
        2.0 * float(sigma) ** 2
    )

    return torch.logsumexp(per_mode_privacy_loss, dim=1) - math.log(
        float(modes_matrix.shape[0])
    )


def sample_balls_in_bins_llr_chunks(
    *,
    coeffs: Sequence[float],
    cycle_length: int,
    horizon: int,
    sigma: float,
    num_samples: int,
    seed: int = 0,
    chunk_size: int | None = None,
    num_workers: int = 0,
    positive_sample: bool = True,
    backend: str = "auto",
    device: str | torch.device | None = None,
    distributed_mode: str | None = "none",
    distributed_dp_runtime: bool = False,
) -> list[torch.Tensor]:
    _validate_bnb_chunking(
        num_samples=int(num_samples),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
    )
    resolved_backend, resolved_device = _resolve_bnb_backend_and_device(
        backend=backend,
        device=device,
    )
    resolved_distributed_mode, _auto_selected = _resolve_bnb_distributed_mode(
        distributed_mode=distributed_mode,
        distributed_dp_runtime=distributed_dp_runtime,
    )
    specs = _derive_bnb_chunk_specs(
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
    )
    if resolved_distributed_mode == "chunk_shard":
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError(
                "bnb_distributed_mode='chunk_shard' requires torch.distributed to be initialized"
            )
        specs = _assign_bnb_chunk_specs_to_shard(
            specs=specs,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )
    modes_matrix = _build_balls_in_bins_modes_matrix(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        device=resolved_device,
    )

    def _one(chunk_num_samples: int, chunk_seed: int) -> torch.Tensor:
        samples = _generate_balls_in_bins_samples_chunk(
            coeffs=coeffs,
            cycle_length=int(cycle_length),
            horizon=int(horizon),
            sigma=float(sigma),
            num_samples=int(chunk_num_samples),
            seed=int(chunk_seed),
            positive_sample=bool(positive_sample),
            modes_matrix=modes_matrix,
            device=resolved_device,
        )
        llr = _compute_balls_in_bins_privacy_loss_chunk(
            samples=samples,
            sigma=float(sigma),
            modes_matrix=modes_matrix,
        )
        if not positive_sample:
            llr = -llr

        return llr

    if len(specs) == 1:
        chunk_num_samples, chunk_seed = specs[0]
        return _reduce_bnb_llr_chunks_to_coordinator(
            local_chunks=[_one(int(chunk_num_samples), int(chunk_seed))],
            distributed_mode=resolved_distributed_mode,
        )

    if num_workers <= 1:
        local_chunks = [
            _one(int(chunk_num_samples), int(chunk_seed))
            for chunk_num_samples, chunk_seed in specs
        ]
        return _reduce_bnb_llr_chunks_to_coordinator(
            local_chunks=local_chunks,
            distributed_mode=resolved_distributed_mode,
        )

    if resolved_backend == "cuda":
        local_chunks = [
            _one(int(chunk_num_samples), int(chunk_seed))
            for chunk_num_samples, chunk_seed in specs
        ]
        return _reduce_bnb_llr_chunks_to_coordinator(
            local_chunks=local_chunks,
            distributed_mode=resolved_distributed_mode,
        )

    max_workers = min(int(num_workers), len(specs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_one, int(chunk_num_samples), int(chunk_seed))
            for chunk_num_samples, chunk_seed in specs
        ]
        local_chunks = [future.result() for future in futures]
    return _reduce_bnb_llr_chunks_to_coordinator(
        local_chunks=local_chunks,
        distributed_mode=resolved_distributed_mode,
    )


def estimate_balls_in_bins_epsilon_monte_carlo(
    *,
    coeffs: Sequence[float],
    cycle_length: int,
    horizon: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    seed: int = 0,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    chunk_size: int | None = None,
    num_workers: int = 0,
    backend: str = "auto",
    device: str | torch.device | None = None,
    distributed_mode: str | None = "none",
    distributed_dp_runtime: bool = False,
) -> float:
    if target_delta < 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in [0, 1)")
    base_delta = get_bnb_base_delta(num_samples=int(num_samples), target_delta=float(target_delta))
    positive_chunks = sample_balls_in_bins_llr_chunks(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
        positive_sample=True,
        backend=backend,
        device=device,
        distributed_mode=distributed_mode,
        distributed_dp_runtime=distributed_dp_runtime,
    )
    negative_chunks = sample_balls_in_bins_llr_chunks(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed) + 1,
        chunk_size=chunk_size,
        num_workers=int(num_workers),
        positive_sample=False,
        backend=backend,
        device=device,
        distributed_mode=distributed_mode,
        distributed_dp_runtime=distributed_dp_runtime,
    )
    if distributed_mode == "chunk_shard" and dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        result = _make_bnb_broadcast_result_tensor(
            float("nan"),
            backend=backend,
            device=device,
        )
        dist.broadcast(result, src=0)
        return float(result.item())
    positive_epsilon = estimate_epsilon_from_llr_chunks(
        target_delta=float(base_delta),
        llr_chunks=[torch.as_tensor(chunk, dtype=torch.float64) for chunk in positive_chunks],
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    negative_epsilon = estimate_epsilon_from_llr_chunks(
        target_delta=float(base_delta),
        llr_chunks=[torch.as_tensor(chunk, dtype=torch.float64) for chunk in negative_chunks],
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    epsilon = float(max(float(positive_epsilon), float(negative_epsilon)))
    if distributed_mode == "chunk_shard" and dist.is_available() and dist.is_initialized():
        result = _make_bnb_broadcast_result_tensor(
            epsilon,
            backend=backend,
            device=device,
        )
        dist.broadcast(result, src=0)
        return float(result.item())

    return epsilon


def estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
    *,
    coeffs: Sequence[float],
    cycle_length: int,
    horizon: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    seed: int = 0,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    chunk_size: int | None = None,
    num_workers: int = 0,
    backend: str = "auto",
    device: str | torch.device | None = None,
    distributed_mode: str | None = "none",
    distributed_dp_runtime: bool = False,
) -> float:
    """
    Estimate epsilon directly from balls-in-bins Monte Carlo LLR samples.

    This is the optimistic point-estimate surface: it inverts `delta(epsilon)`
    directly at the target `delta` without the EVR/base-delta feasibility split.
    """
    if target_delta < 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in [0, 1)")

    positive_chunks = sample_balls_in_bins_llr_chunks(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
        positive_sample=True,
        backend=backend,
        device=device,
        distributed_mode=distributed_mode,
        distributed_dp_runtime=distributed_dp_runtime,
    )
    negative_chunks = sample_balls_in_bins_llr_chunks(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed) + 1,
        chunk_size=chunk_size,
        num_workers=int(num_workers),
        positive_sample=False,
        backend=backend,
        device=device,
        distributed_mode=distributed_mode,
        distributed_dp_runtime=distributed_dp_runtime,
    )
    if (
        distributed_mode == "chunk_shard"
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_rank() != 0
    ):
        result = _make_bnb_broadcast_result_tensor(
            float("nan"),
            backend=backend,
            device=device,
        )
        dist.broadcast(result, src=0)
        return float(result.item())

    positive_epsilon = estimate_epsilon_from_llr_chunks(
        target_delta=float(target_delta),
        llr_chunks=[torch.as_tensor(chunk, dtype=torch.float64) for chunk in positive_chunks],
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    negative_epsilon = estimate_epsilon_from_llr_chunks(
        target_delta=float(target_delta),
        llr_chunks=[torch.as_tensor(chunk, dtype=torch.float64) for chunk in negative_chunks],
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    epsilon = float(max(float(positive_epsilon), float(negative_epsilon)))
    if distributed_mode == "chunk_shard" and dist.is_available() and dist.is_initialized():
        result = _make_bnb_broadcast_result_tensor(
            epsilon,
            backend=backend,
            device=device,
        )
        dist.broadcast(result, src=0)
        return float(result.item())

    return epsilon


def _compute_llr_samples_chunk(
    *,
    up_gm: GaussianMixture,
    lo_gm: GaussianMixture,
    sigma: float,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=up_gm.modes.device).manual_seed(int(seed))
    return compute_llr_samples(
        up_gm=up_gm,
        lo_gm=lo_gm,
        sigma=sigma,
        num_samples=num_samples,
        generator=generator,
    )


def compute_llr_sample_chunks(
    *,
    up_gm: GaussianMixture,
    lo_gm: GaussianMixture,
    sigma: float,
    num_samples: int,
    seed: int = 0,
    chunk_size: int | None = None,
    num_workers: int = 0,
) -> list[torch.Tensor]:
    if up_gm.modes.shape[1] != lo_gm.modes.shape[1]:
        raise ValueError("up_gm and lo_gm must have the same dimension")

    _validate_bnb_chunking(
        num_samples=int(num_samples),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
    )
    specs = _derive_bnb_chunk_specs(
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
    )

    if len(specs) == 1:
        chunk_num_samples, chunk_seed = specs[0]
        return [
            _compute_llr_samples_chunk(
                up_gm=up_gm,
                lo_gm=lo_gm,
                sigma=float(sigma),
                num_samples=int(chunk_num_samples),
                seed=int(chunk_seed),
            )
        ]

    if num_workers <= 1:
        generator = torch.Generator(device=up_gm.modes.device).manual_seed(int(seed))
        return [
            compute_llr_samples(
                up_gm=up_gm,
                lo_gm=lo_gm,
                sigma=float(sigma),
                num_samples=int(chunk_num_samples),
                generator=generator,
            )
            for chunk_num_samples, _chunk_seed in specs
        ]

    max_workers = min(int(num_workers), len(specs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _compute_llr_samples_chunk,
                up_gm=up_gm,
                lo_gm=lo_gm,
                sigma=float(sigma),
                num_samples=int(chunk_num_samples),
                seed=int(chunk_seed),
            )
            for chunk_num_samples, chunk_seed in specs
        ]

        return [future.result() for future in futures]


def sample_b_min_sep_llr(
    *,
    c_matrix: torch.Tensor,
    bands: int | None = None,
    cycle_length: int | None = None,
    sigma: float,
    num_samples: int,
    seed: int = 0,
    reduce_dimensionality: bool = False,
    chunk_size: int | None = None,
    num_workers: int = 0,
) -> torch.Tensor:
    """
    Sample privacy-loss log-likelihood ratios for the BMinSep mechanism.

    We draw outputs from the ``up`` mixture (adjacent dataset with one
    contribution) and evaluate ``LLR = log p_up(y) - log p_lo(y)``, where
    ``lo`` is the zero-mean baseline mixture. These LLR samples are the direct
    input to Monte Carlo estimates of hockey-stick divergence.

    Source: BMinSep (Dong and Ganesh, 2025), Section 5, Equations (2)-(4), Theorem 5.1.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if seed < 0:
        raise ValueError("seed must be >= 0")

    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    up_gm = build_b_min_sep_gaussian_mixture(
        c_matrix=c_matrix,
        bands=bands,
        cycle_length=cycle_length,
        reduce_dimensionality=reduce_dimensionality,
    )
    zero_mode = torch.zeros(
        1,
        up_gm.modes.shape[1],
        dtype=up_gm.modes.dtype,
        device=up_gm.modes.device,
    )
    lo_gm = GaussianMixture(
        modes=zero_mode,
        probs=torch.ones(1, dtype=up_gm.modes.dtype, device=up_gm.modes.device),
    )

    chunks = compute_llr_sample_chunks(
        up_gm=up_gm,
        lo_gm=lo_gm,
        sigma=float(sigma),
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
    )
    if len(chunks) == 1:
        return chunks[0]

    return torch.cat(chunks, dim=0)


def estimate_hockey_stick_delta_from_llr_samples(
    *,
    epsilon: float,
    llr_samples: torch.Tensor,
) -> float:
    if llr_samples.ndim != 1:
        raise ValueError("llr_samples must be a 1-D tensor")

    if llr_samples.numel() == 0:
        raise ValueError("llr_samples must be non-empty")

    vals = torch.clamp(1.0 - torch.exp(epsilon - llr_samples), min=0.0)

    return float(torch.mean(vals))


def estimate_hockey_stick_delta_from_llr_chunks(
    *,
    epsilon: float,
    llr_chunks: Sequence[torch.Tensor],
) -> float:
    if len(llr_chunks) == 0:
        raise ValueError("llr_chunks must be non-empty")

    total = 0.0
    count = 0
    for chunk in llr_chunks:
        if chunk.ndim != 1:
            raise ValueError("each llr chunk must be a 1-D tensor")
        if chunk.numel() == 0:
            continue
        vals = torch.clamp(1.0 - torch.exp(float(epsilon) - chunk), min=0.0)
        total += float(torch.sum(vals))
        count += int(chunk.numel())

    if count <= 0:
        raise ValueError("llr_chunks must contain at least one sample")

    return total / float(count)


def estimate_delta_upper_bound_single_verify_from_llr_samples(
    *,
    epsilon: float,
    llr_samples: torch.Tensor,
    error_probability: float = 1e-6,
) -> SingleVerificationResult:
    """
    Upper-bound ``delta(epsilon)`` for one fixed mechanism via Bernoulli KL inversion.

    Let ``q_hat`` be the empirical hockey-stick divergence estimate from sampled
    privacy losses. This helper returns the smallest ``p >= q_hat`` such that
    ``KL(Bernoulli(q_hat) || Bernoulli(p)) >= log(1 / beta) / n`` where
    ``beta = error_probability`` and ``n`` is the number of samples.

    This is the original single-verification Monte Carlo contract from the
    balls-in-bins paper. It intentionally does not use EVR/base-delta terms.
    """
    if error_probability <= 0.0 or error_probability >= 1.0:
        raise ValueError("error_probability must be in (0, 1)")

    if llr_samples.ndim != 1 or llr_samples.numel() == 0:
        raise ValueError("llr_samples must be a non-empty 1-D tensor")

    samples = llr_samples.to(dtype=torch.float64)
    q_hat = float(
        estimate_hockey_stick_delta_from_llr_samples(
            epsilon=float(epsilon),
            llr_samples=samples,
        )
    )
    n = int(samples.numel())
    threshold = math.log(1.0 / float(error_probability)) / float(n)

    if q_hat >= 1.0:
        upper = 1.0
    else:
        q = torch.tensor(float(q_hat), dtype=torch.float64)

        def _bernoulli_kl_to(candidate: float) -> float:
            p = torch.tensor(float(candidate), dtype=torch.float64)
            return float(kl_divergence(Bernoulli(probs=q), Bernoulli(probs=p)))

        low = float(q_hat)
        high = 1.0
        if _bernoulli_kl_to(high) < threshold:
            upper = 1.0
        else:
            for _ in range(80):
                mid = 0.5 * (low + high)
                if _bernoulli_kl_to(mid) >= threshold:
                    high = mid
                else:
                    low = mid
            upper = float(high)

    return SingleVerificationResult(
        delta_estimate=float(q_hat),
        upper_confidence_bound=float(upper),
        error_probability=float(error_probability),
        accepted=False,
    )


def estimate_delta_upper_bound_single_verify_from_llr_chunks(
    *,
    epsilon: float,
    llr_chunks: Sequence[torch.Tensor],
    error_probability: float = 1e-6,
) -> SingleVerificationResult:
    if len(llr_chunks) == 0:
        raise ValueError("llr_chunks must be non-empty")

    total = sum(int(chunk.numel()) for chunk in llr_chunks if chunk.ndim == 1)
    if total <= 0:
        raise ValueError("llr_chunks must contain at least one sample")

    q_hat = float(
        estimate_hockey_stick_delta_from_llr_chunks(
            epsilon=float(epsilon),
            llr_chunks=llr_chunks,
        )
    )
    threshold = math.log(1.0 / float(error_probability)) / float(total)

    if q_hat >= 1.0:
        upper = 1.0
    else:
        q = torch.tensor(float(q_hat), dtype=torch.float64)

        def _bernoulli_kl_to(candidate: float) -> float:
            p = torch.tensor(float(candidate), dtype=torch.float64)
            return float(kl_divergence(Bernoulli(probs=q), Bernoulli(probs=p)))

        low = float(q_hat)
        high = 1.0
        if _bernoulli_kl_to(high) < threshold:
            upper = 1.0
        else:
            for _ in range(80):
                mid = 0.5 * (low + high)
                if _bernoulli_kl_to(mid) >= threshold:
                    high = mid
                else:
                    low = mid
            upper = float(high)

    return SingleVerificationResult(
        delta_estimate=float(q_hat),
        upper_confidence_bound=float(upper),
        error_probability=float(error_probability),
        accepted=False,
    )


def verify_hockey_stick_delta_hoeffding(
    *,
    epsilon: float,
    llr_samples: torch.Tensor,
    target_delta: float,
    confidence_alpha: float = 1e-6,
) -> DeltaVerificationResult:
    """
    Verify ``delta(epsilon) <= target_delta`` with a Hoeffding confidence bound.

    The empirical estimate from sampled LLRs is converted to an upper
    confidence bound with failure probability ``confidence_alpha``. Acceptance
    is based on that upper bound, not the mean estimate.
    """
    if target_delta <= 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in (0, 1)")

    if confidence_alpha <= 0.0 or confidence_alpha >= 1.0:
        raise ValueError("confidence_alpha must be in (0, 1)")

    if llr_samples.ndim != 1 or llr_samples.numel() == 0:
        raise ValueError("llr_samples must be a non-empty 1-D tensor")

    samples = llr_samples.to(dtype=torch.float64)
    vals = torch.clamp(1.0 - torch.exp(float(epsilon) - samples), min=0.0, max=1.0)
    delta_estimate = float(torch.mean(vals))

    n = float(vals.numel())
    radius = math.sqrt(math.log(1.0 / confidence_alpha) / (2.0 * n))
    upper = min(1.0, delta_estimate + radius)

    return DeltaVerificationResult(
        delta_estimate=delta_estimate,
        upper_confidence_bound=upper,
        confidence_alpha=float(confidence_alpha),
        accepted=upper <= target_delta,
    )


def estimate_epsilon_from_llr_samples(
    *,
    target_delta: float,
    llr_samples: torch.Tensor,
    epsilon_low: float = 0.0,
    epsilon_high: float | None = None,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> float:
    """
    Invert Monte Carlo ``delta(epsilon)`` to estimate epsilon at target delta.

    For fixed LLR samples, ``delta(epsilon)`` is monotone, so we bracket and
    binary-search epsilon until the target delta is matched within tolerance.
    """
    if target_delta < 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in [0, 1)")

    if epsilon_low < 0.0:
        raise ValueError("epsilon_low must be >= 0")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    samples = llr_samples.to(dtype=torch.float64)
    delta_at_low = estimate_hockey_stick_delta_from_llr_samples(
        epsilon=epsilon_low,
        llr_samples=samples,
    )

    if target_delta >= delta_at_low:
        return float(epsilon_low)

    low = float(epsilon_low)
    high = _resolve_epsilon_upper_bound(
        target_delta=target_delta,
        llr_samples=samples,
        epsilon_low=low,
        epsilon_high=epsilon_high,
        max_iterations=max_iterations,
    )

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        d_mid = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=mid,
            llr_samples=samples,
        )

        if abs(d_mid - target_delta) <= tolerance:
            return float(mid)

        if d_mid > target_delta:
            low = mid

        else:
            high = mid

    return float(0.5 * (low + high))


def _resolve_epsilon_upper_bound(
    *,
    target_delta: float,
    llr_samples: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float | None,
    max_iterations: int,
) -> float:
    if epsilon_high is not None:
        high = float(epsilon_high)
        if high <= epsilon_low:
            raise ValueError("epsilon_high must be > epsilon_low")

        d_high = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=high,
            llr_samples=llr_samples,
        )
        if d_high > target_delta:
            raise ValueError("epsilon_high does not satisfy target_delta")

        return high

    high = max(1.0, epsilon_low + 1.0)
    for _ in range(max_iterations):
        d_high = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=high,
            llr_samples=llr_samples,
        )
        if d_high <= target_delta:
            return high

        high *= 2.0

    raise ValueError("could not bracket epsilon; increase max_iterations")


def estimate_epsilon_from_llr_chunks(
    *,
    target_delta: float,
    llr_chunks: Sequence[torch.Tensor],
    epsilon_low: float = 0.0,
    epsilon_high: float | None = None,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> float:
    if target_delta < 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in [0, 1)")

    if epsilon_low < 0.0:
        raise ValueError("epsilon_low must be >= 0")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    delta_at_low = estimate_hockey_stick_delta_from_llr_chunks(
        epsilon=epsilon_low,
        llr_chunks=llr_chunks,
    )
    if target_delta >= delta_at_low:
        return float(epsilon_low)

    low = float(epsilon_low)
    if epsilon_high is not None:
        high = float(epsilon_high)
        if high <= low:
            raise ValueError("epsilon_high must be > epsilon_low")
        d_high = estimate_hockey_stick_delta_from_llr_chunks(
            epsilon=high,
            llr_chunks=llr_chunks,
        )
        if d_high > target_delta:
            raise ValueError("epsilon_high does not satisfy target_delta")
    else:
        high = max(1.0, low + 1.0)
        for _ in range(max_iterations):
            d_high = estimate_hockey_stick_delta_from_llr_chunks(
                epsilon=high,
                llr_chunks=llr_chunks,
            )
            if d_high <= target_delta:
                break
            high *= 2.0
        else:
            raise ValueError("could not bracket epsilon; increase max_iterations")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        d_mid = estimate_hockey_stick_delta_from_llr_chunks(
            epsilon=mid,
            llr_chunks=llr_chunks,
        )
        if abs(d_mid - target_delta) <= tolerance:
            return float(mid)
        if d_mid > target_delta:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


def estimate_b_min_sep_epsilon_monte_carlo(
    *,
    c_matrix: torch.Tensor,
    bands: int | None = None,
    cycle_length: int | None = None,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    seed: int = 0,
    reduce_dimensionality: bool = False,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    chunk_size: int | None = None,
    num_workers: int = 0,
) -> float:
    """
    Estimate epsilon for BMinSep accounting from Monte Carlo LLR samples.

    This is the main one-shot estimator used by the BNB accountant: build LLR
    samples for a fixed ``noise_multiplier`` and then invert ``delta(epsilon)``
    numerically to return epsilon at ``target_delta``.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5, Equations (2)-(4), Theorem 5.1.
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if seed < 0:
        raise ValueError("seed must be >= 0")

    up_gm = build_b_min_sep_gaussian_mixture(
        c_matrix=c_matrix,
        bands=bands,
        cycle_length=cycle_length,
        reduce_dimensionality=reduce_dimensionality,
    )
    zero_mode = torch.zeros(
        1,
        up_gm.modes.shape[1],
        dtype=up_gm.modes.dtype,
        device=up_gm.modes.device,
    )
    lo_gm = GaussianMixture(
        modes=zero_mode,
        probs=torch.ones(1, dtype=up_gm.modes.dtype, device=up_gm.modes.device),
    )
    llr_chunks = compute_llr_sample_chunks(
        up_gm=up_gm,
        lo_gm=lo_gm,
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
    )

    return estimate_epsilon_from_llr_chunks(
        target_delta=float(target_delta),
        llr_chunks=llr_chunks,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def estimate_b_min_sep_delta_monte_carlo(
    *,
    c_matrix: torch.Tensor,
    bands: int | None = None,
    cycle_length: int | None = None,
    noise_multiplier: float,
    epsilon: float,
    num_samples: int,
    seed: int = 0,
    reduce_dimensionality: bool = False,
    chunk_size: int | None = None,
    num_workers: int = 0,
) -> float:
    """
    Estimate hockey-stick ``delta(epsilon)`` for BMinSep from Monte Carlo LLRs.

    This is the direct Monte Carlo estimator used in sigma-calibration loops:
    sample LLR values at fixed ``sigma`` and compute
    ``E[max(0, 1 - exp(epsilon - LLR))]``.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5, Equations (2)-(4), Theorem 5.1.
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")

    if epsilon < 0.0:
        raise ValueError("epsilon must be >= 0")

    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if seed < 0:
        raise ValueError("seed must be >= 0")

    up_gm = build_b_min_sep_gaussian_mixture(
        c_matrix=c_matrix,
        bands=bands,
        cycle_length=cycle_length,
        reduce_dimensionality=reduce_dimensionality,
    )
    zero_mode = torch.zeros(
        1,
        up_gm.modes.shape[1],
        dtype=up_gm.modes.dtype,
        device=up_gm.modes.device,
    )
    lo_gm = GaussianMixture(
        modes=zero_mode,
        probs=torch.ones(1, dtype=up_gm.modes.dtype, device=up_gm.modes.device),
    )
    llr_chunks = compute_llr_sample_chunks(
        up_gm=up_gm,
        lo_gm=lo_gm,
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
    )
    return estimate_hockey_stick_delta_from_llr_chunks(
        epsilon=float(epsilon),
        llr_chunks=llr_chunks,
    )


def calibrate_sigma_evr_binary_search(
    *,
    llr_samples_fn,
    target_epsilon: float,
    target_delta: float,
    confidence_alpha: float = 1e-6,
    sigma_low: float = 1e-7,
    sigma_high: float = 100.0,
    tolerance: float = 1e-4,
    max_iterations: int = 60,
) -> tuple[float, DeltaVerificationResult]:
    """
    Calibrate the smallest sigma that passes EVR verification at target budget.

    The search is performed over sigma, but acceptance is based on the EVR
    upper confidence bound for ``delta(epsilon)``. This provides a formal
    high-confidence guard instead of raw point-estimate thresholding.
    """
    if target_epsilon < 0.0:
        raise ValueError("target_epsilon must be >= 0")

    if sigma_low <= 0.0 or sigma_high <= sigma_low:
        raise ValueError("require 0 < sigma_low < sigma_high")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    low = float(sigma_low)
    high = float(sigma_high)

    # `sigma` is Gaussian noise stddev candidate in EVR calibration.
    verification_at_high = verify_hockey_stick_delta_hoeffding(
        epsilon=target_epsilon,
        llr_samples=llr_samples_fn(high),
        target_delta=target_delta,
        confidence_alpha=confidence_alpha,
    )

    if not verification_at_high.accepted:
        raise ValueError(
            "sigma_high does not satisfy EVR acceptance; increase sigma_high"
        )

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        verification = verify_hockey_stick_delta_hoeffding(
            epsilon=target_epsilon,
            llr_samples=llr_samples_fn(mid),
            target_delta=target_delta,
            confidence_alpha=confidence_alpha,
        )

        if high - low <= tolerance:
            if verification.accepted:
                return mid, verification
            break

        if verification.accepted:
            high = mid
            verification_at_high = verification
        else:
            low = mid

    return high, verification_at_high


def find_sigma_binary_search(
    *,
    delta_fn,
    target_delta: float,
    sigma_low: float = 1e-7,
    sigma_high: float = 100.0,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> float:
    if target_delta <= 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in (0, 1)")

    if sigma_low <= 0.0 or sigma_high <= sigma_low:
        raise ValueError("require 0 < sigma_low < sigma_high")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    low = float(sigma_low)
    high = float(sigma_high)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        delta = float(delta_fn(mid))

        if abs(delta - target_delta) < tolerance:
            return mid

        if delta > target_delta:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def calibrate_b_min_sep_noise_multiplier_monte_carlo(
    *,
    c_matrix: torch.Tensor,
    bands: int | None = None,
    cycle_length: int | None = None,
    target_epsilon: float,
    target_delta: float,
    num_samples: int,
    seed: int = 0,
    reduce_dimensionality: bool = False,
    sigma_low: float = 1e-7,
    sigma_high: float = 100.0,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
    max_sigma: float = 1e6,
) -> float:
    """
    Calibrate BMinSep noise multiplier by searching ``delta(sigma)``.

    This routine expands the high bracket until the Monte Carlo delta estimate
    is below target, then runs binary search for the smallest feasible sigma.
    It is a practical calibration path used when an explicit EVR ladder is not
    required by the caller.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5, Equations (2)-(4), Theorem 5.1.
    """
    if target_epsilon < 0.0:
        raise ValueError("target_epsilon must be >= 0")

    if target_delta <= 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in (0, 1)")

    if max_sigma <= 0.0:
        raise ValueError("max_sigma must be > 0")

    def _delta_fn(sigma: float) -> float:
        return estimate_b_min_sep_delta_monte_carlo(
            c_matrix=c_matrix,
            bands=bands,
            cycle_length=cycle_length,
            noise_multiplier=float(sigma),
            epsilon=float(target_epsilon),
            num_samples=int(num_samples),
            seed=int(seed),
            reduce_dimensionality=reduce_dimensionality,
        )

    low = float(sigma_low)
    high = float(sigma_high)
    delta_at_high = float(_delta_fn(high))
    while delta_at_high > float(target_delta):
        high *= 2.0
        if high > float(max_sigma):
            raise ValueError("The privacy budget is too low.")

        delta_at_high = float(_delta_fn(high))

    return find_sigma_binary_search(
        delta_fn=_delta_fn,
        target_delta=float(target_delta),
        sigma_low=low,
        sigma_high=high,
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
