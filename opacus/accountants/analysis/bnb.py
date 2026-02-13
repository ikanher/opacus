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

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch

BNB_VERIFICATION_CONTRACT = "evr_union_bound_alpha_split_v1"


@dataclass(frozen=True)
class GaussianMixture:
    """
    Finite Gaussian mixture with shared isotropic noise scale.
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
class BNBCalibrationReport:
    """
    Stable, versioned calibration report payload for BNB runs.
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
    Structured runtime status for BNB calibration diagnostics.
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


def validate_bnb_c_matrix_contract(
    *,
    c_matrix: torch.Tensor,
    coeffs: Sequence[float],
    bands: int,
    c_matrix_contract: dict[str, Any],
) -> None:
    """
    Validate explicit C-matrix assumptions and optional numeric derivation checks.
    """
    if not isinstance(c_matrix_contract, dict):
        raise ValueError("bnb consistency check requires c_matrix_contract to be a dict")
    if c_matrix_contract.get("sampling_mode") != "balls_in_bins":
        raise ValueError(
            "bnb consistency check failed: c_matrix_contract['sampling_mode'] "
            "must be 'balls_in_bins'"
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
    if derivation != "lower_toeplitz_from_coeffs":
        raise ValueError(
            "bnb consistency check failed: unsupported c_matrix_contract['derivation']; "
            "supported values: {'lower_toeplitz_from_coeffs'}"
        )

    horizon = int(c_matrix_contract.get("horizon", c_matrix.shape[1]))
    if int(c_matrix.shape[0]) != horizon or int(c_matrix.shape[1]) != horizon:
        raise ValueError(
            "bnb consistency check failed: lower_toeplitz_from_coeffs requires "
            f"square c_matrix with shape [{horizon}, {horizon}]"
        )

    expected = torch.zeros(
        (horizon, horizon),
        dtype=torch.float64,
        device=c_matrix.device,
    )
    coeff_list = [float(c) for c in coeffs]
    for i in range(horizon):
        max_lag = min(i, len(coeff_list) - 1)
        for lag in range(max_lag + 1):
            expected[i, i - lag] = coeff_list[lag]

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
    Parse calibration payloads for the current schema version.
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
    Compact diagnostic summary for logging/debugging calibration outcomes.
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
    Conservative union-bound split for EVR repeated verification checks.
    """
    if total_confidence_alpha <= 0.0 or total_confidence_alpha >= 1.0:
        raise ValueError("total_confidence_alpha must be in (0, 1)")
    if num_checks <= 0:
        raise ValueError("num_checks must be > 0")
    return float(total_confidence_alpha) / float(num_checks)


def verify_evr_confidence_split(
    *,
    llr_samples_seq: Sequence[torch.Tensor],
    epsilon: float,
    target_delta: float,
    total_confidence_alpha: float,
) -> tuple[DeltaVerificationResult, int, float]:
    """
    Multi-check EVR verification using confidence splitting.
    Returns (worst_case_verification, pass_count, per_check_alpha).
    """
    if len(llr_samples_seq) == 0:
        raise ValueError("llr_samples_seq must be non-empty")

    per_check_alpha = split_confidence_alpha(
        total_confidence_alpha=total_confidence_alpha,
        num_checks=len(llr_samples_seq),
    )
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
    Evaluate an ordered sigma ladder and return the first candidate that passes.
    If none pass, return the last candidate and its verification diagnostics.
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
    Two-sided EVR candidate ladder:
    each sigma must pass both forward and reverse verification directions.
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


def build_balls_in_bins_gaussian_mixture(
    *,
    c_matrix: torch.Tensor,
    bands: int,
    reduce_dimensionality: bool = False,
) -> GaussianMixture:
    """
    Builds balls-in-bins Gaussian-mixture means from C.

    Mirrors the structure from the reference Monte Carlo notebook:
    if C is [d, m] and bands | m, produce m/bands equiprobable components by
    summing each bins-aligned stripe across epochs.
    """
    if c_matrix.ndim != 2:
        raise ValueError("c_matrix must have shape [d, m]")
    if bands <= 0:
        raise ValueError("bands must be > 0")

    d, m = c_matrix.shape
    if m % bands != 0:
        raise ValueError(
            "bands must evenly divide c_matrix.shape[1] "
            f"(got m={m}, bands={bands})"
        )

    num_components = m // bands
    modes = c_matrix.reshape(d, bands, num_components).sum(dim=1)  # [d, k]
    if reduce_dimensionality:
        # Keep the same behavior class as notebook-style dimensionality reduction.
        modes = torch.linalg.qr(modes, mode="r").R
    modes = modes.T.contiguous()  # [k, d']
    probs = torch.full((num_components,), 1.0 / float(num_components), dtype=modes.dtype)
    return GaussianMixture(modes=modes, probs=probs)


def _mixture_logpdf(points: torch.Tensor, gm: GaussianMixture, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if points.ndim != 2:
        raise ValueError("points must have shape [n, d]")
    if points.shape[1] != gm.modes.shape[1]:
        raise ValueError("points dimension must match mixture modes")

    sigma_sq = sigma * sigma
    centered = points[:, None, :] - gm.modes[None, :, :]  # [n, k, d]
    sq_dist = torch.sum(centered * centered, dim=2)  # [n, k]
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


def sample_balls_in_bins_llr(
    *,
    c_matrix: torch.Tensor,
    bands: int,
    sigma: float,
    num_samples: int,
    seed: int = 0,
    reduce_dimensionality: bool = False,
) -> torch.Tensor:
    """
    Samples LLR values for balls-in-bins Gaussian-mixture privacy analysis.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    up_gm = build_balls_in_bins_gaussian_mixture(
        c_matrix=c_matrix,
        bands=bands,
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
    generator = torch.Generator(device=up_gm.modes.device).manual_seed(int(seed))
    return compute_llr_samples(
        up_gm=up_gm,
        lo_gm=lo_gm,
        sigma=float(sigma),
        num_samples=int(num_samples),
        generator=generator,
    )


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


def verify_hockey_stick_delta_hoeffding(
    *,
    epsilon: float,
    llr_samples: torch.Tensor,
    target_delta: float,
    confidence_alpha: float = 1e-6,
) -> DeltaVerificationResult:
    """
    Verifies delta <= target_delta using Hoeffding upper confidence bound.
    """
    if target_delta <= 0.0 or target_delta >= 1.0:
        raise ValueError("target_delta must be in (0, 1)")
    if confidence_alpha <= 0.0 or confidence_alpha >= 1.0:
        raise ValueError("confidence_alpha must be in (0, 1)")
    if llr_samples.ndim != 1 or llr_samples.numel() == 0:
        raise ValueError("llr_samples must be a non-empty 1-D tensor")

    samples = llr_samples.to(dtype=torch.float64)
    vals = torch.clamp(1.0 - torch.exp(float(epsilon) - samples), min=0.0, max=1.0)
    n = float(vals.numel())
    delta_estimate = float(torch.mean(vals))
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
    Inverts hockey-stick delta from LLR samples via binary search.

    Uses monotonicity of delta(epsilon) for fixed LLR samples.
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
    if epsilon_high is None:
        high = max(1.0, low + 1.0)
        for _ in range(max_iterations):
            d_high = estimate_hockey_stick_delta_from_llr_samples(
                epsilon=high,
                llr_samples=samples,
            )
            if d_high <= target_delta:
                break
            high *= 2.0
        else:
            raise ValueError("could not bracket epsilon; increase max_iterations")
    else:
        high = float(epsilon_high)
        if high <= low:
            raise ValueError("epsilon_high must be > epsilon_low")
        d_high = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=high,
            llr_samples=samples,
        )
        if d_high > target_delta:
            raise ValueError("epsilon_high does not satisfy target_delta")

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


def estimate_balls_in_bins_epsilon_monte_carlo(
    *,
    c_matrix: torch.Tensor,
    bands: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    seed: int = 0,
    reduce_dimensionality: bool = False,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> float:
    """
    Estimates epsilon for a balls-in-bins mechanism from Monte Carlo PLD samples.
    """
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be > 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if seed < 0:
        raise ValueError("seed must be >= 0")

    llr_samples = sample_balls_in_bins_llr(
        c_matrix=c_matrix,
        bands=bands,
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        reduce_dimensionality=reduce_dimensionality,
    )
    return estimate_epsilon_from_llr_samples(
        target_delta=float(target_delta),
        llr_samples=llr_samples,
        tolerance=tolerance,
        max_iterations=max_iterations,
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
    EVR-style calibration: find smallest sigma whose verified delta is <= target.
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
