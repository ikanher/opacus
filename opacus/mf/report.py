from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from opacus import SamplingSemantics
from opacus.accountants.analysis.bisr import (
    compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs,
    derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs,
    generate_bisr_coeffs_from_sgd_workload,
)
from opacus.accountants.analysis.bsr import (
    compute_bsr_mf_sensitivity_from_coeffs,
    generate_bsr_coeffs_from_sgd_workload,
    resolve_bsr_cyclic_gaussian_contract,
)
from opacus.accountants.blt_inputs import summarize_blt_report_surface
from opacus.accountants.utils import get_noise_multiplier
from opacus.mf.registry import get_mf_family_entry
from opacus.mf.interfaces import SupportsCyclic
from opacus.accountants.blt_fixed_batch import optimize_blt_fixed_batch


BSRReportMethod = Literal["BSR", "BISR"]


@dataclass(frozen=True)
class BLTReportSurface:
    computed_noise_multiplier: float
    accounting_noise_multiplier: float
    blt_horizon: int
    blt_min_separation: int
    blt_max_participations: int
    selected_candidate_index: int
    candidate_count: int
    selected_theta: list[float]
    selected_theta_hat: list[float]


@dataclass(frozen=True)
class BSRFamilyReportInputs:
    method: BSRReportMethod
    mechanism: str
    accountant: str
    mechanism_state: dict[str, object]
    accountant_coeffs: list[float]
    accountant_source: str
    sensitivity: float


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def compute_blt_fixed_batch_report_surface(
    *,
    target_epsilon: float,
    target_delta: float,
    total_steps: int,
    dataset_size: int,
    logical_batch_size: int,
    max_grad_norm: float,
    buffers: int,
) -> BLTReportSurface:
    result = optimize_blt_fixed_batch(
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        total_steps=int(total_steps),
        dataset_size=int(dataset_size),
        logical_batch_size=int(logical_batch_size),
        max_grad_norm=float(max_grad_norm),
        buffers=int(buffers),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    report_surface = summarize_blt_report_surface(dict(result.mechanism_state))
    return BLTReportSurface(
        computed_noise_multiplier=float(report_surface["computed_noise_multiplier"]),
        accounting_noise_multiplier=float(report_surface["accounting_noise_multiplier"]),
        blt_horizon=int(report_surface["blt_horizon"]),
        blt_min_separation=int(report_surface["blt_min_separation"]),
        blt_max_participations=int(report_surface["blt_max_participations"]),
        selected_candidate_index=int(result.selected_candidate_index),
        candidate_count=int(result.candidate_count),
        selected_theta=[float(x) for x in result.selected_theta],
        selected_theta_hat=[float(x) for x in result.selected_theta_hat],
    )


def build_bsr_family_report_inputs(
    *,
    method: BSRReportMethod,
    bands: int,
    momentum: float,
    weight_decay: float,
    total_steps: int,
) -> BSRFamilyReportInputs:
    if method == "BSR":
        runtime_coeffs = generate_bsr_coeffs_from_sgd_workload(
            bands=int(bands),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        mechanism = "bsr"
        accountant = "bsr"
        mechanism_state = {
            "coeffs": [float(c) for c in runtime_coeffs],
            "bsr_bands": int(bands),
        }
        accountant_coeffs = [float(c) for c in runtime_coeffs]
        accountant_source = "raw"
    else:
        inv_coeffs = generate_bisr_coeffs_from_sgd_workload(
            bands=int(bands),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        mechanism = "bisr"
        accountant = "bsr"
        mechanism_state = {
            "bisr_inv_coeffs": [float(c) for c in inv_coeffs],
            "bsr_bands": int(bands),
        }
        accountant_coeffs = [
            float(c)
            for c in derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs(
                coeffs=inv_coeffs,
                steps=int(total_steps),
            )
        ]
        accountant_source = "abs_factor_c_col"

    entry = get_mf_family_entry(mechanism)
    if entry is None:
        raise ValueError(f"Missing MF family entry for mechanism={mechanism}")

    canonical_state = entry.family.canonicalize(mechanism_state)
    return BSRFamilyReportInputs(
        method=method,
        mechanism=mechanism,
        accountant=accountant,
        mechanism_state=canonical_state,
        accountant_coeffs=accountant_coeffs,
        accountant_source=accountant_source,
        sensitivity=_l2_norm(accountant_coeffs),
    )


def resolve_bsr_family_fixed_batch_report_sensitivity(
    *,
    method: BSRReportMethod,
    bands: int,
    momentum: float,
    weight_decay: float,
    total_steps: int,
    max_participations: int,
    min_separation: int,
) -> float:
    if method == "BSR":
        coeffs = generate_bsr_coeffs_from_sgd_workload(
            bands=int(bands),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        return float(
            compute_bsr_mf_sensitivity_from_coeffs(
                coeffs=coeffs,
                steps=int(total_steps),
                max_participations=int(max_participations),
                min_separation=int(min_separation),
            )
        )

    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=int(bands),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    return float(
        compute_bisr_fixed_batch_sensitivity_from_inverse_coeffs(
            coeffs=coeffs,
            steps=int(total_steps),
            max_participations=int(max_participations),
            min_separation=int(min_separation),
        )
    )


def compute_bsr_family_cyclic_report_baseline(
    *,
    method: BSRReportMethod,
    bands: int,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    momentum: float,
    weight_decay: float,
) -> tuple[float, dict[str, float | int]]:
    inputs = build_bsr_family_report_inputs(
        method=method,
        bands=int(bands),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        total_steps=int(steps),
    )
    entry = get_mf_family_entry(inputs.mechanism)
    if entry is None or not isinstance(entry.family, SupportsCyclic):
        raise ValueError(f"Mechanism={inputs.mechanism} does not expose cyclic report support")

    sampling_semantics = SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"bands": int(bands)},
    )
    sensitivity_scale = float(
        entry.family.resolve_cyclic(
            mechanism_state=inputs.mechanism_state,
            context={
                "sampling_semantics": sampling_semantics,
                "steps": int(steps),
                "kwargs": {},
            },
        )
    )
    sigma = float(
        get_noise_multiplier(
            target_epsilon=float(target_epsilon),
            target_delta=float(target_delta),
            sample_rate=float(sample_rate),
            steps=int(steps),
            accountant=inputs.accountant,
            mechanism_state=inputs.mechanism_state,
            sampling_semantics=sampling_semantics,
            bsr_sensitivity_scale=float(sensitivity_scale),
        )
    )
    reduced_contract = resolve_bsr_cyclic_gaussian_contract(
        noise_multiplier=float(sigma) / float(sensitivity_scale),
        steps=int(steps),
        sample_rate=float(sample_rate),
        bands=int(bands),
    )
    return sigma, {
        "cyclic_sensitivity_scale": float(sensitivity_scale),
        "effective_noise_multiplier": float(reduced_contract["effective_noise_multiplier"]),
        "q_eff": float(reduced_contract["sample_rate"]),
        "cycles": int(reduced_contract["steps"]),
    }
