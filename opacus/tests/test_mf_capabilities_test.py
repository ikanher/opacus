from __future__ import annotations

from opacus.mf import (
    BLTFamily,
    BSRFamily,
    build_bsr_family_report_inputs,
    resolve_bsr_family_fixed_batch_report_sensitivity,
    SupportsBallsInBins,
    SupportsCyclic,
    SupportsFixedBatch,
    SupportsOptimization,
    get_mf_family_entry,
    mf_accounting_requires_context,
)
from opacus import SamplingSemantics


def test_mf_registry_exposes_family_local_entries() -> None:
    blt = get_mf_family_entry("blt")
    bsr = get_mf_family_entry("bsr")

    assert blt is not None
    assert bsr is not None
    assert blt.family.name == "blt"
    assert bsr.family.name == "bsr"


def test_blt_family_supports_only_relevant_capabilities() -> None:
    family = BLTFamily()

    assert isinstance(family, SupportsFixedBatch)
    assert isinstance(family, SupportsOptimization)
    assert not isinstance(family, SupportsCyclic)
    assert not isinstance(family, SupportsBallsInBins)


def test_bsr_family_supports_banded_accounting_capabilities() -> None:
    family = BSRFamily("bisr")

    assert isinstance(family, SupportsFixedBatch)
    assert isinstance(family, SupportsCyclic)
    assert isinstance(family, SupportsBallsInBins)
    assert not isinstance(family, SupportsOptimization)


def test_blt_family_build_runtime_and_fixed_batch_resolution_are_local() -> None:
    family = BLTFamily()
    runtime_state = family.build_runtime(
        mechanism_state={
            "theta": [0.8, 0.3],
            "theta_hat": [0.6, 0.1],
            "z_std": 0.03,
        },
        context={
            "metadata": {"blt_horizon": 8},
            "kwargs": {"blt_min_separation": 4, "blt_max_participations": 2, "noise_multiplier_ref": 1.27},
        },
    )

    pair, noise_multiplier_ref, max_participations, min_separation, horizon = family.resolve_fixed_batch(
        mechanism_state=runtime_state,
        context={"metadata": {}, "kwargs": {}, "total_steps": 8},
    )

    assert list(pair.forward.theta) == [0.8, 0.3]
    assert noise_multiplier_ref == 1.27
    assert max_participations == 2
    assert min_separation == 4
    assert horizon == 8


def test_bsr_family_cyclic_and_fixed_batch_capabilities_use_family_context() -> None:
    family = BSRFamily("bsr")
    canonical = family.canonicalize({"coeffs": [1.0, 0.5], "bsr_bands": 2})

    cyclic = family.resolve_cyclic(
        mechanism_state=canonical,
        context={
            "sampling_semantics": SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 2},
            ),
            "steps": 40,
            "kwargs": {},
        },
    )
    fixed = family.resolve_fixed_batch(
        mechanism_state=canonical,
        context={
            "sampling_semantics": SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
            "steps": 40,
            "sample_rate": 0.1,
            "kwargs": {"bsr_max_participations": 4, "bsr_min_separation": 2},
        },
    )

    assert cyclic > 0.0
    assert fixed > 0.0


def test_mf_accounting_requires_context_uses_new_registry() -> None:
    assert mf_accounting_requires_context("blt") is True
    assert mf_accounting_requires_context("bsr") is True
    assert mf_accounting_requires_context("prv") is False


def test_bisr_report_inputs_are_family_facing_and_canonical() -> None:
    inputs = build_bsr_family_report_inputs(
        method="BISR",
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        total_steps=100,
    )

    assert inputs.method == "BISR"
    assert inputs.mechanism == "bisr"
    assert inputs.accountant == "bsr"
    assert inputs.accountant_source == "abs_factor_c_col"
    assert "bisr_inv_coeffs" in inputs.mechanism_state
    assert "coeffs" in inputs.mechanism_state
    assert inputs.sensitivity > 0.0


def test_bsr_fixed_batch_report_sensitivity_is_positive() -> None:
    got = resolve_bsr_family_fixed_batch_report_sensitivity(
        method="BSR",
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        total_steps=50,
        max_participations=4,
        min_separation=2,
    )

    assert got > 0.0
