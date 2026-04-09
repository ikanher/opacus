from __future__ import annotations

import pytest

import opacus.mf.bsr_family as bsr_family_mod
from opacus.accountants.bnb_inputs import (
    attach_accountant_coeff_surface,
    resolve_canonical_bnb_cycle_length,
    resolve_canonical_bsr_bands,
)
from opacus.mf.bsr_family import (
    augment_bsr_family_cyclic_query_state,
    augment_bsr_family_fixed_batch_query_state,
    canonicalize_bsr_family_runtime_state,
    summarize_bsr_runtime_state,
)
from opacus.mf.blt_family import (
    canonicalize_blt_public_or_runtime_state,
    resolve_blt_fixed_batch_accountant_inputs,
    summarize_blt_report_surface,
    summarize_blt_runtime_state,
)
from opacus import SamplingSemantics


def test_canonicalize_blt_public_state_from_decay_pair() -> None:
    state = canonicalize_blt_public_or_runtime_state(
        {
            "theta": [0.8, 0.3],
            "theta_hat": [0.6, 0.1],
            "z_std": 0.03,
        }
    )

    assert {"forward", "inverse", "z_std"}.issubset(state.keys())
    assert state["_blt_distributed_policy"] == "single_process_only"
    assert state["_blt_distributed_runtime"] is False


def test_resolve_blt_fixed_batch_accountant_inputs_reads_runtime_surface() -> None:
    pair, noise_multiplier_ref, max_participations, min_separation, horizon = (
        resolve_blt_fixed_batch_accountant_inputs(
            runtime_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
                "noise_multiplier_ref": 1.27,
                "blt_max_participations": 2,
                "blt_min_separation": 4,
                "blt_horizon": 8,
            },
            metadata={},
            kwargs={},
            total_steps=8,
        )
    )

    assert pair.forward.theta == pytest.approx([0.8, 0.3])
    assert noise_multiplier_ref == pytest.approx(1.27)
    assert max_participations == 2
    assert min_separation == 4
    assert horizon == 8


def test_summarize_blt_runtime_and_report_surfaces_distinguish_quantities() -> None:
    runtime_state = {
        "theta": [0.8, 0.3],
        "theta_hat": [0.6, 0.1],
        "z_std": 0.314,
        "noise_multiplier_ref": 1.27,
        "blt_max_participations": 2,
        "blt_min_separation": 4,
        "blt_horizon": 8,
    }

    telemetry = summarize_blt_runtime_state(runtime_state)
    report = summarize_blt_report_surface(runtime_state)

    assert telemetry["has_noise_multiplier_ref"] is True
    assert report["computed_noise_multiplier"] == pytest.approx(0.314)
    assert report["accounting_noise_multiplier"] == pytest.approx(1.27)
    assert report["blt_max_participations"] == 2


def test_canonicalize_bsr_family_runtime_state_derives_bisr_runtime_coeffs() -> None:
    state = canonicalize_bsr_family_runtime_state(
        mechanism="bisr",
        runtime_state={"bisr_inv_coeffs": [1.0, 0.25], "bsr_bands": 2},
    )

    assert state["_noise_mechanism"] == "bisr"
    assert state["bisr_inv_coeffs"] == [1.0, 0.25]
    assert len(state["coeffs"]) == 2
    assert state["coeff_source"] == "analytical_inv_explicit"


def test_canonicalize_bsr_family_runtime_state_derives_bandinvmf_runtime_coeffs() -> None:
    state = canonicalize_bsr_family_runtime_state(
        mechanism="bandinvmf",
        runtime_state={"bandinvmf_inv_coeffs": [1.0, -0.5], "bsr_bands": 2},
    )

    assert state["_noise_mechanism"] == "bandinvmf"
    assert state["bandinvmf_inv_coeffs"] == [1.0, -0.5]
    assert len(state["coeffs"]) == 2


def test_bsr_family_query_augmentation_and_summary_distinguish_surfaces() -> None:
    semantics = SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={},
    )
    fixed_state = augment_bsr_family_fixed_batch_query_state(
        mechanism="bandmf",
        runtime_state={"coeffs": [1.0, -0.5], "bsr_bands": 2},
        sampling_semantics=semantics,
        steps=40,
        sample_rate=0.1,
        kwargs={"bsr_max_participations": 4, "bsr_min_separation": 2},
    )
    cyclic_state = augment_bsr_family_cyclic_query_state(
        mechanism="bsr",
        runtime_state={"coeffs": [1.0, -0.5], "bsr_bands": 2},
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
        steps=40,
        kwargs={},
    )

    fixed_summary = summarize_bsr_runtime_state(fixed_state)
    cyclic_summary = summarize_bsr_runtime_state(cyclic_state)

    assert fixed_state["bsr_mf_sensitivity"] > 0.0
    assert "bsr_sensitivity_scale" not in fixed_state
    assert cyclic_state["bsr_sensitivity_scale"] > 0.0
    assert "bsr_mf_sensitivity" not in cyclic_state
    assert fixed_summary["bsr_mf_sensitivity"] > 0.0
    assert cyclic_summary["bsr_sensitivity_scale"] > 0.0


def test_attach_accountant_coeff_surface_canonicalizes_coeff_payload() -> None:
    state = attach_accountant_coeff_surface(
        {"coeffs": [1.0], "coeff_source": "explicit_or_precomputed"},
        coeff_key="bnb_accountant_coeffs",
        coeff_source_key="bnb_accountant_coeffs_source",
        coeffs=(1, 2, 3),
        coeff_source="abs_factor_c_col",
    )

    assert state["bnb_accountant_coeffs"] == [1.0, 2.0, 3.0]
    assert state["bnb_accountant_coeffs_source"] == "abs_factor_c_col"


def test_resolve_canonical_bsr_bands_prefers_canonical_precedence() -> None:
    bands = resolve_canonical_bsr_bands(
        runtime_state={"bsr_bands": 7},
        metadata={"bands": 5},
        kwargs={"bsr_bands": 5},
        error_context="unused",
    )

    assert bands == 5


def test_resolve_canonical_bnb_cycle_length_accepts_legacy_bins_only_at_boundary() -> None:
    bins = resolve_canonical_bnb_cycle_length(
        runtime_state={"bnb_bins": 11},
        metadata={},
        kwargs={},
        error_context="unused",
    )

    assert bins == 11


def test_bsr_family_cyclic_resolution_uses_registry_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        bsr_family_mod._BSR_CYCLIC_ACCOUNTANT_INPUT_RESOLVERS,
        "bisr",
        lambda **kwargs: 7.5,
    )

    resolved = bsr_family_mod.resolve_bsr_family_cyclic_accountant_input(
        mechanism="bisr",
        runtime_state={"bisr_inv_coeffs": [1.0, 0.25], "bsr_bands": 2},
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
        steps=20,
        kwargs={},
    )

    assert resolved == pytest.approx(7.5)
