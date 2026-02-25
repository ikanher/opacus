#!/usr/bin/env python3

from __future__ import annotations

import pytest

from opacus.mechanism_contracts import SamplingSemantics
from opacus.privacy_engine import PrivacyEngine


def test_fixed_batch_mf_sensitivity_precedence_kwargs_over_metadata_and_state() -> None:
    mechanism_state = {
        "mf_sensitivity": 7.0,
    }
    sampling_semantics = SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={"mf_sensitivity": 5.0},
    )

    got = PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        steps=200,
        kwargs={"bsr_mf_sensitivity": 3.0},
    )
    assert got == pytest.approx(3.0, rel=0.0, abs=1e-12)


def test_fixed_batch_mf_sensitivity_precedence_metadata_over_state() -> None:
    mechanism_state = {
        "mf_sensitivity": 7.0,
    }
    sampling_semantics = SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={"mf_sensitivity": 5.0},
    )

    got = PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        steps=200,
        kwargs={},
    )
    assert got == pytest.approx(5.0, rel=0.0, abs=1e-12)


def test_fixed_batch_mf_sensitivity_horizon_override_changes_result() -> None:
    mechanism_state = {
        "coeffs": [1.0, 1.0],
        "max_participations": 10,
        "min_separation": 1,
        "iterations_number": 2,
    }

    eps_short = PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=2,
        kwargs={},
    )
    eps_long = PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=2,
        kwargs={"bsr_iterations_number": 10},
    )
    assert eps_long > eps_short


def test_fixed_batch_mf_sensitivity_rejects_inconsistent_explicit_value() -> None:
    mechanism_state = {
        "coeffs": [1.0, 1.0],
        "max_participations": 10,
        "min_separation": 1,
        "iterations_number": 10,
    }

    with pytest.raises(ValueError, match="provided bsr_mf_sensitivity is inconsistent"):
        PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state=mechanism_state,
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_mf_sensitivity": 1.0},
        )


def test_fixed_batch_mf_sensitivity_requires_derivation_inputs_when_no_override() -> None:
    with pytest.raises(ValueError, match="fixed-batch bsr accounting requires MF sensitivity"):
        PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={"coeffs": [1.0, 0.5]},
            sampling_semantics=None,
            steps=100,
            kwargs={},
        )


def test_cyclic_scale_precedence_kwargs_over_metadata_and_state() -> None:
    mechanism_state = {
        "coeffs": [1.0, 0.2],
        "sensitivity_scale": 8.0,
    }
    sampling_semantics = SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"sensitivity_scale": 4.0},
    )

    got = PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        steps=100,
        kwargs={"bsr_sensitivity_scale": 2.0},
    )
    assert got == pytest.approx(2.0, rel=0.0, abs=1e-12)


def test_cyclic_scale_horizon_override_changes_result() -> None:
    mechanism_state = {"coeffs": [1.0, 2.0, 3.0]}
    short = PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=1,
        kwargs={},
    )
    long = PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=1,
        kwargs={"bsr_iterations_number": 3},
    )
    assert long > short


def test_cyclic_scale_rejects_non_positive_override() -> None:
    with pytest.raises(ValueError, match="bsr_sensitivity_scale must be finite and > 0"):
        PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_sensitivity_scale": 0.0},
        )


def test_fixed_batch_mf_sensitivity_rejects_non_finite_override() -> None:
    with pytest.raises(ValueError, match="bsr_mf_sensitivity must be finite and > 0"):
        PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={"coeffs": [1.0], "max_participations": 1, "min_separation": 1},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_mf_sensitivity": float("nan")},
        )


def test_fixed_batch_mf_sensitivity_rejects_invalid_horizon_override() -> None:
    with pytest.raises(ValueError, match="bsr_iterations_number must be >= 1"):
        PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={"coeffs": [1.0], "max_participations": 1, "min_separation": 1},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_iterations_number": 0},
        )


def test_cyclic_scale_rejects_non_finite_override() -> None:
    with pytest.raises(ValueError, match="bsr_sensitivity_scale must be finite and > 0"):
        PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_sensitivity_scale": float("nan")},
        )


def test_cyclic_scale_rejects_invalid_horizon_override() -> None:
    with pytest.raises(ValueError, match="bsr_iterations_number must be >= 1"):
        PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_iterations_number": 0},
        )


def test_cyclic_scale_rejects_fixed_batch_only_kwargs() -> None:
    with pytest.raises(
        ValueError,
        match="cyclic-poisson bsr accounting received fixed-batch-only parameters",
    ):
        PrivacyEngine._resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 2},
            ),
            steps=10,
            kwargs={"bsr_mf_sensitivity": 1.0},
        )


def test_fixed_batch_mf_sensitivity_rejects_cyclic_only_kwargs() -> None:
    with pytest.raises(
        ValueError,
        match="fixed-batch bsr accounting received cyclic-only parameters",
    ):
        PrivacyEngine._resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={"coeffs": [1.0], "max_participations": 1, "min_separation": 1},
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
            steps=10,
            kwargs={"bsr_sensitivity_scale": 1.0},
        )
