#!/usr/bin/env python3

from __future__ import annotations

import pytest

from opacus.accountants.bsr import (
    resolve_bsr_mf_sensitivity_for_fixed_batch,
    resolve_bsr_sensitivity_scale_for_cyclic,
)
from opacus.mechanism_contracts import SamplingSemantics


def test_fixed_batch_mf_sensitivity_precedence_kwargs_over_metadata_and_state() -> None:
    mechanism_state = {
        "bsr_mf_sensitivity": 7.0,
    }
    sampling_semantics = SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={"bsr_mf_sensitivity": 5.0},
    )

    got = resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        steps=200,
        sample_rate=0.01,
        kwargs={"bsr_mf_sensitivity": 3.0},
    )
    assert got == pytest.approx(3.0, rel=0.0, abs=1e-12)


def test_fixed_batch_mf_sensitivity_precedence_metadata_over_state() -> None:
    mechanism_state = {
        "bsr_mf_sensitivity": 7.0,
    }
    sampling_semantics = SamplingSemantics(
        sampling_mode="torch_sampler",
        privacy_metadata={"bsr_mf_sensitivity": 5.0},
    )

    got = resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        steps=200,
        sample_rate=0.01,
        kwargs={},
    )
    assert got == pytest.approx(5.0, rel=0.0, abs=1e-12)


def test_fixed_batch_mf_sensitivity_horizon_override_changes_result() -> None:
    mechanism_state = {
        "coeffs": [1.0, 1.0],
        "bsr_max_participations": 10,
        "bsr_min_separation": 1,
        "bsr_iterations_number": 2,
    }

    eps_short = resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=2,
        sample_rate=0.5,
        kwargs={},
    )
    eps_long = resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=2,
        sample_rate=0.5,
        kwargs={"bsr_iterations_number": 10},
    )
    assert eps_long > eps_short


def test_fixed_batch_mf_sensitivity_rejects_inconsistent_explicit_value() -> None:
    mechanism_state = {
        "coeffs": [1.0, 1.0],
        "bsr_max_participations": 10,
        "bsr_min_separation": 1,
        "bsr_iterations_number": 10,
    }

    with pytest.raises(ValueError, match="provided bsr_mf_sensitivity is inconsistent"):
        resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state=mechanism_state,
            sampling_semantics=None,
            steps=10,
            sample_rate=0.5,
            kwargs={"bsr_mf_sensitivity": 1.0},
        )


def test_fixed_batch_mf_sensitivity_derives_from_sample_rate_when_metadata_missing() -> None:
    got = resolve_bsr_mf_sensitivity_for_fixed_batch(
        mechanism_state={"coeffs": [1.0, 0.5]},
        sampling_semantics=None,
        steps=100,
        sample_rate=0.01,
        kwargs={},
    )
    assert got > 0.0


def test_cyclic_scale_precedence_kwargs_over_metadata_and_state() -> None:
    mechanism_state = {
        "coeffs": [1.0, 0.2],
        "bsr_sensitivity_scale": 8.0,
    }
    sampling_semantics = SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"bsr_sensitivity_scale": 4.0},
    )

    got = resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        steps=100,
        kwargs={"bsr_sensitivity_scale": 2.0},
    )
    assert got == pytest.approx(2.0, rel=0.0, abs=1e-12)


def test_cyclic_scale_horizon_override_is_stable_when_both_horizons_are_valid() -> None:
    mechanism_state = {"coeffs": [1.0, 2.0, 3.0]}
    short = resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=3,
        kwargs={},
    )
    long = resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state=mechanism_state,
        sampling_semantics=None,
        steps=3,
        kwargs={"bsr_iterations_number": 6},
    )
    assert long == pytest.approx(short, rel=0.0, abs=1e-12)


def test_cyclic_scale_rejects_non_positive_override() -> None:
    with pytest.raises(ValueError, match="bsr_sensitivity_scale must be finite and > 0"):
        resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_sensitivity_scale": 0.0},
        )


def test_fixed_batch_mf_sensitivity_rejects_non_finite_override() -> None:
    with pytest.raises(ValueError, match="bsr_mf_sensitivity must be finite and > 0"):
        resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={
                "coeffs": [1.0],
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
            sampling_semantics=None,
            steps=10,
            sample_rate=0.1,
            kwargs={"bsr_mf_sensitivity": float("nan")},
        )


def test_fixed_batch_mf_sensitivity_rejects_invalid_horizon_override() -> None:
    with pytest.raises(ValueError, match="bsr_iterations_number must be >= 1"):
        resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={
                "coeffs": [1.0],
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
            sampling_semantics=None,
            steps=10,
            sample_rate=0.1,
            kwargs={"bsr_iterations_number": 0},
        )


def test_cyclic_scale_rejects_non_finite_override() -> None:
    with pytest.raises(ValueError, match="bsr_sensitivity_scale must be finite and > 0"):
        resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_sensitivity_scale": float("nan")},
        )


def test_cyclic_scale_rejects_invalid_horizon_override() -> None:
    with pytest.raises(ValueError, match="bsr_iterations_number must be >= 1"):
        resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2]},
            sampling_semantics=None,
            steps=10,
            kwargs={"bsr_iterations_number": 0},
        )


def test_cyclic_scale_rejects_steps_below_bands() -> None:
    with pytest.raises(ValueError, match="steps >= bands"):
        resolve_bsr_sensitivity_scale_for_cyclic(
            mechanism_state={"coeffs": [1.0, 0.2, 0.1]},
            sampling_semantics=SamplingSemantics(
                sampling_mode="cyclic_poisson",
                privacy_metadata={"bands": 3},
            ),
            steps=2,
            kwargs={},
        )


def test_cyclic_scale_rejects_fixed_batch_only_kwargs() -> None:
    with pytest.raises(
        ValueError,
        match="cyclic-poisson bandmf accounting received fixed-batch-only parameters",
    ):
        resolve_bsr_sensitivity_scale_for_cyclic(
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
        resolve_bsr_mf_sensitivity_for_fixed_batch(
            mechanism_state={
                "coeffs": [1.0],
                "bsr_max_participations": 1,
                "bsr_min_separation": 1,
            },
            sampling_semantics=SamplingSemantics(
                sampling_mode="torch_sampler",
                privacy_metadata={},
            ),
            steps=10,
            sample_rate=0.1,
            kwargs={"bsr_sensitivity_scale": 1.0},
        )


def test_cyclic_scale_ignores_legacy_sensitivity_scale_alias() -> None:
    baseline = resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state={"coeffs": [1.0, 0.2]},
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
        steps=10,
        kwargs={},
    )
    resolved = resolve_bsr_sensitivity_scale_for_cyclic(
        mechanism_state={"coeffs": [1.0, 0.2]},
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": 2},
        ),
        steps=10,
        kwargs={"sensitivity_scale": 1.0},
    )
    assert resolved == pytest.approx(baseline, rel=0.0, abs=1e-12)
