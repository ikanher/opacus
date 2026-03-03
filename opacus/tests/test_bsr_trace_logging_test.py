#!/usr/bin/env python3

from __future__ import annotations

import json
import logging

from opacus.mechanism_contracts import NoiseMechanismConfig, SamplingSemantics
from opacus.privacy_engine import PrivacyEngine


def test_log_bsr_trace_emits_json_payload_for_bsr(caplog) -> None:
    config = NoiseMechanismConfig(
        mechanism="bsr",
        accounting_mode="bsr_accountant",
        mechanism_state={
            "coeffs": [1.0, 0.5, 0.25],
            "z_std": 0.02,
            "sensitivity_scale": 1.1,
            "iterations_number": 2000,
        },
    )
    semantics = SamplingSemantics(
        sampling_mode="cyclic_poisson",
        privacy_metadata={"bands": 100, "sample_rate": 0.01},
    )

    with caplog.at_level(logging.INFO):
        PrivacyEngine._log_bsr_trace(
            stage="unit_test",
            mechanism_config=config,
            sampling_semantics=semantics,
            sample_rate=0.01,
            expected_batch_size=500,
            noise_multiplier=1.23,
            target_epsilon=8.0,
            target_delta=1e-5,
            total_steps=2000,
            epochs=None,
            loss_reduction="mean",
            correlated_denominator=500.0,
        )

    trace_records = [r.message for r in caplog.records if r.message.startswith("MF_TRACE ")]
    assert len(trace_records) == 1
    payload = json.loads(trace_records[0][len("MF_TRACE "):])
    assert payload["stage"] == "unit_test"
    assert payload["sampling_mode"] == "cyclic_poisson"
    assert payload["mechanism_state"]["coeff_count"] == 3
    assert payload["mechanism_state"]["coeff_head"] == [1.0, 0.5, 0.25]


def test_log_bsr_trace_is_noop_for_non_bsr(caplog) -> None:
    config = NoiseMechanismConfig(
        mechanism="gaussian",
        accounting_mode="standard_step_accountant",
        mechanism_state={},
    )

    with caplog.at_level(logging.INFO):
        PrivacyEngine._log_bsr_trace(
            stage="unit_test",
            mechanism_config=config,
            sampling_semantics=None,
            sample_rate=None,
            expected_batch_size=None,
            noise_multiplier=None,
            target_epsilon=None,
            target_delta=None,
            total_steps=None,
            epochs=None,
            loss_reduction=None,
            correlated_denominator=None,
        )

    assert not [r for r in caplog.records if r.message.startswith("MF_TRACE ")]
