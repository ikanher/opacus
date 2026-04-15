from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from opacus import NoiseMechanismConfig
from opacus.mechanism_contracts import SamplingSemantics
from opacus.mf.interfaces import SupportsBallsInBins

from opacus.mf import (
    BLTFamilyState,
    BSRFamilyState,
    get_mf_family_entry,
    mf_accounting_requires_context,
)


def test_blt_family_state_canonicalizes_decay_pair_and_preserves_metadata() -> None:
    state = BLTFamilyState.from_input_state(
        {
            "theta": [0.8, 0.3],
            "theta_hat": [0.6, 0.1],
            "z_std": 0.03,
            "blt_horizon": 8,
        }
    )

    canonical = state.to_state_dict()
    assert {"forward", "inverse", "z_std"}.issubset(canonical.keys())
    assert canonical["blt_horizon"] == 8
    assert canonical["_blt_distributed_policy"] == "ddp_flat_only"
    assert canonical["_blt_distributed_runtime"] is False


def test_blt_family_state_applies_torch_sampler_defaults() -> None:
    state = BLTFamilyState.from_input_state(
        {
            "theta": [0.8, 0.3],
            "theta_hat": [0.6, 0.1],
            "z_std": 0.03,
        }
    )

    changed = state.apply_torch_sampler_defaults(
        total_steps=8,
        dataset_size=32,
        logical_batch_size=8,
        max_grad_norm=1.0,
        calibration_denominator=32.0,
    )

    assert changed is True
    assert state.metadata["blt_horizon"] == 8
    assert state.metadata["blt_min_separation"] == 4
    assert state.metadata["blt_max_participations"] == 2
    assert math.isclose(state.metadata["noise_multiplier_ref"], 0.96)


def test_bisr_family_state_derives_runtime_coeffs_from_inverse() -> None:
    state = BSRFamilyState.from_state(
        mechanism="bisr",
        mechanism_state={"bisr_inv_coeffs": [1.0, 0.25]},
    )

    changed = state.ensure_bisr_runtime_coeffs_from_inverse()

    assert changed is True
    assert state.state["bisr_inv_coeffs"] == [1.0, 0.25]
    assert len(state.state["coeffs"]) == 2


def test_bsr_family_state_rejects_conflicting_band_inputs() -> None:
    state = BSRFamilyState.from_state(
        mechanism="bsr",
        mechanism_state={"bsr_bands": 2},
    )

    with pytest.raises(
        ValueError,
        match="conflicting canonical inputs: `bsr_bands` must match",
    ):
        state.resolve_bands(metadata={"bands": 2}, kwargs={"bsr_bands": 3})


def test_mf_registry_exposes_blt_and_banded_entries() -> None:
    assert get_mf_family_entry("blt") is not None
    assert get_mf_family_entry("bsr") is not None
    assert get_mf_family_entry("bandmf") is not None
    assert get_mf_family_entry("gaussian") is None


def test_mf_accounting_requires_context_matches_mf_accountants() -> None:
    assert mf_accounting_requires_context("blt") is True
    assert mf_accounting_requires_context("bsr") is True
    assert mf_accounting_requires_context("prv") is False


def test_bandmf_family_owns_sampling_validation() -> None:
    entry = get_mf_family_entry("bandmf")
    assert entry is not None

    with pytest.raises(
        ValueError,
        match="bandmf mechanism requires explicit sampling_semantics",
    ):
        entry.family.validate_sampling_compatibility(
            mechanism_config=NoiseMechanismConfig(
                mechanism="bandmf",
                accounting_mode="bandmf_accountant",
                mechanism_state={"coeffs": [1.0], "z_std": 0.1},
            ),
            poisson_sampling=False,
            sampling_semantics=None,
            validate_cyclic_poisson_mode=True,
        )


def test_blt_entry_requests_default_local_sampling_semantics() -> None:
    entry = get_mf_family_entry("blt")
    assert entry is not None
    assert entry.needs_default_local_sampling_semantics is True


def test_blt_family_exposes_balls_in_bins_accountant_state() -> None:
    entry = get_mf_family_entry("blt")
    assert entry is not None
    assert entry.supports_balls_in_bins is True
    assert isinstance(entry.family, SupportsBallsInBins)

    resolved = entry.family.resolve_balls_in_bins(
        mechanism_state={
            "theta": [0.8],
            "theta_hat": [0.6],
            "z_std": 0.03,
            "blt_min_separation": 4,
        },
        context={
            "metadata": {"bins": 8},
            "kwargs": {},
            "total_steps": 12,
        },
    )

    assert resolved["bnb_bands"] == 4
    assert resolved["bnb_cycle_length"] == 8
    assert resolved["bnb_horizon"] == 12
    assert resolved["bnb_accountant_coeffs_source"] == "normalized_forward_c_col"
    assert len(resolved["bnb_accountant_coeffs"]) == 12


def test_bifr_family_exposes_bnb_accountant_state() -> None:
    entry = get_mf_family_entry("bifr")
    assert entry is not None
    assert entry.supports_balls_in_bins is True
    assert isinstance(entry.family, SupportsBallsInBins)

    resolved = entry.family.resolve_balls_in_bins(
        mechanism_state={
            "coeffs": [1.0, 0.2],
            "bsr_bands": 2,
            "bifr_frac": 1.0,
        },
        context={
            "metadata": {"bins": 8, "bands": 2},
            "kwargs": {},
            "total_steps": 12,
        },
    )

    assert resolved["bnb_bands"] == 2
    assert resolved["bnb_cycle_length"] == 8
    assert resolved["bnb_horizon"] == 12
    assert resolved["bnb_accountant_coeffs_source"] == "abs_exact_factor_c_col"
    assert resolved["bifr_frac"] == pytest.approx(1.0)


def test_blt_family_augment_query_mechanism_config_applies_fixed_batch_defaults() -> None:
    entry = get_mf_family_entry("blt")
    assert entry is not None

    config = entry.family.augment_query_mechanism_config(
        mechanism_config=NoiseMechanismConfig(
            mechanism="blt",
            accounting_mode="blt_accountant",
            mechanism_state={
                "theta": [0.8, 0.3],
                "theta_hat": [0.6, 0.1],
                "z_std": 0.03,
            },
        ),
        local_sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
        total_steps=8,
        epochs=None,
        poisson_sampling=False,
        data_loader=None,
        kwargs={},
        resolve_total_steps_sample_rate=lambda **_: 0.25,
        query_runtime_context={
            "dataset_size": 32,
            "logical_batch_size": 8,
            "max_grad_norm": 1.0,
            "loss_reduction": "mean",
            "total_steps": 8,
        },
    )

    state = config.mechanism_state
    assert state["blt_horizon"] == 8
    assert state["blt_min_separation"] == 4
    assert state["blt_max_participations"] == 2
    assert math.isclose(state["noise_multiplier_ref"], 0.24)


def test_bsr_family_augment_query_mechanism_config_builds_balls_in_bins_state() -> None:
    entry = get_mf_family_entry("bsr")
    assert entry is not None
    optimizer = torch.optim.SGD(torch.nn.Linear(4, 3).parameters(), lr=0.05)

    config = entry.family.augment_query_mechanism_config(
        mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bnb_accountant",
            mechanism_state={},
        ),
        local_sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bands": 2, "bins": 8},
        ),
        total_steps=12,
        epochs=None,
        poisson_sampling=False,
        data_loader=None,
        kwargs={"total_steps": 12},
        resolve_total_steps_sample_rate=lambda **_: 0.25,
        optimizer=optimizer,
        query_runtime_context={"total_steps": 12},
    )

    state = config.mechanism_state
    assert state["bsr_bands"] == 2
    assert state["bnb_bands"] == 2
    assert state["bnb_cycle_length"] == 8
    assert state["bnb_horizon"] == 12
    assert len(state["bnb_accountant_coeffs"]) == 2
    assert tuple(state["bnb_c_matrix"].shape) == (12, 12)


def test_bifr_accountant_module_does_not_import_provider_canonicalizer() -> None:
    text = (Path(__file__).resolve().parents[1] / "accountants" / "bifr.py").read_text()
    assert "from opacus.mf.bifr_family import canonicalize_bifr_runtime_state" not in text
