"""Payload bytes are storage ownership, not tensor values or allocator peaks."""

import pytest
import torch
from opacus.accountants.analysis.blt import BLTPairedParams, BLTParams
from opacus.noise_mechanisms import (
    BufferedToeplitzNoiseMechanism,
    GaussianNoiseMechanism,
)
from opacus.tests.test_inverse_noise_regeneration_test import (
    context,
    inventory,
    mechanism,
    release,
)


def test_gaussian_has_no_historical_noise_payload():
    assert inventory(GaussianNoiseMechanism()) == {
        "history": {},
        "diagnostics": {},
        "scratch": {},
    }


def test_inventory_counts_actual_cpu_rng_snapshot_payload_without_cloning(monkeypatch):
    ctx, noiser = context(shapes=((8192,),)), mechanism(4)
    for _ in range(6):
        release(noiser, ctx)

    def forbidden(*args, **kwargs):
        pytest.fail("inventory must not clone state or synchronize the device")

    monkeypatch.setattr(noiser, "state_dict", forbidden)
    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    payload = inventory(noiser)
    assert payload["history"] == {"cpu": 3 * ctx.generator.get_state().numel()}
    assert payload["diagnostics"] == {}


def test_blt_inventory_counts_shared_backing_storage_once():
    noiser = BufferedToeplitzNoiseMechanism(
        pair=BLTPairedParams(
            forward=BLTParams(theta=[0.8, 0.3], omega=[0.1, 0.2]),
            inverse=BLTParams(theta=[0.6], omega=[0.4]),
        ),
        z_std=0.2,
    )
    backing = torch.zeros(64)
    # Two buffer views keep the whole allocation alive, even when disjoint.
    noiser._buffers = [backing[:4], backing[8:12]]
    payload = inventory(noiser)
    assert payload["history"] == {"cpu": backing.untyped_storage().nbytes()}


def test_inventory_separates_diagnostic_payload_from_history():
    noiser = mechanism(1)
    noiser.last_flat_z = torch.ones(8)
    noiser.last_flat_u = noiser.last_flat_z[:2]
    payload = inventory(noiser)
    assert sum(payload["history"].values()) == 0
    assert payload["diagnostics"] == {"cpu": 32}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real accelerator")
def test_inventory_does_not_label_cpu_snapshots_as_gpu_storage():
    noiser, ctx = mechanism(), context("cuda")
    for _ in range(5):
        release(noiser, ctx)
    value = inventory(noiser)
    assert value["history"] == {"cpu": 3 * ctx.generator.get_state().numel()}


@pytest.mark.parametrize("diagnostics", [None, False, True])
def test_blt_diagnostics_are_opt_in_and_do_not_change_noise(diagnostics):
    from opacus.accountants.analysis.blt import blt_pair_from_theta_pair

    pair = blt_pair_from_theta_pair(theta=[0.8, 0.3], theta_hat=[0.6, 0.1])
    options = {} if diagnostics is None else {"record_noise_diagnostics": diagnostics}
    noiser = BufferedToeplitzNoiseMechanism(pair=pair, z_std=0.2, **options)
    reference = BufferedToeplitzNoiseMechanism(
        pair=pair, z_std=0.2, record_noise_diagnostics=False
    )
    ctx, ref_ctx = context(), context()
    for _ in range(4):
        actual = release(noiser, ctx)
        torch.testing.assert_close(actual, release(reference, ref_ctx), rtol=0, atol=0)
        torch.testing.assert_close(
            ctx.generator.get_state(), ref_ctx.generator.get_state(), rtol=0, atol=0
        )
        for buf, ref_buf in zip(noiser._buffers, reference._buffers):
            torch.testing.assert_close(buf, ref_buf, rtol=0, atol=0)
    assert noiser.record_noise_diagnostics is (diagnostics is True)
    if diagnostics:
        torch.testing.assert_close(noiser.last_flat_u, actual, rtol=0, atol=0)
        assert noiser.last_flat_z is not None
        assert inventory(noiser)["diagnostics"] == {
            "cpu": 2 * actual.numel() * actual.element_size()
        }
    else:
        assert noiser.last_flat_z is None and noiser.last_flat_u is None
        assert inventory(noiser)["diagnostics"] == {}


@pytest.mark.parametrize("diagnostics", [None, False, True])
def test_blt_builder_routes_optional_diagnostics(diagnostics):
    from opacus import NoiseMechanismConfig, PrivacyEngine

    state = {
        "forward": {"theta": [0.8], "omega": [0.2]},
        "inverse": {"theta": [0.6], "omega": [-0.2]},
        "z_std": 0.2,
    }
    if diagnostics is not None:
        state["record_noise_diagnostics"] = diagnostics
    noiser = PrivacyEngine._build_noise_mechanism_from_config(
        NoiseMechanismConfig(
            mechanism="blt", accounting_mode="blt_accountant", mechanism_state=state
        )
    )
    assert noiser.record_noise_diagnostics is (diagnostics is True)


def test_storage_inventory_deduplicates_across_categories_without_copies(monkeypatch):
    from opacus.noise_mechanisms import _utils

    read = getattr(_utils, "storage_inventory", None)
    assert callable(read), "shared tensor-storage byte counter must be available"
    backing, diagnostic, scratch = torch.zeros(16), torch.zeros(3), torch.zeros(2)

    def forbidden(*args, **kwargs):
        pytest.fail("inventory must read metadata without copies or synchronization")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    result = read(
        history=[backing[:1]],
        diagnostics=[backing[2:], diagnostic],
        scratch=[diagnostic[:1], scratch],
    )
    assert result == {
        "history": {"cpu": 64},
        "diagnostics": {"cpu": 12},
        "scratch": {"cpu": 8},
    }
