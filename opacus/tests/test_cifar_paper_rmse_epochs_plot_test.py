from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB_PATH = _REPO_ROOT / "local-scripts" / "lib" / "cifar_mf_paper_rmse.py"
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_paper_rmse_vs_epochs.py"

_LIB_SPEC = importlib.util.spec_from_file_location("cifar_mf_paper_rmse", _LIB_PATH)
assert _LIB_SPEC is not None and _LIB_SPEC.loader is not None
_LIB = importlib.util.module_from_spec(_LIB_SPEC)
sys.modules[_LIB_SPEC.name] = _LIB
_LIB_SPEC.loader.exec_module(_LIB)

_PLOT_SPEC = importlib.util.spec_from_file_location("plot_cifar_paper_rmse_vs_epochs", _PLOT_PATH)
assert _PLOT_SPEC is not None and _PLOT_SPEC.loader is not None
_PLOT = importlib.util.module_from_spec(_PLOT_SPEC)
sys.modules[_PLOT_SPEC.name] = _PLOT
_PLOT_SPEC.loader.exec_module(_PLOT)


def test_library_returns_unsupported_for_blt() -> None:
    point = _LIB.evaluate_fixed_batch_paper_rmse_point(
        scenario=_LIB.CIFARFixedBatchScenario(),
        family="BLT",
        epochs=1,
        backend="prv",
    )
    assert point.status == "unsupported"
    assert point.reason == "unsupported_blt_rmse_surface"
    assert point.paper_rmse is None


def test_library_builds_fixed_batch_backend_privacy_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _LIB,
        "_fixed_batch_base_sigma",
        lambda *, target_epsilon, target_delta, backend: 2.0,
    )
    monkeypatch.setattr(_LIB, "eps_from_mu", lambda *, mu, delta: mu + delta)
    reference = _LIB.build_fixed_batch_backend_privacy_reference(
        scenario=_LIB.CIFARFixedBatchScenario(),
        backend="prv",
        delta_grid=[1e-3, 1e-5],
    )
    assert reference.contract == "fixed_batch_gaussian_mu_reference_v1"
    assert reference.context_kind == "shared_fixed_batch_backend_reference"
    assert reference.base_sigma == pytest.approx(2.0)
    assert reference.mu == pytest.approx(0.5)
    assert reference.epsilon_by_delta == {
        "1e-03": pytest.approx(0.501),
        "1e-05": pytest.approx(0.50001),
    }


def test_library_evaluates_normalized_paper_rmse_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _LIB,
        "_fixed_batch_base_sigma",
        lambda *, target_epsilon, target_delta, backend: 2.0,
    )
    monkeypatch.setattr(
        _LIB,
        "_strategy_matrix_and_source",
        lambda *, scenario, family, epochs, bandwidth, bifr_frac: (
            _LIB.torch.eye(1, dtype=_LIB.torch.float64),
            "identity_test_matrix",
        ),
    )
    monkeypatch.setattr(
        _LIB,
        "_fixed_batch_sensitivity_for_family",
        lambda *, scenario, family, epochs, bandwidth, bifr_frac: 3.0,
    )
    scenario = _LIB.CIFARFixedBatchScenario(dataset_size=1, batch_size=1)
    point = _LIB.evaluate_fixed_batch_paper_rmse_point(
        scenario=scenario,
        family="BSR",
        epochs=1,
        backend="prv",
        bandwidth=4,
    )
    assert point.status == "computed"
    assert point.noise_multiplier == pytest.approx(6.0)
    assert point.paper_rmse == pytest.approx(6.0)
    assert point.rmse_contract == "paper_normalized_prefix_workload_v1"


def test_build_epoch_rmse_report_records_bifr_policy_and_omissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_family_curve(*, scenario, family, backend, epoch_grid, bandwidth=None, bifr_frac=None):
        return [
            _LIB.FamilyEpochRMSEPoint(
                status="computed",
                family=str(family),
                backend=str(backend),
                epochs=int(epoch),
                total_steps=int(epoch * 10),
                bandwidth=int(4 if bandwidth is None else bandwidth),
                noise_multiplier=float(epoch),
                sensitivity=1.0,
                paper_rmse=float(epoch),
                matrix_source="mock_matrix",
                rmse_contract=_LIB.RMSE_CONTRACT_ID,
                reason=None,
                bifr_frac=None,
            )
            for epoch in epoch_grid
        ]

    def fake_bifr_point(*, scenario, family, epochs, backend, bandwidth=None, bifr_frac=None):
        assert family == "BIFR"
        frac = float(bifr_frac)
        return _LIB.FamilyEpochRMSEPoint(
            status="computed",
            family="BIFR",
            backend=str(backend),
            epochs=int(epochs),
            total_steps=int(epochs * 10),
            bandwidth=int(4 if bandwidth is None else bandwidth),
            noise_multiplier=float(epochs),
            sensitivity=1.0,
            paper_rmse=float(epochs) + abs(frac - 0.25),
            matrix_source="mock_bifr_matrix",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
            bifr_frac=frac,
        )

    monkeypatch.setattr(_PLOT, "evaluate_family_paper_rmse_over_epochs", fake_family_curve)
    monkeypatch.setattr(_PLOT, "evaluate_fixed_batch_paper_rmse_point", fake_bifr_point)
    monkeypatch.setattr(
        _PLOT,
        "build_fixed_batch_backend_privacy_reference",
        lambda *, scenario, backend: _LIB.FixedBatchBackendPrivacyReference(
            backend=str(backend),
            contract="fixed_batch_gaussian_mu_reference_v1",
            context_kind="shared_fixed_batch_backend_reference",
            base_sigma=2.0,
            mu=0.5,
            epsilon_by_delta={"1e-05": 9.0},
        ),
    )

    report = _PLOT.build_epoch_rmse_report(
        scenario=_LIB.CIFARFixedBatchScenario(),
        epoch_grid=[1, 2],
        backends=["prv"],
    )

    assert report["metadata"]["fixed_batch_only"] is True
    assert report["metadata"]["omitted_families"][0]["family"] == "BLT"
    assert report["metadata"]["bifr_policy"]["kind"] == "best_over_canonical_frac_sweep_v1"
    assert report["metadata"]["fixed_batch_privacy_reference_contract"]["kind"] == "fixed_batch_gaussian_mu_reference_v1"
    assert report["metadata"]["fixed_batch_privacy_reference"]["prv"]["mu"] == pytest.approx(0.5)
    bifr_series = next(
        entry
        for entry in report["series"]
        if entry["family"] == "BIFR" and entry["backend"] == "prv"
    )
    assert bifr_series["curve_policy"] == "best_over_canonical_frac_sweep_v1"
    assert [point["selected_bifr_frac"] for point in bifr_series["points"]] == [0.25, 0.25]


def test_render_epoch_rmse_plot_writes_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_family_curve(*, scenario, family, backend, epoch_grid, bandwidth=None, bifr_frac=None):
        return [
            _LIB.FamilyEpochRMSEPoint(
                status="computed",
                family=str(family),
                backend=str(backend),
                epochs=int(epoch),
                total_steps=int(epoch * 10),
                bandwidth=int(4 if bandwidth is None else bandwidth),
                noise_multiplier=float(epoch),
                sensitivity=1.0,
                paper_rmse=float(epoch),
                matrix_source="mock_matrix",
                rmse_contract=_LIB.RMSE_CONTRACT_ID,
                reason=None,
                bifr_frac=None,
            )
            for epoch in epoch_grid
        ]

    def fake_bifr_point(*, scenario, family, epochs, backend, bandwidth=None, bifr_frac=None):
        return _LIB.FamilyEpochRMSEPoint(
            status="computed",
            family="BIFR",
            backend=str(backend),
            epochs=int(epochs),
            total_steps=int(epochs * 10),
            bandwidth=int(4 if bandwidth is None else bandwidth),
            noise_multiplier=float(epochs),
            sensitivity=1.0,
            paper_rmse=float(epochs) + float(bifr_frac or 0.0),
            matrix_source="mock_bifr_matrix",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
            bifr_frac=float(bifr_frac or 0.0),
        )

    monkeypatch.setattr(_PLOT, "evaluate_family_paper_rmse_over_epochs", fake_family_curve)
    monkeypatch.setattr(_PLOT, "evaluate_fixed_batch_paper_rmse_point", fake_bifr_point)
    monkeypatch.setattr(
        _PLOT,
        "build_fixed_batch_backend_privacy_reference",
        lambda *, scenario, backend: _LIB.FixedBatchBackendPrivacyReference(
            backend=str(backend),
            contract="fixed_batch_gaussian_mu_reference_v1",
            context_kind="shared_fixed_batch_backend_reference",
            base_sigma=2.0,
            mu=0.5,
            epsilon_by_delta={"1e-05": 9.0},
        ),
    )

    report = _PLOT.build_epoch_rmse_report(
        scenario=_LIB.CIFARFixedBatchScenario(),
        epoch_grid=[1, 2],
        backends=["prv", "rdp"],
    )
    output_path = tmp_path / "paper_rmse_vs_epochs.png"
    _PLOT.render_epoch_rmse_plot(report, output_path=output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_main_prints_fixed_batch_privacy_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "paper_rmse_vs_epochs.json"
    output_png = tmp_path / "paper_rmse_vs_epochs.png"
    monkeypatch.setattr(
        _PLOT,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "epochs": None,
                "backends": ["prv"],
                "output_json": output_json,
                "output_png": output_png,
            },
        )(),
    )
    monkeypatch.setattr(
        _PLOT,
        "build_epoch_rmse_report",
        lambda *, scenario, epoch_grid, backends: {
            "metadata": {
                "backends": ["prv"],
                "fixed_batch_privacy_reference": {
                    "prv": {
                        "base_sigma": 2.0,
                        "mu": 0.5,
                        "epsilon_by_delta": {"1e-05": 9.0},
                    }
                },
            },
            "series": [],
        },
    )
    monkeypatch.setattr(
        _PLOT,
        "render_epoch_rmse_plot",
        lambda report, output_path: output_path.write_bytes(b"png"),
    )

    _PLOT.main()

    stdout = capsys.readouterr().out
    assert "PRV fixed-batch privacy reference:" in stdout
    assert "base_sigma=2.000000" in stdout
    assert "mu=0.500000" in stdout
    assert "1e-05->9.00" in stdout
    assert json.loads(output_json.read_text(encoding="utf-8"))["metadata"]["backends"] == ["prv"]
