from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB_PATH = _REPO_ROOT / "local-scripts" / "lib" / "cifar_mf_paper_rmse.py"
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_paper_rmse_vs_epochs.py"
_PLOTTING_LIB_PATH = _REPO_ROOT / "local-scripts" / "lib" / "mf_plotting.py"
_RENDER_PATH = _REPO_ROOT / "local-scripts" / "render_mf_plot_bundle.py"

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

_PLOTTING_SPEC = importlib.util.spec_from_file_location("mf_plotting", _PLOTTING_LIB_PATH)
assert _PLOTTING_SPEC is not None and _PLOTTING_SPEC.loader is not None
_PLOTTING = importlib.util.module_from_spec(_PLOTTING_SPEC)
sys.modules[_PLOTTING_SPEC.name] = _PLOTTING
_PLOTTING_SPEC.loader.exec_module(_PLOTTING)

_RENDER_SPEC = importlib.util.spec_from_file_location("render_mf_plot_bundle", _RENDER_PATH)
assert _RENDER_SPEC is not None and _RENDER_SPEC.loader is not None
_RENDER = importlib.util.module_from_spec(_RENDER_SPEC)
sys.modules[_RENDER_SPEC.name] = _RENDER
_RENDER_SPEC.loader.exec_module(_RENDER)


def test_library_supports_blt_fixed_batch_baseline(
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
            "mock_blt_strategy",
        ),
    )
    monkeypatch.setattr(
        _LIB,
        "compute_blt_fixed_batch_report_surface",
        lambda **kwargs: type(
            "MockBLTReportSurface",
            (),
            {
                "computed_noise_multiplier": 1.5,
                "accounting_noise_multiplier": 6.0,
            },
        )(),
    )
    point = _LIB.evaluate_fixed_batch_paper_rmse_point(
        scenario=_LIB.CIFARFixedBatchScenario(),
        family="BLT",
        epochs=1,
        backend="prv",
    )
    assert point.status == "computed"
    assert point.reason is None
    assert point.noise_multiplier == pytest.approx(1.5)
    assert point.sensitivity is None
    assert point.paper_rmse == pytest.approx(1.5)


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
    assert point.rmse_contract == "normalized_prefix_workload_v1"


def test_library_uses_direct_inverse_family_rmse_path_for_bisr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _LIB,
        "_fixed_batch_base_sigma",
        lambda *, target_epsilon, target_delta, backend: 2.0,
    )
    monkeypatch.setattr(
        _LIB,
        "_fixed_batch_sensitivity_for_family",
        lambda *, scenario, family, epochs, bandwidth, bifr_frac: 3.0,
    )
    monkeypatch.setattr(
        _LIB,
        "_inverse_family_rmse_and_source",
        lambda *, scenario, family, epochs, bandwidth, bifr_frac, noise_multiplier: (
            7.0,
            "direct_inverse_family_rmse_from_bisr_inv_coeffs",
        ),
    )

    def _unexpected_strategy_matrix(**kwargs):
        raise AssertionError("inverse-family RMSE should not reconstruct a strategy matrix")

    monkeypatch.setattr(_LIB, "_strategy_matrix_and_source", _unexpected_strategy_matrix)
    scenario = _LIB.CIFARFixedBatchScenario(dataset_size=1, batch_size=1)
    point = _LIB.evaluate_fixed_batch_paper_rmse_point(
        scenario=scenario,
        family="BISR",
        epochs=1,
        backend="prv",
        bandwidth=4,
    )
    assert point.status == "computed"
    assert point.noise_multiplier == pytest.approx(6.0)
    assert point.paper_rmse == pytest.approx(7.0)
    assert point.matrix_source == "direct_inverse_family_rmse_from_bisr_inv_coeffs"


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
    )
    output_path = tmp_path / "paper_rmse_vs_epochs.png"
    _PLOT.render_epoch_rmse_plot(report, output_path=output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_epoch_rmse_figure_data_records_panels_and_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _PLOT,
        "evaluate_family_paper_rmse_over_epochs",
        lambda *, scenario, family, backend, epoch_grid, bandwidth=None, bifr_frac=None: [
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
        ],
    )
    monkeypatch.setattr(
        _PLOT,
        "evaluate_fixed_batch_paper_rmse_point",
        lambda *, scenario, family, epochs, backend, bandwidth=None, bifr_frac=None: _LIB.FamilyEpochRMSEPoint(
            status="computed",
            family="BIFR",
            backend=str(backend),
            epochs=int(epochs),
            total_steps=int(epochs * 10),
            bandwidth=int(4 if bandwidth is None else bandwidth),
            noise_multiplier=float(epochs),
            sensitivity=1.0,
            paper_rmse=float(epochs),
            matrix_source="mock_bifr_matrix",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
            bifr_frac=float(bifr_frac or 0.25),
        ),
    )
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
    )
    figure_data = _PLOT.build_epoch_rmse_figure_data(report)
    assert figure_data["plot_bundle_contract"] == "local_mf_plot_render_data_v1"
    assert figure_data["figure_contract"] == "cifar_fixed_batch_epoch_rmse_plot_v1"
    assert len(figure_data["panels"]) == 1
    first_series = figure_data["panels"][0]["series"][0]
    assert first_series["role"] == "curve"
    assert first_series["data"][0] == {"x": 1, "y": pytest.approx(1.0)}


def test_main_prints_fixed_batch_privacy_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "paper_rmse_vs_epochs.json"
    output_png = tmp_path / "paper_rmse_vs_epochs.png"
    output_svg = tmp_path / "paper_rmse_vs_epochs.svg"
    output_figure_data = tmp_path / "paper_rmse_vs_epochs_figure_data.json"
    output_bundle_manifest = tmp_path / "plot_bundle_manifest.json"
    monkeypatch.setattr(
        _PLOT,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "epochs": None,
                "output_json": output_json,
                "output_png": output_png,
                "output_svg": output_svg,
                "output_figure_data": output_figure_data,
                "output_bundle_manifest": output_bundle_manifest,
            },
        )(),
    )
    monkeypatch.setattr(
        _PLOT,
        "build_epoch_rmse_report",
        lambda *, scenario, epoch_grid: {
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
        "build_epoch_rmse_figure_data",
        lambda report: {
            "plot_bundle_contract": "local_mf_plot_render_data_v1",
            "figure_contract": "cifar_fixed_batch_epoch_rmse_plot_v1",
            "figure_slug": "paper_rmse_vs_epochs",
            "title": "title",
            "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
            "scenario_metadata": report["metadata"],
            "panels": [{"panel_id": "backend:prv", "title": "PRV", "x_label": "Epochs", "y_label": "Paper RMSE", "series": []}],
        },
    )
    monkeypatch.setattr(
        _PLOT,
        "write_plot_bundle",
        lambda **kwargs: (
            kwargs["output_dir"].mkdir(parents=True, exist_ok=True),
            kwargs["output_dir"].joinpath(kwargs["figure_data_filename"]).write_text("{}", encoding="utf-8"),
            kwargs["output_dir"].joinpath(kwargs["manifest_filename"]).write_text("{}", encoding="utf-8"),
            output_png.write_bytes(b"png"),
            output_svg.write_bytes(b"svg"),
        ),
    )

    _PLOT.main()

    stdout = capsys.readouterr().out
    assert "PRV fixed-batch privacy reference:" in stdout
    assert "Wrote plot bundle manifest:" in stdout
    assert "Wrote SVG plot:" in stdout
    assert "Backend: prv" in stdout
    assert "base_sigma=2.000000" in stdout
    assert "mu=0.500000" in stdout
    assert "1e-05->9.00" in stdout
    assert json.loads(output_json.read_text(encoding="utf-8"))["metadata"]["backends"] == ["prv"]


def test_write_plot_bundle_writes_manifest_and_outputs(tmp_path: Path) -> None:
    figure_data = {
        "plot_bundle_contract": "local_mf_plot_render_data_v1",
        "figure_contract": "cifar_fixed_batch_epoch_rmse_plot_v1",
        "figure_slug": "paper_rmse_vs_epochs",
        "title": "CIFAR-10 Fixed-Batch Paper RMSE vs Epochs",
        "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
        "scenario_metadata": {"dataset": "CIFAR-10"},
        "panels": [{"panel_id": "backend:prv", "title": "PRV", "x_label": "Epochs", "y_label": "Paper RMSE", "series": []}],
    }

    manifest = _PLOTTING.write_plot_bundle(
        output_dir=tmp_path,
        figure_data_filename="figure_data.json",
        manifest_filename="plot_bundle_manifest.json",
        figure_data=figure_data,
        source_artifacts=[
            _PLOTTING.PlotSourceArtifact(
                path="report.json",
                kind="epoch_sweep_report",
                contract="normalized_prefix_workload_v1",
            )
        ],
        generator_entrypoint="plot.py",
        renderer_entrypoint="render.py",
        render_outputs=[
            ("rendered_figure", tmp_path / "plot.png"),
            ("publication_figure", tmp_path / "plot.svg"),
        ],
        render_fn=lambda payload, path: path.write_bytes(b"data"),
    )
    assert manifest.contract == "local_mf_plot_bundle_v1"
    manifest_payload = json.loads((tmp_path / "plot_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["figure_contract"] == "cifar_fixed_batch_epoch_rmse_plot_v1"
    assert (tmp_path / "figure_data.json").exists()
    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.svg").exists()


def test_render_only_path_renders_from_precomputed_figure_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "figure_data.json"
    input_path.write_text(
        json.dumps(
            {
                "plot_bundle_contract": "local_mf_plot_render_data_v1",
                "figure_contract": "cifar_fixed_batch_epoch_rmse_plot_v1",
                "figure_slug": "paper_rmse_vs_epochs",
                "title": "title",
                "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
                "scenario_metadata": {"dataset": "CIFAR-10"},
                "panels": [{"panel_id": "backend:prv", "title": "PRV", "x_label": "Epochs", "y_label": "Paper RMSE", "series": []}],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle"
    monkeypatch.setattr(
        _RENDER,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "input": input_path,
                "output_dir": output_dir,
                "output_png": None,
                "output_svg": None,
                "output_manifest": None,
            },
        )(),
    )
    monkeypatch.setattr(
        _RENDER,
        "render_plot_from_figure_data",
        lambda payload, output_path: output_path.write_bytes(b"plot"),
    )
    _RENDER.main()
    manifest = json.loads((output_dir / "plot_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_artifacts"][0]["path"] == str(input_path)
    assert manifest["source_artifacts"][0]["contract"] == "local_mf_plot_render_data_v1"
    assert (output_dir / "paper_rmse_vs_epochs.png").exists()
    assert (output_dir / "paper_rmse_vs_epochs.svg").exists()


def test_render_only_path_rejects_wrong_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "figure_data.json"
    input_path.write_text(
        json.dumps(
            {
                "plot_bundle_contract": "wrong_contract",
                "figure_contract": "cifar_fixed_batch_epoch_rmse_plot_v1",
                "panels": [{"panel_id": "x", "series": []}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _RENDER,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "input": input_path,
                "output_dir": tmp_path / "bundle",
                "output_png": None,
                "output_svg": None,
                "output_manifest": None,
            },
        )(),
    )
    with pytest.raises(ValueError, match="Unexpected plot render data contract"):
        _RENDER.main()
