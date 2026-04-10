from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB_PATH = _REPO_ROOT / "local-scripts" / "lib" / "cifar_mf_paper_rmse.py"
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_fixed_batch_p_vs_paper_rmse.py"
_RENDER_PATH = _REPO_ROOT / "local-scripts" / "render_mf_plot_bundle.py"

_LIB_SPEC = importlib.util.spec_from_file_location("cifar_mf_paper_rmse", _LIB_PATH)
assert _LIB_SPEC is not None and _LIB_SPEC.loader is not None
_LIB = importlib.util.module_from_spec(_LIB_SPEC)
sys.modules[_LIB_SPEC.name] = _LIB
_LIB_SPEC.loader.exec_module(_LIB)

_PLOT_SPEC = importlib.util.spec_from_file_location("plot_cifar_fixed_batch_p_vs_paper_rmse", _PLOT_PATH)
assert _PLOT_SPEC is not None and _PLOT_SPEC.loader is not None
_PLOT = importlib.util.module_from_spec(_PLOT_SPEC)
sys.modules[_PLOT_SPEC.name] = _PLOT
_PLOT_SPEC.loader.exec_module(_PLOT)

_RENDER_SPEC = importlib.util.spec_from_file_location("render_mf_plot_bundle", _RENDER_PATH)
assert _RENDER_SPEC is not None and _RENDER_SPEC.loader is not None
_RENDER = importlib.util.module_from_spec(_RENDER_SPEC)
sys.modules[_RENDER_SPEC.name] = _RENDER
_RENDER_SPEC.loader.exec_module(_RENDER)


def test_build_fixed_batch_p_rmse_report_marks_bifr_endpoint_and_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_family_curve(*, scenario, family, backend, epochs, bandwidth_grid):
        return [
            _LIB.FamilyEpochRMSEPoint(
                status="computed",
                family=str(family),
                backend=str(backend),
                epochs=int(epochs),
                total_steps=int(epochs * 10),
                bandwidth=int(bandwidth),
                noise_multiplier=float(bandwidth),
                sensitivity=1.0,
                paper_rmse=float(bandwidth) / 10.0,
                matrix_source="mock_matrix",
                rmse_contract=_LIB.RMSE_CONTRACT_ID,
                reason=None,
                bifr_frac=None,
            )
            for bandwidth in bandwidth_grid
        ]

    def fake_point(*, scenario, family, epochs, backend, bandwidth=None, bifr_frac=None):
        resolved_bandwidth = int(1 if bandwidth is None else bandwidth)
        if family == "BIFR":
            frac = float(bifr_frac)
            return _LIB.FamilyEpochRMSEPoint(
                status="computed",
                family="BIFR",
                backend=str(backend),
                epochs=int(epochs),
                total_steps=int(epochs * 10),
                bandwidth=resolved_bandwidth,
                noise_multiplier=float(resolved_bandwidth),
                sensitivity=1.0,
                paper_rmse=float(resolved_bandwidth) + abs(frac - 0.25),
                matrix_source="mock_bifr_matrix",
                rmse_contract=_LIB.RMSE_CONTRACT_ID,
                reason=None,
                bifr_frac=frac,
            )
        return _LIB.FamilyEpochRMSEPoint(
            status="computed",
            family=str(family),
            backend=str(backend),
            epochs=int(epochs),
            total_steps=int(epochs * 10),
            bandwidth=resolved_bandwidth,
            noise_multiplier=float(resolved_bandwidth),
            sensitivity=1.0,
            paper_rmse=9.0 if family == "DP-SGD" else 7.0,
            matrix_source="mock_baseline_matrix",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
            bifr_frac=None,
        )

    monkeypatch.setattr(_PLOT, "evaluate_family_paper_rmse_over_bandwidths", fake_family_curve)
    monkeypatch.setattr(_PLOT, "evaluate_fixed_batch_paper_rmse_point", fake_point)

    report = _PLOT.build_fixed_batch_p_rmse_report(
        scenario=_LIB.CIFARFixedBatchScenario(),
        epochs=10,
        bandwidth_grid=[2, 4],
    )

    assert report["metadata"]["p_grid"] == [2, 4]
    bifr_series = next(
        entry
        for entry in report["series"]
        if entry["family"] == "BIFR" and entry["backend"] == "prv"
    )
    assert bifr_series["curve_policy"] == "best_over_canonical_frac_sweep_v1"
    assert [point["selected_bifr_frac"] for point in bifr_series["points"]] == [0.25, 0.25]
    assert "endpoint_label" not in bifr_series["points"][0]

    baseline_families = [
        entry["family"]
        for entry in report["series"]
        if entry["backend"] == "prv" and entry["series_role"] == "baseline"
    ]
    assert baseline_families == ["DP-SGD", "BLT"]


def test_build_fixed_batch_p_rmse_report_only_marks_lambda_cgd_endpoint_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_family_curve(*, scenario, family, backend, epochs, bandwidth_grid):
        return []

    def fake_point(*, scenario, family, epochs, backend, bandwidth=None, bifr_frac=None):
        resolved_bandwidth = int(1 if bandwidth is None else bandwidth)
        if family == "BIFR":
            frac = float(bifr_frac)
            return _LIB.FamilyEpochRMSEPoint(
                status="computed",
                family="BIFR",
                backend=str(backend),
                epochs=int(epochs),
                total_steps=int(epochs * 10),
                bandwidth=resolved_bandwidth,
                noise_multiplier=float(resolved_bandwidth),
                sensitivity=1.0,
                paper_rmse=0.5 if (resolved_bandwidth == 2 and frac == 1.0) else 1.0 + frac,
                matrix_source="mock_bifr_matrix",
                rmse_contract=_LIB.RMSE_CONTRACT_ID,
                reason=None,
                bifr_frac=frac,
            )
        return _LIB.FamilyEpochRMSEPoint(
            status="computed",
            family=str(family),
            backend=str(backend),
            epochs=int(epochs),
            total_steps=int(epochs * 10),
            bandwidth=resolved_bandwidth,
            noise_multiplier=float(resolved_bandwidth),
            sensitivity=1.0,
            paper_rmse=7.0,
            matrix_source="mock_baseline_matrix",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
            bifr_frac=None,
        )

    monkeypatch.setattr(_PLOT, "evaluate_family_paper_rmse_over_bandwidths", fake_family_curve)
    monkeypatch.setattr(_PLOT, "evaluate_fixed_batch_paper_rmse_point", fake_point)

    report = _PLOT.build_fixed_batch_p_rmse_report(
        scenario=_LIB.CIFARFixedBatchScenario(),
        epochs=10,
        bandwidth_grid=[2, 4],
    )

    bifr_series = next(
        entry
        for entry in report["series"]
        if entry["family"] == "BIFR" and entry["backend"] == "prv"
    )
    assert bifr_series["points"][0]["selected_bifr_frac"] == pytest.approx(1.0)
    assert bifr_series["points"][0]["endpoint_label"] == "DP-lambdaCGD endpoint"
    assert "endpoint_label" not in bifr_series["points"][1]


def test_build_fixed_batch_p_rmse_figure_data_uses_roles_and_endpoint_markers() -> None:
    report = {
        "metadata": {
            "epochs": 10,
            "backends": ["prv"],
            "p_grid": [2, 4],
        },
        "series": [
            {
                "family": "BIFR",
                "backend": "prv",
                "series_role": "p_curve",
                "points": [
                    {
                        "status": "computed",
                        "p": 2,
                        "paper_rmse": 1.5,
                        "endpoint_label": "DP-lambdaCGD endpoint",
                    },
                    {
                        "status": "computed",
                        "p": 4,
                        "paper_rmse": 1.0,
                    },
                ],
            },
            {
                "family": "BLT",
                "backend": "prv",
                "series_role": "baseline",
                "points": [
                    {
                        "status": "computed",
                        "p": 2,
                        "paper_rmse": 0.75,
                    }
                ],
            },
        ],
    }

    figure_data = _PLOT.build_fixed_batch_p_rmse_figure_data(report)
    assert figure_data["figure_contract"] == _PLOT.CIFAR_FIXED_BATCH_P_RMSE_FIGURE_CONTRACT
    assert figure_data["scenario_metadata"]["p_grid"] == [2, 4]

    panel = figure_data["panels"][0]
    assert panel["x_scale"] == "log"
    roles = [series["role"] for series in panel["series"]]
    assert roles == ["p_curve", "endpoint_marker", "baseline"]

    baseline = panel["series"][-1]
    assert baseline["family"] == "BLT"
    assert baseline["data"] == [{"x": 2.0, "y": 0.75}, {"x": 4.0, "y": 0.75}]


def test_library_uses_runtime_blt_noise_multiplier_instead_of_accountant_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        epochs=10,
        backend="prv",
        bandwidth=2,
    )

    assert point.status == "computed"
    assert point.noise_multiplier == pytest.approx(1.5)
    assert point.sensitivity is None
    assert point.paper_rmse == pytest.approx(1.5)


def test_render_only_path_accepts_fixed_batch_p_figure_contract(tmp_path: Path) -> None:
    figure_data = {
        "plot_bundle_contract": _PLOT.PLOT_RENDER_DATA_CONTRACT_ID,
        "figure_contract": _PLOT.CIFAR_FIXED_BATCH_P_RMSE_FIGURE_CONTRACT,
        "figure_slug": "fixed_batch_p_vs_paper_rmse",
        "title": "Fixed-Batch p vs Paper RMSE",
        "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
        "scenario_metadata": {"epochs": 10, "backends": ["prv"], "p_grid": [2, 4]},
        "panels": [
            {
                "panel_id": "backend:prv",
                "title": "PRV",
                "x_label": "p",
                "y_label": "Paper RMSE",
                "x_scale": "log",
                "series": [
                    {
                        "family": "BIFR",
                        "backend": "prv",
                        "role": "p_curve",
                        "label": "BIFR",
                        "color": "#e45756",
                        "data": [{"x": 2, "y": 1.5}, {"x": 4, "y": 1.0}],
                    },
                    {
                        "family": "BLT",
                        "backend": "prv",
                        "role": "baseline",
                        "label": "BLT baseline",
                        "color": "#ff9da6",
                        "data": [{"x": 2, "y": 0.75}, {"x": 4, "y": 0.75}],
                    },
                ],
            }
        ],
    }
    input_path = tmp_path / "fixed_batch_p_vs_paper_rmse_figure_data.json"
    input_path.write_text(json.dumps(figure_data), encoding="utf-8")

    output_dir = tmp_path / "rendered"
    argv = sys.argv[:]
    sys.argv = [
        "render_mf_plot_bundle.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
    ]
    try:
        _RENDER.main()
    finally:
        sys.argv = argv

    output_path = output_dir / "fixed_batch_p_vs_paper_rmse.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    manifest_path = output_dir / "plot_bundle_manifest.json"
    assert manifest_path.exists()
