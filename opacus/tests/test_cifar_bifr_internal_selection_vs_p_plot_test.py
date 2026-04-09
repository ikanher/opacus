from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_bifr_internal_selection_vs_p.py"
_RENDER_PATH = _REPO_ROOT / "local-scripts" / "render_mf_plot_bundle.py"

_PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_cifar_bifr_internal_selection_vs_p", _PLOT_PATH
)
assert _PLOT_SPEC is not None and _PLOT_SPEC.loader is not None
_PLOT = importlib.util.module_from_spec(_PLOT_SPEC)
sys.modules[_PLOT_SPEC.name] = _PLOT
_PLOT_SPEC.loader.exec_module(_PLOT)

_RENDER_SPEC = importlib.util.spec_from_file_location("render_mf_plot_bundle", _RENDER_PATH)
assert _RENDER_SPEC is not None and _RENDER_SPEC.loader is not None
_RENDER = importlib.util.module_from_spec(_RENDER_SPEC)
sys.modules[_RENDER_SPEC.name] = _RENDER
_RENDER_SPEC.loader.exec_module(_RENDER)


def test_build_bifr_internal_selection_report_preserves_selected_frac_and_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        _PLOT,
        "_build_bifr_series",
        lambda **kwargs: {
            "family": "BIFR",
            "backend": "prv",
            "series_role": "p_curve",
            "curve_policy": "best_over_canonical_frac_sweep_v1",
            "points": [
                {
                    "status": "computed",
                    "p": 2,
                    "paper_rmse": 1.5,
                    "selected_bifr_frac": 1.0,
                    "endpoint_label": "DP-lambdaCGD endpoint",
                    "bifr_frac_candidates": [0.0, 0.25, 0.5, 1.0],
                    "computed_candidate_count": 4,
                },
                {
                    "status": "computed",
                    "p": 4,
                    "paper_rmse": 1.0,
                    "selected_bifr_frac": 0.25,
                    "bifr_frac_candidates": [0.0, 0.25, 0.5, 1.0],
                    "computed_candidate_count": 4,
                },
            ],
        },
    )
    report = _PLOT.build_bifr_internal_selection_report(
        scenario=_PLOT.CIFARFixedBatchScenario(),
        epochs=10,
        bandwidth_grid=[2, 4],
    )
    assert report["metadata"]["curve_policy"] == "best_over_canonical_frac_sweep_v1"
    point0 = report["series"][0]["points"][0]
    assert point0["selected_bifr_frac"] == 1.0
    assert point0["endpoint_label"] == "DP-lambdaCGD endpoint"


def test_build_bifr_internal_selection_figure_data_uses_two_panels() -> None:
    report = {
        "metadata": {
            "epochs": 10,
            "backends": ["prv"],
        },
        "series": [
            {
                "family": "BIFR",
                "backend": "prv",
                "points": [
                    {
                        "status": "computed",
                        "p": 2,
                        "selected_bifr_frac": 1.0,
                        "paper_rmse": 1.5,
                        "endpoint_label": "DP-lambdaCGD endpoint",
                    },
                    {
                        "status": "computed",
                        "p": 4,
                        "selected_bifr_frac": 0.25,
                        "paper_rmse": 1.0,
                    },
                ],
            }
        ],
    }
    figure_data = _PLOT.build_bifr_internal_selection_figure_data(report)
    assert figure_data["figure_contract"] == _PLOT.BIFR_INTERNAL_SELECTION_FIGURE_CONTRACT
    assert len(figure_data["panels"]) == 2
    assert [series["role"] for series in figure_data["panels"][0]["series"]] == ["p_curve", "endpoint_marker"]
    assert [series["role"] for series in figure_data["panels"][1]["series"]] == ["p_curve", "endpoint_marker"]


def test_render_only_path_accepts_bifr_internal_selection_contract(tmp_path: Path) -> None:
    figure_data = {
        "plot_bundle_contract": _PLOT.PLOT_RENDER_DATA_CONTRACT_ID,
        "figure_contract": _PLOT.BIFR_INTERNAL_SELECTION_FIGURE_CONTRACT,
        "figure_slug": "bifr_internal_selection_vs_p",
        "title": "BIFR internal selection",
        "panel_layout": {"rows": 1, "cols": 2, "shared_legend": False},
        "scenario_metadata": {"p_grid": [2, 4]},
        "panels": [
            {
                "panel_id": "backend:prv:frac",
                "title": "PRV Selected frac",
                "x_label": "p",
                "y_label": "Selected frac",
                "series": [
                    {
                        "family": "BIFR",
                        "backend": "prv",
                        "role": "p_curve",
                        "label": "Selected BIFR frac",
                        "color": "#e45756",
                        "data": [{"x": 2, "y": 1.0}, {"x": 4, "y": 0.25}],
                    }
                ],
            },
            {
                "panel_id": "backend:prv:rmse",
                "title": "PRV Paper RMSE",
                "x_label": "p",
                "y_label": "Paper RMSE",
                "series": [
                    {
                        "family": "BIFR",
                        "backend": "prv",
                        "role": "p_curve",
                        "label": "Selected-point paper RMSE",
                        "color": "#e45756",
                        "data": [{"x": 2, "y": 1.5}, {"x": 4, "y": 1.0}],
                    }
                ],
            },
        ],
    }
    input_path = tmp_path / "bifr_internal_selection_vs_p_figure_data.json"
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

    assert (output_dir / "bifr_internal_selection_vs_p.png").exists()
    assert (output_dir / "plot_bundle_manifest.json").exists()
