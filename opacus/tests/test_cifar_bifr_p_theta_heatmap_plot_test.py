from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_bifr_p_theta_heatmap.py"
_RENDER_PATH = _REPO_ROOT / "local-scripts" / "render_mf_plot_bundle.py"

_PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_cifar_bifr_p_theta_heatmap", _PLOT_PATH
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


def test_build_bifr_p_theta_heatmap_report_preserves_explicit_grids(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_report(**kwargs):
        captured.update(kwargs)
        return {
            "metadata": {"target_epsilon": 9.0, "target_delta": 1e-5},
            "summary": {
                "bifr": {
                    "amplified_candidate_audit": {
                        "balls_in_bins:p2": {
                            "selected_frac": 1.0,
                            "selected_paper_rmse": 5.0,
                            "candidates": [
                                {"frac": 0.0, "status": "computed", "paper_rmse": 8.0, "computed_noise_multiplier": 1.2},
                                {"frac": 1.0, "status": "computed", "paper_rmse": 5.0, "computed_noise_multiplier": 0.8},
                            ],
                        }
                    }
                }
            },
        }

    monkeypatch.setattr(_PLOT, "build_report", fake_build_report)
    report = _PLOT.build_bifr_p_theta_heatmap_report(
        bandwidth_grid=[1, 2],
        theta_grid=[0.0, 0.5, 1.0],
        bnb_num_samples=10_000,
    )

    assert captured["bifr_fracs"] == [0.0, 0.5, 1.0]
    assert [row.bandwidth for row in captured["paper_rows"]] == [1, 2]
    assert report["p_grid"] == [1, 2]
    assert report["theta_grid"] == [0.0, 0.5, 1.0]
    assert report["metadata"]["canonical_p_grid"] == [1, 2]
    assert report["metadata"]["canonical_theta_grid"] == [0.0, 0.5, 1.0]
    assert report["selected_theta_path"] == report["selected_path"]
    assert report["selected_path"][0]["is_lambda_cgd_endpoint"] is True


def test_build_bifr_p_theta_heatmap_figure_data_emits_plain_heatmap() -> None:
    report = {
        "metadata": {
            "canonical_p_grid": [1, 2, 4],
            "canonical_theta_grid": [0.0, 0.5, 1.0],
        },
        "cells": [
            {"p": 1, "theta": 0.0, "paper_rmse": 9.0},
            {"p": 2, "theta": 0.0, "paper_rmse": 8.0},
            {"p": 4, "theta": 0.0, "paper_rmse": 7.0},
            {"p": 1, "theta": 0.5, "paper_rmse": 6.0},
            {"p": 2, "theta": 0.5, "paper_rmse": 5.0},
            {"p": 4, "theta": 0.5, "paper_rmse": 4.0},
            {"p": 1, "theta": 1.0, "paper_rmse": 7.5},
            {"p": 2, "theta": 1.0, "paper_rmse": 4.5},
            {"p": 4, "theta": 1.0, "paper_rmse": 4.2},
        ],
        "selected_path": [
            {"p": 1, "theta": 0.5, "is_bsr_slice": True, "is_lambda_cgd_endpoint": False},
            {"p": 2, "theta": 1.0, "is_bsr_slice": False, "is_lambda_cgd_endpoint": True},
            {"p": 4, "theta": 0.5, "is_bsr_slice": True, "is_lambda_cgd_endpoint": False},
        ],
    }
    figure_data = _PLOT.build_bifr_p_theta_heatmap_figure_data(report)
    panel = figure_data["panels"][0]
    assert figure_data["figure_contract"] == _PLOT.BIFR_P_THETA_HEATMAP_FIGURE_CONTRACT
    assert panel["x_scale"] == "log"
    assert panel["y_label"] == "theta"
    assert panel["heatmap"]["x_values"] == [1, 2, 4]
    assert panel["heatmap"]["y_values"] == [0.0, 0.5, 1.0]
    assert (
        panel["heatmap"]["colorbar_label"]
        == "log10(Paper RMSE), values above 10^3 masked as bad"
    )
    assert panel["heatmap"]["cmap"] == "viridis_r"
    assert panel["heatmap"]["vmin"] < panel["heatmap"]["vmax"]
    assert panel["heatmap"]["vmax"] == math.log10(9.0)
    assert panel["heatmap"]["bad_color"] == "#2b2b2b"
    assert panel["series"] == []


def test_build_bifr_p_theta_heatmap_figure_data_masks_overflow_cells_as_bad() -> None:
    report = {
        "metadata": {
            "canonical_p_grid": [2],
            "canonical_theta_grid": [0.5, 1.0],
        },
        "cells": [
            {"p": 2, "theta": 0.5, "paper_rmse": 9.0},
            {"p": 2, "theta": 1.0, "paper_rmse": 1e5},
        ],
        "selected_path": [],
    }
    figure_data = _PLOT.build_bifr_p_theta_heatmap_figure_data(report)
    z_matrix = figure_data["panels"][0]["heatmap"]["z_matrix"]
    assert math.isfinite(z_matrix[0][0])
    assert math.isnan(z_matrix[1][0])


def test_build_bifr_p_theta_heatmap_report_marks_nonfinite_cells_noncomputed(monkeypatch) -> None:
    def fake_build_report(**kwargs):
        return {
            "metadata": {"target_epsilon": 9.0, "target_delta": 1e-5},
            "summary": {
                "bifr": {
                    "amplified_candidate_audit": {
                        "balls_in_bins:p2": {
                            "selected_frac": 0.5,
                            "selected_paper_rmse": 5.0,
                            "candidates": [
                                {"frac": 0.0, "status": "computed", "paper_rmse": 8.0, "computed_noise_multiplier": 1.2},
                                {"frac": 0.5, "status": "computed", "paper_rmse": 5.0, "computed_noise_multiplier": 0.8},
                                {"frac": 1.0, "status": "computed", "paper_rmse": float("inf"), "computed_noise_multiplier": 0.7},
                            ],
                        }
                    }
                }
            },
        }

    monkeypatch.setattr(_PLOT, "build_report", fake_build_report)
    report = _PLOT.build_bifr_p_theta_heatmap_report(
        bandwidth_grid=[2],
        theta_grid=[0.0, 0.5, 1.0],
        bnb_num_samples=10_000,
    )
    nonfinite = [cell for cell in report["cells"] if cell["theta"] == 1.0][0]
    assert nonfinite["status"] == "nonfinite_paper_rmse"
    assert nonfinite["paper_rmse"] is None
    assert nonfinite["paper_rmse_reason"] == "nonfinite_paper_rmse"


def test_render_only_path_accepts_bifr_p_theta_heatmap_contract(tmp_path: Path) -> None:
    figure_data = {
        "plot_bundle_contract": _PLOT.PLOT_RENDER_DATA_CONTRACT_ID,
        "figure_contract": _PLOT.BIFR_P_THETA_HEATMAP_FIGURE_CONTRACT,
        "figure_slug": "bifr_p_theta_heatmap",
        "title": "BIFR p x theta heatmap",
        "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
        "scenario_metadata": {"canonical_p_grid": [1, 2, 4], "canonical_theta_grid": [0.0, 0.5, 1.0]},
        "panels": [
            {
                "panel_id": "backend:balls_in_bins:bifr_heatmap",
                "title": "Balls-in-Bins BIFR paper RMSE",
                "x_label": "p",
                "y_label": "theta",
                "x_scale": "log",
                "y_scale": "linear",
                "heatmap": {
                    "x_values": [1, 2, 4],
                    "y_values": [0.0, 0.5, 1.0],
                    "z_matrix": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [7.5, 4.5, 4.2]],
                    "colorbar_label": "log10(Paper RMSE), values above 10^3 masked as bad",
                    "cmap": "viridis_r",
                    "vmin": 4.0,
                    "vmax": 9.0,
                    "bad_color": "#2b2b2b",
                },
                "series": [],
            }
        ],
    }
    input_path = tmp_path / "bifr_p_theta_heatmap_figure_data.json"
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

    assert (output_dir / "bifr_p_theta_heatmap.png").exists()
    assert (output_dir / "plot_bundle_manifest.json").exists()
