from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORT_PATH = _REPO_ROOT / "local-scripts" / "replicate_bisr_paper_cifar_noise_multipliers.py"
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_amplified_bnb_p_vs_paper_rmse.py"

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "replicate_bisr_paper_cifar_noise_multipliers", _REPORT_PATH
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
_REPORT = importlib.util.module_from_spec(_REPORT_SPEC)
sys.modules[_REPORT_SPEC.name] = _REPORT
_REPORT_SPEC.loader.exec_module(_REPORT)

_PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_cifar_amplified_bnb_p_vs_paper_rmse", _PLOT_PATH
)
assert _PLOT_SPEC is not None and _PLOT_SPEC.loader is not None
_PLOT = importlib.util.module_from_spec(_PLOT_SPEC)
sys.modules[_PLOT_SPEC.name] = _PLOT
_PLOT_SPEC.loader.exec_module(_PLOT)


def _fake_amplified_blt_plot_row(*, rank: int = 4, paper_rmse: float = 0.9) -> dict[str, object]:
    return {
        "regime": "amplified",
        "backend": "balls_in_bins",
        "method": "BLT",
        "status": "computed",
        "bandwidth": rank,
        "blt_rank": rank,
        "blt_selection_mode": "optimizer_selected",
        "accounting_source": "opacus_blt_amplified_bnb_accountant_contract",
        "paper_rmse": paper_rmse,
    }


def test_build_amplified_bnb_p_sweep_rows_includes_baselines_and_curve_grid() -> None:
    rows = _REPORT.build_amplified_bnb_p_sweep_rows(bandwidth_grid=[2, 4])
    baseline_rows = [row for row in rows if row.method in _REPORT.AMPLIFIED_BNB_BASELINE_METHODS]
    assert [(row.method, row.bandwidth) for row in baseline_rows] == [("DP-SGD", 1)]

    bifr_rows = [row for row in rows if row.method == "BIFR"]
    assert [row.bandwidth for row in bifr_rows] == [2, 4]
    assert all(row.regime == "amplified" for row in rows)


def test_build_report_marks_paper_row_override_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_REPORT, "compute_comparison_rows", lambda **kwargs: [])
    paper_rows = _REPORT.build_amplified_bnb_p_sweep_rows(bandwidth_grid=[2, 4])

    report = _REPORT.build_report(
        include_amplified=True,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=["balls_in_bins"],
        skip_bandinvmf=False,
        methods=["DP-SGD", "BSR"],
        paper_rows=paper_rows,
    )

    assert report["metadata"]["paper_rows_source"] == "caller_supplied_override"
    assert len(report["metadata"]["paper_rows"]) == len(paper_rows)


def test_build_amplified_bnb_p_rmse_report_uses_plot_local_num_samples_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(_PLOT, "build_amplified_bnb_p_sweep_rows", lambda bandwidth_grid: [])

    def fake_build_report(**kwargs):
        captured.update(kwargs)
        return {"metadata": {}, "rows": [_fake_amplified_blt_plot_row()]}

    monkeypatch.setattr(_PLOT, "build_report", fake_build_report)

    _PLOT.build_amplified_bnb_p_rmse_report(bandwidth_grid=[2, 4])

    assert captured["bnb_num_samples"] == 200_000
    assert captured["optimizer_momentum"] == _PLOT.DEFAULT_OPTIMIZER_MOMENTUM
    assert captured["optimizer_weight_decay"] == _PLOT.DEFAULT_OPTIMIZER_WEIGHT_DECAY


def test_build_amplified_bnb_p_rmse_report_propagates_optimizer_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(_PLOT, "build_amplified_bnb_p_sweep_rows", lambda bandwidth_grid: [])

    def fake_build_report(**kwargs):
        captured.update(kwargs)
        return {"metadata": {}, "rows": [_fake_amplified_blt_plot_row()]}

    monkeypatch.setattr(_PLOT, "build_report", fake_build_report)

    report = _PLOT.build_amplified_bnb_p_rmse_report(
        bandwidth_grid=[2, 4],
        optimizer_momentum=0.0,
        optimizer_weight_decay=0.0,
    )

    assert captured["optimizer_momentum"] == 0.0
    assert captured["optimizer_weight_decay"] == 0.0
    assert report["metadata"]["plot_optimizer_workload_override"] == {
        "momentum": 0.0,
        "weight_decay": 0.0,
    }


def test_build_amplified_bnb_p_rmse_figure_data_uses_roles_and_baselines() -> None:
    report = {
        "metadata": {
            "canonical_p_grid": [2, 4],
            "plot_backend": "balls_in_bins",
        },
        "rows": [
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BSR",
                "status": "computed",
                "bandwidth": 2,
                "paper_rmse": 1.5,
            },
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BSR",
                "status": "computed",
                "bandwidth": 4,
                "paper_rmse": 1.0,
            },
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "DP-SGD",
                "status": "computed",
                "bandwidth": 1,
                "paper_rmse": 0.8,
            },
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BLT",
                "status": "computed",
                "bandwidth": 4,
                "blt_rank": 4,
                "paper_rmse": 0.9,
            },
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BIFR",
                "status": "computed",
                "bandwidth": 2,
                "bifr_frac": 0.5,
                "paper_rmse": 1.25,
            },
        ],
    }

    figure_data = _PLOT.build_amplified_bnb_p_rmse_figure_data(report)
    assert figure_data["figure_contract"] == _PLOT.AMPLIFIED_BNB_P_RMSE_FIGURE_CONTRACT
    panel = figure_data["panels"][0]
    assert panel["x_scale"] == "log"
    roles = [series["role"] for series in panel["series"]]
    assert roles == ["p_curve", "p_curve", "baseline", "comparison"]
    assert panel["series"][0]["family"] == "BIFR"
    assert panel["series"][1]["family"] == "BSR"
    assert panel["series"][2]["family"] == "DP-SGD"
    assert panel["series"][3]["family"] == "BLT"
    assert panel["series"][3]["label"] == "BLT (buffer=4)"


def test_build_amplified_bnb_p_rmse_report_requests_blt_as_comparison_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_report(**kwargs):
        captured.update(kwargs)
        return {"metadata": {}, "rows": [_fake_amplified_blt_plot_row(rank=8)]}

    monkeypatch.setattr(_PLOT, "build_report", fake_build_report)

    report = _PLOT.build_amplified_bnb_p_rmse_report(bandwidth_grid=[2, 4])

    assert report["metadata"]["plot_family_roles"]["baseline_methods"] == ["DP-SGD"]
    assert report["metadata"]["plot_family_roles"]["comparison_methods"] == ["BLT"]
    assert report["metadata"]["plot_baseline_contract"]["plotted_baseline_backend"] == "balls_in_bins"
    assert report["metadata"]["plot_baseline_contract"]["report_reference_only_rows"][0]["backend"] == "poisson_prv"
    assert report["metadata"]["plot_comparison_contract"]["BLT"] == {
        "buffer_count": 8,
        "selection_surface": "opacus_blt_amplified_bnb_accountant_contract",
        "selection_mode": "optimizer_selected",
    }
    assert captured["methods"] == ["DP-SGD", "BLT", "BIFR", "BISR", "Band-MF", "Band-Inv-MF", "BSR"]


def test_build_amplified_bnb_p_rmse_report_does_not_import_jax_privacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _REPORT,
        "compute_comparison_rows",
        lambda **kwargs: [
            _REPORT._comparison_row(
                row=_REPORT.PaperRow("amplified", "BLT", float("nan"), 0.1, 4, 10.0),
                backend="balls_in_bins",
                status="computed",
                computed=0.75,
                sensitivity=1.0,
                reason_code="computed_bnb_accountant_blt_forward_c_col",
                reference_noise_multiplier=1.0,
                reference_source="poisson_prv",
                accounting_noise_multiplier=0.75,
                accounting_source="opacus_blt_amplified_bnb_accountant_contract",
                blt_selection_mode="optimizer_selected",
                blt_rank=4,
                blt_selected_candidate_index=0,
                blt_candidate_count=1,
                blt_selected_theta=[0.8, 0.6, 0.4, 0.2],
                blt_selected_theta_hat=[0.7, 0.5, 0.3, 0.1],
                comparison_noise_multiplier=1.0,
                comparison_source="poisson_prv",
                notes="test fixture amplified BLT row",
                paper_rmse=0.9,
            )
        ],
    )
    monkeypatch.setattr(
        _REPORT,
        "_attach_paper_rmse",
        lambda rows, *, optimizer_workload=_REPORT.DEFAULT_OPTIMIZER_WORKLOAD: {
            "computed_rows": len(rows),
            "missing_rows": 0,
        },
    )

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax_privacy" or name.startswith("jax_privacy."):
            raise AssertionError("plot path must not import jax_privacy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    _PLOT.build_amplified_bnb_p_rmse_report(bandwidth_grid=[2, 4])


def test_build_amplified_bnb_p_rmse_report_refreshes_only_requested_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    existing_report = {
        "metadata": {
            "canonical_p_grid": [2, 4],
            "paper_rows": [
                {
                    "regime": "amplified",
                    "method": "BLT",
                    "paper_noise_multiplier": float("nan"),
                    "learning_rate": 0.1,
                    "bandwidth": 8,
                    "clip_norm": 10.0,
                },
                {
                    "regime": "amplified",
                    "method": "BSR",
                    "paper_noise_multiplier": float("nan"),
                    "learning_rate": 0.3,
                    "bandwidth": 4,
                    "clip_norm": 10.0,
                },
            ],
            "paper_rows_source": "caller_supplied_override",
            "non_plot_field": "keep-me",
        },
        "rows": [
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BSR",
                "status": "computed",
                "bandwidth": 4,
                "paper_rmse": 1.25,
            },
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BLT",
                "status": "computed",
                "bandwidth": 4,
                "blt_rank": 4,
                "blt_selection_mode": "optimizer_selected",
                "accounting_source": "opacus_blt_amplified_bnb_accountant_contract",
                "paper_rmse": 0.9,
            },
        ],
    }

    def fake_build_report(**kwargs):
        captured.update(kwargs)
        return {
            "metadata": {
                "paper_rows": kwargs["paper_rows"],
                "paper_rows_source": "caller_supplied_override",
                "optimizer_momentum": kwargs["optimizer_momentum"],
                "optimizer_weight_decay": kwargs["optimizer_weight_decay"],
            },
            "rows": [_fake_amplified_blt_plot_row(rank=8, paper_rmse=0.7)],
        }

    monkeypatch.setattr(_PLOT, "build_report", fake_build_report)

    report = _PLOT.build_amplified_bnb_p_rmse_report(
        bandwidth_grid=[2, 4],
        existing_report=existing_report,
        refresh_methods=["BLT"],
    )

    assert captured["methods"] == ["BLT"]
    assert [(row.method, row.bandwidth) for row in captured["paper_rows"]] == [("BLT", 8)]
    assert report["metadata"]["non_plot_field"] == "keep-me"
    assert any(row["method"] == "BSR" and row["paper_rmse"] == 1.25 for row in report["rows"])
    blt_rows = [row for row in report["rows"] if row["method"] == "BLT"]
    assert len(blt_rows) == 1
    assert blt_rows[0]["blt_rank"] == 8
    assert blt_rows[0]["paper_rmse"] == 0.7
    assert report["metadata"]["plot_comparison_contract"]["BLT"]["buffer_count"] == 8


def test_build_amplified_bnb_p_rmse_figure_data_only_marks_true_lambda_cgd_endpoint() -> None:
    report = {
        "metadata": {
            "canonical_p_grid": [2, 4],
            "plot_backend": "balls_in_bins",
        },
        "rows": [
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BIFR",
                "status": "computed",
                "bandwidth": 2,
                "bifr_frac": 0.5,
                "paper_rmse": 1.5,
            },
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BIFR",
                "status": "computed",
                "bandwidth": 4,
                "bifr_frac": 0.25,
                "paper_rmse": 1.0,
            },
        ],
    }

    figure_data = _PLOT.build_amplified_bnb_p_rmse_figure_data(report)
    panel = figure_data["panels"][0]
    assert [series["role"] for series in panel["series"]] == ["p_curve"]


def test_build_amplified_bnb_p_rmse_figure_data_marks_true_lambda_cgd_endpoint() -> None:
    report = {
        "metadata": {
            "canonical_p_grid": [2, 4],
            "plot_backend": "balls_in_bins",
        },
        "rows": [
            {
                "regime": "amplified",
                "backend": "balls_in_bins",
                "method": "BIFR",
                "status": "computed",
                "bandwidth": 2,
                "bifr_frac": 1.0,
                "paper_rmse": 1.5,
            },
        ],
    }

    figure_data = _PLOT.build_amplified_bnb_p_rmse_figure_data(report)
    panel = figure_data["panels"][0]
    assert [series["role"] for series in panel["series"]] == ["p_curve", "endpoint_marker"]


def test_render_only_path_accepts_amplified_bnb_p_figure_contract(tmp_path: Path) -> None:
    figure_data = {
        "plot_bundle_contract": _PLOT.PLOT_RENDER_DATA_CONTRACT_ID,
        "figure_contract": _PLOT.AMPLIFIED_BNB_P_RMSE_FIGURE_CONTRACT,
        "figure_slug": "amplified_bnb_p_vs_paper_rmse",
        "title": "Amplified Balls-in-Bins p vs Paper RMSE",
        "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
        "scenario_metadata": {"canonical_p_grid": [2, 4]},
        "panels": [
            {
                "panel_id": "backend:balls_in_bins",
                "title": "Balls-in-Bins",
                "x_label": "p",
                "y_label": "Paper RMSE",
                "x_scale": "log",
                "series": [
                    {
                        "family": "BSR",
                        "backend": "balls_in_bins",
                        "role": "p_curve",
                        "label": "BSR",
                        "color": "#f58518",
                        "data": [{"x": 2, "y": 1.5}, {"x": 4, "y": 1.0}],
                    },
                    {
                        "family": "DP-SGD",
                        "backend": "balls_in_bins",
                        "role": "baseline",
                        "label": "DP-SGD baseline",
                        "color": "#4c78a8",
                        "data": [{"x": 2, "y": 0.8}, {"x": 4, "y": 0.8}],
                    },
                ],
            }
        ],
    }
    input_path = tmp_path / "amplified_bnb_p_vs_paper_rmse_figure_data.json"
    input_path.write_text(json.dumps(figure_data), encoding="utf-8")

    render_spec = importlib.util.spec_from_file_location(
        "render_mf_plot_bundle", _REPO_ROOT / "local-scripts" / "render_mf_plot_bundle.py"
    )
    assert render_spec is not None and render_spec.loader is not None
    render_module = importlib.util.module_from_spec(render_spec)
    sys.modules[render_spec.name] = render_module
    render_spec.loader.exec_module(render_module)

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
        render_module.main()
    finally:
        sys.argv = argv

    assert (output_dir / "amplified_bnb_p_vs_paper_rmse.png").exists()
    assert (output_dir / "plot_bundle_manifest.json").exists()
