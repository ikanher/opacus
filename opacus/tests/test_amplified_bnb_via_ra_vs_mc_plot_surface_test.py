from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORT_PATH = (
    _REPO_ROOT
    / "local-scripts"
    / "replicate_bisr_paper_cifar_noise_multipliers.py"
)
_PLOT_PATH = (
    _REPO_ROOT
    / "local-scripts"
    / "plot_cifar_amplified_bnb_via_ra_vs_mc_paper_rmse.py"
)

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "replicate_bisr_paper_cifar_noise_multipliers",
    _REPORT_PATH,
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
_REPORT = importlib.util.module_from_spec(_REPORT_SPEC)
sys.modules[_REPORT_SPEC.name] = _REPORT
_REPORT_SPEC.loader.exec_module(_REPORT)

_PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_cifar_amplified_bnb_via_ra_vs_mc_paper_rmse",
    _PLOT_PATH,
)
assert _PLOT_SPEC is not None and _PLOT_SPEC.loader is not None
_PLOT = importlib.util.module_from_spec(_PLOT_SPEC)
sys.modules[_PLOT_SPEC.name] = _PLOT
_PLOT_SPEC.loader.exec_module(_PLOT)


def _fake_blt_row(*, backend: str, sigma: float, rmse: float, buffers: int = 4) -> dict[str, object]:
    return {
        "regime": "amplified",
        "backend": backend,
        "method": "BLT",
        "status": "computed",
        "bandwidth": buffers,
        "blt_buffers": buffers,
        "blt_selection_mode": "optimizer_selected",
        "computed_noise_multiplier": sigma,
        "paper_rmse": rmse,
    }


def test_build_amplified_bnb_via_ra_vs_mc_report_matches_blt_across_backends(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        _PLOT,
        "build_report",
        lambda **kwargs: {
            "metadata": {
                "paper_rows": kwargs["paper_rows"],
                "paper_rows_source": "caller_supplied_override",
                "optimizer_momentum": kwargs["optimizer_momentum"],
                "optimizer_weight_decay": kwargs["optimizer_weight_decay"],
            },
            "rows": [
                _fake_blt_row(backend="balls_in_bins", sigma=0.75, rmse=0.9, buffers=4),
                _fake_blt_row(backend="ra_fixed_bin_bnb", sigma=0.7, rmse=0.8, buffers=4),
            ],
        },
    )

    report = _PLOT.build_amplified_bnb_via_ra_vs_mc_report(
        bandwidth_grid=[2, 4, 8],
        refresh_methods=["BLT"],
    )

    comparison_rows = report["metadata"]["comparison_rows"]
    assert len(comparison_rows) == 1
    assert comparison_rows[0]["family"] == "BLT"
    assert comparison_rows[0]["bandwidth"] == 4
    assert report["metadata"]["plot_comparison_contract"]["unmatched_notes"] == []


def test_build_amplified_bnb_via_ra_vs_mc_figure_data_renders_blt_on_both_panels_and_gap(
) -> None:
    report = {
        "metadata": {
            "canonical_p_grid": [2, 4, 8],
            "plot_mode": "comparison",
            "comparison_rows": [
                {
                    "family": "BLT",
                    "bandwidth": 4,
                    "mc_paper_rmse": 0.9,
                    "ra_paper_rmse": 0.8,
                    "absolute_rmse_diff": -0.1,
                    "relative_rmse_diff": -0.1 / 0.9,
                    "mc_noise_multiplier": 0.75,
                    "ra_noise_multiplier": 0.7,
                }
            ],
        },
        "rows": [
            _fake_blt_row(backend="balls_in_bins", sigma=0.75, rmse=0.9, buffers=4),
            _fake_blt_row(backend="ra_fixed_bin_bnb", sigma=0.7, rmse=0.8, buffers=4),
        ],
    }

    figure_data = _PLOT.build_amplified_bnb_via_ra_vs_mc_figure_data(report)
    mc_panel, ra_panel, gap_panel = figure_data["panels"]

    mc_blt = [series for series in mc_panel["series"] if series["family"] == "BLT"]
    ra_blt = [series for series in ra_panel["series"] if series["family"] == "BLT"]
    gap_blt = [series for series in gap_panel["series"] if series["family"] == "BLT"]

    assert len(mc_blt) == 1
    assert len(ra_blt) == 1
    assert len(gap_blt) == 1
    assert mc_blt[0]["role"] == "comparison"
    assert ra_blt[0]["role"] == "comparison"
    assert gap_blt[0]["role"] == "comparison"
    assert mc_blt[0]["label"] == "BLT (buffer=4)"
    assert ra_blt[0]["label"] == "BLT (buffer=4)"
    assert gap_blt[0]["data"] == [
        {"x": 2.0, "y": -0.1 / 0.9},
        {"x": 8.0, "y": -0.1 / 0.9},
    ]
