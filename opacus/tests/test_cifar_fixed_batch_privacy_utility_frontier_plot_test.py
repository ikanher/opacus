from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB_PATH = _REPO_ROOT / "local-scripts" / "lib" / "cifar_mf_paper_rmse.py"
_PLOT_PATH = _REPO_ROOT / "local-scripts" / "plot_cifar_fixed_batch_privacy_utility_frontier.py"
_RENDER_PATH = _REPO_ROOT / "local-scripts" / "render_mf_plot_bundle.py"

_LIB_SPEC = importlib.util.spec_from_file_location("cifar_mf_paper_rmse", _LIB_PATH)
assert _LIB_SPEC is not None and _LIB_SPEC.loader is not None
_LIB = importlib.util.module_from_spec(_LIB_SPEC)
sys.modules[_LIB_SPEC.name] = _LIB
_LIB_SPEC.loader.exec_module(_LIB)

_PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_cifar_fixed_batch_privacy_utility_frontier", _PLOT_PATH
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


def test_evaluate_family_frontier_best_over_bandwidths_selects_best_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_bandwidth_curve(*, scenario, family, backend, epochs, bandwidth_grid, bifr_frac=None):
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
                paper_rmse={2: 1.5, 4: 1.0}[int(bandwidth)],
                matrix_source="mock_matrix",
                rmse_contract=_LIB.RMSE_CONTRACT_ID,
                reason=None,
                bifr_frac=None,
            )
            for bandwidth in bandwidth_grid
        ]

    monkeypatch.setattr(_LIB, "evaluate_family_paper_rmse_over_bandwidths", fake_bandwidth_curve)

    point = _LIB.evaluate_family_frontier_best_over_bandwidths(
        scenario=_LIB.CIFARFixedBatchScenario(),
        family="BSR",
        backend="prv",
        epochs=10,
        target_epsilon=1.0,
        bandwidth_grid=[2, 4],
    )

    assert point.status == "computed"
    assert point.selected_bandwidth == 4
    assert point.paper_rmse == pytest.approx(1.0)


def test_build_fixed_batch_frontier_report_records_envelopes_and_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _PLOT,
        "evaluate_family_frontier_best_over_bandwidths",
        lambda **kwargs: _LIB.FamilyFrontierRMSEPoint(
            status="computed",
            family=str(kwargs["family"]),
            backend=str(kwargs["backend"]),
            target_epsilon=float(kwargs["target_epsilon"]),
            target_delta=1e-5,
            epochs=int(kwargs["epochs"]),
            selected_bandwidth=4,
            selected_bifr_frac=0.25 if kwargs["family"] == "BIFR" else None,
            noise_multiplier=1.0,
            sensitivity=1.0,
            paper_rmse=float(kwargs["target_epsilon"]),
            matrix_source="mock_envelope",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
        ),
    )
    monkeypatch.setattr(
        _PLOT,
        "evaluate_fixed_batch_paper_rmse_point",
        lambda **kwargs: _LIB.FamilyEpochRMSEPoint(
            status="computed",
            family=str(kwargs["family"]),
            backend=str(kwargs["backend"]),
            epochs=int(kwargs["epochs"]),
            total_steps=int(kwargs["epochs"] * 10),
            bandwidth=int(kwargs.get("bandwidth", 1)),
            noise_multiplier=1.0,
            sensitivity=1.0,
            paper_rmse=0.5,
            matrix_source="mock_baseline",
            rmse_contract=_LIB.RMSE_CONTRACT_ID,
            reason=None,
            bifr_frac=None,
        ),
    )

    report = _PLOT.build_fixed_batch_frontier_report(
        scenario=_LIB.CIFARFixedBatchScenario(),
        epochs=10,
        epsilon_grid=[0.5, 1.0],
        bandwidth_grid=[2, 4],
    )

    assert report["metadata"]["epsilon_grid"] == [0.5, 1.0]
    assert report["metadata"]["selection_policy"]["kind"] == "best_over_fixed_batch_p_grid_v1"
    envelope_series = next(
        entry for entry in report["series"] if entry["family"] == "BIFR" and entry["backend"] == "prv"
    )
    assert envelope_series["series_role"] == "envelope"
    assert [point["selected_bandwidth"] for point in envelope_series["points"]] == [4, 4]
    assert [point["selected_bifr_frac"] for point in envelope_series["points"]] == [0.25, 0.25]


def test_build_fixed_batch_frontier_figure_data_uses_envelope_role() -> None:
    report = {
        "metadata": {
            "epochs": 10,
            "backends": ["prv"],
        },
        "series": [
            {
                "family": "BSR",
                "backend": "prv",
                "series_role": "envelope",
                "points": [
                    {"status": "computed", "target_epsilon": 0.5, "paper_rmse": 1.5},
                    {"status": "computed", "target_epsilon": 1.0, "paper_rmse": 1.0},
                ],
            },
            {
                "family": "DP-SGD",
                "backend": "prv",
                "series_role": "baseline",
                "points": [
                    {"status": "computed", "target_epsilon": 0.5, "paper_rmse": 0.8},
                    {"status": "computed", "target_epsilon": 1.0, "paper_rmse": 0.6},
                ],
            },
        ],
    }

    figure_data = _PLOT.build_fixed_batch_frontier_figure_data(report)
    assert figure_data["figure_contract"] == _PLOT.FIXED_BATCH_FRONTIER_FIGURE_CONTRACT
    panel = figure_data["panels"][0]
    assert panel["x_scale"] == "log"
    roles = [series["role"] for series in panel["series"]]
    assert roles == ["envelope", "baseline"]


def test_render_only_path_accepts_fixed_batch_frontier_contract(tmp_path: Path) -> None:
    figure_data = {
        "plot_bundle_contract": _PLOT.PLOT_RENDER_DATA_CONTRACT_ID,
        "figure_contract": _PLOT.FIXED_BATCH_FRONTIER_FIGURE_CONTRACT,
        "figure_slug": "fixed_batch_privacy_utility_frontier",
        "title": "Fixed-Batch Frontier",
        "panel_layout": {"rows": 1, "cols": 1, "shared_legend": False},
        "scenario_metadata": {"epsilon_grid": [0.5, 1.0]},
        "panels": [
            {
                "panel_id": "backend:prv",
                "title": "PRV",
                "x_label": "epsilon",
                "y_label": "Paper RMSE",
                "x_scale": "log",
                "series": [
                    {
                        "family": "BSR",
                        "backend": "prv",
                        "role": "envelope",
                        "label": "BSR best-over-p",
                        "color": "#f58518",
                        "data": [{"x": 0.5, "y": 1.5}, {"x": 1.0, "y": 1.0}],
                    },
                    {
                        "family": "DP-SGD",
                        "backend": "prv",
                        "role": "baseline",
                        "label": "DP-SGD baseline",
                        "color": "#4c78a8",
                        "data": [{"x": 0.5, "y": 0.8}, {"x": 1.0, "y": 0.6}],
                    },
                ],
            }
        ],
    }
    input_path = tmp_path / "fixed_batch_privacy_utility_frontier_figure_data.json"
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

    assert (output_dir / "fixed_batch_privacy_utility_frontier.png").exists()
    assert (output_dir / "plot_bundle_manifest.json").exists()
