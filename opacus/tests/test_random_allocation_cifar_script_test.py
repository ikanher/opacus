from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "local-scripts"
    / "replicate_bisr_paper_cifar_noise_multipliers.py"
)
_SPEC = importlib.util.spec_from_file_location("replicate_bisr_paper_cifar_noise_multipliers", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _bridge_row_for(
    row,
    *,
    status: str = "failed",
    computed: float | None = None,
    reason_code: str = "known_missing_fixed_bin_bridge_quantitative_window_realization",
    route: str | None = None,
    initial_package_route: str | None = None,
) :
    return _MODULE._comparison_row(
        row=row,
        backend="random_allocation",
        status=status,
        computed=computed,
        reason_code=reason_code,
        source_law_kind="balls_in_bins_fixed_bin",
        accountant_engine_kind="deterministic_random_allocation",
        route=route
        if route is not None
        else (
            "fixed_bin_bridge_exact_pair_package"
            if status == "computed"
            else "fixed_bin_bridge_ambient_quantitative_window_realization_package"
        ),
        exact_law_route="exact_fixed_bin_gaussian_mixture_pair",
        initial_package_route=initial_package_route
        if initial_package_route is not None
        else (
            "pair_driven_exact_initial_package"
            if status == "computed"
            else "pair_driven_ambient_quantitative_window_realization_package"
        ),
        notes="fixed-bin bridge row",
    )


def _paper_row(method: str):
    return next(
        paper_row
        for paper_row in _MODULE.PAPER_ROWS
        if paper_row.regime == "amplified" and paper_row.method == method
    )


def _install_fast_bridge_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method: str,
    computed: float,
    route: str,
    initial_package_route: str,
    exact_law_route: str = "exact_fixed_bin_gaussian_mixture_pair",
    source_law_kind: str = "balls_in_bins_fixed_bin",
    accountant_engine_kind: str = "deterministic_random_allocation",
    mechanism: str = "gaussian",
    coeff_source: str = "test_fixture",
    sensitivity: float = 1.0,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_resolve_fixed_bin_bridge_workload",
        lambda *, method, bands: (_MODULE.np.eye(4, dtype=float), mechanism, coeff_source, sensitivity),
    )
    monkeypatch.setattr(
        _MODULE,
        "resolve_fixed_bin_random_allocation_bridge_inputs",
        lambda **kwargs: SimpleNamespace(
            source_law_kind=source_law_kind,
            accountant_engine_kind=accountant_engine_kind,
            route=route,
            exact_law_route=exact_law_route,
            initial_package_route=initial_package_route,
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_fixed_bin_bridge_noise_multiplier",
        lambda **kwargs: (
            computed,
            {
                "source_law_kind": source_law_kind,
                "accountant_engine_kind": accountant_engine_kind,
                "route": route,
                "exact_law_route": exact_law_route,
                "initial_package_route": initial_package_route,
            },
        ),
    )


def test_parse_args_exposes_include_amplified_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["replicate_bisr_paper_cifar_noise_multipliers.py", "--include-amplified-deterministic"],
    )
    args = _MODULE.parse_args()
    assert bool(args.include_amplified_deterministic) is True
    assert bool(args.include_amplified) is False


def test_amplified_random_allocation_rows_are_absent_by_default() -> None:
    rows = _MODULE.compute_comparison_rows(include_amplified=False, methods=["DP-SGD"])
    amplified_rows = [row for row in rows if row.regime == "amplified"]
    assert all(row.backend != "random_allocation" for row in amplified_rows)


def test_amplified_random_allocation_rows_require_the_new_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_amplified_deterministic=True,
        methods=["DP-SGD"],
    )

    random_allocation_rows = [row for row in rows if row.regime == "amplified" and row.backend == "random_allocation"]
    assert len(random_allocation_rows) == 1
    assert random_allocation_rows[0].parity_status == "known_missing"


def test_amplified_random_allocation_rows_use_bridge_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_bnb_delta",
        lambda **kwargs: _MODULE.SingleVerifyProbeResult(
            status="computed",
            delta_estimate=1e-6,
            delta_upper_confidence_bound=1e-6,
            error_probability=1e-6,
            notes="stub",
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_bnb_noise_multiplier_for_coeffs",
        lambda **kwargs: 2.34,
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_amplified_deterministic=True,
        methods=["DP-SGD"],
    )

    random_allocation_rows = [row for row in rows if row.regime == "amplified" and row.backend == "random_allocation"]
    assert len(random_allocation_rows) == 1
    row = random_allocation_rows[0]
    assert row.parity_status == "known_missing"
    assert row.reason_code == "known_missing_fixed_bin_bridge_quantitative_window_realization"
    assert row.source_law_kind == "balls_in_bins_fixed_bin"
    assert row.accountant_engine_kind == "deterministic_random_allocation"
    assert row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"

    amplified_bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "bnb"]
    assert len(amplified_bnb_rows) == 1
    bnb_row = amplified_bnb_rows[0]
    assert bnb_row.backend == "bnb"
    assert "random_allocation" not in bnb_row.notes


def test_amplified_random_allocation_method_matrix_keeps_family_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    sigma_by_mechanism = {
        "gaussian": 0.61,
        "bsr": 1.11,
        "bisr": 1.96,
        "bandmf": 0.61,
        "bandinvmf": 2.71,
    }

    def _fake_get_noise_multiplier(*, accountant: str, mechanism_state=None, **kwargs):
        if accountant != "random_allocation":
            raise AssertionError(f"unexpected accountant: {accountant}")
        state = {} if mechanism_state is None else dict(mechanism_state)
        mechanism = str(state.get("mechanism", state.get("name", "gaussian")))
        return sigma_by_mechanism[mechanism]

    monkeypatch.setattr(_MODULE, "get_noise_multiplier", _fake_get_noise_multiplier)
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_amplified_deterministic=True,
        methods=["DP-SGD", "BSR", "BISR", "Band-MF", "Band-Inv-MF"],
    )

    random_allocation_rows = [row for row in rows if row.regime == "amplified" and row.backend == "random_allocation"]
    assert {row.method for row in random_allocation_rows} == {
        "DP-SGD",
        "BSR",
        "BISR",
        "Band-MF",
        "Band-Inv-MF",
    }
    for row in random_allocation_rows:
        assert row.parity_status == "known_missing"
        assert row.source_law_kind == "balls_in_bins_fixed_bin"
        assert row.accountant_engine_kind == "deterministic_random_allocation"
        assert row.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"
        assert row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"


def test_fixed_bin_random_allocation_rows_use_canonical_backend_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )
    report = _MODULE.build_report(
        include_amplified=False,
        include_amplified_deterministic=True,
        skip_bandinvmf=False,
        methods=["DP-SGD"],
    )
    rows = report["rows"]
    random_allocation_rows = [row for row in rows if row["backend"] == "random_allocation"]
    assert len(random_allocation_rows) == 1
    random_allocation = random_allocation_rows[0]
    assert random_allocation["parity_status"] == "known_missing"
    assert random_allocation["source_law_kind"] == "balls_in_bins_fixed_bin"
    assert random_allocation["accountant_engine_kind"] == "deterministic_random_allocation"
    assert random_allocation["route"] == "fixed_bin_bridge_ambient_quantitative_window_realization_package"


def test_dpsgd_fixed_bin_bridge_workload_uses_reduced_full_horizon_mode_norm_control() -> None:
    c_matrix, mechanism, coeff_source, sensitivity = _MODULE._resolve_fixed_bin_bridge_workload(
        method="DP-SGD",
        bands=1,
    )
    assert c_matrix.shape == (1, _MODULE.TOTAL_STEPS)
    assert coeff_source == "full_horizon_mode_norm_control"
    assert mechanism == "gaussian"
    assert sensitivity == pytest.approx(_MODULE.EPOCHS ** 0.5, rel=0.0, abs=1e-12)


def test_dpsgd_fixed_bin_bridge_row_reports_exact_pair_route_without_live_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _paper_row("DP-SGD")
    _install_fast_bridge_success(
        monkeypatch,
        method="DP-SGD",
        computed=0.39642333984375,
        route="fixed_bin_bridge_exact_pair_package",
        initial_package_route="pair_driven_exact_initial_package",
        mechanism="gaussian",
        coeff_source="full_horizon_mode_norm_control",
        sensitivity=_MODULE.EPOCHS ** 0.5,
    )

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "computed"
    assert bridge_row.parity_status == "computed_far"
    assert bridge_row.reason_code == "computed_fixed_bin_bridge_paper_contract"
    assert bridge_row.source_law_kind == "balls_in_bins_fixed_bin"
    assert bridge_row.accountant_engine_kind == "deterministic_random_allocation"
    assert bridge_row.route == "fixed_bin_bridge_exact_pair_package"
    assert bridge_row.initial_package_route == "pair_driven_exact_initial_package"
    assert bridge_row.computed_noise_multiplier == pytest.approx(0.39642333984375)


def test_bsr_fixed_bin_bridge_row_reports_close_ambient_route_without_live_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _paper_row("BSR")
    _install_fast_bridge_success(
        monkeypatch,
        method="BSR",
        computed=2.314453125,
        route="fixed_bin_bridge_ambient_quantitative_window_realization_package",
        initial_package_route="pair_driven_ambient_quantitative_window_realization_package",
        mechanism="bsr",
        coeff_source="bsr_runtime_coeffs",
    )

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "computed"
    assert bridge_row.parity_status == "computed_close"
    assert bridge_row.reason_code == "computed_fixed_bin_bridge_paper_contract"
    assert bridge_row.source_law_kind == "balls_in_bins_fixed_bin"
    assert bridge_row.accountant_engine_kind == "deterministic_random_allocation"
    assert bridge_row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge_row.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert bridge_row.computed_noise_multiplier == pytest.approx(2.314453125)


def test_bisr_fixed_bin_bridge_row_reports_close_ambient_route_without_live_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _paper_row("BISR")
    _install_fast_bridge_success(
        monkeypatch,
        method="BISR",
        computed=4.26483154296875,
        route="fixed_bin_bridge_ambient_quantitative_window_realization_package",
        initial_package_route="pair_driven_ambient_quantitative_window_realization_package",
        mechanism="bisr",
        coeff_source="bisr_runtime_coeffs",
    )

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "computed"
    assert bridge_row.parity_status == "computed_close"
    assert bridge_row.reason_code == "computed_fixed_bin_bridge_paper_contract"
    assert bridge_row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge_row.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert bridge_row.computed_noise_multiplier == pytest.approx(4.26483154296875)


def test_band_mf_fixed_bin_bridge_row_reports_ambient_route_without_live_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _paper_row("Band-MF")
    _install_fast_bridge_success(
        monkeypatch,
        method="Band-MF",
        computed=1.246564005037385,
        route="fixed_bin_bridge_ambient_quantitative_window_realization_package",
        initial_package_route="pair_driven_ambient_quantitative_window_realization_package",
        mechanism="band_mf",
        coeff_source="band_mf_runtime_coeffs",
    )

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "computed"
    assert bridge_row.reason_code == "computed_fixed_bin_bridge_paper_contract"
    assert bridge_row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge_row.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert bridge_row.parity_status == "computed_far"
    assert bridge_row.computed_noise_multiplier == pytest.approx(1.246564005037385)


def test_band_inv_mf_fixed_bin_bridge_row_reports_ambient_route_without_live_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _paper_row("Band-Inv-MF")
    _install_fast_bridge_success(
        monkeypatch,
        method="Band-Inv-MF",
        computed=6.89601366667091,
        route="fixed_bin_bridge_ambient_quantitative_window_realization_package",
        initial_package_route="pair_driven_ambient_quantitative_window_realization_package",
        mechanism="bandinvmf",
        coeff_source="bandinvmf_runtime_coeffs",
    )

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "computed"
    assert bridge_row.reason_code == "computed_fixed_bin_bridge_paper_contract"
    assert bridge_row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge_row.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert bridge_row.parity_status == "computed_far"
    assert bridge_row.computed_noise_multiplier == pytest.approx(6.89601366667091)


def test_ambient_quantitative_window_route_maps_nonfinite_upper_bound_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = next(
        paper_row
        for paper_row in _MODULE.PAPER_ROWS
        if paper_row.regime == "amplified" and paper_row.method == "BSR"
    )

    def _raise_nonfinite_upper(**kwargs):
        raise _MODULE.NoiseSearchConvergenceError(
            accountant="fixed_bin_random_allocation_bridge",
            target_epsilon=9.0,
            target_delta=1e-5,
            last_finite_sigma=None,
            last_finite_epsilon=None,
            last_nonfinite_sigma=2.0,
            iterations=3,
        )

    monkeypatch.setattr(_MODULE, "_compute_fixed_bin_bridge_noise_multiplier", _raise_nonfinite_upper)
    monkeypatch.setattr(_MODULE, "_diagnose_fixed_bin_ambient_nonfinite_upper_bound", lambda **kwargs: None)

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "failed"
    assert bridge_row.reason_code == "known_missing_fixed_bin_bridge_nonfinite_upper_bound"
    assert bridge_row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge_row.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert "deterministic upper bound stays non-finite" in bridge_row.notes
