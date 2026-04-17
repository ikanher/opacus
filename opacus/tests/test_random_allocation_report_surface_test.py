from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from opacus.accountants.analysis.random_allocation import (
    resolve_random_allocation_gaussian_runtime_config,
)


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
        backend="ra_fixed_bin_bnb",
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
        lambda *, method, bands, bifr_frac=None, optimizer_workload=_MODULE.DEFAULT_OPTIMIZER_WORKLOAD: (
            _MODULE.np.eye(4, dtype=float),
            mechanism,
            coeff_source,
            sensitivity,
        ),
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
                "runtime_policy_name": "fixed_bin_bridge_candidate_grid_1e-2",
                "runtime_loss_discretization": 1e-2,
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
    assert bool(args.include_non_amplified) is False


def test_parse_args_exposes_amplified_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--amplified-backends",
            "b_min_sep,ra",
        ],
    )
    args = _MODULE.parse_args()
    assert args.amplified_backends == ["b_min_sep,ra"]


def test_parse_args_exposes_amplified_direct_verification_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--amplified-execution-mode",
            "verify_table_sigma",
            "--amplified-verify-budgets",
            "200000,500000",
        ],
    )
    args = _MODULE.parse_args()
    assert args.amplified_execution_mode == "verify_table_sigma"
    assert args.amplified_verify_budgets == ["200000,500000"]


def test_parse_args_accepts_fixed_bin_loss_discretization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--fixed-bin-loss-discretization",
            "0.02",
        ],
    )
    args = _MODULE.parse_args()
    assert args.fixed_bin_loss_discretization == pytest.approx(0.02)


def test_parse_args_accepts_blt_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--methods",
            "BLT",
        ],
    )
    args = _MODULE.parse_args()
    assert args.methods == ["BLT"]


def test_parse_args_accepts_bifr_method_and_frac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--methods",
            "BIFR",
            "--bifr-frac",
            "0.3",
        ],
    )
    args = _MODULE.parse_args()
    assert args.methods == ["BIFR"]
    assert args.bifr_frac == pytest.approx(0.3)


def test_parse_args_accepts_blt_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--methods",
            "BLT",
            "--blt-lambda",
            "0.6",
        ],
    )
    args = _MODULE.parse_args()
    assert args.methods == ["BLT"]
    assert args.blt_lambda == pytest.approx(0.6)


def test_blt_amplified_diagnostic_row_is_added_only_when_requested() -> None:
    rows = _MODULE._selected_report_rows(selected_methods={"BLT"})
    blt_regimes = {(row.method, row.regime) for row in rows if row.method == "BLT"}
    assert ("BLT", "non-amplified") in blt_regimes
    assert ("BLT", "amplified") not in blt_regimes

    original = _MODULE._resolve_blt_amplified_accountant_coeffs
    _MODULE._resolve_blt_amplified_accountant_coeffs = (
        lambda *, buffers, blt_lambda: (
            [1.0] + [0.0] * (int(_MODULE.TOTAL_STEPS) - 1),
            "normalized_forward_c_col",
            {
                "noise_multiplier_ref": 1.0,
                "blt_horizon": int(_MODULE.TOTAL_STEPS),
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 1,
                "selected_theta": [0.8],
                "selected_theta_hat": [0.6],
                "lambda_anchor": blt_lambda,
            },
        )
    )
    old_compute = _MODULE._compute_bnb_noise_multiplier_for_coeffs
    _MODULE._compute_bnb_noise_multiplier_for_coeffs = lambda **kwargs: 0.75
    try:
        rows = _MODULE.compute_comparison_rows(
            include_amplified=True,
            include_non_amplified=False,
            methods=["BLT"],
            amplified_backends=["balls_in_bins"],
            bnb_calibration_mode="optimistic",
            bnb_num_samples=64,
            bnb_num_workers=0,
        )
    finally:
        _MODULE._resolve_blt_amplified_accountant_coeffs = original
        _MODULE._compute_bnb_noise_multiplier_for_coeffs = old_compute

    amplified_row = next(row for row in rows if row.method == "BLT" and row.regime == "amplified")
    assert amplified_row.blt_buffers == _MODULE.BLT_AMPLIFIED_DIAGNOSTIC_ROW.bandwidth
    assert amplified_row.accounting_source == "opacus_blt_amplified_bnb_accountant_contract"


def test_blt_amplified_row_uses_caller_supplied_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_blt_amplified_accountant_coeffs(*, buffers, blt_lambda):
        captured["buffers"] = int(buffers)
        return (
            [1.0] + [0.0] * (int(_MODULE.TOTAL_STEPS) - 1),
            "normalized_forward_c_col",
            {
                "noise_multiplier_ref": 1.0,
                "blt_horizon": int(_MODULE.TOTAL_STEPS),
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 1,
                "selected_theta": [0.8] * int(buffers),
                "selected_theta_hat": [0.6] * int(buffers),
                "lambda_anchor": blt_lambda,
            },
        )

    monkeypatch.setattr(
        _MODULE,
        "_resolve_blt_amplified_accountant_coeffs",
        _fake_resolve_blt_amplified_accountant_coeffs,
    )
    monkeypatch.setattr(_MODULE, "_compute_bnb_noise_multiplier_for_coeffs", lambda **kwargs: 0.75)

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BLT"],
        paper_rows=[_MODULE.PaperRow("amplified", "BLT", float("nan"), 0.1, 8, 10.0)],
        amplified_backends=["balls_in_bins"],
        bnb_calibration_mode="optimistic",
        bnb_num_samples=64,
        bnb_num_workers=0,
    )

    amplified_row = next(row for row in rows if row.method == "BLT" and row.regime == "amplified")
    assert captured["buffers"] == 8
    assert amplified_row.bandwidth == 8
    assert amplified_row.blt_buffers == 8


def test_resolve_blt_amplified_accountant_coeffs_uses_forward_c_col(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_blt_fixed_batch_runtime_z_std",
        lambda *, buffers, blt_lambda=None: (
            0.5,
            {
                "noise_multiplier_ref": 1.0,
                "blt_horizon": int(_MODULE.TOTAL_STEPS),
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 1,
                "selected_theta": [0.8],
                "selected_theta_hat": [0.6],
                "lambda_anchor": blt_lambda,
            },
        ),
    )
    coeffs, source, meta = _MODULE._resolve_blt_amplified_accountant_coeffs(
        buffers=2,
        blt_lambda=0.4,
    )

    assert source == "normalized_forward_c_col"
    assert meta["selected_theta"] == [0.8]
    assert meta["selected_theta_hat"] == [0.6]
    assert len(coeffs) == int(_MODULE.TOTAL_STEPS)
    assert all(float(c) >= 0.0 for c in coeffs)


def test_parse_args_defaults_bifr_frac_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--methods",
            "BIFR",
        ],
    )
    args = _MODULE.parse_args()
    assert args.methods == ["BIFR"]
    assert args.bifr_frac is None


def test_parse_args_defaults_blt_lambda_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_bisr_paper_cifar_noise_multipliers.py",
            "--methods",
            "BLT",
        ],
    )
    args = _MODULE.parse_args()
    assert args.methods == ["BLT"]
    assert args.blt_lambda is None


def test_bnb_parallel_defaults_are_adaptive() -> None:
    assert _MODULE.BNB_NUM_WORKERS_DEFAULT == 8
    assert _MODULE.BNB_CHUNK_SIZE_DEFAULT is None
    assert _MODULE._resolve_effective_bnb_chunk_size(
        num_samples=10_000,
        chunk_size=None,
        num_workers=8,
    ) == 1250


def test_amplified_random_allocation_rows_are_absent_by_default() -> None:
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["DP-SGD"],
    )
    amplified_rows = [row for row in rows if row.regime == "amplified"]
    assert all(row.backend != "ra_fixed_bin_bnb" for row in amplified_rows)


def test_non_amplified_rows_are_absent_by_default() -> None:
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["DP-SGD"],
    )
    non_amplified_rows = [row for row in rows if row.regime == "non-amplified"]
    assert len(non_amplified_rows) == 1
    assert non_amplified_rows[0].backend == "skipped"
    assert non_amplified_rows[0].status == "skipped"
    assert non_amplified_rows[0].reason_code == "known_skip_non_amplified_disabled"


def test_non_amplified_rows_require_explicit_flag() -> None:
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=True,
        methods=["DP-SGD"],
    )
    non_amplified_rows = [row for row in rows if row.regime == "non-amplified"]
    assert len(non_amplified_rows) == 1
    assert non_amplified_rows[0].backend == "fixed_batch_prv"
    assert non_amplified_rows[0].status == "computed"


def test_blt_rows_are_not_included_unless_explicitly_requested() -> None:
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=None,
    )
    assert all(row.method != "BLT" for row in rows)


def test_bifr_rows_are_not_included_unless_explicitly_requested() -> None:
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=None,
    )
    assert all(row.method != "BIFR" for row in rows)


def test_non_amplified_blt_row_uses_nonpaper_fixed_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_blt_fixed_batch_runtime_z_std",
        lambda *, buffers, blt_lambda=None: (
            0.314 if blt_lambda is None else 0.3 + float(blt_lambda),
            {
                "noise_multiplier_ref": 1.27,
                "blt_horizon": 8,
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 3,
                "selected_theta": [0.8, 0.3],
                "selected_theta_hat": [0.6, 0.1],
                "lambda_anchor": blt_lambda,
            },
        ),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["BLT"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "BLT"
    assert row.regime == "non-amplified"
    assert row.backend == "blt"
    assert row.status == "computed"
    assert row.blt_lambda is None
    assert row.blt_selection_mode == "optimizer_selected"
    assert row.blt_buffers == 2
    assert row.blt_selected_candidate_index == 0
    assert row.blt_candidate_count == 3
    assert row.blt_selected_theta == [0.8, 0.3]
    assert row.blt_selected_theta_hat == [0.6, 0.1]
    assert row.reference_source == "fixed_batch_identity_gaussian_prv"
    assert row.reference_noise_multiplier == pytest.approx(1.7255103931826974)
    assert row.computed_noise_multiplier == pytest.approx(0.314)
    assert row.accounting_noise_multiplier == pytest.approx(1.27)
    assert row.accounting_source == "opacus_blt_fixed_batch_accountant_contract"
    assert row.comparison_noise_multiplier is not None
    assert row.comparison_source == "fixed_batch_identity_gaussian_prv"
    assert row.reason_code == "computed_opacus_blt_fixed_batch_contract"
    assert "Non-paper BLT optimizer-selected row" in row.notes


def test_non_amplified_blt_rows_honor_explicit_lambda_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_blt_fixed_batch_runtime_z_std",
        lambda *, buffers, blt_lambda=None: (
            0.3 + float(blt_lambda),
            {
                "noise_multiplier_ref": 1.27 + float(blt_lambda),
                "blt_horizon": 8,
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 3,
                "selected_theta": [float(blt_lambda)],
                "selected_theta_hat": [0.75 * float(blt_lambda)],
                "lambda_anchor": blt_lambda,
            },
        ),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["BLT"],
        blt_lambda=0.6,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.blt_lambda == pytest.approx(0.6)
    assert row.computed_noise_multiplier == pytest.approx(0.9)
    assert row.accounting_noise_multiplier == pytest.approx(1.87)


def test_non_amplified_bifr_rows_use_nonpaper_fixed_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_method_fixed_batch_sensitivity(
        method: str,
        bands: int,
        bifr_frac: float | None = None,
    ) -> float:
        assert bands == 4
        if method == "BIFR":
            assert bifr_frac == pytest.approx(0.25)
            return 2.5
        if method == "BSR":
            return 2.0
        raise AssertionError(f"unexpected method {method}")

    monkeypatch.setattr(_MODULE, "_method_fixed_batch_sensitivity", _fake_method_fixed_batch_sensitivity)
    monkeypatch.setattr(
        _MODULE,
        "_fixed_batch_base_sigma",
        lambda *, backend: 1.0 if backend == "prv" else 1.5,
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["BIFR"],
        bifr_frac=0.25,
    )

    assert len(rows) == 2
    prv_row = next(row for row in rows if row.backend == "prv")
    rdp_row = next(row for row in rows if row.backend == "rdp")
    for row, base_sigma in ((prv_row, 1.0), (rdp_row, 1.5)):
        assert row.method == "BIFR"
        assert row.regime == "non-amplified"
        assert row.status == "computed"
        assert row.parity_status == "non_paper_diagnostic"
        assert row.bifr_frac == pytest.approx(0.25)
        assert row.paper_noise_multiplier is None
        assert row.computed_noise_multiplier == pytest.approx(base_sigma * 2.5)
        assert row.reference_noise_multiplier == pytest.approx(base_sigma * 2.0)
        assert row.reference_source == "bsr_half_slice_fixed_batch_contract"
        assert row.comparison_noise_multiplier == pytest.approx(base_sigma * (_MODULE.FIXED_BATCH_K ** 0.5))
        assert row.comparison_source == f"fixed_batch_identity_gaussian_{row.backend}"
        assert row.reason_code == "computed_bifr_fixed_batch_exact_contract"
        assert "exact finite-horizon BIFR factor-side family" in row.notes
        assert "BSR half-slice" in row.notes


def test_non_amplified_bifr_unstable_exact_slice_is_known_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_method_fixed_batch_sensitivity(
        method: str,
        bands: int,
        bifr_frac: float | None = None,
    ) -> float:
        assert bands == 4
        if method == "BIFR":
            raise ValueError("unstable_exact_bifr_slice: test fixture")
        if method == "BSR":
            return 2.0
        raise AssertionError(f"unexpected method {method}")

    monkeypatch.setattr(_MODULE, "_method_fixed_batch_sensitivity", _fake_method_fixed_batch_sensitivity)
    monkeypatch.setattr(
        _MODULE,
        "_fixed_batch_base_sigma",
        lambda *, backend: 1.0 if backend == "prv" else 1.5,
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["BIFR"],
        bifr_frac=0.625,
    )

    assert len(rows) == 2
    for row in rows:
        assert row.method == "BIFR"
        assert row.status == "skipped"
        assert row.parity_status == "known_skipped"
        assert row.reason_code == "known_unstable_exact_bifr_slice"
        assert row.computed_noise_multiplier is None


def test_non_amplified_bifr_rows_default_to_canonical_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_method_fixed_batch_sensitivity(
        method: str,
        bands: int,
        bifr_frac: float | None = None,
    ) -> float:
        assert bands == 4
        if method == "BIFR":
            assert bifr_frac is not None
            return 10.0 + float(bifr_frac)
        if method == "BSR":
            return 2.0
        raise AssertionError(f"unexpected method {method}")

    monkeypatch.setattr(_MODULE, "_method_fixed_batch_sensitivity", _fake_method_fixed_batch_sensitivity)
    monkeypatch.setattr(
        _MODULE,
        "_fixed_batch_base_sigma",
        lambda *, backend: 1.0 if backend == "prv" else 1.5,
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        methods=["BIFR"],
    )

    assert len(rows) == 8
    fracs = sorted({row.bifr_frac for row in rows})
    assert fracs == [0.0, 0.25, 0.5, 1.0]
    for frac in fracs:
        frac_rows = [row for row in rows if row.bifr_frac == pytest.approx(frac)]
        assert len(frac_rows) == 2
        assert {row.backend for row in frac_rows} == {"prv", "rdp"}


def test_bifr_fixed_batch_sensitivity_uses_disjoint_fallback_when_bsr_guard_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coeffs = [1.0, 2.0, 3.0, 4.0]
    k_eff = min(_MODULE.FIXED_BATCH_K, (_MODULE.TOTAL_STEPS - 1) // _MODULE.FIXED_BATCH_B + 1)
    expected = (k_eff ** 0.5) * sum(c * c for c in coeffs) ** 0.5

    class _FakeFamily:
        def fixed_batch_sensitivity(self, *, max_participations, min_separation, allow_disjoint_fallback):
            assert max_participations == _MODULE.FIXED_BATCH_K
            assert min_separation == _MODULE.FIXED_BATCH_B
            assert allow_disjoint_fallback is True
            return expected

    monkeypatch.setattr(
        _MODULE,
        "build_bifr_exact_factor_family_from_sgd_workload",
        lambda **kwargs: _FakeFamily(),
    )
    _MODULE._resolve_bifr_fixed_batch_sensitivity.cache_clear()
    try:
        got = _MODULE._resolve_bifr_fixed_batch_sensitivity(bands=4, frac=1.0)
    finally:
        _MODULE._resolve_bifr_fixed_batch_sensitivity.cache_clear()

    assert got == pytest.approx(expected)


def test_non_amplified_bifr_full_endpoint_rows_compute_under_disjoint_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sensitivity = (_MODULE.FIXED_BATCH_K ** 0.5) * (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2) ** 0.5

    class _FakeFamily:
        def fixed_batch_sensitivity(self, *, max_participations, min_separation, allow_disjoint_fallback):
            assert max_participations == _MODULE.FIXED_BATCH_K
            assert min_separation == _MODULE.FIXED_BATCH_B
            assert allow_disjoint_fallback is True
            return expected_sensitivity

    monkeypatch.setattr(
        _MODULE,
        "build_bifr_exact_factor_family_from_sgd_workload",
        lambda **kwargs: _FakeFamily(),
    )
    monkeypatch.setattr(
        _MODULE,
        "_fixed_batch_base_sigma",
        lambda *, backend: 1.0 if backend == "prv" else 1.5,
    )
    monkeypatch.setitem(_MODULE._METHOD_FIXED_BATCH_SENSITIVITY_RESOLVERS, "BSR", lambda bands: 2.0)
    _MODULE._resolve_bifr_fixed_batch_sensitivity.cache_clear()
    try:
        rows = _MODULE.compute_comparison_rows(
            include_amplified=False,
            include_non_amplified=False,
            methods=["BIFR"],
            bifr_frac=1.0,
        )
    finally:
        _MODULE._resolve_bifr_fixed_batch_sensitivity.cache_clear()

    assert len(rows) == 2
    prv_row = next(row for row in rows if row.backend == "prv")
    rdp_row = next(row for row in rows if row.backend == "rdp")
    for row, base_sigma in ((prv_row, 1.0), (rdp_row, 1.5)):
        assert row.status == "computed"
        assert row.bifr_frac == pytest.approx(1.0)
        assert row.reason_code == "computed_bifr_fixed_batch_exact_contract"
        assert row.computed_noise_multiplier == pytest.approx(base_sigma * expected_sensitivity)
        assert row.reference_noise_multiplier == pytest.approx(base_sigma * 2.0)
        assert row.comparison_noise_multiplier == pytest.approx(base_sigma * (_MODULE.FIXED_BATCH_K ** 0.5))


def test_print_summary_calls_out_blt_diagnostic(capsys: pytest.CaptureFixture[str]) -> None:
    report = {
        "metadata": {
            "target_epsilon": 9.0,
            "target_delta": 1e-5,
            "batch_size": 512,
            "total_steps": 980,
            "include_amplified": False,
            "include_non_amplified": False,
            "include_amplified_deterministic": False,
            "amplified_backends": [],
            "blt_selection_policy": "optimizer_selected_default_v1",
            "baseline_interpretation": {
                "cyclic_poisson": {
                    "contract": {
                        "q_eff": "bands * sample_rate",
                        "cycles": "ceil(steps / bands)",
                    }
                },
                "structured_amplified": {
                    "backends": ["balls_in_bins", "b_min_sep", "ra_fixed_bin_bnb"],
                },
                "inverse_family_boundary": {
                    "notes": "inverse rows are not apples-to-apples with structured accountant rows",
                },
            },
        },
        "summary": {
            "parity_status_counts": {"computed_close": 1},
            "row_count": 1,
            "best_paper_rmse_by_family": {
                "BLT": {
                    "parameter": "lambda",
                    "best_by_backend": {
                        "blt": {
                            "backend": "blt",
                            "lambda": 0.4,
                            "paper_rmse": 0.8125,
                        }
                    },
                }
            },
            "blt": {
                "selection_modes": ["optimizer_selected"],
                "buffers": [2],
                "rows": [],
                "paper_rmse": {
                    "best_by_backend": {
                        "blt": {
                            "backend": "blt",
                            "lambda": 0.4,
                            "paper_rmse": 0.8125,
                        }
                    }
                },
            },
        },
        "rows": [
            {
                    "regime": "non-amplified",
                    "method": "BLT",
                    "backend": "blt",
                    "blt_selection_mode": "optimizer_selected",
                    "blt_buffers": 2,
                    "status": "computed",
                "parity_status": "computed_close",
                "paper_noise_multiplier": None,
                "computed_noise_multiplier": 0.390625,
                "comparison_noise_multiplier": 1.7255103931826974,
                "comparison_relative_error": 0.77,
                "sensitivity": None,
                "absolute_error": None,
                "relative_error": None,
            }
        ],
    }

    _MODULE.print_summary(report)

    out = capsys.readouterr().out
    assert "Methods in report: BLT" in out
    assert "BLT selection: policy=optimizer_selected_default_v1 modes=optimizer_selected buffers=2" in out
    assert "Paper RMSE family decisions: BLT" in out
    assert "BLT paper RMSE best: blt:lambda=0.4@rmse=0.8125" in out
    assert "Baseline semantics: cyclic_poisson = reduced sampled-Gaussian baseline" in out
    assert "Structured routes: balls_in_bins, b_min_sep, ra_fixed_bin_bnb = structured amplified accountant routes" in out
    assert "Inverse-family boundary: inverse rows are not apples-to-apples with structured accountant rows" in out
    assert "Diagnostics:" in out
    assert "BLT (non-amplified/blt)" in out


def test_print_summary_surfaces_param_column_and_amplified_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "metadata": {
            "target_epsilon": 9.0,
            "target_delta": 1e-5,
            "batch_size": 512,
            "total_steps": 980,
            "include_amplified": True,
            "include_non_amplified": True,
            "include_amplified_deterministic": False,
            "amplified_backends": ["balls_in_bins", "b_min_sep"],
            "amplified_execution_mode": "calibrate",
            "baseline_interpretation": {},
            "blt_selection_policy": "optimizer_selected_default_v1",
        },
        "summary": {
            "parity_status_counts": {"non_paper_diagnostic": 2},
            "row_count": 2,
            "paper_rmse": {"computed_rows": 0, "missing_rows": 2, "row_count": 2},
            "best_paper_rmse_by_family": {},
            "blt": {"selection_modes": ["optimizer_selected"], "buffers": [2]},
        },
        "rows": [
            {
                "regime": "non-amplified",
                "method": "BLT",
                "backend": "blt",
                "blt_selection_mode": "optimizer_selected",
                "blt_buffers": 2,
                "bifr_frac": None,
                "status": "computed",
                "parity_status": "non_paper_diagnostic",
                "paper_noise_multiplier": None,
                "computed_noise_multiplier": 0.8,
                "comparison_noise_multiplier": 1.7,
                "comparison_relative_error": 0.53,
                "sensitivity": None,
                "absolute_error": None,
                "relative_error": None,
            },
            {
                "regime": "amplified",
                "method": "BLT",
                "backend": "balls_in_bins",
                "blt_selection_mode": "optimizer_selected",
                "blt_buffers": 2,
                "bifr_frac": None,
                "status": "computed",
                "parity_status": "non_paper_diagnostic",
                "paper_noise_multiplier": None,
                "computed_noise_multiplier": 0.43,
                "comparison_noise_multiplier": 0.57,
                "comparison_relative_error": 0.24,
                "sensitivity": 1.0,
                "absolute_error": None,
                "relative_error": None,
            },
        ],
    }

    _MODULE.print_summary(report)

    out = capsys.readouterr().out
    assert "param" in out
    assert "buffers=2" in out
    amplified_idx = out.index("amplified       BLT")
    non_amplified_idx = out.index("non-amplified   BLT")
    assert amplified_idx < non_amplified_idx
    assert "normalized accountant c_col norm (=1)" in out


def test_print_summary_calls_out_bifr_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    report = {
        "metadata": {
            "target_epsilon": 9.0,
            "target_delta": 1e-5,
            "batch_size": 512,
            "total_steps": 980,
            "include_amplified": False,
            "include_non_amplified": False,
            "include_amplified_deterministic": False,
            "amplified_backends": [],
            "amplified_execution_mode": "calibrate",
            "baseline_interpretation": {},
            "bifr_sweep_policy": "canonical_default_v1",
        },
        "summary": {
            "parity_status_counts": {"non_paper_diagnostic": 8},
            "row_count": 8,
            "paper_rmse": {"computed_rows": 8, "missing_rows": 0, "row_count": 8},
            "best_paper_rmse_by_family": {
                "BIFR": {
                    "parameter": "frac",
                    "best_by_backend": {
                        "prv": {
                            "backend": "prv",
                            "frac": 0.25,
                            "paper_rmse": 1.2345,
                        }
                    },
                }
            },
            "bifr": {
                "fracs": [0.0, 0.25, 0.5, 1.0],
                "anchor_slices": {
                    "identity": 0.0,
                    "bsr_half_slice": 0.5,
                    "full_workload": 1.0,
                },
                "paper_rmse": {
                    "best_by_backend": {
                        "prv": {
                            "backend": "prv",
                            "frac": 0.25,
                            "paper_rmse": 1.2345,
                            "anchors": {
                                "identity": {
                                    "frac": 0.0,
                                    "paper_rmse": 1.5,
                                    "delta_to_best": 0.2655,
                                },
                                "bsr_half_slice": {
                                    "frac": 0.5,
                                    "paper_rmse": 1.7,
                                    "delta_to_best": 0.4655,
                                },
                                "full_workload": {
                                    "frac": 1.0,
                                    "paper_rmse": 2.0,
                                    "delta_to_best": 0.7655,
                                },
                            },
                        }
                    }
                },
                "rows": [],
            },
        },
        "rows": [],
    }

    _MODULE.print_summary(report)

    out = capsys.readouterr().out
    assert "BIFR sweep: policy=canonical_default_v1 fracs=0,0.25,0.5,1" in out
    assert "Paper RMSE family decisions: BIFR" in out
    assert (
        "BIFR paper RMSE best: "
        "prv:frac=0.25@rmse=1.2345 (bsr_half_slice:+0.4655, full_workload:+0.7655, identity:+0.2655)"
        in out
    )
    assert "Paper RMSE coverage: computed=8 missing=0 rows=8" in out


def test_build_report_exposes_baseline_interpretation_metadata() -> None:
    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BLT"],
        bnb_calibration_mode="optimistic",
        bnb_chunk_size=None,
        bnb_num_workers=1,
        bnb_num_samples=1000,
        bnb_probe_num_samples=1000,
        bnb_backend="cpu",
        bnb_device=None,
        bnb_distributed_mode=None,
        amplified_execution_mode="calibrate",
        amplified_verify_budgets=None,
        distributed_launch=_MODULE.DistributedLaunch(
            initialized=False,
            backend=None,
            device=None,
            distributed_runtime=False,
            rank=0,
            local_rank=0,
            world_size=1,
        ),
    )

    baseline = report["metadata"]["baseline_interpretation"]
    assert baseline["cyclic_poisson"]["kind"] == "reduced_sampled_gaussian_baseline"
    assert baseline["cyclic_poisson"]["contract"]["q_eff"] == "bands * sample_rate"
    assert baseline["structured_amplified"]["backends"] == [
        "balls_in_bins",
        "b_min_sep",
        "ra_fixed_bin_bnb",
    ]
    assert "not apples-to-apples" in baseline["inverse_family_boundary"]["notes"]
    assert baseline["sigma_interpretation_contract"]["amplified"].startswith(
        "Amplified direct-verification rows use the paper-table sigma directly"
    )
    assert report["metadata"]["amplified_execution_mode"] == "calibrate"
    assert report["metadata"]["amplified_verify_budgets"] == [200000, 500000, 1000000]


def test_build_report_forwards_bnb_overrides_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compute_comparison_rows(**kwargs):
        captured.update(kwargs)
        return [
            _MODULE._comparison_row(
                row=_MODULE.BLT_DIAGNOSTIC_ROW,
                backend="blt",
                status="computed",
                computed=0.390625,
                sensitivity=None,
                reference_noise_multiplier=0.390625,
                reference_source="opacus_blt_optimized_fixed_batch_contract",
                reason_code="test",
                notes="test row",
            )
        ]

    monkeypatch.setattr(_MODULE, "compute_comparison_rows", _fake_compute_comparison_rows)
    _MODULE.compute_comparison_rows.last_amplified_bsr_scale_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandinvmf_accountant_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandmf_matrix_family_probe = {}
    _MODULE.compute_comparison_rows.last_non_amplified_bandinvmf_probe = None

    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BLT"],
        bnb_calibration_mode="optimistic",
        bnb_chunk_size=111,
        bnb_num_workers=7,
        bnb_num_samples=12345,
        bnb_backend="cpu",
        bnb_device="cpu",
        bnb_distributed_mode="chunk_shard",
        distributed_launch=_MODULE.DistributedLaunch(
            initialized=False,
            backend=None,
            device=None,
            distributed_runtime=False,
            rank=0,
            local_rank=0,
            world_size=1,
        ),
    )

    assert captured["bnb_calibration_mode"] == "optimistic"
    assert captured["bnb_chunk_size"] == 111
    assert captured["bnb_num_workers"] == 7
    assert captured["bnb_num_samples"] == 12345
    assert captured["bnb_backend"] == "cpu"
    assert captured["bnb_device"] == "cpu"
    assert captured["bnb_distributed_mode"] == "chunk_shard"
    assert captured["bnb_distributed_dp_runtime"] is False
    assert captured["fixed_bin_loss_discretization"] is None
    assert report["rows"][0]["method"] == "BLT"


def test_build_report_forwards_fixed_bin_loss_discretization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compute_comparison_rows(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(_MODULE, "compute_comparison_rows", _fake_compute_comparison_rows)
    _MODULE.compute_comparison_rows.last_amplified_bsr_scale_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandinvmf_accountant_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandmf_matrix_family_probe = {}
    _MODULE.compute_comparison_rows.last_non_amplified_bandinvmf_probe = None

    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=True,
        amplified_backends=["ra_fixed_bin_bnb"],
        skip_bandinvmf=False,
        methods=["BISR"],
        fixed_bin_loss_discretization=0.02,
    )

    assert captured["fixed_bin_loss_discretization"] == pytest.approx(0.02)
    assert report["metadata"]["fixed_bin_loss_discretization"] == pytest.approx(0.02)


def test_direct_amplified_bsr_verification_rows_record_single_verify_and_evr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", lambda method, bands: [1.0])
    monkeypatch.setattr(
        _MODULE,
        "_method_amplified_accountant_coeffs",
        lambda method, bands: ([1.0], "raw"),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda *, method, bands: (
            1.23,
            {
                "cyclic_sensitivity_scale": 1.5,
                "effective_noise_multiplier": 0.82,
                "q_eff": 0.04,
                "cycles": 245,
            },
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_bnb_delta",
        lambda **kwargs: _MODULE.SingleVerifyProbeResult(
            status="computed",
            delta_estimate=8e-6,
            delta_upper_confidence_bound=9e-6,
            error_probability=1e-6,
            notes="single verify ok",
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_opacus_bnb_base_delta",
        lambda **kwargs: _MODULE.OpacusBNBProbeResult(
            status="computed",
            base_delta=7e-6,
            num_samples=int(kwargs["num_samples"]),
            feasible_budget=int(kwargs["num_samples"]),
            notes="evr ok",
        ),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BSR"],
        amplified_execution_mode="verify_table_sigma",
        amplified_verify_budgets=[200000],
    )

    bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "balls_in_bins"]
    assert len(bnb_rows) == 1
    row = bnb_rows[0]
    assert row.computed_noise_multiplier == pytest.approx(_paper_row("BSR").paper_noise_multiplier)
    assert row.amplified_execution_mode == "verify_table_sigma"
    assert row.verification_budget == 200000
    assert row.single_verify_status == "computed"
    assert row.single_verify_delta_upper_bound == pytest.approx(9e-6)
    assert row.opacus_bnb_status == "computed"
    assert row.opacus_bnb_base_delta == pytest.approx(7e-6)
    assert row.reason_code == "computed_direct_paper_sigma_verification"


def test_direct_amplified_bandmf_verification_rows_preserve_family_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", lambda method, bands: [1.0])
    monkeypatch.setattr(
        _MODULE,
        "_method_amplified_accountant_coeffs",
        lambda method, bands: ([1.0], "raw"),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda *, method, bands: (
            1.23,
            {
                "cyclic_sensitivity_scale": 1.5,
                "effective_noise_multiplier": 0.82,
                "q_eff": 0.04,
                "cycles": 245,
            },
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_bnb_delta",
        lambda **kwargs: _MODULE.SingleVerifyProbeResult(
            status="computed",
            delta_estimate=8e-6,
            delta_upper_confidence_bound=9e-6,
            error_probability=1e-6,
            notes="single verify ok",
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_opacus_bnb_base_delta",
        lambda **kwargs: _MODULE.OpacusBNBProbeResult(
            status="computed",
            base_delta=7e-6,
            num_samples=int(kwargs["num_samples"]),
            feasible_budget=int(kwargs["num_samples"]),
            notes="evr ok",
        ),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["Band-MF"],
        amplified_execution_mode="verify_table_sigma",
        amplified_verify_budgets=[200000],
    )

    bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "balls_in_bins"]
    assert len(bnb_rows) == 1
    row = bnb_rows[0]
    assert row.reason_code == "computed_direct_paper_sigma_verification"
    assert row.parity_status == "computed_close"
    assert row.bandmf_family == "normalized_equal_column_norm"
    assert row.verification_budget == 200000


def test_direct_amplified_budget_sweep_retains_budget_per_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", lambda method, bands: [1.0])
    monkeypatch.setattr(
        _MODULE,
        "_method_amplified_accountant_coeffs",
        lambda method, bands: ([1.0], "raw"),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda **kwargs: (1.23, {"cyclic_sensitivity_scale": 1.5, "effective_noise_multiplier": 0.82, "q_eff": 0.04, "cycles": 245}),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_bnb_delta",
        lambda **kwargs: _MODULE.SingleVerifyProbeResult(
            status="computed",
            delta_estimate=8e-6,
            delta_upper_confidence_bound=9e-6,
            error_probability=1e-6,
            notes="single verify ok",
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_opacus_bnb_base_delta",
        lambda **kwargs: _MODULE.OpacusBNBProbeResult(
            status="computed",
            base_delta=7e-6,
            num_samples=int(kwargs["num_samples"]),
            feasible_budget=int(kwargs["num_samples"]),
            notes="evr ok",
        ),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BSR"],
        amplified_execution_mode="verify_table_sigma",
        amplified_verify_budgets=[200000, 500000],
    )

    bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "balls_in_bins"]
    assert [row.verification_budget for row in bnb_rows] == [200000, 500000]
    assert all(
        row.computed_noise_multiplier == pytest.approx(_paper_row("BSR").paper_noise_multiplier)
        for row in bnb_rows
    )


def test_direct_amplified_bisr_and_bandinvmf_fail_with_explicit_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", lambda method, bands: [1.0])

    def _accountant_coeffs(method: str, bands: int):
        if method == "BISR":
            return ([-1.0], "raw")
        if method == "Band-Inv-MF":
            raise RuntimeError(
                "BandInvMF optimization produced a non-finite final candidate"
            )
        return ([1.0], "raw")

    monkeypatch.setattr(_MODULE, "_method_amplified_accountant_coeffs", _accountant_coeffs)

    bisr_rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BISR"],
        amplified_execution_mode="verify_table_sigma",
        amplified_verify_budgets=[200000, 500000],
    )
    bisr_failures = [row for row in bisr_rows if row.backend == "balls_in_bins"]
    assert len(bisr_failures) == 2
    assert all(row.reason_code == "known_missing_signed_c_contract" for row in bisr_failures)
    assert [row.verification_budget for row in bisr_failures] == [200000, 500000]

    bandinvmf_rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["Band-Inv-MF"],
        amplified_execution_mode="verify_table_sigma",
        amplified_verify_budgets=[200000, 500000],
    )
    bandinvmf_failures = [row for row in bandinvmf_rows if row.backend == "balls_in_bins"]
    assert len(bandinvmf_failures) == 2
    assert all(
        row.reason_code == "known_missing_bandinvmf_optimizer_instability"
        for row in bandinvmf_failures
    )
    assert all(
        "BandInvMF optimization produced a non-finite final candidate" in row.notes
        for row in bandinvmf_failures
    )


def test_amplified_bifr_rows_use_real_bnb_path_and_preserve_selected_frac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resolve_row_paper_rmse = _MODULE._resolve_row_paper_rmse

    def _fake_coeffs(method: str, bands: int, bifr_frac: float | None = None, **kwargs):
        assert method == "BIFR"
        assert bifr_frac is not None
        return [1.0, float(bifr_frac)]

    def _fake_accountant_coeffs(method: str, bands: int, bifr_frac: float | None = None, **kwargs):
        assert method == "BIFR"
        assert bifr_frac is not None
        return ([1.0, float(bifr_frac)], "abs_factor_c_col")

    def _fake_sigma(**kwargs):
        coeffs = list(kwargs["coeffs"])
        return 10.0 - float(coeffs[1])

    def _fake_paper_rmse(row, **kwargs):
        frac = 0.0 if row.bifr_frac is None else float(row.bifr_frac)
        if row.method == "BIFR" and row.regime == "amplified":
            return (
                20.0 - (10.0 * frac),
                "direct_inverse_family_rmse_from_bifr_inv_coeffs",
                None,
            )
        return original_resolve_row_paper_rmse(row)

    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", _fake_coeffs)
    monkeypatch.setattr(_MODULE, "_method_amplified_accountant_coeffs", _fake_accountant_coeffs)
    monkeypatch.setattr(_MODULE, "_compute_bnb_noise_multiplier_for_coeffs", _fake_sigma)
    monkeypatch.setattr(_MODULE, "_resolve_row_paper_rmse", _fake_paper_rmse)

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BIFR"],
        amplified_backends=["balls_in_bins"],
        paper_rows=_MODULE.build_amplified_bnb_p_sweep_rows(bandwidth_grid=[2]),
    )

    bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "balls_in_bins"]
    assert len(bnb_rows) == 1
    row = bnb_rows[0]
    assert row.status == "computed"
    assert row.reason_code == "computed_bnb_accountant_bifr_factor_c_col"
    assert row.bifr_frac == pytest.approx(1.0)
    assert row.opacus_bnb_c_col_scale == "abs_factor_c_col"
    assert "Unexpected amplified method: BIFR" not in row.notes


def test_bisr_amplified_accountant_coeffs_use_exact_factor_column() -> None:
    workload = _MODULE._resolve_optimizer_workload(momentum=0.0, weight_decay=0.0)
    accountant_coeffs, accountant_source = _MODULE._resolve_bisr_amplified_accountant_coeffs(
        4,
        optimizer_workload=workload,
    )
    inverse_coeffs = _MODULE.generate_bisr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=0.0,
    )
    exact_factor_coeffs = _MODULE.derive_bisr_factor_coeffs_from_inverse_coeffs(
        coeffs=inverse_coeffs,
        steps=int(_MODULE.TOTAL_STEPS),
    )

    assert accountant_source == "abs_factor_c_col"
    assert accountant_coeffs == pytest.approx([abs(float(c)) for c in exact_factor_coeffs])
    assert len(accountant_coeffs) == len(exact_factor_coeffs)

def test_amplified_cycle_endpoint_cyclic_rows_are_known_skips_not_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_bnb_noise_multiplier_for_coeffs",
        lambda **kwargs: 1.0,
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cyclic helper should not be called")),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BSR"],
        amplified_backends=["balls_in_bins"],
        paper_rows=_MODULE.build_amplified_bnb_p_sweep_rows(
            bandwidth_grid=[_MODULE.STEPS_PER_EPOCH]
        ),
        bnb_num_workers=0,
        bnb_num_samples=16,
    )

    cyclic_rows = [row for row in rows if row.backend == "cyclic_poisson"]
    assert len(cyclic_rows) == 1
    assert all(row.status == "skipped" for row in cyclic_rows)
    assert all(row.reason_code == "known_unsupported_cyclic_endpoint_contract" for row in cyclic_rows)


def test_amplified_bifr_unstable_exact_slice_is_known_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_coeffs(method: str, bands: int, bifr_frac: float | None = None):
        assert method == "BIFR"
        assert bands == 2
        assert bifr_frac == pytest.approx(0.625)
        raise ValueError("unstable_exact_bifr_slice: test fixture")

    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", _fake_coeffs)
    monkeypatch.setattr(
        _MODULE,
        "_method_amplified_accountant_coeffs",
        lambda method, bands, bifr_frac=None: (_fake_coeffs(method, bands, bifr_frac), "unused"),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        methods=["BIFR"],
        bifr_frac=0.625,
        amplified_backends=["balls_in_bins"],
        paper_rows=_MODULE.build_amplified_bnb_p_sweep_rows(bandwidth_grid=[2]),
    )

    amplified_rows = [row for row in rows if row.regime == "amplified"]
    assert len(amplified_rows) == 1
    row = amplified_rows[0]
    assert row.method == "BIFR"
    assert row.status == "skipped"
    assert row.parity_status == "known_skipped"
    assert row.reason_code == "known_unstable_exact_bifr_slice"
    assert row.computed_noise_multiplier is None


def test_non_amplified_bandinvmf_optimizer_instability_uses_specific_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fixed_batch = _MODULE._method_fixed_batch_sensitivity

    def _fixed_batch(method: str, bands: int) -> float:
        if method == "Band-Inv-MF":
            raise RuntimeError("non-finite inverse-side runtime coefficients")
        return original_fixed_batch(method, bands)

    monkeypatch.setattr(_MODULE, "_method_fixed_batch_sensitivity", _fixed_batch)

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=True,
        methods=["Band-Inv-MF"],
    )

    failures = [
        row
        for row in rows
        if row.method == "Band-Inv-MF" and row.backend in {"prv", "rdp"}
    ]
    assert len(failures) == 2
    assert all(
        row.reason_code == "known_missing_bandinvmf_optimizer_instability"
        for row in failures
    )
    assert all("non-finite inverse-side runtime coefficients" in row.notes for row in failures)


def test_blt_runtime_helper_delegates_to_mf_report_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "compute_blt_fixed_batch_report_surface",
        lambda **kwargs: SimpleNamespace(
            computed_noise_multiplier=0.314,
            accounting_noise_multiplier=1.27,
            blt_horizon=8,
            blt_min_separation=4,
            blt_max_participations=2,
            selected_candidate_index=0,
            candidate_count=3,
            selected_theta=[0.8, 0.3],
            selected_theta_hat=[0.6, 0.1],
        ),
    )

    sigma, meta = _MODULE._compute_blt_fixed_batch_runtime_z_std(buffers=2)

    assert sigma == pytest.approx(0.314)
    assert meta["noise_multiplier_ref"] == pytest.approx(1.27)
    assert meta["blt_horizon"] == 8


def test_amplified_random_allocation_rows_require_the_new_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=True,
        methods=["DP-SGD"],
    )

    bridge_rows = [row for row in rows if row.regime == "amplified" and row.backend == "ra_fixed_bin_bnb"]
    assert len(bridge_rows) == 1
    assert bridge_rows[0].parity_status == "known_missing"


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
        include_non_amplified=False,
        include_amplified_deterministic=True,
        methods=["DP-SGD"],
    )

    bridge_rows = [row for row in rows if row.regime == "amplified" and row.backend == "ra_fixed_bin_bnb"]
    assert len(bridge_rows) == 1
    row = bridge_rows[0]
    assert row.parity_status == "known_missing"
    assert row.reason_code == "known_missing_fixed_bin_bridge_quantitative_window_realization"
    assert row.source_law_kind == "balls_in_bins_fixed_bin"
    assert row.accountant_engine_kind == "deterministic_random_allocation"
    assert row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"

    amplified_bnb_rows = [
        row
        for row in rows
        if row.regime == "amplified" and row.backend in {"balls_in_bins", "b_min_sep"}
    ]
    assert len(amplified_bnb_rows) == 1
    bnb_row = amplified_bnb_rows[0]
    assert bnb_row.backend in {"balls_in_bins", "b_min_sep"}
    assert "ra_fixed_bin_bnb" not in bnb_row.notes


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
        include_non_amplified=False,
        include_amplified_deterministic=True,
        methods=["DP-SGD", "BSR", "BISR", "Band-MF", "Band-Inv-MF"],
    )

    bridge_rows = [row for row in rows if row.regime == "amplified" and row.backend == "ra_fixed_bin_bnb"]
    assert {row.method for row in bridge_rows} == {
        "DP-SGD",
        "BSR",
        "BISR",
        "Band-MF",
        "Band-Inv-MF",
    }
    for row in bridge_rows:
        assert row.parity_status == "known_missing"
        assert row.source_law_kind == "balls_in_bins_fixed_bin"
        assert row.accountant_engine_kind == "deterministic_random_allocation"
        assert row.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"
        assert row.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"


def test_amplified_banded_methods_include_cyclic_poisson_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda *, method, bands: (
            1.23,
            {
                "cyclic_sensitivity_scale": 1.5,
                "effective_noise_multiplier": 0.82,
                "q_eff": 0.04,
                "cycles": 245,
            },
        ),
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
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["BSR"],
    )

    cyclic_rows = [
        row for row in rows if row.regime == "amplified" and row.backend == "cyclic_poisson"
    ]
    assert len(cyclic_rows) == 1
    row = cyclic_rows[0]
    assert row.method == "BSR"
    assert row.status == "computed"
    assert row.computed_noise_multiplier == pytest.approx(1.23)
    assert row.source_law_kind == "cyclic_poisson"
    assert row.accountant_engine_kind == "cyclic_gaussian_reduction"
    assert row.route == "cyclic_poisson_mf_accountant"
    assert row.cyclic_sensitivity_scale == pytest.approx(1.5)
    assert row.effective_noise_multiplier == pytest.approx(0.82)
    assert row.q_eff == pytest.approx(0.04)
    assert row.cycles == 245

    bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "balls_in_bins"]
    assert len(bnb_rows) == 1
    bnb_row = bnb_rows[0]
    assert bnb_row.comparison_noise_multiplier == pytest.approx(1.23)
    assert bnb_row.comparison_source == "cyclic_poisson"
    assert bnb_row.comparison_relative_error is not None


def test_inverse_family_cyclic_poisson_rows_are_marked_local_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda *, method, bands: (
            1.23,
            {
                "cyclic_sensitivity_scale": 1.5,
                "effective_noise_multiplier": 0.82,
                "q_eff": 0.04,
                "cycles": 245,
            },
        ),
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
    monkeypatch.setattr(
        _MODULE,
        "_compute_amplified_bisr_accountant_probe",
        lambda **kwargs: {
                "accountant_coeff_source": "stub",
                "accountant_first_coeff": 1.0,
                "accountant_second_coeff": 0.1,
                "accountant_coeff_l2": 1.0,
                "accountant_coeff_norm": 1.0,
                "factor_col_l2": 1.0,
                "inverse_runtime_l2": 1.0,
                "factor_runtime_sigma_ratio": 1.0,
            },
        )
    monkeypatch.setattr(
        _MODULE,
        "_compute_amplified_bandinvmf_accountant_probe",
        lambda **kwargs: {
                "accountant_coeff_source": "stub",
                "accountant_first_coeff": 1.0,
                "accountant_second_coeff": 0.1,
                "accountant_coeff_l2": 1.0,
                "accountant_coeff_norm": 1.0,
                "factor_col_l2": 1.0,
                "inverse_runtime_l2": 1.0,
                "factor_runtime_sigma_ratio": 1.0,
            },
        )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["BISR", "Band-Inv-MF"],
        amplified_backends=["balls_in_bins"],
    )

    bisr_row = next(
        row for row in rows if row.regime == "amplified" and row.method == "BISR" and row.backend == "cyclic_poisson"
    )
    assert "local baseline only" in bisr_row.notes
    assert "different privacy objects" in bisr_row.notes
    assert "|C^p[:,0]|" in bisr_row.notes

    bandinvmf_row = next(
        row
        for row in rows
        if row.regime == "amplified" and row.method == "Band-Inv-MF" and row.backend == "cyclic_poisson"
    )
    assert "local baseline only" in bandinvmf_row.notes
    assert "different privacy objects" in bandinvmf_row.notes
    assert "|C[:,0]|" in bandinvmf_row.notes


def test_amplified_b_min_sep_rows_compare_against_cyclic_poisson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_cyclic_poisson_noise_multiplier",
        lambda *, method, bands: (
            1.5,
            {
                "cyclic_sensitivity_scale": 1.6,
                "effective_noise_multiplier": 0.93,
                "q_eff": 0.04,
                "cycles": 245,
            },
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_b_min_sep_delta",
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
        lambda **kwargs: 2.5,
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["BSR"],
        amplified_backends=["b_min_sep"],
    )

    cyclic_rows = [
        row for row in rows if row.regime == "amplified" and row.backend == "cyclic_poisson"
    ]
    assert len(cyclic_rows) == 1
    bminsep_rows = [
        row for row in rows if row.regime == "amplified" and row.backend == "b_min_sep"
    ]
    assert len(bminsep_rows) == 1
    row = bminsep_rows[0]
    assert row.comparison_noise_multiplier == pytest.approx(1.5)
    assert row.comparison_source == "cyclic_poisson"
    assert row.comparison_relative_error is not None


def test_bisr_cyclic_poisson_uses_bisr_specific_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "compute_bsr_family_cyclic_report_baseline",
        lambda **kwargs: (
            1.25,
            {
                "cyclic_sensitivity_scale": 1.25,
                "effective_noise_multiplier": 1.0,
                "q_eff": 0.04096,
                "cycles": 245,
            },
        ),
    )

    sigma, metadata = _MODULE._compute_cyclic_poisson_noise_multiplier(
        method="BISR",
        bands=4,
    )

    assert sigma == pytest.approx(1.25)
    assert metadata["cyclic_sensitivity_scale"] == pytest.approx(1.25)
    assert metadata["effective_noise_multiplier"] == pytest.approx(1.0)


def test_bsr_cyclic_poisson_delegates_to_mf_report_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "compute_bsr_family_cyclic_report_baseline",
        lambda **kwargs: (
            1.7,
            {
                "cyclic_sensitivity_scale": 1.7,
                "effective_noise_multiplier": 1.0,
                "q_eff": 0.04096,
                "cycles": 245,
            },
        ),
    )

    sigma, metadata = _MODULE._compute_cyclic_poisson_noise_multiplier(
        method="BSR",
        bands=4,
    )

    assert sigma == pytest.approx(1.7)
    assert metadata["cyclic_sensitivity_scale"] == pytest.approx(1.7)
    assert metadata["effective_noise_multiplier"] == pytest.approx(1.0)


def test_bandinvmf_cyclic_poisson_uses_bandinvmf_specific_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "resolve_bsr_family_cyclic_accountant_input",
        lambda **kwargs: 1.4 if kwargs["mechanism"] == "bandinvmf" else 1.8,
    )
    monkeypatch.setattr(
        _MODULE,
        "get_noise_multiplier",
        lambda **kwargs: kwargs["bsr_sensitivity_scale"],
    )
    monkeypatch.setattr(
        _MODULE,
        "_bandinvmf_inv_coeffs",
        lambda *, bands, optimizer_workload=_MODULE.DEFAULT_OPTIMIZER_WORKLOAD: (1.0, -0.9, 0.03, -0.08),
    )
    monkeypatch.setattr(
        _MODULE,
        "_bandinvmf_runtime_coeffs",
        lambda *, bands, optimizer_workload=_MODULE.DEFAULT_OPTIMIZER_WORKLOAD: (1.0, 0.9, 0.8, 0.8),
    )

    sigma, metadata = _MODULE._compute_cyclic_poisson_noise_multiplier(
        method="Band-Inv-MF",
        bands=4,
    )

    assert sigma == pytest.approx(1.4)
    assert metadata["cyclic_sensitivity_scale"] == pytest.approx(1.4)
    assert metadata["effective_noise_multiplier"] == pytest.approx(1.0)


def test_bandmf_cyclic_poisson_uses_bsr_family_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "resolve_bsr_family_cyclic_accountant_input",
        lambda **kwargs: 1.7,
    )
    monkeypatch.setattr(
        _MODULE,
        "get_noise_multiplier",
        lambda **kwargs: kwargs["bsr_sensitivity_scale"],
    )
    monkeypatch.setattr(_MODULE, "_method_amplified_coeffs", lambda method, bands: (0.58, 0.5, 0.46, 0.42))

    sigma, metadata = _MODULE._compute_cyclic_poisson_noise_multiplier(
        method="Band-MF",
        bands=4,
    )

    assert sigma == pytest.approx(1.7)
    assert metadata["cyclic_sensitivity_scale"] == pytest.approx(1.7)
    assert metadata["effective_noise_multiplier"] == pytest.approx(1.0)


def test_method_random_allocation_mechanism_uses_registry_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        _MODULE._METHOD_RANDOM_ALLOCATION_MECHANISMS,
        "BSR",
        "patched_bsr",
    )

    assert _MODULE._method_random_allocation_mechanism("BSR") == "patched_bsr"


def test_amplified_dp_sgd_rows_keep_poisson_prv_comparison_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["DP-SGD"],
        amplified_backends=["balls_in_bins"],
    )

    bnb_rows = [row for row in rows if row.regime == "amplified" and row.backend == "balls_in_bins"]
    assert len(bnb_rows) == 1
    row = bnb_rows[0]
    assert row.comparison_noise_multiplier is not None
    assert row.comparison_source == "poisson_prv"
    assert row.comparison_relative_error is not None


def test_fixed_bin_random_allocation_rows_use_canonical_backend_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )
    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=True,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["DP-SGD"],
    )
    rows = report["rows"]
    bridge_rows = [row for row in rows if row["backend"] == "ra_fixed_bin_bnb"]
    assert len(bridge_rows) == 1
    bridge_row = bridge_rows[0]
    assert bridge_row["parity_status"] == "known_missing"
    assert bridge_row["source_law_kind"] == "balls_in_bins_fixed_bin"
    assert bridge_row["accountant_engine_kind"] == "deterministic_random_allocation"
    assert bridge_row["route"] == "fixed_bin_bridge_ambient_quantitative_window_realization_package"


def test_build_report_records_amplified_backends_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda paper_row: _bridge_row_for(paper_row),
    )
    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=True,
        amplified_backends=["b_min_sep", "random_allocation"],
        skip_bandinvmf=False,
        methods=["DP-SGD"],
    )
    assert report["metadata"]["amplified_backends"] == ["b_min_sep", "ra_fixed_bin_bnb"]
    assert report["metadata"]["include_non_amplified"] is False


def test_build_report_records_effective_and_requested_include_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "compute_comparison_rows",
        lambda **kwargs: [
            _MODULE._comparison_row(
                row=_paper_row("BSR"),
                backend="b_min_sep",
                status="computed",
                computed=2.1,
                sensitivity=1.0,
                reason_code="test",
                notes="test row",
            ),
            _MODULE._comparison_row(
                row=_MODULE.BLT_DIAGNOSTIC_ROW,
                backend="blt",
                status="computed",
                computed=0.39,
                sensitivity=None,
                reference_noise_multiplier=0.39,
                reference_source="opacus_blt_optimized_fixed_batch_contract",
                reason_code="test",
                notes="test row",
            ),
        ],
    )
    _MODULE.compute_comparison_rows.last_amplified_bsr_scale_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandinvmf_accountant_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandmf_matrix_family_probe = {}
    _MODULE.compute_comparison_rows.last_non_amplified_bandinvmf_probe = None

    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=["b_min_sep"],
        skip_bandinvmf=True,
        methods=["BSR", "BLT"],
        distributed_launch=None,
    )

    meta = report["metadata"]
    assert meta["requested_include_amplified"] is False
    assert meta["requested_include_non_amplified"] is False
    assert meta["requested_include_amplified_deterministic"] is False
    assert meta["include_amplified"] is True
    assert meta["include_non_amplified"] is True


def test_build_report_does_not_count_skipped_amplified_rows_as_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "compute_comparison_rows",
        lambda **kwargs: [
            _MODULE._comparison_row(
                row=_paper_row("BSR"),
                backend="skipped",
                status="skipped",
                computed=None,
                reason_code="known_skip_amplified_disabled",
                notes="amplified skipped",
            ),
            _MODULE._comparison_row(
                row=_MODULE.BLT_DIAGNOSTIC_ROW,
                backend="blt",
                status="computed",
                computed=0.39,
                sensitivity=None,
                reference_noise_multiplier=0.39,
                reference_source="opacus_blt_optimized_fixed_batch_contract",
                reason_code="test",
                notes="test row",
            ),
        ],
    )
    _MODULE.compute_comparison_rows.last_amplified_bsr_scale_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandinvmf_accountant_probe = None
    _MODULE.compute_comparison_rows.last_amplified_bandmf_matrix_family_probe = {}
    _MODULE.compute_comparison_rows.last_non_amplified_bandinvmf_probe = None

    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=["b_min_sep"],
        skip_bandinvmf=True,
        methods=["BSR", "BLT"],
        distributed_launch=None,
    )

    meta = report["metadata"]
    assert meta["requested_include_amplified"] is False
    assert meta["include_amplified"] is False
    assert meta["include_non_amplified"] is True


def test_optimistic_balls_in_bins_calibration_uses_two_sided_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_directions: list[bool] = []

    monkeypatch.setattr(
        _MODULE,
        "build_bnb_toeplitz_c_matrix_and_contract",
        lambda **kwargs: (_MODULE.np.eye(2, dtype=float), {"sampling_mode": "balls_in_bins"}),
    )
    monkeypatch.setattr(
        _MODULE,
        "sample_balls_in_bins_llr_chunks",
        lambda *, positive_sample, **kwargs: (
            seen_directions.append(bool(positive_sample))
            or [(bool(positive_sample), float(kwargs["sigma"]))]
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "estimate_hockey_stick_delta_from_llr_chunks",
        lambda *, epsilon, llr_chunks: (
            2e-5
            if (not bool(llr_chunks[0][0]) and float(llr_chunks[0][1]) < 2.0)
            else 1e-8
        ),
    )

    sigma = _MODULE._compute_bnb_noise_multiplier_for_coeffs(
        coeffs=[1.0, 0.5],
        bands=2,
        mechanism="bisr",
        bnb_calibration_mode="optimistic",
        bnb_chunk_size=4,
        bnb_num_workers=1,
        bnb_num_samples=8,
        bnb_backend="cpu",
        bnb_device=None,
        bnb_distributed_mode="none",
        bnb_distributed_dp_runtime=False,
        bnb_sampling_mode="balls_in_bins",
    )

    assert sigma > 0.0
    assert True in seen_directions
    assert False in seen_directions


def test_build_report_records_blt_nonpaper_metadata() -> None:
    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BLT"],
    )
    blt_meta = report["metadata"]["non_paper_methods"]["BLT"]
    assert blt_meta["regime"] == "non-amplified"
    assert blt_meta["default_rank"] == 2
    assert blt_meta["default_amplified_rank"] == 4
    assert blt_meta["reference_source"] == "fixed_batch_identity_gaussian_prv"
    assert "--methods BLT" in blt_meta["notes"]


def test_build_report_records_bifr_nonpaper_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_method_fixed_batch_sensitivity",
        lambda method, bands, bifr_frac=None: 2.5 if method == "BIFR" else 2.0,
    )
    monkeypatch.setattr(
        _MODULE,
        "_fixed_batch_base_sigma",
        lambda *, backend: 1.0,
    )
    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BIFR"],
        bifr_frac=0.3,
    )
    bifr_meta = report["metadata"]["non_paper_methods"]["BIFR"]
    assert bifr_meta["regime"] == "non-amplified"
    assert bifr_meta["configured_frac"] == pytest.approx(0.3)
    assert bifr_meta["configured_fracs"] == [pytest.approx(0.3)]
    assert bifr_meta["sweep_policy"] == "explicit_single_slice"
    assert bifr_meta["default_bandwidth"] == 4
    assert bifr_meta["reference_source"] == "bsr_half_slice_fixed_batch_contract"
    assert "--methods BIFR" in bifr_meta["notes"]
    assert "exact finite-horizon BIFR" in bifr_meta["notes"]
    rows = report["rows"]
    assert all(row["bifr_frac"] == pytest.approx(0.3) for row in rows)


def test_build_report_uses_default_bifr_sweep_when_override_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_method_fixed_batch_sensitivity",
        lambda method, bands, bifr_frac=None: (
            10.0 + float(bifr_frac) if method == "BIFR" else 2.0
        ),
    )
    monkeypatch.setattr(_MODULE, "_fixed_batch_base_sigma", lambda *, backend: 1.0)

    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BIFR"],
    )

    meta = report["metadata"]
    bifr_meta = meta["non_paper_methods"]["BIFR"]
    assert meta["bifr_frac"] is None
    assert meta["bifr_fracs"] == [0.0, 0.25, 0.5, 1.0]
    assert meta["bifr_sweep_policy"] == "canonical_default_v1"
    assert bifr_meta["configured_frac"] is None
    assert bifr_meta["configured_fracs"] == [0.0, 0.25, 0.5, 1.0]
    assert bifr_meta["sweep_policy"] == "canonical_default_v1"
    assert report["summary"]["bifr"]["fracs"] == [0.0, 0.25, 0.5, 1.0]
    assert len(report["rows"]) == 8


def test_build_report_attaches_paper_rmse_and_missing_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_paper_rmse_direct_inverse_for_method",
        lambda method, bandwidth, bifr_frac: (
            ([1.0], "direct_inverse_family_rmse_from_bifr_inv_coeffs")
            if method == "BIFR"
            else (None, None)
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "compute_prefix_workload_normalized_rmse_from_inverse_coeffs",
        lambda *, inv_coeffs, steps, noise_multiplier: float(noise_multiplier),
    )
    monkeypatch.setattr(
        _MODULE,
        "_method_fixed_batch_sensitivity",
        lambda method, bands, bifr_frac=None: 2.5 if method == "BIFR" else 2.0,
    )
    monkeypatch.setattr(_MODULE, "_fixed_batch_base_sigma", lambda *, backend: 1.0)
    monkeypatch.setattr(
        _MODULE,
        "_compute_blt_fixed_batch_runtime_z_std",
        lambda *, buffers, blt_lambda=None: (
            0.314 if blt_lambda is None else 0.3 + float(blt_lambda),
            {"noise_multiplier_ref": 1.27, "blt_horizon": 8, "blt_min_separation": 4,
             "blt_max_participations": 2, "selected_candidate_index": 0, "candidate_count": 1,
             "selected_theta": [1.0], "selected_theta_hat": [1.0], "lambda_anchor": blt_lambda},
        ),
    )

    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BIFR", "BLT"],
        bifr_frac=0.3,
        blt_lambda=0.6,
    )

    rows = report["rows"]
    bifr_rows = [row for row in rows if row["method"] == "BIFR"]
    blt_rows = [row for row in rows if row["method"] == "BLT"]
    assert len(bifr_rows) == 2
    assert len(blt_rows) == 1
    assert blt_rows[0]["blt_lambda"] == pytest.approx(0.6)
    assert all(row["paper_rmse"] is not None for row in bifr_rows)
    assert all(
        row["paper_rmse_source"] == "direct_inverse_family_rmse_from_bifr_inv_coeffs"
        for row in bifr_rows
    )
    assert all(row["paper_rmse_reason"] is None for row in bifr_rows)
    assert blt_rows[0]["paper_rmse"] is None
    assert blt_rows[0]["paper_rmse_reason"] == "unsupported_blt_rmse_surface"
    assert report["summary"]["paper_rmse"] == {
        "computed_rows": 2,
        "missing_rows": 1,
        "row_count": 3,
    }
    bifr_rmse = report["summary"]["bifr"]["paper_rmse"]
    assert report["summary"]["best_paper_rmse_by_family"]["BIFR"] == {
        "parameter": "frac",
        "best_by_backend": bifr_rmse["best_by_backend"],
    }
    assert bifr_rmse["best_by_backend"]["prv"]["frac"] == pytest.approx(0.3)
    assert bifr_rmse["best_by_backend"]["rdp"]["frac"] == pytest.approx(0.3)
    assert report["summary"]["blt"].get("paper_rmse") is None


def test_bandinvmf_direct_inverse_rmse_uses_requested_optimizer_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    workload = _MODULE.OptimizerWorkloadConfig(momentum=0.0, weight_decay=0.0)

    def _fake_bandinvmf_inv_coeffs(*, bands, optimizer_workload):
        captured["bands"] = bands
        captured["optimizer_workload"] = optimizer_workload
        return (1.0, -0.5)

    monkeypatch.setattr(_MODULE, "_bandinvmf_inv_coeffs", _fake_bandinvmf_inv_coeffs)
    _MODULE._paper_rmse_direct_inverse_for_method.cache_clear()
    try:
        coeffs, source = _MODULE._paper_rmse_direct_inverse_for_method(
            "Band-Inv-MF",
            4,
            None,
            optimizer_workload=workload,
        )
    finally:
        _MODULE._paper_rmse_direct_inverse_for_method.cache_clear()

    assert coeffs == [1.0, -0.5]
    assert source == "direct_inverse_family_rmse_from_bandinvmf_inv_coeffs"
    assert captured["bands"] == 4
    assert captured["optimizer_workload"] == workload


def test_resolve_row_paper_rmse_supports_amplified_blt_selected_pair() -> None:
    theta = [0.8, 0.6, 0.4, 0.2]
    theta_hat = [0.7, 0.5, 0.3, 0.1]
    row = _MODULE._comparison_row(
        row=_MODULE.BLT_AMPLIFIED_DIAGNOSTIC_ROW,
        backend="balls_in_bins",
        status="computed",
        computed=1.25,
        sensitivity=1.0,
        reason_code="computed_bnb_accountant_blt_forward_c_col",
        blt_selection_mode="optimizer_selected",
        blt_buffers=_MODULE.BLT_AMPLIFIED_DIAGNOSTIC_ROW.bandwidth,
        blt_selected_candidate_index=0,
        blt_candidate_count=1,
        blt_selected_theta=theta,
        blt_selected_theta_hat=theta_hat,
        accounting_noise_multiplier=1.25,
        accounting_source="opacus_blt_amplified_bnb_accountant_contract",
        comparison_noise_multiplier=2.0,
        comparison_source="poisson_prv",
        source_law_kind="balls_in_bins",
        accountant_engine_kind="bnb_monte_carlo",
        route="bnb_accountant_balls_in_bins",
        notes="test amplified BLT row",
    )

    pair = _MODULE.blt_pair_from_theta_pair(theta=theta, theta_hat=theta_hat)
    expected_matrix = _MODULE.build_lower_toeplitz_matrix_from_coeffs(
        coeffs=_MODULE.blt_forward_coeffs_for_amplified_accounting(
            pair=pair,
            horizon=int(_MODULE.TOTAL_STEPS),
        ),
        steps=int(_MODULE.TOTAL_STEPS),
    )
    expected_rmse = _MODULE.compute_prefix_workload_normalized_rmse_from_matrix(
        c_matrix=expected_matrix,
        noise_multiplier=float(row.computed_noise_multiplier),
    )

    paper_rmse, source, reason = _MODULE._resolve_row_paper_rmse(row)
    assert paper_rmse == pytest.approx(expected_rmse)
    assert source == "toeplitz_strategy_from_blt_amplified_raw_forward_c_col"
    assert reason is None


def test_build_report_serializes_blt_row_as_nonpaper_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_blt_fixed_batch_runtime_z_std",
        lambda *, buffers, blt_lambda=None: (
            0.314 if blt_lambda is None else 0.3 + float(blt_lambda),
            {
                "noise_multiplier_ref": 1.27,
                "blt_horizon": 8,
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 3,
                "selected_theta": [0.8, 0.3],
                "selected_theta_hat": [0.6, 0.1],
                "lambda_anchor": blt_lambda,
            },
        ),
    )
    report = _MODULE.build_report(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=None,
        skip_bandinvmf=False,
        methods=["BLT"],
    )
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["method"] == "BLT"
    assert row["blt_lambda"] is None
    assert row["blt_selection_mode"] == "optimizer_selected"
    assert row["blt_buffers"] == 2
    assert row["blt_selected_candidate_index"] == 0
    assert row["blt_candidate_count"] == 3
    assert row["blt_selected_theta"] == [0.8, 0.3]
    assert row["blt_selected_theta_hat"] == [0.6, 0.1]
    assert row["paper_noise_multiplier"] is None
    assert row["reference_noise_multiplier"] == pytest.approx(1.7255103931826974)
    assert row["accounting_noise_multiplier"] == pytest.approx(1.27)
    assert row["accounting_source"] == "opacus_blt_fixed_batch_accountant_contract"
    assert row["comparison_noise_multiplier"] is not None
    assert row["comparison_source"] == "fixed_batch_identity_gaussian_prv"
    assert report["metadata"]["blt_lambdas"] == []
    assert report["metadata"]["blt_selection_policy"] == "optimizer_selected_default_v1"
    blt_meta = report["metadata"]["non_paper_methods"]["BLT"]
    assert blt_meta["configured_lambda"] is None
    assert blt_meta["configured_lambdas"] == []
    assert blt_meta["selection_policy"] == "optimizer_selected_default_v1"
    assert report["summary"]["blt"].get("paper_rmse") is None
    assert report["summary"]["best_paper_rmse_by_family"] == {}


def test_build_report_metadata_records_inverse_family_cyclic_boundary() -> None:
    report = _MODULE.build_report(
        include_amplified=True,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        amplified_backends=["b_min_sep", "balls_in_bins"],
        skip_bandinvmf=False,
        methods=["BISR", "Band-Inv-MF"],
    )

    note = report["metadata"]["comparison_column_notes"]["inverse_family_cyclic_boundary"]
    assert "local reduced sampled-Gaussian baseline only" in note
    assert "kappa(T)-style cyclic sensitivity scale" in note
    assert "|C^p[:,0]|" in note
    assert "|C[:,0]|" in note


def test_b_min_sep_rows_use_b_min_sep_backend_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_b_min_sep_delta",
        lambda **kwargs: _MODULE.SingleVerifyProbeResult(
            status="computed",
            delta_estimate=1e-6,
            delta_upper_confidence_bound=None,
            error_probability=None,
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
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["DP-SGD"],
        amplified_backends=["b_min_sep"],
    )
    bminsep_rows = [row for row in rows if row.regime == "amplified" and row.backend == "b_min_sep"]
    assert len(bminsep_rows) == 1
    row = bminsep_rows[0]
    assert row.source_law_kind == "b_min_sep"
    assert row.route == "bnb_analysis_b_min_sep_direct_monte_carlo"


def test_optimistic_amplified_rows_skip_single_verify_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_bnb_delta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("single_verify_bnb should be skipped")),
    )
    monkeypatch.setattr(
        _MODULE,
        "_probe_single_verify_b_min_sep_delta",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("single_verify_b_min_sep should be skipped")),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_bnb_noise_multiplier_for_coeffs",
        lambda **kwargs: 2.34,
    )
    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["DP-SGD"],
        amplified_backends=["balls_in_bins", "b_min_sep"],
        bnb_calibration_mode="optimistic",
    )
    amplified_rows = [row for row in rows if row.regime == "amplified" and row.backend in {"balls_in_bins", "b_min_sep"}]
    assert len(amplified_rows) == 2
    for row in amplified_rows:
        assert row.single_verify_status == "skipped"
        assert row.single_verify_delta_estimate is None
        assert "skipped because it is only used" in str(row.single_verify_notes)


def test_ra_rows_use_repeated_random_allocation_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_compute_random_allocation_noise_multiplier_for_coeffs",
        lambda **kwargs: (
            1.23,
            {
                "source_law_kind": "repeated_k_out_of_t",
                "accountant_engine_kind": "deterministic_random_allocation",
                "route": "pair_driven_public_exact_initial_package",
                "coeff_source": "identity_c_col",
                "repeated_runtime_policy": _MODULE.REPORT_GRADE_REPEATED_RA_RUNTIME_POLICY,
                "repeated_runtime_mode": "report_grade",
                "runtime_loss_discretization": _MODULE.REPORT_GRADE_REPEATED_RA_OUTPUT_LOSS,
                "runtime_policy_basis": _MODULE.REPORT_GRADE_REPEATED_RA_ACCOUNTANT_POLICY,
            },
        ),
    )
    rows = _MODULE.compute_comparison_rows(
        include_amplified=False,
        include_non_amplified=False,
        include_amplified_deterministic=False,
        methods=["DP-SGD"],
        amplified_backends=["ra"],
    )
    ra_rows = [row for row in rows if row.regime == "amplified" and row.backend == "ra"]
    assert len(ra_rows) == 1
    row = ra_rows[0]
    assert row.source_law_kind == "repeated_k_out_of_t"
    assert row.accountant_engine_kind == "deterministic_random_allocation"
    assert row.route == "pair_driven_public_exact_initial_package"
    assert row.route != "fixed_bin_bridge_exact_pair_package"
    assert row.repeated_runtime_policy == _MODULE.REPORT_GRADE_REPEATED_RA_RUNTIME_POLICY
    assert row.repeated_runtime_mode == "report_grade"
    assert row.runtime_loss_discretization == pytest.approx(
        _MODULE.REPORT_GRADE_REPEATED_RA_OUTPUT_LOSS
    )
    assert row.computed_noise_multiplier == pytest.approx(1.23)
    assert "report-grade approximation policy" in row.notes
    assert _MODULE.REPORT_GRADE_REPEATED_RA_ACCOUNTANT_POLICY in row.notes


def test_report_grade_repeated_ra_runtime_config_is_distinct_from_library_policy() -> None:
    runtime = _MODULE._resolve_report_grade_repeated_ra_runtime_config(
        reduced_num_steps_per_round=391,
        reduced_num_rounds=25,
    )
    assert runtime["report_runtime_policy"] == _MODULE.REPORT_GRADE_REPEATED_RA_RUNTIME_POLICY
    assert runtime["accountant_runtime_policy"] == _MODULE.REPORT_GRADE_REPEATED_RA_ACCOUNTANT_POLICY
    assert runtime["accountant_loss_discretization"] == pytest.approx(
        _MODULE.REPORT_GRADE_REPEATED_RA_OUTPUT_LOSS
    )
    assert runtime["estimated_inner_loss_discretization"] > 0.0021


def test_probe_single_verify_b_min_sep_delta_uses_direct_monte_carlo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "estimate_b_min_sep_delta_monte_carlo",
        lambda **kwargs: 7.5e-6,
    )
    probe = _MODULE._probe_single_verify_b_min_sep_delta(
        c_matrix=_MODULE.np.eye(4, dtype=float),
        bands=2,
        sigma=1.5,
        epsilon=9.0,
        num_samples=1000,
        chunk_size=100,
        num_workers=0,
    )
    assert probe.status == "computed"
    assert probe.delta_estimate == pytest.approx(7.5e-6)
    assert probe.delta_upper_confidence_bound is None
    assert "b_min_sep Monte Carlo point estimate" in str(probe.notes)


def test_b_min_sep_noise_calibration_uses_direct_delta_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "estimate_b_min_sep_delta_monte_carlo",
        lambda **kwargs: 1e-4 if float(kwargs["noise_multiplier"]) < 2.0 else 1e-7,
    )
    monkeypatch.setattr(
        _MODULE,
        "sample_balls_in_bins_llr_chunks",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("balls_in_bins path should not be used")),
    )
    sigma = _MODULE._compute_bnb_noise_multiplier_for_coeffs(
        coeffs=[1.0],
        bands=1,
        mechanism="gaussian",
        bsr_accountant_scale="raw",
        bnb_calibration_mode="optimistic",
        bnb_chunk_size=100,
        bnb_num_workers=0,
        bnb_num_samples=1000,
        bnb_backend="cpu",
        bnb_device=None,
        bnb_distributed_mode=None,
        bnb_distributed_dp_runtime=False,
        bnb_sampling_mode="b_min_sep",
    )
    assert 1.9 <= sigma <= 2.1


def test_dpsgd_fixed_bin_bridge_workload_uses_reduced_full_horizon_mode_norm_control() -> None:
    c_matrix, mechanism, coeff_source, sensitivity = _MODULE._resolve_fixed_bin_bridge_workload(
        method="DP-SGD",
        bands=1,
    )
    assert c_matrix.shape == (1, _MODULE.TOTAL_STEPS)
    assert coeff_source == "full_horizon_mode_norm_control"
    assert mechanism == "gaussian"
    assert sensitivity == pytest.approx(_MODULE.EPOCHS ** 0.5, rel=0.0, abs=1e-12)


def test_fixed_bin_bridge_workload_trims_bnb_padding_to_logical_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_horizon = int(_MODULE.TOTAL_STEPS)
    padded_horizon = logical_horizon + 4
    padded = _MODULE.np.arange(float(padded_horizon * padded_horizon), dtype=float).reshape(
        padded_horizon,
        padded_horizon,
    )

    monkeypatch.setattr(
        _MODULE,
        "_method_amplified_accountant_coeffs",
        lambda method, bands, bifr_frac=None, optimizer_workload=_MODULE.DEFAULT_OPTIMIZER_WORKLOAD: (
            [1.0],
            "test_accountant_coeffs",
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "build_bnb_toeplitz_c_matrix_and_contract",
        lambda **kwargs: (
            padded,
            {
                "horizon": logical_horizon,
                "padded_horizon": padded_horizon,
                "padding_columns": padded_horizon - logical_horizon,
            },
        ),
    )

    c_matrix, mechanism, coeff_source, sensitivity = _MODULE._resolve_fixed_bin_bridge_workload(
        method="BISR",
        bands=8,
    )

    assert c_matrix.shape == (logical_horizon, logical_horizon)
    assert _MODULE.np.array_equal(c_matrix, padded[:logical_horizon, :logical_horizon])
    assert mechanism == "bisr"
    assert coeff_source == "test_accountant_coeffs"
    assert sensitivity is None


def test_blt_fixed_bin_bridge_workload_reuses_accountant_owned_toeplitz_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_horizon = int(_MODULE.TOTAL_STEPS)
    padded_horizon = logical_horizon + 4
    padded = _MODULE.np.arange(float(padded_horizon * padded_horizon), dtype=float).reshape(
        padded_horizon,
        padded_horizon,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        _MODULE,
        "_compute_blt_fixed_batch_runtime_z_std",
        lambda *, buffers, blt_lambda=None: (
            0.314,
            {
                "noise_multiplier_ref": 1.27,
                "blt_horizon": logical_horizon,
                "blt_min_separation": 4,
                "blt_max_participations": 2,
                "selected_candidate_index": 0,
                "candidate_count": 3,
                "selected_theta": [0.8, 0.3],
                "selected_theta_hat": [0.6, 0.1],
            },
        ),
    )

    def _resolve_state(**kwargs):
        captured.update(kwargs)
        return {
            "bnb_c_matrix": padded,
            "bnb_c_matrix_contract": {
                "horizon": logical_horizon,
                "padded_horizon": padded_horizon,
            },
            "bnb_accountant_coeffs_source": "normalized_forward_c_col",
        }

    monkeypatch.setattr(_MODULE, "resolve_blt_balls_in_bins_accountant_state", _resolve_state)

    c_matrix, coeff_source, blt_meta = _MODULE._resolve_blt_fixed_bin_bridge_workload(
        buffers=4,
        blt_lambda=None,
    )

    assert c_matrix.shape == (logical_horizon, logical_horizon)
    assert _MODULE.np.array_equal(c_matrix, padded[:logical_horizon, :logical_horizon])
    assert coeff_source == "normalized_forward_c_col"
    assert blt_meta["selected_candidate_index"] == 0
    assert captured["kwargs"]["bnb_cycle_length"] == int(_MODULE.STEPS_PER_EPOCH)
    assert captured["kwargs"]["bnb_bands"] == 4


def test_blt_fixed_bin_bridge_row_preserves_selected_theta_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _MODULE.BLT_AMPLIFIED_DIAGNOSTIC_ROW
    monkeypatch.setattr(
        _MODULE,
        "_resolve_blt_fixed_bin_bridge_workload",
        lambda *, buffers, blt_lambda=None: (
            _MODULE.np.eye(4, dtype=float),
            "normalized_forward_c_col",
            {
                "selected_candidate_index": 1,
                "candidate_count": 5,
                "selected_theta": [0.8, 0.3],
                "selected_theta_hat": [0.6, 0.1],
            },
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "resolve_fixed_bin_random_allocation_bridge_inputs",
        lambda **kwargs: SimpleNamespace(
            source_law_kind="balls_in_bins_fixed_bin",
            accountant_engine_kind="deterministic_random_allocation",
            route="fixed_bin_bridge_exact_pair_package",
            exact_law_route="exact_fixed_bin_gaussian_mixture_pair",
            initial_package_route="pair_driven_exact_initial_package",
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_compute_fixed_bin_bridge_noise_multiplier",
        lambda **kwargs: (
            0.8125,
            {
                "source_law_kind": "balls_in_bins_fixed_bin",
                "accountant_engine_kind": "deterministic_random_allocation",
                "route": "fixed_bin_bridge_exact_pair_package",
                "exact_law_route": "exact_fixed_bin_gaussian_mixture_pair",
                "initial_package_route": "pair_driven_exact_initial_package",
                "runtime_policy_name": "fixed_bin_bridge_candidate_grid_1e-2",
                "runtime_loss_discretization": 1e-2,
            },
        ),
    )

    bridge_row = _MODULE._build_amplified_fixed_bin_bridge_row(row)

    assert bridge_row.status == "computed"
    assert bridge_row.backend == "ra_fixed_bin_bnb"
    assert bridge_row.route == "fixed_bin_bridge_exact_pair_package"
    assert bridge_row.blt_selection_mode == "optimizer_selected"
    assert bridge_row.blt_buffers == int(row.bandwidth)
    assert bridge_row.blt_selected_candidate_index == 1
    assert bridge_row.blt_candidate_count == 5
    assert bridge_row.blt_selected_theta == [0.8, 0.3]
    assert bridge_row.blt_selected_theta_hat == [0.6, 0.1]
    assert bridge_row.computed_noise_multiplier == pytest.approx(0.8125)


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
    assert "Bridge runtime policy=fixed_bin_bridge_candidate_grid_1e-2" in bridge_row.notes
    assert bridge_row.computed_noise_multiplier == pytest.approx(0.39642333984375)
    assert bridge_row.sensitivity is None
    assert "not the repeated k_out_of_t random_allocation route" in bridge_row.notes


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


def test_blt_amplified_report_keeps_explicit_failed_fixed_bin_bridge_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_resolve_blt_amplified_accountant_coeffs",
        lambda *, buffers, blt_lambda: (
            [1.0] + [0.0] * (int(_MODULE.TOTAL_STEPS) - 1),
            "normalized_forward_c_col",
            {
                "selected_candidate_index": 0,
                "candidate_count": 1,
                "selected_theta": [0.8, 0.3],
                "selected_theta_hat": [0.6, 0.1],
                "blt_min_separation": 4,
                "blt_horizon": int(_MODULE.TOTAL_STEPS),
            },
        ),
    )
    monkeypatch.setattr(_MODULE, "_compute_bnb_noise_multiplier_for_coeffs", lambda **kwargs: 0.75)
    monkeypatch.setattr(
        _MODULE,
        "_build_amplified_fixed_bin_bridge_row",
        lambda row, **kwargs: _MODULE._comparison_row(
            row=row,
            backend="ra_fixed_bin_bnb",
            status="failed",
            computed=None,
            sensitivity=None,
            reason_code="known_missing_fixed_bin_bridge_quantitative_window_realization",
            blt_selection_mode="optimizer_selected",
            blt_buffers=int(row.bandwidth),
            source_law_kind="balls_in_bins_fixed_bin",
            accountant_engine_kind="deterministic_random_allocation",
            route="fixed_bin_bridge_ambient_quantitative_window_realization_package",
            notes="test unsupported BLT fixed-bin row",
        ),
    )

    rows = _MODULE.compute_comparison_rows(
        include_amplified=True,
        include_non_amplified=False,
        include_amplified_deterministic=True,
        methods=["BLT"],
        amplified_backends=["balls_in_bins"],
    )

    bridge_rows = [row for row in rows if row.method == "BLT" and row.backend == "ra_fixed_bin_bnb"]
    assert len(bridge_rows) == 1
    assert bridge_rows[0].status == "failed"
    assert (
        bridge_rows[0].reason_code
        == "known_missing_fixed_bin_bridge_quantitative_window_realization"
    )
    assert bridge_rows[0].route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"


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
