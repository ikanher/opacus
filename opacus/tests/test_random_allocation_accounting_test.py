from __future__ import annotations

import builtins
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import opacus.accountants.utils as accountant_utils_module
import opacus.accountants.analysis.random_allocation.accountant as random_allocation_module
import opacus.accountants.analysis.random_allocation.fixed_bin as fixed_bin_random_allocation_module
import opacus.accountants.analysis.random_allocation.initial_package as random_allocation_initial_package_module
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.accountants.analysis.bnb import build_bnb_toeplitz_c_matrix_and_contract
from opacus.accountants.analysis.bandinvmf import (
    derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs,
    optimize_bandinvmf_inv_coeffs_for_sgd_workload,
)
from opacus.accountants.analysis.bandmf import generate_bandmf_coeffs_from_sgd_workload
from opacus.accountants.analysis.bisr import generate_bisr_coeffs_from_sgd_workload
from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload
from opacus.accountants.analysis.random_allocation.fixed_bin import (
    _aggregate_fixed_bin_mode_family,
    build_fixed_bin_exact_law_pair,
    _build_fixed_bin_gaussian_mixture_pair,
    _diagnose_fixed_bin_ambient_nonfinite_upper_bound,
    estimate_epsilon_range_fixed_bin_random_allocation,
    estimate_epsilon_upper_fixed_bin_random_allocation,
    get_noise_multiplier_fixed_bin_random_allocation,
    resolve_fixed_bin_random_allocation_bridge_inputs,
    resolve_fixed_bin_random_allocation_bridge_runtime_config,
)
from opacus.accountants.analysis.random_allocation.exact_laws import (
    FiniteGaussianMixtureNeighboringPair,
    build_poisson_gaussian_mixture_neighboring_pair,
    build_product_gaussian_mixture_neighboring_pair,
    build_realizable_gaussian_one_step_neighboring_pair,
)
from opacus.accountants.analysis.random_allocation.initial_package import (
    build_ambient_quantitative_window_package_from_exact_law,
    build_deterministic_initial_package_from_exact_law,
    _build_exact_family_accountant_contract_from_exact_law,
    _build_exact_family_round_pair_from_contract,
    _build_deterministic_witness_family_package_from_exact_law,
    estimate_epsilon_random_allocation_from_initial_package,
    _estimate_epsilon_range_random_allocation_from_exact_family_round_pair_package,
    estimate_epsilon_range_random_allocation_from_initial_package,
    resolve_pair_driven_ambient_quantitative_window_inputs,
    _resolve_exact_family_round_pair_inputs,
    _resolve_exact_family_round_pair_package_inputs,
    _resolve_pair_driven_exact_family_random_allocation_inputs,
    resolve_pair_driven_random_allocation_inputs,
)
from opacus.accountants.analysis.random_allocation import (
    build_gaussian_random_allocation_realization,
    estimate_epsilon_random_allocation,
    estimate_epsilon_range_random_allocation,
    resolve_random_allocation_accountant_inputs,
    resolve_random_allocation_gaussian_runtime_config,
)
from opacus.accountants.utils import (
    NoiseSearchConvergenceError,
    get_noise_multiplier,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "opacus" / "opacus" / "tests" / "fixtures" / "random_allocation_pld_accounting_goldens.json"
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_random_allocation_pld_accounting_goldens.py"
OLD_GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_bnb_deterministic_pld_accounting_goldens.py"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _fixture_case(name: str) -> dict:
    fixture = _load_fixture()
    for case in fixture["cases"]:
        if case["name"] == name:
            return case
    raise AssertionError(f"Missing fixture case: {name}")


def _blocked_pld_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PLD_accounting" or name.startswith("PLD_accounting."):
            raise ModuleNotFoundError("blocked by test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _blocked_dp_accounting_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dp_accounting" or name.startswith("dp_accounting."):
            raise ModuleNotFoundError("blocked by test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _tiny_loader(*, n_samples: int = 32, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8) -> DataLoader:
    gen = torch.Generator().manual_seed(20260327)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True)


def _amplified_bsr_random_allocation_state(*, bands: int = 4) -> tuple[dict, SamplingSemantics]:
    coeffs = [
        float(c)
        for c in generate_bsr_coeffs_from_sgd_workload(
            bands=int(bands),
            momentum=0.9,
            weight_decay=0.9999,
        )
    ]
    state = {
        "mechanism": "bsr",
        "_noise_mechanism": "bsr",
        "coeffs": coeffs,
        "random_allocation_accountant_coeffs": list(coeffs),
        "random_allocation_accountant_coeffs_source": "raw_c_col",
        "bsr_bands": int(bands),
    }
    semantics = SamplingSemantics(
        sampling_mode="k_out_of_t",
        privacy_metadata={"num_steps": 98, "num_selected": 1},
    )
    return state, semantics


def _live_bisr_fixed_bin_c_matrix(*, bands: int) -> np.ndarray:
    coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=bands,
        momentum=0.0,
        weight_decay=0.0,
    )
    c_matrix, _contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=coeffs,
        bands=bands,
        horizon=980,
    )
    return np.asarray(c_matrix, dtype=np.float64)


def _live_bandinvmf_fixed_bin_c_matrix(*, bands: int) -> np.ndarray:
    inv_coeffs = optimize_bandinvmf_inv_coeffs_for_sgd_workload(
        bands=bands,
        momentum=0.0,
        weight_decay=0.0,
        steps=980,
        max_participations=10,
        min_separation=bands,
        optimizer_steps=20,
    )
    coeffs = derive_bandinvmf_amplified_accountant_coeffs_from_inv_coeffs(
        inv_coeffs=inv_coeffs,
        steps=980,
    )
    c_matrix, _contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=coeffs,
        bands=bands,
        horizon=980,
    )
    return np.asarray(c_matrix, dtype=np.float64)


def _toy_lower_toeplitz_c_matrix(coeffs: list[float] | tuple[float, ...]) -> np.ndarray:
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    n = int(coeffs_arr.shape[0])
    c_matrix = np.zeros((n, n), dtype=np.float64)
    for row in range(n):
        for col in range(row + 1):
            c_matrix[row, col] = coeffs_arr[row - col]
    return c_matrix


def _one_window_pair_driven_vs_fixed_bin_epsilons(
    *, c_matrix: np.ndarray, noise_multiplier: float = 1.0, target_delta: float = 1e-5
) -> tuple[float, float]:
    runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=target_delta,
        loss_discretization=1e-3,
        tail_truncation=1e-12,
        max_grid_fft=200000,
        max_grid_mult=200000,
    )
    matrix = np.asarray(c_matrix, dtype=np.float64)
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=matrix[:, 0],
        reverse_mean=np.zeros(matrix.shape[0], dtype=np.float64),
        noise_multiplier=noise_multiplier,
    )
    pair_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=matrix.shape[1],
        num_selected=1,
        num_epochs=1,
    )
    pair_epsilon = estimate_epsilon_random_allocation_from_initial_package(
        inputs=pair_inputs,
        target_delta=target_delta,
        runtime_config=runtime,
    )
    fixed_bin_epsilon = estimate_epsilon_upper_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=matrix,
        bins=matrix.shape[1],
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        runtime_config=runtime,
    )
    return pair_epsilon, fixed_bin_epsilon


def _simple_gaussian_random_allocation_inputs(*, noise_multiplier: float = 1.0):
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[1.0],
        cycle_length=5,
        horizon=20,
        noise_multiplier=noise_multiplier,
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=1e-5)
    return inputs, runtime


def test_random_allocation_runtime_does_not_require_pld_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)

    case = _fixture_case("multi_step_gaussian_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=case["target_delta"],
        **case["runtime_config"],
    )
    epsilon = estimate_epsilon_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    upper, lower = estimate_epsilon_range_random_allocation(
        inputs=inputs,
        target_delta=case["target_delta"],
        runtime_config=runtime,
    )
    assert math.isfinite(epsilon)
    assert math.isfinite(upper)
    assert math.isfinite(lower)


def test_random_allocation_runtime_does_not_require_dp_accounting_subprocess() -> None:
    script = textwrap.dedent(
        f"""
        import builtins
        import math
        import sys

        sys.path.insert(0, {str((REPO_ROOT / "opacus").resolve())!r})
        original_import = builtins.__import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "dp_accounting" or name.startswith("dp_accounting."):
                raise ModuleNotFoundError("blocked by subprocess test")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _guarded_import

        from opacus.accountants.analysis.random_allocation import (
            estimate_epsilon_random_allocation,
            resolve_random_allocation_accountant_inputs,
            resolve_random_allocation_gaussian_runtime_config,
        )

        inputs = resolve_random_allocation_accountant_inputs(
            mechanism="gaussian",
            accountant_coeffs=[3.0, 4.0],
            cycle_length=5,
            horizon=20,
            noise_multiplier=10.0,
        )
        runtime = resolve_random_allocation_gaussian_runtime_config(
            target_delta=1e-5,
            loss_discretization=5e-2,
            tail_truncation=1e-6,
            max_grid_fft=1_000_000,
            max_grid_mult=30_000,
            convolution_method="geometric",
        )
        epsilon = estimate_epsilon_random_allocation(
            inputs=inputs,
            target_delta=1e-5,
            runtime_config=runtime,
        )
        assert math.isfinite(epsilon)
        print(epsilon)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_random_allocation_debug_timing_is_gated_for_direct_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs, runtime = _simple_gaussian_random_allocation_inputs()
    build_gaussian_random_allocation_realization.cache_clear()

    monkeypatch.delenv("DEBUG_TIMING", raising=False)
    estimate_epsilon_random_allocation(inputs=inputs, target_delta=1e-5, runtime_config=runtime)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("DEBUG_TIMING", "1")
    build_gaussian_random_allocation_realization.cache_clear()
    estimate_epsilon_random_allocation(inputs=inputs, target_delta=1e-5, runtime_config=runtime)
    out = capsys.readouterr().out
    assert "[opacus.random_allocation] [timing] start estimate_epsilon_random_allocation_bound" in out
    assert "allocation_remove_pmf" in out
    assert "allocation_add_pmf" in out
    assert "compose_linear_pmfs iter" in out
    assert "pair_driven_exact_initial_package:exact_common_covariance_gaussian_one_step_pair" in out
    assert "epsilon_query result" in out


def test_linear_dist_hockey_stick_delta_matches_hand_computation() -> None:
    dist = random_allocation_module._LinearDiscreteDist(
        x_min=0.0,
        x_gap=1.0,
        pmf=np.array([0.4, 0.6], dtype=np.float64),
        p_pos_inf=0.1,
    )
    observed = random_allocation_module._delta_from_linear_dist_for_epsilon(dist, 0.5)
    expected = 0.1 + 0.6 * (1.0 - math.exp(0.5 - 1.0))
    assert observed == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_linear_dist_epsilon_for_delta_is_infinite_when_positive_infinity_mass_exceeds_delta() -> None:
    dist = random_allocation_module._LinearDiscreteDist(
        x_min=0.0,
        x_gap=1.0,
        pmf=np.array([0.4, 0.6], dtype=np.float64),
        p_pos_inf=0.1,
    )
    assert math.isinf(random_allocation_module._epsilon_from_linear_dist_for_delta(dist, 0.1))
    assert math.isinf(random_allocation_module._epsilon_from_linear_dist_for_delta(dist, 0.05))


def test_linear_dist_epsilon_for_large_delta_clamps_to_zero() -> None:
    dist = random_allocation_module._LinearDiscreteDist(
        x_min=0.0,
        x_gap=1.0,
        pmf=np.array([0.4, 0.6], dtype=np.float64),
        p_pos_inf=0.1,
    )
    assert random_allocation_module._epsilon_from_linear_dist_for_delta(dist, 1.0) == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_linear_dist_epsilon_handles_zero_weighted_tail_branch() -> None:
    dist = random_allocation_module._LinearDiscreteDist(
        x_min=800.0,
        x_gap=50.0,
        pmf=np.array([0.5, 0.5], dtype=np.float64),
        p_pos_inf=0.0,
    )
    observed = random_allocation_module._epsilon_from_linear_dist_for_delta(dist, 0.25)
    assert observed == pytest.approx(850.0 + math.log(0.5), rel=0.0, abs=1e-12)


def test_remove_add_epsilon_uses_the_maximum_side() -> None:
    remove = random_allocation_module._LinearDiscreteDist(
        x_min=0.0,
        x_gap=1.0,
        pmf=np.array([0.4, 0.6], dtype=np.float64),
        p_pos_inf=0.0,
    )
    add = random_allocation_module._LinearDiscreteDist(
        x_min=0.0,
        x_gap=1.0,
        pmf=np.array([0.8, 0.2], dtype=np.float64),
        p_pos_inf=0.0,
    )
    epsilon_remove = random_allocation_module._epsilon_from_linear_dist_for_delta(remove, 0.2)
    epsilon_add = random_allocation_module._epsilon_from_linear_dist_for_delta(add, 0.2)
    observed = random_allocation_module._epsilon_from_remove_add_pmfs_for_delta(remove, add, 0.2)
    assert observed == pytest.approx(max(epsilon_remove, epsilon_add), rel=0.0, abs=1e-12)


def test_convolve_infinite_masses_accepts_exact_unit_endpoint_without_warning() -> None:
    with np.errstate(all="raise"):
        p_neg_inf, p_pos_inf = random_allocation_module._convolve_infinite_masses(1.0, 0.25, 0.4, 1.0)
    assert p_neg_inf == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert p_pos_inf == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_random_allocation_debug_timing_preserves_direct_probe_epsilon(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs, runtime = _simple_gaussian_random_allocation_inputs()

    monkeypatch.delenv("DEBUG_TIMING", raising=False)
    baseline = estimate_epsilon_random_allocation(inputs=inputs, target_delta=1e-5, runtime_config=runtime)
    capsys.readouterr()

    monkeypatch.setenv("DEBUG_TIMING", "1")
    observed = estimate_epsilon_random_allocation(inputs=inputs, target_delta=1e-5, runtime_config=runtime)
    capsys.readouterr()

    assert observed == pytest.approx(baseline, rel=0.0, abs=1e-12)


def test_random_allocation_module_docstring_pins_base_theorem_layer() -> None:
    assert random_allocation_module.__doc__ is not None
    assert "randomAllocationTransforms" in random_allocation_module.__doc__
    assert "exactKOutOfTReductionTheoremTarget" in random_allocation_module.__doc__
    assert "PLDRandomAllocationNumerics.lean" in random_allocation_module.__doc__
    assert "BNBDeterministicRandomAllocationBridge.lean" in random_allocation_module.__doc__


def test_resolved_random_allocation_inputs_expose_exact_package_contract() -> None:
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[3.0, 4.0],
        cycle_length=5,
        horizon=20,
        noise_multiplier=10.0,
    )
    assert inputs.contract_kind == "pld_accounting_random_allocation"
    assert inputs.package_alignment_kind == "repeated_k_out_of_t"
    assert inputs.route == "pair_driven_public_exact_initial_package"
    assert inputs.exact_law_route == "exact_common_covariance_gaussian_one_step_pair"
    assert inputs.initial_package_route == "pair_driven_exact_initial_package"
    assert inputs.pld_num_steps == 5
    assert inputs.pld_num_selected == 1
    assert inputs.pld_num_epochs == 4
    assert "exact common-covariance one-step neighboring law" in inputs.package_alignment_notes
    assert "deterministic initial-package evaluator" in inputs.package_alignment_notes
    assert "fixed-bin balls-in-bins bridge" in inputs.package_alignment_notes


def test_fixed_bin_bridge_aggregates_modes_by_epoch_bin_for_dpsgd_control() -> None:
    c_matrix = np.eye(6, dtype=np.float64)
    modes = _aggregate_fixed_bin_mode_family(c_matrix=c_matrix, bins=2)
    assert modes == (
        (1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    )


def test_exp_neg_loss_moment_rescaling_avoids_overflow_warnings() -> None:
    pmf = np.array([0.5, 0.5], dtype=np.float64)
    x_array = np.array([-1000.0, 0.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scaled = random_allocation_module._rescale_pmf_to_exp_neg_loss_moment_at_most_one(
            pmf,
            x_array,
            atol=1e-12,
        )

    assert np.all(np.isfinite(scaled))
    assert float(np.sum(scaled, dtype=np.float64)) <= 1.0 + 1e-12


def test_initial_package_dominating_realization_avoids_overflow_warnings() -> None:
    x_array = np.array([-1000.0, 0.0], dtype=np.float64)
    cdf_lower = np.array([0.5, 1.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        realization = (
            random_allocation_initial_package_module._build_dominating_realization_from_cdf_lower_bounds(
                x_array=x_array,
                cdf_lower=cdf_lower,
            )
        )

    assert np.all(np.isfinite(realization.PMF_array))
    assert realization.p_loss_inf >= 0.0


def test_compute_bin_ratio_accepts_geometric_grid_with_infinite_tail() -> None:
    x_array = np.array([2.5e307, 5.0e307, 1.0e308, np.inf], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ratio = random_allocation_module._compute_bin_ratio(x_array)  # type: ignore[attr-defined]

    assert ratio == pytest.approx(2.0, rel=0.0, abs=1e-12)


def test_geometric_convolution_avoids_overflow_warnings_on_large_grids() -> None:
    pmf = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float64)
    dist_1 = random_allocation_module._GeometricDiscreteDist(  # type: ignore[attr-defined]
        x_min=2.5e307,
        ratio=2.0,
        pmf=pmf,
    )
    dist_2 = random_allocation_module._GeometricDiscreteDist(  # type: ignore[attr-defined]
        x_min=5.0e307,
        ratio=2.0,
        pmf=pmf,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = random_allocation_module._geometric_convolve(  # type: ignore[attr-defined]
            dist_1,
            dist_2,
            tail_truncation=1e-8,
            bound_type=random_allocation_module._BoundType.DOMINATES,  # type: ignore[attr-defined]
        )

    assert out.x_min == pytest.approx(7.5e307, rel=1e-12)
    assert out.ratio == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert np.all(np.isfinite(out.PMF_array))


def _reference_geometric_convolve(
    dist_1,
    dist_2,
    *,
    tail_truncation: float,
    bound_type,
):
    if not math.isclose(
        float(dist_1.ratio),
        float(dist_2.ratio),
        rel_tol=random_allocation_module.SPACING_RTOL,  # type: ignore[attr-defined]
        abs_tol=random_allocation_module.SPACING_ATOL,  # type: ignore[attr-defined]
    ):
        raise ValueError("Grid ratios must match")

    ratio = 0.5 * (dist_1.ratio + dist_2.ratio)
    x1_min = dist_1.x_min
    x2_min = dist_2.x_min
    p1 = dist_1.PMF_array
    p2 = dist_2.PMF_array
    if x1_min > x2_min:
        x1_min, p1, x2_min, p2 = x2_min, p2, x1_min, p1

    scale = float(x2_min / x1_min)
    n = max(p1.size, p2.size)
    if p1.size < n:
        p1 = np.pad(p1, (0, n - p1.size), mode="constant")
    if p2.size < n:
        p2 = np.pad(p2, (0, n - p2.size), mode="constant")

    if n == 1:
        pmf_out = np.array([float(p1[0] * p2[0])], dtype=np.float64)
        x_out_min = x1_min + x2_min
    else:
        log_r = math.log(ratio)
        log_scale = math.log(scale)
        log_ap1 = math.log(scale + 1.0)
        d_vec = np.arange(n, dtype=np.float64)
        log_r_d = d_vec * log_r
        log_lohi = np.logaddexp(0.0, log_scale + log_r_d)
        tau_lohi = (log_lohi - log_ap1) / log_r
        log_hilo = np.logaddexp(log_scale, log_r_d)
        tau_hilo = (log_hilo - log_ap1) / log_r
        rounding_eps = 1e-16

        if bound_type == random_allocation_module._BoundType.DOMINATES:  # type: ignore[attr-defined]
            delta_lohi = np.ceil(tau_lohi - rounding_eps).astype(np.int64)
            delta_hilo = np.ceil(tau_hilo - rounding_eps).astype(np.int64)
        else:
            delta_lohi = np.floor(tau_lohi + rounding_eps).astype(np.int64)
            delta_hilo = np.floor(tau_hilo + rounding_eps).astype(np.int64)

        pmf_out = np.zeros(n, dtype=np.float64)
        comp = np.zeros(n, dtype=np.float64)
        for i in range(n):
            mass = float(p1[i] * p2[i])
            y = mass - comp[i]
            t = pmf_out[i] + y
            comp[i] = (t - pmf_out[i]) - y
            pmf_out[i] = t

        for d in range(1, n):
            imax = n - d
            kshift1 = int(delta_lohi[d])
            kshift2 = int(delta_hilo[d])

            for i in range(imax):
                k1 = i + kshift1
                mass1 = float(p1[i] * p2[i + d])
                if 0 <= k1 < n:
                    y = mass1 - comp[k1]
                    t = pmf_out[k1] + y
                    comp[k1] = (t - pmf_out[k1]) - y
                    pmf_out[k1] = t

                k2 = i + kshift2
                mass2 = float(p1[i + d] * p2[i])
                if 0 <= k2 < n:
                    y = mass2 - comp[k2]
                    t = pmf_out[k2] + y
                    comp[k2] = (t - pmf_out[k2]) - y
                    pmf_out[k2] = t

        x_out_min = x1_min + x2_min

    expected_neg_inf, expected_pos_inf = random_allocation_module._convolve_infinite_masses(  # type: ignore[attr-defined]
        dist_1.p_neg_inf,
        dist_1.p_pos_inf,
        dist_2.p_neg_inf,
        dist_2.p_pos_inf,
    )
    pmf_out, p_neg_inf, p_pos_inf = random_allocation_module._enforce_mass_conservation(  # type: ignore[attr-defined]
        pmf_out,
        expected_neg_inf,
        expected_pos_inf,
        bound_type,
    )
    return random_allocation_module._GeometricDiscreteDist(  # type: ignore[attr-defined]
        float(x_out_min),
        ratio,
        pmf_out,
        p_neg_inf,
        p_pos_inf,
    ).truncate_edges(tail_truncation, bound_type)


@pytest.mark.parametrize(
    ("size_1", "size_2", "scale"),
    [
        (16, 16, 1.0),
        (16, 16, 2.0),
        (16, 16, 4.0),
        (16, 16, 8.0),
        (16, 16, 9.0),
        (21, 13, 4.0),
    ],
)
@pytest.mark.parametrize(
    "bound_type",
    [
        random_allocation_module._BoundType.DOMINATES,  # type: ignore[attr-defined]
        random_allocation_module._BoundType.IS_DOMINATED,  # type: ignore[attr-defined]
    ],
)
def test_geometric_convolution_matches_reference_for_representative_scales(
    size_1: int,
    size_2: int,
    scale: float,
    bound_type,
) -> None:
    gen = np.random.default_rng(20260416 + size_1 + size_2)
    pmf_1 = gen.random(size_1)
    pmf_1 /= np.sum(pmf_1, dtype=np.float64)
    pmf_2 = gen.random(size_2)
    pmf_2 /= np.sum(pmf_2, dtype=np.float64)

    dist_1 = random_allocation_module._GeometricDiscreteDist(  # type: ignore[attr-defined]
        x_min=1.0,
        ratio=1.02,
        pmf=pmf_1,
        p_neg_inf=0.05,
        p_pos_inf=0.0,
    )
    dist_2 = random_allocation_module._GeometricDiscreteDist(  # type: ignore[attr-defined]
        x_min=float(scale),
        ratio=1.02,
        pmf=pmf_2,
        p_neg_inf=0.0,
        p_pos_inf=0.02,
    )

    observed = random_allocation_module._geometric_convolve(  # type: ignore[attr-defined]
        dist_1,
        dist_2,
        tail_truncation=0.0,
        bound_type=bound_type,
    )
    expected = _reference_geometric_convolve(
        dist_1,
        dist_2,
        tail_truncation=0.0,
        bound_type=bound_type,
    )

    assert observed.x_min == pytest.approx(expected.x_min, rel=0.0, abs=1e-12)
    assert observed.ratio == pytest.approx(expected.ratio, rel=0.0, abs=1e-12)
    assert observed.p_neg_inf == pytest.approx(expected.p_neg_inf, rel=0.0, abs=1e-12)
    assert observed.p_pos_inf == pytest.approx(expected.p_pos_inf, rel=0.0, abs=1e-12)
    assert observed.PMF_array.shape == expected.PMF_array.shape
    assert np.max(np.abs(observed.PMF_array - expected.PMF_array)) <= 1e-12


def test_fixed_bin_bridge_keeps_source_law_distinct_from_repeated_random_allocation() -> None:
    repeated = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[1.0],
        cycle_length=3,
        horizon=6,
        noise_multiplier=1.0,
    )
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="gaussian",
        c_matrix=np.array([[1.0] * 6], dtype=np.float64),
        bins=3,
        noise_multiplier=1.0,
    )
    assert repeated.package_alignment_kind == "repeated_k_out_of_t"
    assert bridge.source_law_kind == "balls_in_bins_fixed_bin"
    assert bridge.accountant_engine_kind == "deterministic_random_allocation"
    assert bridge.route != repeated.route


def test_fixed_bin_bridge_route_metadata_is_not_repeated_random_allocation() -> None:
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="bsr",
        c_matrix=np.array([[1.0] * 8], dtype=np.float64),
        bins=4,
        noise_multiplier=2.0,
    )
    assert bridge.route == "fixed_bin_bridge_exact_pair_package"
    assert bridge.source_law_kind == "balls_in_bins_fixed_bin"
    assert bridge.accountant_engine_kind == "deterministic_random_allocation"
    assert bridge.epochs == 2
    assert bridge.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"
    assert bridge.initial_package_route == "pair_driven_exact_initial_package"
    assert bridge.pair_driven_route == "pair_driven_exact_initial_package"


def test_fixed_bin_bridge_exposes_explicit_forward_and_reverse_mixture_sides() -> None:
    pair = _build_fixed_bin_gaussian_mixture_pair(
        c_matrix=np.eye(6, dtype=np.float64),
        bins=2,
    )
    assert pair.forward_modes == (
        (1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    )
    assert pair.reverse_modes == ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),)
    assert pair.component_probs == (0.5, 0.5)


def test_fixed_bin_bridge_consumes_upstream_exact_law_and_initial_package() -> None:
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="gaussian",
        c_matrix=np.array([[1.0] * 6], dtype=np.float64),
        bins=3,
        noise_multiplier=1.5,
    )
    assert bridge.route == "fixed_bin_bridge_exact_pair_package"
    assert bridge.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"
    assert bridge.initial_package_route == "pair_driven_exact_initial_package"
    assert bridge.pair_driven_route == "pair_driven_exact_initial_package"


def test_fixed_bin_bridge_nondegenerate_fixture_surfaces_exact_family_round_pair_pending_route() -> None:
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    assert bridge.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"
    assert bridge.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert bridge.pair_driven_route == "pair_driven_exact_initial_package"


def test_fixed_bin_bridge_first_dpsgd_control_supports_package_backed_interval() -> None:
    upper, lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=np.array([[1.0] * 6], dtype=np.float64),
        bins=3,
        noise_multiplier=2.0,
        target_delta=1e-5,
    )
    assert math.isfinite(lower)
    assert upper >= lower or math.isinf(upper)


def test_fixed_bin_bridge_upper_only_matches_interval_upper_for_exact_pair_route() -> None:
    runtime = resolve_fixed_bin_random_allocation_bridge_runtime_config(target_delta=1e-5)
    upper = estimate_epsilon_upper_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=np.array([[1.0] * 6], dtype=np.float64),
        bins=3,
        noise_multiplier=2.0,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    interval_upper, interval_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=np.array([[1.0] * 6], dtype=np.float64),
        bins=3,
        noise_multiplier=2.0,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    assert math.isfinite(upper)
    assert upper == pytest.approx(interval_upper, rel=0.0, abs=1e-12)
    assert interval_upper >= interval_lower or math.isinf(interval_upper)


def test_fixed_bin_bridge_first_mf_extension_fixture_stays_separate_from_repeated_route() -> None:
    upper, lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=np.array([[1.0, 0.5, 1.0, 0.5, 1.0, 0.5]], dtype=np.float64),
        bins=3,
        noise_multiplier=2.5,
        target_delta=1e-5,
    )
    repeated = resolve_random_allocation_accountant_inputs(
        mechanism="bsr",
        accountant_coeffs=[1.0],
        cycle_length=3,
        horizon=6,
        noise_multiplier=2.5,
    )
    assert repeated.route != "fixed_bin_bridge_exact_pair_package"
    assert math.isfinite(lower)
    assert upper >= lower or math.isinf(upper)


def test_fixed_bin_bridge_upper_only_matches_interval_upper_for_ambient_route() -> None:
    runtime = resolve_fixed_bin_random_allocation_bridge_runtime_config(target_delta=1e-5)
    c_matrix = np.array([[1.0, 0.5, 1.0, 0.5, 1.0, 0.5]], dtype=np.float64)
    upper = estimate_epsilon_upper_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.5,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    interval_upper, interval_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.5,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    assert math.isfinite(upper)
    assert upper == pytest.approx(interval_upper, rel=0.0, abs=1e-12)
    assert interval_upper >= interval_lower or math.isinf(interval_upper)


def test_fixed_bin_bridge_runtime_policy_defaults_to_candidate_grid() -> None:
    runtime = resolve_fixed_bin_random_allocation_bridge_runtime_config(target_delta=1e-5)
    assert runtime.policy_name == "fixed_bin_bridge_candidate_grid_1e-2"
    assert runtime.loss_discretization == pytest.approx(1e-2, rel=0.0, abs=1e-12)
    assert runtime.tail_truncation == pytest.approx(1e-8, rel=0.0, abs=1e-12)


def test_fixed_bin_bridge_runtime_policy_respects_explicit_override() -> None:
    runtime = resolve_fixed_bin_random_allocation_bridge_runtime_config(
        target_delta=1e-5,
        loss_discretization=2e-2,
        convolution_method="geometric",
    )
    assert runtime.loss_discretization == pytest.approx(2e-2, rel=0.0, abs=1e-12)
    assert runtime.convolution_method == "geometric"
    assert runtime.policy_name != "fixed_bin_bridge_candidate_grid_1e-2"


def test_random_allocation_runtime_policy_resolver_supports_strict_and_efficient_modes() -> None:
    strict_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        runtime_policy="strict_exact_package",
    )
    efficient_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        runtime_policy="efficient_staged_grid",
    )

    assert strict_runtime.runtime_policy == "strict_exact_package"
    assert strict_runtime.clamp_to_package_grid is True
    assert strict_runtime.refinement_rounds == 0
    assert efficient_runtime.runtime_policy == "efficient_staged_grid"
    assert efficient_runtime.clamp_to_package_grid is False
    assert efficient_runtime.refinement_rounds == 2


def test_fixed_bin_bridge_candidate_grid_is_between_fast_and_fine_for_dpsgd_control() -> None:
    c_matrix = np.array([[1.0] * 6], dtype=np.float64)
    coarse_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        loss_discretization=5e-2,
    )
    candidate_runtime = resolve_fixed_bin_random_allocation_bridge_runtime_config(
        target_delta=1e-5
    )
    fine_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        loss_discretization=5e-3,
    )

    coarse_upper, coarse_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.0,
        target_delta=1e-5,
        runtime_config=coarse_runtime,
    )
    candidate_upper, candidate_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.0,
        target_delta=1e-5,
        runtime_config=candidate_runtime,
    )
    fine_upper, fine_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.0,
        target_delta=1e-5,
        runtime_config=fine_runtime,
    )

    assert coarse_upper >= candidate_upper >= fine_upper
    assert coarse_lower <= candidate_lower <= fine_lower


def test_fixed_bin_bridge_candidate_grid_is_between_fast_and_fine_for_small_bsr_fixture() -> None:
    c_matrix = np.array([[1.0, 0.5, 1.0, 0.5, 1.0, 0.5]], dtype=np.float64)
    coarse_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        loss_discretization=5e-2,
    )
    candidate_runtime = resolve_fixed_bin_random_allocation_bridge_runtime_config(
        target_delta=1e-5
    )
    fine_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        loss_discretization=5e-3,
    )

    coarse_upper, coarse_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.5,
        target_delta=1e-5,
        runtime_config=coarse_runtime,
    )
    candidate_upper, candidate_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.5,
        target_delta=1e-5,
        runtime_config=candidate_runtime,
    )
    fine_upper, fine_lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=c_matrix,
        bins=3,
        noise_multiplier=2.5,
        target_delta=1e-5,
        runtime_config=fine_runtime,
    )

    assert coarse_upper >= candidate_upper >= fine_upper
    assert coarse_lower <= candidate_lower <= fine_lower


def test_fixed_bin_bridge_ambient_realization_runtime_failure_translates_to_route_local_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(**kwargs):
        raise ValueError("Cannot enforce mass conservation with zero finite mass")

    monkeypatch.setattr(
        fixed_bin_random_allocation_module,
        "estimate_epsilon_range_random_allocation_from_initial_package",
        _raise_value_error,
    )

    with pytest.raises(
        NotImplementedError,
        match="ambient quantitative-window realization package is constructed",
    ):
        estimate_epsilon_range_fixed_bin_random_allocation(
            mechanism="bsr",
            c_matrix=np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            bins=2,
            noise_multiplier=1.0,
            target_delta=1e-5,
        )


def test_fixed_bin_bridge_ambient_nonfinite_upper_diagnostic_identifies_remove_all_infinity_mass() -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
    )
    c_matrix, _contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=coeffs,
        bands=4,
        horizon=980,
    )
    diagnostic = _diagnose_fixed_bin_ambient_nonfinite_upper_bound(
        mechanism="bsr",
        c_matrix=np.asarray(c_matrix, dtype=np.float64),
        bins=98,
        noise_multiplier=1_000.0,
        target_delta=1e-5,
    )
    assert diagnostic is not None
    assert diagnostic.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert diagnostic.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert diagnostic.blocker == "remove_composed_positive_infinity_mass_exceeds_delta"
    assert diagnostic.remove_final_pos_inf > 1e-5
    assert diagnostic.add_final_pos_inf < 1e-5


def test_fixed_bin_bridge_live_bsr_fixture_computes_after_explicit_remove_dual_reuse() -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
    )
    c_matrix, _contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=coeffs,
        bands=4,
        horizon=980,
    )
    sigma = get_noise_multiplier_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=np.asarray(c_matrix, dtype=np.float64),
        bins=98,
        target_epsilon=9.0,
        target_delta=1e-5,
    )
    upper, lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="bsr",
        c_matrix=np.asarray(c_matrix, dtype=np.float64),
        bins=98,
        noise_multiplier=sigma,
        target_delta=1e-5,
    )
    assert 2.0 < sigma < 2.6
    assert math.isfinite(upper)
    assert math.isfinite(lower)
    assert lower <= upper <= 9.01


def test_fixed_bin_bridge_noise_search_returns_bracketed_sigma_at_interval_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_estimate_upper(**kwargs):
        sigma = float(kwargs["noise_multiplier"])
        return 8.963280189895514 if sigma >= 1.25 else 9.25

    monkeypatch.setattr(
        fixed_bin_random_allocation_module,
        "estimate_epsilon_upper_fixed_bin_random_allocation",
        _fake_estimate_upper,
    )
    monkeypatch.setattr(fixed_bin_random_allocation_module, "MIN_NOISE_SEARCH_SIGMA_INTERVAL", 0.05)

    sigma = get_noise_multiplier_fixed_bin_random_allocation(
        mechanism="band_mf",
        c_matrix=np.eye(2, dtype=np.float64),
        bins=1,
        target_epsilon=9.0,
        target_delta=1e-5,
    )

    assert sigma == pytest.approx(1.25, rel=0.0, abs=1e-12)


def test_fixed_bin_bridge_noise_search_uses_upper_only_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def _fake_upper(**kwargs):
        sigma = float(kwargs["noise_multiplier"])
        calls.append(sigma)
        return 8.5 if sigma >= 1.5 else 9.5

    monkeypatch.setattr(
        fixed_bin_random_allocation_module,
        "estimate_epsilon_upper_fixed_bin_random_allocation",
        _fake_upper,
    )
    monkeypatch.setattr(
        fixed_bin_random_allocation_module,
        "estimate_epsilon_range_fixed_bin_random_allocation",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("full interval evaluator should not be used by search")
        ),
    )

    sigma = get_noise_multiplier_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=np.eye(2, dtype=np.float64),
        bins=1,
        target_epsilon=9.0,
        target_delta=1e-5,
    )

    assert sigma > 0.0
    assert calls


def test_fixed_bin_bridge_live_band_mf_fixture_computes_after_interval_floor_return() -> None:
    coeffs = generate_bandmf_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9999,
        steps=980,
    )
    c_matrix, _contract = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=coeffs,
        bands=4,
        horizon=980,
    )
    sigma = get_noise_multiplier_fixed_bin_random_allocation(
        mechanism="band_mf",
        c_matrix=np.asarray(c_matrix, dtype=np.float64),
        bins=98,
        target_epsilon=9.0,
        target_delta=1e-5,
    )
    upper, lower = estimate_epsilon_range_fixed_bin_random_allocation(
        mechanism="band_mf",
        c_matrix=np.asarray(c_matrix, dtype=np.float64),
        bins=98,
        noise_multiplier=sigma,
        target_delta=1e-5,
    )
    assert 1.0 < sigma < 1.5
    assert math.isfinite(upper)
    assert math.isfinite(lower)
    assert lower <= upper <= 9.01


def test_fixed_bin_bridge_live_bisr_p8_fixture_keeps_certified_ambient_route() -> None:
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="bisr",
        c_matrix=_live_bisr_fixed_bin_c_matrix(bands=8),
        bins=98,
        noise_multiplier=1.0,
        logical_horizon=980,
    )

    assert bridge.route == "fixed_bin_bridge_ambient_quantitative_window_realization_package"
    assert bridge.initial_package_route == "pair_driven_ambient_quantitative_window_realization_package"
    assert bridge.pair_driven_route == "pair_driven_exact_initial_package"


def test_fixed_bin_bridge_live_bandinvmf_p8_fixture_stays_explicit_until_conservative_route_exists() -> None:
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="bandinvmf",
        c_matrix=_live_bandinvmf_fixed_bin_c_matrix(bands=8),
        bins=98,
        noise_multiplier=1.0,
        logical_horizon=980,
    )

    assert (
        bridge.route
        == "fixed_bin_bridge_ambient_quantitative_window_uncertified_conservativity"
    )
    assert bridge.initial_package_route is None
    assert bridge.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"


def test_geometric_mass_conservation_accepts_all_positive_infinity_mass() -> None:
    pmf = np.zeros(5, dtype=np.float64)
    adjusted_pmf, p_neg_inf, p_pos_inf = random_allocation_module._enforce_mass_conservation(  # type: ignore[attr-defined]
        pmf,
        0.0,
        1.0,
        random_allocation_module._BoundType.DOMINATES,  # type: ignore[attr-defined]
    )
    dist = random_allocation_module._GeometricDiscreteDist(  # type: ignore[attr-defined]
        1.0,
        2.0,
        adjusted_pmf,
        p_neg_inf,
        p_pos_inf,
    ).truncate_edges(1e-8, random_allocation_module._BoundType.DOMINATES)  # type: ignore[attr-defined]
    assert np.allclose(dist.PMF_array, 0.0)
    assert dist.p_pos_inf == pytest.approx(1.0)
    assert dist.p_neg_inf == pytest.approx(0.0)


def test_fixed_bin_bridge_cifar_identity_control_collapses_to_random_allocation_gaussian_base() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="gaussian",
        c_matrix=np.eye(980, dtype=np.float64),
        bins=98,
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert "orthogonal equal-norm one-hot mode family" in package.bounds.justification
    assert package.route == "pair_driven_exact_initial_package"
    assert package.add.realization.p_loss_inf < 1e-5


def test_one_window_identity_control_matches_pair_driven_exact_route() -> None:
    pair_epsilon, fixed_bin_epsilon = _one_window_pair_driven_vs_fixed_bin_epsilons(
        c_matrix=np.eye(2, dtype=np.float64),
    )
    assert pair_epsilon == pytest.approx(3.699782020260986, rel=0.0, abs=1e-12)
    assert fixed_bin_epsilon == pytest.approx(pair_epsilon, rel=0.0, abs=1e-12)


def test_one_window_orthogonal_equal_norm_one_hot_family_matches_pair_driven_exact_route() -> None:
    controls = [
        (
            np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64),
            9.307319169419463,
        ),
        (
            np.array(
                [
                    [0.0, 2.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=np.float64,
            ),
            8.901385390575294,
        ),
    ]

    for c_matrix, expected_epsilon in controls:
        pair_epsilon, fixed_bin_epsilon = _one_window_pair_driven_vs_fixed_bin_epsilons(
            c_matrix=c_matrix,
        )
        assert pair_epsilon == pytest.approx(expected_epsilon, rel=0.0, abs=1e-12)
        assert fixed_bin_epsilon == pytest.approx(expected_epsilon, rel=0.0, abs=1e-12)


def test_one_window_overlapping_toeplitz_stays_separate_from_pair_driven_exact_route() -> None:
    two_window = _toy_lower_toeplitz_c_matrix([1.0, 1.0])
    three_window = _toy_lower_toeplitz_c_matrix([1.0, 1.0, 0.0])

    pair_two, fixed_two = _one_window_pair_driven_vs_fixed_bin_epsilons(c_matrix=two_window)
    pair_three, fixed_three = _one_window_pair_driven_vs_fixed_bin_epsilons(c_matrix=three_window)

    assert pair_two == pytest.approx(5.883839981786562, rel=0.0, abs=1e-12)
    assert fixed_two == pytest.approx(5.712149901082944, rel=0.0, abs=1e-12)
    assert pair_two - fixed_two > 0.1

    assert pair_three == pytest.approx(5.479830251366031, rel=0.0, abs=1e-12)
    assert fixed_three == pytest.approx(5.411037801724155, rel=0.0, abs=1e-12)
    assert pair_three - fixed_three > 0.05


def test_fixed_bin_bridge_accepts_logical_horizon_for_padded_identity_control() -> None:
    padded = np.eye(984, dtype=np.float64)

    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="gaussian",
        c_matrix=padded,
        bins=98,
        noise_multiplier=1.0,
        logical_horizon=980,
    )

    assert bridge.horizon == 980
    assert bridge.epochs == 10
    assert bridge.route == "fixed_bin_bridge_exact_pair_package"

    mode_family = _aggregate_fixed_bin_mode_family(
        c_matrix=padded,
        bins=98,
        logical_horizon=980,
    )
    assert len(mode_family) == 98
    assert len(mode_family[0]) == 980


def test_fixed_bin_bridge_noise_search_propagates_logical_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float | None] = []

    def _fake_estimate_upper(**kwargs):
        calls.append(kwargs.get("logical_horizon"))
        sigma = float(kwargs["noise_multiplier"])
        return 1.0 / sigma

    monkeypatch.setattr(
        fixed_bin_random_allocation_module,
        "estimate_epsilon_upper_fixed_bin_random_allocation",
        _fake_estimate_upper,
    )

    sigma = get_noise_multiplier_fixed_bin_random_allocation(
        mechanism="gaussian",
        c_matrix=np.eye(984, dtype=np.float64),
        bins=98,
        logical_horizon=980,
        target_epsilon=2.0,
        target_delta=1e-5,
    )

    assert sigma > 0.0
    assert calls
    assert all(call == 980 for call in calls)


def test_nonorthogonal_fixed_bin_exact_mixture_builds_ambient_quantitative_window_realization_package() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="gaussian",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert package.route == "pair_driven_ambient_quantitative_window_realization_package"
    assert package.remove_lower is not None
    assert package.add_lower is not None
    assert package.bounds.beta > 0.0


def test_ambient_quantitative_window_package_keeps_direct_pair_geometry_explicit() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    package = build_ambient_quantitative_window_package_from_exact_law(pair)
    assert package.route == "pair_driven_ambient_quantitative_window_policy"
    assert package.num_modes == 2
    assert package.policy.grid_size >= 2
    assert package.policy.lower <= package.policy.upper
    assert package.policy.per_event_tail > 0.0
    assert package.bounds.has_initial_bounds is True
    assert package.bounds.alpha == pytest.approx(package.policy.alpha)
    assert package.bounds.beta == pytest.approx(package.policy.beta)
    assert "AmbientFiniteGaussianMixtureQuantitativeWindows.lean" in package.theorem_alignment


def test_ambient_quantitative_window_inputs_are_distinct_from_evaluable_initial_package() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    inputs = resolve_pair_driven_ambient_quantitative_window_inputs(
        pair=pair,
        num_steps=2,
        num_selected=1,
        num_epochs=1,
    )
    assert inputs.route == "pair_driven_ambient_quantitative_window_policy"
    assert inputs.quantitative_window_package.route == "pair_driven_ambient_quantitative_window_policy"
    assert not hasattr(inputs.quantitative_window_package, "remove")
    assert "does not yet construct an evaluable PLD realization package" in (
        inputs.quantitative_window_package.bounds.justification
    )


def test_ambient_quantitative_window_realization_package_keeps_lower_and_upper_realizations_explicit() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert package.route == "pair_driven_ambient_quantitative_window_realization_package"
    assert package.remove_lower is not None
    assert package.add_lower is not None
    assert package.remove.realization.x_gap == pytest.approx(package.remove_lower.realization.x_gap)
    assert package.add.realization.x_gap == pytest.approx(package.add_lower.realization.x_gap)
    assert package.bounds.alpha > 0.0
    assert package.bounds.beta > 0.0


def test_exact_witness_family_package_is_distinct_from_single_realization_package() -> None:
    gaussian_pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[1.0, 0.0],
        reverse_mean=[0.0, 0.0],
        noise_multiplier=1.0,
    )
    exact_package = build_deterministic_initial_package_from_exact_law(gaussian_pair)
    family_pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    family_package = _build_deterministic_witness_family_package_from_exact_law(family_pair)
    assert exact_package.route == "pair_driven_exact_initial_package"
    assert family_package.route == "pair_driven_exact_witness_family_package"
    assert family_package.layer_kind == "deterministic_exact_witness_family_package"
    assert family_package.exact_law_route == "exact_fixed_bin_gaussian_mixture_pair"
    assert family_package.bounds.has_initial_bounds is True


def test_exact_witness_family_package_keeps_weighted_component_realizations_explicit() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    family_package = _build_deterministic_witness_family_package_from_exact_law(pair)
    assert len(family_package.family) == 2
    assert tuple(member.source_index for member in family_package.family) == (0, 1)
    assert tuple(member.weight for member in family_package.family) == pytest.approx((0.5, 0.5))
    assert tuple(member.forward_mode for member in family_package.family) == pair.forward_modes
    assert all(member.remove.direction == "remove" for member in family_package.family)
    assert all(member.add.direction == "add" for member in family_package.family)
    assert all(member.remove.realization.p_loss_inf == pytest.approx(0.0, abs=1e-12) for member in family_package.family)


def test_exact_witness_family_package_requires_singleton_centered_reverse_reference() -> None:
    pair = FiniteGaussianMixtureNeighboringPair(
        metadata=build_fixed_bin_exact_law_pair(
            mechanism="bsr",
            c_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            bins=2,
            noise_multiplier=1.0,
        ).metadata,
        mechanism="bsr",
        covariance_kind="common_covariance_sigma_squared_identity",
        noise_multiplier=1.0,
        centered_reference_mean=(0.0, 0.0),
        forward_modes=((1.0, 0.0), (0.0, 1.0)),
        forward_weights=(0.5, 0.5),
        reverse_modes=((0.0, 0.0), (1.0, 0.0)),
        reverse_weights=(0.5, 0.5),
    )
    with pytest.raises(NotImplementedError, match="singleton reverse reference law"):
        _build_deterministic_witness_family_package_from_exact_law(pair)


def test_exact_family_accountant_contract_is_distinct_from_single_realization_route() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    family_inputs = _resolve_pair_driven_exact_family_random_allocation_inputs(
        pair=pair,
        num_steps=2,
        num_selected=1,
        num_epochs=1,
    )
    exact_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=build_realizable_gaussian_one_step_neighboring_pair(
            mechanism="gaussian",
            forward_mean=[1.0, 0.0],
            reverse_mean=[0.0, 0.0],
            noise_multiplier=1.0,
        ),
        num_steps=1,
        num_selected=1,
        num_epochs=1,
    )
    assert family_inputs.route == "pair_driven_exact_family_accountant_contract"
    assert family_inputs.exact_family_contract.route == "pair_driven_exact_family_accountant_contract"
    assert exact_inputs.route == "pair_driven_exact_initial_package"


def test_exact_family_accountant_contract_keeps_source_law_family_explicit() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    contract = _build_exact_family_accountant_contract_from_exact_law(pair)
    assert contract.layer_kind == "exact_family_deterministic_accountant_contract"
    assert contract.witness_family_route == "pair_driven_exact_witness_family_package"
    assert tuple(member.source_index for member in contract.family) == (0, 1)
    assert tuple(member.weight for member in contract.family) == pytest.approx((0.5, 0.5))
    assert tuple(member.forward_mean for member in contract.family) == pair.forward_modes
    assert all(member.reference_mean == pair.centered_reference_mean for member in contract.family)
    assert all(member.source_law_kind == "forward_component_gaussian_source_law" for member in contract.family)


def test_exact_family_accountant_contract_does_not_present_weighted_pld_average_as_realization() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="bsr",
        c_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bins=2,
        noise_multiplier=1.0,
    )
    contract = _build_exact_family_accountant_contract_from_exact_law(pair)
    assert not hasattr(contract, "remove")
    assert not hasattr(contract, "add")
    assert contract.bounds.justification.startswith("exact ambient finite Gaussian-mixture pair represented")


def test_exact_family_round_pair_embeds_source_means_into_round_blocks() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="gaussian",
        c_matrix=np.array([[2.0]], dtype=np.float64),
        bins=1,
        noise_multiplier=1.0,
    )
    contract = _build_exact_family_accountant_contract_from_exact_law(pair)
    round_inputs = _resolve_exact_family_round_pair_inputs(
        contract=contract,
        num_steps_per_round=3,
        num_rounds=2,
    )
    assert round_inputs.route == "pair_driven_exact_family_round_pair"
    assert round_inputs.round_pair.metadata.route == "exact_family_random_allocation_round_pair"
    assert round_inputs.round_pair.noise_multiplier == pytest.approx(1.0)
    assert round_inputs.round_pair.centered_reference_mean == (0.0, 0.0, 0.0)
    assert round_inputs.round_pair.forward_modes == (
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
    )
    assert round_inputs.round_pair.forward_weights == pytest.approx((1.0 / 3.0,) * 3)


def test_exact_family_round_pair_keeps_family_explicit_instead_of_averaging() -> None:
    seed_pair = build_fixed_bin_exact_law_pair(
        mechanism="gaussian",
        c_matrix=np.array([[1.0]], dtype=np.float64),
        bins=1,
        noise_multiplier=1.0,
    )
    pair = FiniteGaussianMixtureNeighboringPair(
        metadata=seed_pair.metadata,
        mechanism="gaussian",
        covariance_kind="common_covariance_sigma_squared_identity",
        noise_multiplier=1.0,
        centered_reference_mean=(0.0,),
        forward_modes=((1.0,), (2.0,)),
        forward_weights=(0.25, 0.75),
        reverse_modes=((0.0,),),
        reverse_weights=(1.0,),
    )
    contract = _build_exact_family_accountant_contract_from_exact_law(pair)
    round_pair = _build_exact_family_round_pair_from_contract(
        contract=contract,
        num_steps_per_round=2,
    )
    assert round_pair.forward_modes == (
        (1.0, 0.0),
        (2.0, 0.0),
        (0.0, 1.0),
        (0.0, 2.0),
    )
    assert round_pair.forward_weights == pytest.approx((0.125, 0.375, 0.125, 0.375))
    assert (1.75, 0.0) not in round_pair.forward_modes
    package_inputs = _resolve_exact_family_round_pair_package_inputs(
        contract=contract,
        num_steps_per_round=2,
        num_rounds=1,
    )
    assert package_inputs.route == "pair_driven_exact_family_round_pair_package"
    assert package_inputs.pair_driven_inputs.initial_package.route == (
        "pair_driven_ambient_quantitative_window_realization_package"
    )


def test_exact_family_round_pair_package_reuses_pair_driven_evaluator_for_supported_slice() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="gaussian",
        c_matrix=np.array([[2.0]], dtype=np.float64),
        bins=1,
        noise_multiplier=1.0,
    )
    contract = _build_exact_family_accountant_contract_from_exact_law(pair)
    package_inputs = _resolve_exact_family_round_pair_package_inputs(
        contract=contract,
        num_steps_per_round=3,
        num_rounds=2,
    )
    assert package_inputs.route == "pair_driven_exact_family_round_pair_package"
    assert package_inputs.pair_driven_inputs.route == "pair_driven_exact_initial_package"
    assert package_inputs.pair_driven_inputs.initial_package.exact_law_route == "exact_family_random_allocation_round_pair"
    upper, lower = _estimate_epsilon_range_random_allocation_from_exact_family_round_pair_package(
        inputs=package_inputs,
        target_delta=1e-5,
    )
    assert math.isfinite(upper)
    assert math.isfinite(lower)
    assert 0.0 <= lower <= upper


def test_get_noise_multiplier_fixed_bin_random_allocation_raises_convergence_error_on_nonfinite_upper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opacus.accountants.analysis.random_allocation.fixed_bin.estimate_epsilon_upper_fixed_bin_random_allocation",
        lambda **kwargs: float("inf"),
    )
    with pytest.raises(NoiseSearchConvergenceError) as exc_info:
        get_noise_multiplier_fixed_bin_random_allocation(
            mechanism="gaussian",
            c_matrix=np.array([[1.0] * 6], dtype=np.float64),
            bins=3,
            target_epsilon=3.0,
            target_delta=1e-5,
        )
    exc = exc_info.value
    assert exc.accountant == "fixed_bin_random_allocation_bridge"
    assert exc.last_finite_sigma is None
    assert exc.last_finite_epsilon is None
    assert exc.last_nonfinite_sigma is not None


def test_exact_common_covariance_gaussian_one_step_pair_keeps_forward_and_reverse_explicit() -> None:
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[1.0, -2.0, 3.0],
        reverse_mean=[0.0, 0.0, 0.0],
        noise_multiplier=1.5,
    )
    assert pair.metadata.layer_kind == "exact_law_level_input"
    assert pair.metadata.accountant_package_kind == "not_initialized"
    assert pair.metadata.sampler_bridge_kind == "none"
    assert pair.metadata.route == "exact_common_covariance_gaussian_one_step_pair"
    assert pair.forward_mean == (1.0, -2.0, 3.0)
    assert pair.reverse_mean == (0.0, 0.0, 0.0)
    assert pair.covariance_kind == "common_covariance_sigma_squared_identity"


def test_exact_poisson_gaussian_mixture_pair_uses_centered_reference_law() -> None:
    pair = build_poisson_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=np.array([[1.0, 2.0]], dtype=np.float64),
        sampling_probability=0.25,
        noise_multiplier=2.0,
    )
    assert pair.metadata.layer_kind == "exact_law_level_input"
    assert pair.metadata.route == "exact_poisson_gaussian_mixture_pair"
    assert pair.centered_reference_mean == (0.0,)
    assert pair.reverse_modes == ((0.0,),)
    assert pair.reverse_weights == (1.0,)
    assert pair.forward_modes == ((0.0,), (2.0,), (1.0,), (3.0,))
    assert pair.forward_weights == pytest.approx((0.5625, 0.1875, 0.1875, 0.0625))


def test_exact_product_gaussian_mixture_reduces_to_poisson_when_probabilities_are_constant() -> None:
    c_matrix = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float64)
    poisson_pair = build_poisson_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=c_matrix,
        sampling_probability=0.3,
        noise_multiplier=1.25,
    )
    product_pair = build_product_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=c_matrix,
        participation_probabilities=[0.3, 0.3],
        noise_multiplier=1.25,
    )
    assert product_pair.forward_modes == poisson_pair.forward_modes
    assert product_pair.forward_weights == pytest.approx(poisson_pair.forward_weights, rel=0.0, abs=1e-12)
    assert product_pair.centered_reference_mean == poisson_pair.centered_reference_mean
    assert product_pair.reverse_modes == poisson_pair.reverse_modes
    assert product_pair.reverse_weights == poisson_pair.reverse_weights


def test_exact_law_module_docstring_pins_lean_and_theory_sources() -> None:
    import opacus.accountants.analysis.random_allocation.exact_laws as exact_laws_module

    assert exact_laws_module.__doc__ is not None
    assert "RealizableGaussianOneStep.lean" in exact_laws_module.__doc__
    assert "PoissonGaussianMixturePLD.lean" in exact_laws_module.__doc__
    assert "ProductGaussianMixturePLD.lean" in exact_laws_module.__doc__
    assert "PRV.tex" in exact_laws_module.__doc__


def test_dpsgd_style_control_uses_exact_common_covariance_pair_without_effective_sigma_shortcut() -> None:
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[1.0, 0.0, 0.0, 0.0],
        reverse_mean=[0.0, 0.0, 0.0, 0.0],
        noise_multiplier=0.9,
    )
    assert not hasattr(pair, "effective_sigma")
    assert pair.forward_mean != pair.reverse_mean
    assert pair.metadata.accountant_package_kind == "not_initialized"


def test_exact_mixture_control_matrix_covers_uncapped_poisson_and_constant_product_reduction() -> None:
    c_matrix = np.array([[1.0, -1.0, 2.0]], dtype=np.float64)
    poisson_pair = build_poisson_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=c_matrix,
        sampling_probability=0.5,
        noise_multiplier=1.0,
    )
    product_pair = build_product_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=c_matrix,
        participation_probabilities=[0.5, 0.5, 0.5],
        noise_multiplier=1.0,
    )
    assert len(poisson_pair.forward_modes) == 8
    assert sum(poisson_pair.forward_weights) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert product_pair.forward_modes == poisson_pair.forward_modes
    assert product_pair.forward_weights == pytest.approx(poisson_pair.forward_weights, rel=0.0, abs=1e-12)


def test_deterministic_initial_package_exposes_explicit_metadata_for_gaussian_control() -> None:
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[1.0, 0.0],
        reverse_mean=[0.0, 0.0],
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert package.layer_kind == "deterministic_initial_accountant_package"
    assert package.route == "pair_driven_exact_initial_package"
    assert package.exact_law_route == "exact_common_covariance_gaussian_one_step_pair"
    assert package.bounds.has_initial_bounds is True


def test_deterministic_initial_package_keeps_remove_remove_dual_and_add_explicit() -> None:
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[1.0, 0.0],
        reverse_mean=[0.0, 0.0],
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert package.remove.direction == "remove"
    assert package.remove.source_law_kind == "forward"
    assert package.remove_dual.direction == "remove_dual"
    assert package.remove_dual.source_law_kind == "reverse"
    assert package.add.direction == "add"
    assert package.add.source_law_kind == "reverse"


def test_deterministic_initial_package_exposes_explicit_domination_metadata() -> None:
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[2.0, 0.0],
        reverse_mean=[0.0, 0.0],
        noise_multiplier=2.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert package.bounds.alpha == 0.0
    assert package.bounds.beta == 0.0
    assert package.bounds.has_initial_bounds is True
    assert "zero slack" in package.bounds.justification


def test_pair_driven_deterministic_route_exercises_package_backed_gaussian_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _blocked_pld_import(monkeypatch)
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism="gaussian",
        forward_mean=[1.0],
        reverse_mean=[0.0],
        noise_multiplier=1.0,
    )
    pair_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=5,
        num_selected=1,
        num_epochs=4,
    )
    public_inputs = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[1.0],
        cycle_length=5,
        horizon=20,
        noise_multiplier=1.0,
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=1e-5)
    pair_upper, pair_lower = estimate_epsilon_range_random_allocation_from_initial_package(
        inputs=pair_inputs,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    public_upper, public_lower = estimate_epsilon_range_random_allocation(
        inputs=public_inputs,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    assert pair_inputs.route == "pair_driven_exact_initial_package"
    assert math.isfinite(pair_upper)
    assert math.isfinite(pair_lower)
    assert pair_lower <= pair_upper
    assert math.isfinite(public_upper)
    assert math.isfinite(public_lower)
    assert public_inputs.route == "pair_driven_public_exact_initial_package"
    assert public_inputs.exact_law_route == "exact_common_covariance_gaussian_one_step_pair"
    assert public_inputs.initial_package_route == "pair_driven_exact_initial_package"
    assert public_upper == pytest.approx(pair_upper, rel=0.0, abs=1e-12)
    assert public_lower == pytest.approx(pair_lower, rel=0.0, abs=1e-12)


def test_supported_public_repeated_family_keeps_repeated_contract_metadata() -> None:
    state = {
        "mechanism": "bandinvmf",
        "_noise_mechanism": "bandinvmf",
        "bandinvmf_inv_coeffs": [1.0, -0.1, -0.02],
        "coeffs": [1.0, 0.5, 0.25],
        "random_allocation_accountant_coeffs": [1.0, 0.5, 0.25],
        "random_allocation_accountant_coeffs_source": "abs_factor_c_col",
    }
    semantics = SamplingSemantics(
        sampling_mode="k_out_of_t",
        privacy_metadata={"num_steps": 5, "num_selected": 1},
    )
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism="bandinvmf",
        mechanism_state=state,
        sampling_semantics=semantics,
        kwargs={"bnb_horizon": 20, "num_selected": 1},
        noise_multiplier=2.0,
    )
    bridge = resolve_fixed_bin_random_allocation_bridge_inputs(
        mechanism="bandinvmf",
        c_matrix=np.array([[1.0] * 20], dtype=np.float64),
        bins=5,
        noise_multiplier=2.0,
    )
    assert inputs.package_alignment_kind == "repeated_k_out_of_t"
    assert inputs.route == "pair_driven_public_exact_initial_package"
    assert inputs.exact_law_route == "exact_common_covariance_gaussian_one_step_pair"
    assert inputs.initial_package_route == "pair_driven_exact_initial_package"
    assert "fixed-bin balls-in-bins bridge" in inputs.package_alignment_notes
    assert bridge.route != inputs.route
    assert bridge.source_law_kind == "balls_in_bins_fixed_bin"


def test_unsupported_public_repeated_family_fails_clearly() -> None:
    with pytest.raises(ValueError, match="public exact-law route does not support"):
        resolve_random_allocation_accountant_inputs(
            mechanism="unsupported_mechanism",
            accountant_coeffs=[1.0],
            cycle_length=5,
            horizon=20,
            noise_multiplier=1.0,
        )


@pytest.mark.parametrize("mechanism", ["bifr", "blt"])
def test_supported_public_repeated_family_resolves_for_widened_matrix(mechanism: str) -> None:
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=mechanism,
        accountant_coeffs=[1.0, 0.5],
        cycle_length=5,
        horizon=20,
        noise_multiplier=1.0,
    )

    assert inputs.mechanism == mechanism
    assert inputs.route == "pair_driven_public_exact_initial_package"
    assert inputs.package_alignment_kind == "repeated_k_out_of_t"


def test_exact_mixture_fixture_builds_initial_package_for_first_1d_control() -> None:
    pair = build_poisson_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=np.array([[1.0, 2.0]], dtype=np.float64),
        sampling_probability=0.25,
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert package.layer_kind == "deterministic_initial_accountant_package"
    assert package.exact_law_route == "exact_poisson_gaussian_mixture_pair"
    assert package.bounds.has_initial_bounds is True
    assert "finite Gaussian-mixture pair" in package.bounds.justification


def test_degenerate_fixed_bin_exact_mixture_collapses_to_exact_gaussian_initial_package() -> None:
    pair = build_fixed_bin_exact_law_pair(
        mechanism="gaussian",
        c_matrix=np.array([[0.5] * 6], dtype=np.float64),
        bins=3,
        noise_multiplier=1.0,
    )
    package = build_deterministic_initial_package_from_exact_law(pair)
    assert "collapses to an exact common-covariance Gaussian" in package.bounds.justification
    assert package.add.realization.p_loss_inf < 1e-9


def test_pair_driven_deterministic_route_supports_first_exact_mixture_control(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    pair = build_poisson_gaussian_mixture_neighboring_pair(
        mechanism="gaussian",
        c_matrix=np.array([[1.0, 2.0]], dtype=np.float64),
        sampling_probability=0.25,
        noise_multiplier=1.0,
    )
    inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=3,
        num_selected=1,
        num_epochs=2,
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=1e-5)
    upper, lower = estimate_epsilon_range_random_allocation_from_initial_package(
        inputs=inputs,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    assert math.isfinite(lower)
    assert upper >= lower or math.isinf(upper)


def test_initial_package_module_docstring_pins_numerics_theorem_surface() -> None:
    import opacus.accountants.analysis.random_allocation.initial_package as initial_package_module

    assert initial_package_module.__doc__ is not None
    assert "PLDRandomAllocationNumerics.lean" in initial_package_module.__doc__
    assert "ConcreteRandomAllocationInitialApprox" in initial_package_module.__doc__
    assert "InitialBounds" in initial_package_module.__doc__
    assert "valid_of_pair_sources" in initial_package_module.__doc__
    assert "tight_of_pair_bounds" in initial_package_module.__doc__


def test_resolved_random_allocation_inputs_expose_general_k_contract_and_reduction() -> None:
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[3.0, 4.0],
        cycle_length=10,
        horizon=20,
        noise_multiplier=10.0,
        kwargs={"num_selected": 3},
    )
    assert inputs.pld_num_steps == 10
    assert inputs.pld_num_selected == 3
    assert inputs.pld_num_epochs == 2
    assert inputs.reduced_num_steps_per_round == 3
    assert inputs.reduced_num_rounds == 6
    assert "exactKOutOfTReductionTheoremTarget" in inputs.package_alignment_notes


def test_general_k_floor_reduction_matches_reduced_one_out_of_t_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        loss_discretization=5e-2,
        tail_truncation=1e-6,
        max_grid_fft=1_000_000,
        max_grid_mult=30_000,
        convolution_method="geometric",
    )
    general_k = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[3.0, 4.0],
        cycle_length=10,
        horizon=20,
        noise_multiplier=10.0,
        kwargs={"num_selected": 3},
    )
    reduced = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[3.0, 4.0],
        cycle_length=3,
        horizon=18,
        noise_multiplier=10.0,
    )
    general_upper, general_lower = estimate_epsilon_range_random_allocation(
        inputs=general_k,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    reduced_upper, reduced_lower = estimate_epsilon_range_random_allocation(
        inputs=reduced,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    assert general_upper == pytest.approx(reduced_upper, rel=0.0, abs=1e-12)
    assert general_lower == pytest.approx(reduced_lower, rel=0.0, abs=1e-12)


def test_general_k_random_allocation_matches_frozen_interval_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    case = _fixture_case("general_k_gaussian_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
        kwargs={"num_selected": case["num_selected"]},
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=case["target_delta"], **case["runtime_config"])
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism=case["mechanism"],
        forward_mean=case["accountant_coeffs"],
        reverse_mean=[0.0] * len(case["accountant_coeffs"]),
        noise_multiplier=case["noise_multiplier"],
    )
    pair_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=case["cycle_length"],
        num_selected=case["num_selected"],
        num_epochs=case["horizon"] // case["cycle_length"],
    )
    upper, lower = estimate_epsilon_range_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    pair_upper, pair_lower = estimate_epsilon_range_random_allocation_from_initial_package(
        inputs=pair_inputs,
        target_delta=case["target_delta"],
        runtime_config=runtime,
    )
    assert upper == pytest.approx(pair_upper, rel=0.0, abs=1e-12)
    assert lower == pytest.approx(pair_lower, rel=0.0, abs=1e-12)


def test_general_k_random_allocation_runtime_does_not_require_pld_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    case = _fixture_case("general_k_gaussian_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
        kwargs={"num_selected": case["num_selected"]},
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=case["target_delta"], **case["runtime_config"])
    epsilon = estimate_epsilon_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    upper, lower = estimate_epsilon_range_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    assert math.isfinite(epsilon)
    assert math.isfinite(upper)
    assert math.isfinite(lower)


def test_random_allocation_range_is_not_placeholder_degenerate_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    case = _fixture_case("multi_step_gaussian_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=case["target_delta"], **case["runtime_config"])
    upper, lower = estimate_epsilon_range_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    assert math.isfinite(upper)
    assert math.isfinite(lower)
    assert lower < upper


def test_random_allocation_range_matches_frozen_interval_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    case = _fixture_case("multi_step_gaussian_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=case["target_delta"], **case["runtime_config"])
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism=case["mechanism"],
        forward_mean=case["accountant_coeffs"],
        reverse_mean=[0.0] * len(case["accountant_coeffs"]),
        noise_multiplier=case["noise_multiplier"],
    )
    pair_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=case["cycle_length"],
        num_selected=1,
        num_epochs=case["horizon"] // case["cycle_length"],
    )
    upper, lower = estimate_epsilon_range_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    pair_upper, pair_lower = estimate_epsilon_range_random_allocation_from_initial_package(
        inputs=pair_inputs,
        target_delta=case["target_delta"],
        runtime_config=runtime,
    )
    assert upper == pytest.approx(pair_upper, rel=0.0, abs=1e-12)
    assert lower == pytest.approx(pair_lower, rel=0.0, abs=1e-12)


def test_random_allocation_interval_surface_is_the_local_numerical_accountant_layer() -> None:
    assert random_allocation_module.__doc__ is not None
    assert "PLDRandomAllocationNumerics.lean" in random_allocation_module.__doc__


def test_random_allocation_range_preserves_basic_noise_monotonicity(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    low_noise_inputs = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[3.0, 4.0],
        cycle_length=5,
        horizon=20,
        noise_multiplier=10.0,
    )
    high_noise_inputs = resolve_random_allocation_accountant_inputs(
        mechanism="gaussian",
        accountant_coeffs=[3.0, 4.0],
        cycle_length=5,
        horizon=20,
        noise_multiplier=12.0,
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=1e-5, loss_discretization=5e-2, tail_truncation=1e-6, max_grid_fft=1_000_000, max_grid_mult=30_000, convolution_method="geometric")
    low_upper, low_lower = estimate_epsilon_range_random_allocation(inputs=low_noise_inputs, target_delta=1e-5, runtime_config=runtime)
    high_upper, high_lower = estimate_epsilon_range_random_allocation(inputs=high_noise_inputs, target_delta=1e-5, runtime_config=runtime)
    assert not (high_upper > low_upper and high_lower > low_lower)


def test_multi_step_gaussian_random_allocation_matches_frozen_upper_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    case = _fixture_case("multi_step_gaussian_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=case["target_delta"], **case["runtime_config"])
    observed = estimate_epsilon_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism=case["mechanism"],
        forward_mean=case["accountant_coeffs"],
        reverse_mean=[0.0] * len(case["accountant_coeffs"]),
        noise_multiplier=case["noise_multiplier"],
    )
    pair_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=case["cycle_length"],
        num_selected=1,
        num_epochs=case["horizon"] // case["cycle_length"],
    )
    pair_observed = estimate_epsilon_random_allocation_from_initial_package(
        inputs=pair_inputs,
        target_delta=case["target_delta"],
        runtime_config=runtime,
    )
    assert math.isfinite(observed)
    assert observed == pytest.approx(pair_observed, rel=0.0, abs=1e-12)


def test_single_step_realization_random_allocation_matches_frozen_upper_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    _blocked_pld_import(monkeypatch)
    case = _fixture_case("single_step_realization_case")
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=case["mechanism"],
        accountant_coeffs=case["accountant_coeffs"],
        cycle_length=case["cycle_length"],
        horizon=case["horizon"],
        noise_multiplier=case["noise_multiplier"],
    )
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=case["target_delta"], **case["runtime_config"])
    observed = estimate_epsilon_random_allocation(inputs=inputs, target_delta=case["target_delta"], runtime_config=runtime)
    pair = build_realizable_gaussian_one_step_neighboring_pair(
        mechanism=case["mechanism"],
        forward_mean=case["accountant_coeffs"],
        reverse_mean=[0.0] * len(case["accountant_coeffs"]),
        noise_multiplier=case["noise_multiplier"],
    )
    pair_inputs = resolve_pair_driven_random_allocation_inputs(
        pair=pair,
        num_steps=case["cycle_length"],
        num_selected=1,
        num_epochs=case["horizon"] // case["cycle_length"],
    )
    pair_observed = estimate_epsilon_random_allocation_from_initial_package(
        inputs=pair_inputs,
        target_delta=case["target_delta"],
        runtime_config=runtime,
    )
    assert math.isfinite(observed)
    assert observed == pytest.approx(pair_observed, rel=0.0, abs=1e-12)


def test_random_allocation_fixture_metadata_records_package_provenance() -> None:
    fixture = _load_fixture()
    assert fixture["source"]["author_package_root"] == "src/PLD_accounting"
    assert fixture["source"]["generator"] == "scripts/generate_random_allocation_pld_accounting_goldens.py"
    assert GENERATOR_PATH.exists()
    assert not OLD_GENERATOR_PATH.exists()
    seen_general_k = False
    for case in fixture["cases"]:
        assert case["author_source"] == "PLD_accounting.general_allocation_epsilon"
        assert "expected_epsilon_upper" in case
        assert "expected_epsilon_lower" in case
        if case["name"] == "general_k_gaussian_case":
            seen_general_k = True
            assert case["num_selected"] == 3
        assert set(case["runtime_config"].keys()) == {
            "loss_discretization",
            "tail_truncation",
            "max_grid_fft",
            "max_grid_mult",
            "convolution_method",
        }
    assert seen_general_k


def test_amplified_bsr_random_allocation_old_tail_policy_has_finite_and_nonfinite_sigma_probes() -> None:
    state, semantics = _amplified_bsr_random_allocation_state()
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=1e-5, tail_truncation=1e-6)

    finite_inputs = resolve_random_allocation_accountant_inputs(
        mechanism="bsr",
        mechanism_state=state,
        sampling_semantics=semantics,
        kwargs={"num_selected": 1, "bnb_horizon": 980},
        noise_multiplier=2.4505,
    )
    nonfinite_inputs = resolve_random_allocation_accountant_inputs(
        mechanism="bsr",
        mechanism_state=state,
        sampling_semantics=semantics,
        kwargs={"num_selected": 1, "bnb_horizon": 980},
        noise_multiplier=2.45,
    )

    finite_eps = estimate_epsilon_random_allocation(
        inputs=finite_inputs,
        target_delta=1e-5,
        runtime_config=runtime,
    )
    nonfinite_eps = estimate_epsilon_random_allocation(
        inputs=nonfinite_inputs,
        target_delta=1e-5,
        runtime_config=runtime,
    )

    assert math.isfinite(finite_eps)
    assert finite_eps < 9.0
    assert math.isinf(nonfinite_eps)


def test_amplified_bsr_random_allocation_default_runtime_reaches_beyond_false_ceiling() -> None:
    state, semantics = _amplified_bsr_random_allocation_state()
    runtime = resolve_random_allocation_gaussian_runtime_config(target_delta=1e-5)
    supported_inputs = resolve_random_allocation_accountant_inputs(
        mechanism="bsr",
        mechanism_state=state,
        sampling_semantics=semantics,
        kwargs={"num_selected": 1, "bnb_horizon": 980},
        noise_multiplier=1.0,
    )

    supported_eps = estimate_epsilon_random_allocation(
        inputs=supported_inputs,
        target_delta=1e-5,
        runtime_config=runtime,
    )

    assert math.isfinite(supported_eps)
    assert supported_eps > 9.0


def test_get_noise_multiplier_random_allocation_bsr_old_tail_cliff_raises_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, semantics = _amplified_bsr_random_allocation_state()
    monkeypatch.setattr("opacus.accountants.utils.MAX_NOISE_SEARCH_BINARY_STEPS", 48)

    with pytest.raises(NoiseSearchConvergenceError) as exc_info:
        get_noise_multiplier(
            target_epsilon=9.0,
            target_delta=1e-5,
            sample_rate=1.0 / 98.0,
            steps=980,
            accountant="random_allocation",
            mechanism_state=state,
            sampling_semantics=semantics,
            random_allocation_tail_truncation=1e-6,
        )

    exc = exc_info.value
    assert exc.accountant == "random_allocation"
    assert exc.target_epsilon == pytest.approx(9.0)
    assert exc.target_delta == pytest.approx(1e-5)
    assert exc.last_finite_sigma is not None
    assert exc.last_finite_epsilon is not None
    assert exc.last_finite_epsilon < 9.0
    assert exc.last_nonfinite_sigma is not None
    assert exc.last_nonfinite_sigma < exc.last_finite_sigma
    assert exc.iterations >= 1


def test_random_allocation_debug_timing_preserves_convergence_error_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state, semantics = _amplified_bsr_random_allocation_state()
    monkeypatch.setattr("opacus.accountants.utils.MAX_NOISE_SEARCH_BINARY_STEPS", 48)

    monkeypatch.delenv("DEBUG_TIMING", raising=False)
    with pytest.raises(NoiseSearchConvergenceError) as baseline_info:
        get_noise_multiplier(
            target_epsilon=9.0,
            target_delta=1e-5,
            sample_rate=1.0 / 98.0,
            steps=980,
            accountant="random_allocation",
            mechanism_state=state,
            sampling_semantics=semantics,
            random_allocation_tail_truncation=1e-6,
        )
    capsys.readouterr()

    monkeypatch.setenv("DEBUG_TIMING", "1")
    with pytest.raises(NoiseSearchConvergenceError) as debug_info:
        get_noise_multiplier(
            target_epsilon=9.0,
            target_delta=1e-5,
            sample_rate=1.0 / 98.0,
            steps=980,
            accountant="random_allocation",
            mechanism_state=state,
            sampling_semantics=semantics,
            random_allocation_tail_truncation=1e-6,
        )
    out = capsys.readouterr().out

    baseline = baseline_info.value
    observed = debug_info.value
    assert observed.accountant == baseline.accountant
    assert observed.target_epsilon == pytest.approx(baseline.target_epsilon)
    assert observed.target_delta == pytest.approx(baseline.target_delta)
    assert observed.last_finite_sigma == pytest.approx(baseline.last_finite_sigma, rel=0.0, abs=1e-12)
    assert observed.last_finite_epsilon == pytest.approx(baseline.last_finite_epsilon, rel=0.0, abs=1e-12)
    assert observed.last_nonfinite_sigma == pytest.approx(baseline.last_nonfinite_sigma, rel=0.0, abs=1e-12)
    assert observed.iterations == baseline.iterations
    assert "random_allocation search_context" in out


def test_get_noise_multiplier_random_allocation_amplified_bsr_returns_finite_sigma_with_default_runtime() -> None:
    state, semantics = _amplified_bsr_random_allocation_state()

    sigma = get_noise_multiplier(
        target_epsilon=9.0,
        target_delta=1e-5,
        sample_rate=1.0 / 98.0,
        steps=980,
        accountant="random_allocation",
        mechanism_state=state,
        sampling_semantics=semantics,
    )

    assert math.isfinite(sigma)
    assert sigma > 0.0


def test_get_noise_multiplier_random_allocation_debug_output_exposes_probe_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DEBUG_TIMING", "1")

    sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0 / 5.0,
        steps=20,
        accountant="random_allocation",
        mechanism_state={
            "mechanism": "gaussian",
            "_noise_mechanism": "gaussian",
            "coeffs": [1.0],
            "random_allocation_accountant_coeffs": [1.0],
            "random_allocation_accountant_coeffs_source": "identity_c_col",
        },
        sampling_semantics=SamplingSemantics(
            sampling_mode="k_out_of_t",
            privacy_metadata={"num_steps": 5, "num_selected": 1},
        ),
    )
    out = capsys.readouterr().out

    assert math.isfinite(sigma)
    assert "random_allocation search_context" in out
    assert "mechanism=gaussian" in out
    assert "sample_rate=0.2" in out
    assert "steps=20" in out
    assert "num_steps=5" in out
    assert "num_selected=1" in out
    assert "reduced_num_steps_per_round=5" in out
    assert "reduced_num_rounds=4" in out
    assert "binary_iter=" in out


def test_get_noise_multiplier_random_allocation_returns_finite_sigma_on_convergent_input() -> None:
    sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0 / 5.0,
        steps=20,
        accountant="random_allocation",
        mechanism_state={
            "mechanism": "gaussian",
            "_noise_mechanism": "gaussian",
            "coeffs": [1.0],
            "random_allocation_accountant_coeffs": [1.0],
            "random_allocation_accountant_coeffs_source": "identity_c_col",
        },
        sampling_semantics=SamplingSemantics(
            sampling_mode="k_out_of_t",
            privacy_metadata={"num_steps": 5, "num_selected": 1},
        ),
    )

    assert math.isfinite(sigma)
    assert sigma > 0.0


@pytest.mark.parametrize(
    ("mechanism", "coeffs"),
    [
        ("gaussian", [1.0, 0.5]),
        ("bsr", [1.0, 0.5]),
        ("bisr", [1.0, 0.5]),
        ("bandmf", [1.0, 0.5]),
        ("bandinvmf", [1.0, 0.5]),
    ],
)
def test_repeated_random_allocation_efficient_runtime_keeps_finite_interval_against_strict(
    mechanism: str,
    coeffs: list[float],
) -> None:
    inputs = resolve_random_allocation_accountant_inputs(
        mechanism=mechanism,
        accountant_coeffs=coeffs,
        cycle_length=1,
        horizon=2,
        noise_multiplier=1.0,
    )
    strict_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        runtime_policy="strict_exact_package",
    )
    efficient_runtime = resolve_random_allocation_gaussian_runtime_config(
        target_delta=1e-5,
        runtime_policy="efficient_staged_grid",
    )

    strict_upper, strict_lower = estimate_epsilon_range_random_allocation(
        inputs=inputs,
        target_delta=1e-5,
        runtime_config=strict_runtime,
    )
    efficient_upper, efficient_lower = estimate_epsilon_range_random_allocation(
        inputs=inputs,
        target_delta=1e-5,
        runtime_config=efficient_runtime,
    )

    assert math.isfinite(strict_upper)
    assert math.isfinite(strict_lower)
    assert math.isfinite(efficient_upper)
    assert math.isfinite(efficient_lower)
    assert strict_upper >= strict_lower
    assert efficient_upper >= efficient_lower
    assert efficient_upper >= strict_lower - 1e-9


def test_get_noise_multiplier_random_allocation_strict_runtime_never_bootstraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        accountant_utils_module,
        "_bootstrap_random_allocation_sigma",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("strict calibration must not call bootstrap")
        ),
    )

    sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0,
        steps=2,
        accountant="random_allocation",
        mechanism_state={
            "mechanism": "gaussian",
            "_noise_mechanism": "gaussian",
            "coeffs": [1.0],
            "random_allocation_accountant_coeffs": [1.0],
            "random_allocation_accountant_coeffs_source": "identity_c_col",
        },
        sampling_semantics=SamplingSemantics(
            sampling_mode="k_out_of_t",
            privacy_metadata={"num_steps": 1, "num_selected": 1},
        ),
        random_allocation_runtime_policy="strict_exact_package",
        epsilon_tolerance=0.1,
    )

    assert math.isfinite(sigma)
    assert sigma > 0.0


def test_get_noise_multiplier_random_allocation_efficient_runtime_returns_finite_sigma() -> None:
    mechanism_state = {
        "mechanism": "gaussian",
        "_noise_mechanism": "gaussian",
        "coeffs": [1.0],
        "random_allocation_accountant_coeffs": [1.0],
        "random_allocation_accountant_coeffs_source": "identity_c_col",
    }
    semantics = SamplingSemantics(
        sampling_mode="k_out_of_t",
        privacy_metadata={"num_steps": 1, "num_selected": 1},
    )
    strict_sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0,
        steps=2,
        accountant="random_allocation",
        mechanism_state=mechanism_state,
        sampling_semantics=semantics,
        random_allocation_runtime_policy="strict_exact_package",
        epsilon_tolerance=0.1,
    )
    efficient_sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0,
        steps=2,
        accountant="random_allocation",
        mechanism_state=mechanism_state,
        sampling_semantics=semantics,
        random_allocation_runtime_policy="efficient_staged_grid",
        epsilon_tolerance=0.1,
    )

    assert math.isfinite(strict_sigma)
    assert math.isfinite(efficient_sigma)
    assert strict_sigma > 0.0
    assert efficient_sigma > 0.0
    assert abs(strict_sigma - efficient_sigma) / strict_sigma < 0.5


@pytest.mark.parametrize(
    ("mechanism", "state", "semantics"),
    [
        (
            "gaussian",
            {
                "mechanism": "gaussian",
                "_noise_mechanism": "gaussian",
                "coeffs": [1.0],
                "random_allocation_accountant_coeffs": [1.0],
                "random_allocation_accountant_coeffs_source": "identity_c_col",
            },
            SamplingSemantics(
                sampling_mode="k_out_of_t",
                privacy_metadata={"num_steps": 1, "num_selected": 1},
            ),
        ),
        (
            "bsr",
            {
                "mechanism": "bsr",
                "_noise_mechanism": "bsr",
                "coeffs": [1.0, 0.5],
                "random_allocation_accountant_coeffs": [1.0, 0.5],
                "random_allocation_accountant_coeffs_source": "raw_c_col",
                "bsr_bands": 2,
            },
            SamplingSemantics(
                sampling_mode="k_out_of_t",
                privacy_metadata={"num_steps": 1, "num_selected": 1},
            ),
        ),
    ],
)
def test_get_noise_multiplier_random_allocation_efficient_runtime_stays_close_to_strict_for_representative_rows(
    mechanism: str,
    state: dict,
    semantics: SamplingSemantics,
) -> None:
    strict_sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0,
        steps=2,
        accountant="random_allocation",
        mechanism_state=state,
        sampling_semantics=semantics,
        random_allocation_runtime_policy="strict_exact_package",
        epsilon_tolerance=0.1,
    )
    efficient_sigma = get_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=1.0,
        steps=2,
        accountant="random_allocation",
        mechanism_state=state,
        sampling_semantics=semantics,
        random_allocation_runtime_policy="efficient_staged_grid",
        epsilon_tolerance=0.1,
    )

    assert math.isfinite(strict_sigma)
    assert math.isfinite(efficient_sigma)
    assert strict_sigma > 0.0
    assert efficient_sigma > 0.0
    assert abs(strict_sigma - efficient_sigma) / strict_sigma < 0.35


def test_bisr_get_epsilon_random_allocation_uses_persisted_accountant_coeffs_with_blocked_package_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _blocked_pld_import(monkeypatch)

    pe = PrivacyEngine(accountant="bnb")
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.9999)

    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_tiny_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="flat",
        grad_sample_mode="hooks",
        total_steps=16,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bisr",
            accounting_mode="bnb_accountant",
            mechanism_state={"bsr_bands": 2},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bands": 2, "bins": 4},
        ),
    )

    state = pe.noise_mechanism_config.mechanism_state
    assert all(float(c) >= 0.0 for c in state["bnb_accountant_coeffs"])
    original_accountant_coeffs = tuple(float(c) for c in state["bnb_accountant_coeffs"])
    state["coeffs"] = [-abs(float(c)) for c in state["bnb_accountant_coeffs"]]
    pe.accountant.step(noise_multiplier=1.0, sample_rate=1.0)

    epsilon = pe.get_epsilon(1e-5, bnb_accounting_backend="deterministic")

    assert math.isfinite(float(epsilon))
    used_inputs = resolve_random_allocation_accountant_inputs(
        mechanism=str(state.get("mechanism", state.get("name", "gaussian"))),
        mechanism_state=state,
        sampling_semantics=pe.sampling_semantics,
        kwargs={
            "bnb_accountant_coeffs": state["bnb_accountant_coeffs"],
            "bnb_cycle_length": int(pe.sampling_semantics.privacy_metadata["bins"]),
            "bnb_horizon": int(state["bnb_c_matrix"].shape[1]),
        },
        noise_multiplier=1.0,
    )
    assert tuple(float(c) for c in used_inputs.accountant_coeffs) == original_accountant_coeffs
    assert tuple(float(c) for c in used_inputs.accountant_coeffs) != tuple(float(c) for c in state["coeffs"])
