#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import unittest
from unittest import mock

import torch

from opacus.accountants.analysis.bisr import (
    derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs,
    generate_bisr_coeffs_from_sgd_workload,
)
from opacus.accountants.bnb import BNBAccountant
from opacus.accountants.analysis.bifr import (
    derive_bifr_amplified_accountant_coeffs_from_factor_coeffs,
    resolve_bifr_exact_factor_coeffs_for_accounting,
)
from opacus.accountants.analysis.bnb import (
    _build_balls_in_bins_modes_matrix,
    _assign_bnb_chunk_specs_to_shard,
    _make_bnb_broadcast_result_tensor,
    _reduce_bnb_llr_chunks_to_coordinator,
    _resolve_bnb_distributed_mode,
    BNBCalibrationReport,
    build_balls_in_bins_sigma_reuse_state,
    DeltaVerificationResult,
    GaussianMixture,
    SingleVerificationResult,
    build_bnb_toeplitz_c_matrix_and_contract,
    build_b_min_sep_gaussian_mixture,
    estimate_balls_in_bins_epsilon_monte_carlo,
    estimate_balls_in_bins_epsilon_monte_carlo_optimistic,
    build_lower_toeplitz_c_matrix_from_coeffs,
    calibrate_sigma_evr_binary_search,
    compute_llr_sample_chunks,
    compute_llr_samples,
    describe_bnb_calibration_report,
    estimate_b_min_sep_epsilon_monte_carlo,
    estimate_delta_upper_bound_single_verify_from_llr_chunks,
    estimate_delta_upper_bound_single_verify_from_llr_samples,
    estimate_epsilon_from_llr_chunks,
    estimate_epsilon_from_llr_samples,
    estimate_hockey_stick_delta_from_llr_chunks,
    estimate_hockey_stick_delta_from_llr_samples,
    find_sigma_binary_search,
    make_bnb_calibration_report,
    make_bnb_toeplitz_c_matrix_contract,
    normalize_bnb_accountant_coeffs,
    parse_bnb_calibration_report,
    generate_mixture_samples,
    get_bnb_base_delta,
    sample_b_min_sep_llr,
    sample_balls_in_bins_llr_chunks,
    select_evr_candidate_ladder,
    select_evr_candidate_ladder_two_sided,
    split_confidence_alpha,
    verify_evr_confidence_split,
    verify_hockey_stick_delta_hoeffding,
)
from opacus.accountants.utils import get_noise_multiplier
from opacus.mechanism_contracts import SamplingSemantics


def _legacy_balls_in_bins_epsilon_optimistic_baseline(
    *,
    coeffs,
    cycle_length: int,
    horizon: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    seed: int = 0,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    chunk_size: int | None = None,
    num_workers: int = 0,
) -> float:
    positive_chunks = sample_balls_in_bins_llr_chunks(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed),
        chunk_size=chunk_size,
        num_workers=int(num_workers),
        positive_sample=True,
        backend="cpu",
        device="cpu",
        distributed_mode="none",
        distributed_dp_runtime=False,
    )
    negative_chunks = sample_balls_in_bins_llr_chunks(
        coeffs=coeffs,
        cycle_length=int(cycle_length),
        horizon=int(horizon),
        sigma=float(noise_multiplier),
        num_samples=int(num_samples),
        seed=int(seed) + 1,
        chunk_size=chunk_size,
        num_workers=int(num_workers),
        positive_sample=False,
        backend="cpu",
        device="cpu",
        distributed_mode="none",
        distributed_dp_runtime=False,
    )
    positive_epsilon = estimate_epsilon_from_llr_chunks(
        target_delta=float(target_delta),
        llr_chunks=[torch.as_tensor(chunk, dtype=torch.float64) for chunk in positive_chunks],
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    negative_epsilon = estimate_epsilon_from_llr_chunks(
        target_delta=float(target_delta),
        llr_chunks=[torch.as_tensor(chunk, dtype=torch.float64) for chunk in negative_chunks],
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    return float(max(float(positive_epsilon), float(negative_epsilon)))


def _representative_high_memory_bisr_proxy():
    steps_per_epoch = math.ceil(50_000 / 512)
    horizon = steps_per_epoch * 20
    bands = 64
    inverse_coeffs = generate_bisr_coeffs_from_sgd_workload(
        bands=bands,
        momentum=0.0,
        weight_decay=0.0,
    )
    accountant_coeffs = derive_bisr_amplified_accountant_coeffs_from_inverse_coeffs(
        coeffs=inverse_coeffs,
        steps=horizon,
    )
    c_matrix, _ = build_bnb_toeplitz_c_matrix_and_contract(
        coeffs=accountant_coeffs,
        bands=bands,
        horizon=horizon,
    )
    return {
        "cycle_length": steps_per_epoch,
        "horizon": horizon,
        "bands": bands,
        "coeffs": accountant_coeffs,
        "c_matrix": c_matrix,
        "num_samples": 20_000,
        "noise_multiplier": 5.0,
        "target_delta": 1e-3,
        "seed": 123,
        "tolerance": 1e-4,
        "max_iterations": 80,
        "chunk_size": 1_000,
        "num_workers": 0,
    }


class BNBAnalysisTest(unittest.TestCase):
    def test_get_noise_multiplier_validates_stable_bnb_contract_once_per_search(
        self,
    ) -> None:
        coeffs = [1.0, 0.5]
        c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=coeffs,
            bands=2,
            horizon=4,
        )
        sampling_semantics = SamplingSemantics(
            sampling_mode="balls_in_bins",
            privacy_metadata={"bins": 2, "bands": 2},
        )
        mechanism_state = {
            "mechanism": "bsr",
            "coeffs": coeffs,
            "bnb_accountant_coeffs": coeffs,
            "bnb_bands": 2,
            "bnb_cycle_length": 2,
            "bnb_c_matrix": c_matrix,
            "bnb_c_matrix_contract": c_matrix_contract,
        }

        def _fake_estimator(**kwargs) -> float:
            return 60.0 / float(kwargs["noise_multiplier"])

        mocked_estimator = mock.Mock(side_effect=_fake_estimator)
        mocked_estimator.__name__ = (
            "estimate_balls_in_bins_epsilon_reduced_mixture_optimistic"
        )
        with mock.patch.object(
            BNBAccountant,
            "_validate_builtin_b_min_sep_consistency",
            side_effect=lambda **_kwargs: None,
        ) as mocked_validate:
            with mock.patch(
                "opacus.accountants.bnb.estimate_balls_in_bins_epsilon_reduced_mixture_optimistic",
                new=mocked_estimator,
            ):
                sigma = get_noise_multiplier(
                    target_epsilon=8.0,
                    target_delta=1e-5,
                    sample_rate=0.5,
                    steps=4,
                    accountant="bnb",
                    mechanism_state=mechanism_state,
                    sampling_semantics=sampling_semantics,
                    bnb_c_matrix=c_matrix,
                    bnb_c_matrix_contract=c_matrix_contract,
                    bnb_bands=2,
                    bnb_cycle_length=2,
                    bnb_accountant_coeffs=coeffs,
                    bnb_calibration_mode="optimistic",
                    bnb_num_samples=16,
                    bnb_chunk_size=8,
                    bnb_backend="cpu",
                    bnb_device="cpu",
                )

        self.assertGreater(mocked_estimator.call_count, 1)
        self.assertEqual(mocked_validate.call_count, 1)
        self.assertGreaterEqual(float(sigma), 7.5)
        self.assertLessEqual(float(sigma), 7.51)

    def test_build_balls_in_bins_modes_matrix_matches_explicit_periodic_shifts(self) -> None:
        coeffs = [1.0, 0.375, 0.140625, 0.052734375]
        cycle_length = 3
        horizon = 10

        actual = _build_balls_in_bins_modes_matrix(
            coeffs=coeffs,
            cycle_length=cycle_length,
            horizon=horizon,
        )

        first_mode = torch.zeros(horizon, dtype=torch.float64)
        coeff_t = torch.tensor(coeffs, dtype=torch.float64)
        for offset in range(0, horizon, cycle_length):
            take = min(len(coeffs), horizon - offset)
            first_mode[offset : offset + take] += coeff_t[:take]

        expected = torch.zeros((cycle_length, horizon), dtype=torch.float64)
        for row in range(cycle_length):
            expected[row, row:] = first_mode[: horizon - row]

        self.assertTrue(torch.allclose(actual, expected))
        self.assertAlmostEqual(float(actual[0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(actual[0, 3]), float(expected[0, 3]), places=12)
        self.assertAlmostEqual(float(actual[0, 6]), float(expected[0, 6]), places=12)
        self.assertAlmostEqual(float(actual[0, 9]), float(expected[0, 9]), places=12)

    def test_balls_in_bins_sigma_reuse_matches_direct_path_on_representative_bifr_case(
        self,
    ) -> None:
        factor_coeffs, _source = resolve_bifr_exact_factor_coeffs_for_accounting(
            bands=4,
            steps=16,
            momentum=0.9,
            weight_decay=0.9999,
            frac=0.25,
        )
        accountant_coeffs = derive_bifr_amplified_accountant_coeffs_from_factor_coeffs(
            coeffs=factor_coeffs
        )
        reuse_state = build_balls_in_bins_sigma_reuse_state(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            num_samples=32,
            seed=11,
            chunk_size=8,
            backend="cpu",
            device="cpu",
        )

        direct_positive = sample_balls_in_bins_llr_chunks(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            sigma=1.25,
            num_samples=32,
            seed=11,
            chunk_size=8,
            positive_sample=True,
            backend="cpu",
            device="cpu",
        )
        reuse_positive = sample_balls_in_bins_llr_chunks(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            sigma=1.25,
            num_samples=32,
            seed=999,
            chunk_size=8,
            positive_sample=True,
            backend="cpu",
            device="cpu",
            sigma_reuse_state=reuse_state,
        )
        self.assertEqual(len(direct_positive), len(reuse_positive))
        for direct_chunk, reuse_chunk in zip(direct_positive, reuse_positive):
            self.assertTrue(torch.equal(direct_chunk, reuse_chunk))

        direct_negative = sample_balls_in_bins_llr_chunks(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            sigma=1.25,
            num_samples=32,
            seed=12,
            chunk_size=8,
            positive_sample=False,
            backend="cpu",
            device="cpu",
        )
        reuse_negative = sample_balls_in_bins_llr_chunks(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            sigma=1.25,
            num_samples=32,
            seed=999,
            chunk_size=8,
            positive_sample=False,
            backend="cpu",
            device="cpu",
            sigma_reuse_state=reuse_state,
        )
        self.assertEqual(len(direct_negative), len(reuse_negative))
        for direct_chunk, reuse_chunk in zip(direct_negative, reuse_negative):
            self.assertTrue(torch.equal(direct_chunk, reuse_chunk))

        epsilon_direct = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            noise_multiplier=1.25,
            target_delta=1e-4,
            num_samples=32,
            seed=11,
            chunk_size=8,
            backend="cpu",
            device="cpu",
        )
        epsilon_reuse = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
            coeffs=accountant_coeffs,
            cycle_length=4,
            horizon=16,
            noise_multiplier=1.25,
            target_delta=1e-4,
            num_samples=32,
            seed=11,
            chunk_size=8,
            backend="cpu",
            device="cpu",
            sigma_reuse_state=reuse_state,
        )
        self.assertAlmostEqual(float(epsilon_direct), float(epsilon_reuse), places=12)

    def test_resolve_bnb_distributed_mode_auto_selects_chunk_shard_for_dp_runtime(self) -> None:
        mode, auto_selected = _resolve_bnb_distributed_mode(
            distributed_mode=None,
            distributed_dp_runtime=True,
        )
        self.assertEqual(mode, "chunk_shard")
        self.assertTrue(auto_selected)

    def test_assign_bnb_chunk_specs_to_shard_round_robins_chunks(self) -> None:
        chunk_specs = [(0, 4), (4, 4), (8, 4), (12, 4), (16, 4)]
        shard_0 = _assign_bnb_chunk_specs_to_shard(
            specs=chunk_specs,
            rank=0,
            world_size=2,
        )
        shard_1 = _assign_bnb_chunk_specs_to_shard(
            specs=chunk_specs,
            rank=1,
            world_size=2,
        )
        self.assertEqual(shard_0, [chunk_specs[0], chunk_specs[2], chunk_specs[4]])
        self.assertEqual(shard_1, [chunk_specs[1], chunk_specs[3]])

    def test_reduce_bnb_llr_chunks_to_coordinator_uses_tensor_gather(self) -> None:
        local_chunks = [torch.tensor([1.0, 2.0], dtype=torch.float32)]
        gather_calls: list[tuple[torch.Tensor, object, int]] = []

        def _fake_gather(payload, gather_list=None, dst=0):
            gather_calls.append((payload.clone(), gather_list, dst))
            self.assertEqual(dst, 0)
            self.assertIsNotNone(gather_list)
            if payload.dtype == torch.int64:
                gather_list[0].copy_(payload)
                gather_list[1].copy_(torch.tensor([1], dtype=torch.int64, device=payload.device))
            else:
                gather_list[0].copy_(payload)
                remote = torch.zeros_like(payload)
                remote[0] = 3.0
                gather_list[1].copy_(remote)

        with mock.patch("opacus.accountants.analysis.bnb.dist.is_available", return_value=True):
            with mock.patch("opacus.accountants.analysis.bnb.dist.is_initialized", return_value=True):
                with mock.patch("opacus.accountants.analysis.bnb.dist.get_rank", return_value=0):
                    with mock.patch("opacus.accountants.analysis.bnb.dist.get_world_size", return_value=2):
                        with mock.patch(
                            "opacus.accountants.analysis.bnb.dist.gather",
                            side_effect=_fake_gather,
                        ) as mocked_gather:
                            with mock.patch(
                                "opacus.accountants.analysis.bnb.dist.all_reduce",
                                side_effect=lambda tensor, op=None: tensor.fill_(2),
                            ) as mocked_all_reduce:
                                reduced = _reduce_bnb_llr_chunks_to_coordinator(
                                    local_chunks=local_chunks,
                                    distributed_mode="chunk_shard",
                                    backend="cpu",
                                    device="cpu",
                                )

        self.assertEqual(mocked_gather.call_count, 2)
        mocked_all_reduce.assert_called_once()
        self.assertEqual(len(gather_calls), 2)
        self.assertEqual(len(reduced), 2)
        self.assertEqual(reduced[0].dtype, torch.float64)
        self.assertTrue(torch.equal(reduced[0], torch.tensor([1.0, 2.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(reduced[1], torch.tensor([3.0], dtype=torch.float64)))

    def test_make_bnb_broadcast_result_tensor_uses_cpu_for_cpu_backend(self) -> None:
        tensor = _make_bnb_broadcast_result_tensor(
            1.25,
            backend="cpu",
            device="cpu",
        )
        self.assertEqual(tensor.device.type, "cpu")
        self.assertAlmostEqual(float(tensor.item()), 1.25, places=12)

    def test_make_bnb_broadcast_result_tensor_requests_resolved_cuda_device(self) -> None:
        seen: dict[str, object] = {}
        orig_tensor = torch.tensor

        def _fake_tensor(*args, **kwargs):
            seen["device"] = kwargs.get("device")
            safe_kwargs = dict(kwargs)
            safe_kwargs.pop("device", None)
            return orig_tensor(*args, **safe_kwargs)

        with mock.patch(
            "opacus.accountants.analysis.bnb._resolve_bnb_backend_and_device",
            return_value=("cuda", torch.device("cuda:7")),
        ):
            with mock.patch("opacus.accountants.analysis.bnb.torch.tensor", side_effect=_fake_tensor):
                tensor = _make_bnb_broadcast_result_tensor(
                    2.5,
                    backend="auto",
                    device=None,
                )

        self.assertEqual(seen["device"], torch.device("cuda:7"))
        self.assertEqual(tensor.device.type, "cpu")
        self.assertAlmostEqual(float(tensor.item()), 2.5, places=12)

    def test_build_lower_toeplitz_c_matrix_from_coeffs_expected_entries(self) -> None:
        c = build_lower_toeplitz_c_matrix_from_coeffs(
            coeffs=[1.0, 0.5, 0.25],
            horizon=4,
        )
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.25, 0.5, 1.0, 0.0],
                [0.0, 0.25, 0.5, 1.0],
            ],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(c, expected))

    def test_make_bnb_toeplitz_c_matrix_contract_schema(self) -> None:
        c = build_lower_toeplitz_c_matrix_from_coeffs(coeffs=[1.0], horizon=3)
        contract = make_bnb_toeplitz_c_matrix_contract(
            c_matrix=c,
            bands=2,
            horizon=3,
            atol=1e-8,
        )
        self.assertEqual(contract["sampling_mode"], "b_min_sep")
        self.assertEqual(contract["bands"], 2)
        self.assertEqual(contract["granularity"], "single_participation")
        self.assertEqual(contract["matrix_columns"], 3)
        self.assertEqual(contract["derivation"], "lower_toeplitz_from_coeffs")
        self.assertEqual(contract["horizon"], 3)
        self.assertAlmostEqual(float(contract["atol"]), 1e-8, places=20)

    def test_build_bnb_toeplitz_c_matrix_and_contract_round_trip(self) -> None:
        coeffs = [1.0, 0.2]
        c, contract = build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=coeffs,
            bands=2,
            horizon=4,
        )
        self.assertEqual(tuple(c.shape), (4, 4))
        self.assertEqual(contract["bands"], 2)
        self.assertEqual(contract["matrix_columns"], 4)
        self.assertEqual(contract["horizon"], 4)
        self.assertEqual(contract["derivation"], "lower_toeplitz_from_coeffs")

    def test_build_bnb_toeplitz_c_matrix_and_contract_right_pads_nondivisible_horizon(self) -> None:
        coeffs = [1.0, 0.2, 0.1, 0.05]
        c, contract = build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=coeffs,
            bands=4,
            horizon=98,
        )
        self.assertEqual(tuple(c.shape), (100, 100))
        self.assertEqual(contract["bands"], 4)
        self.assertEqual(contract["matrix_columns"], 100)
        self.assertEqual(contract["horizon"], 98)
        self.assertEqual(contract["padded_horizon"], 100)
        self.assertEqual(contract["padding_columns"], 2)
        self.assertEqual(contract["derivation"], "lower_toeplitz_from_coeffs_right_padded")

    def test_normalize_bnb_accountant_coeffs_returns_unit_l2_coeffs(self) -> None:
        coeffs = normalize_bnb_accountant_coeffs(coeffs=[1.0, 2.0, 2.0])
        self.assertAlmostEqual(sum(c * c for c in coeffs), 1.0, places=12)
        self.assertTrue(all(c >= 0.0 for c in coeffs))

    def test_build_b_min_sep_gaussian_mixture_accepts_paper_padded_shape(self) -> None:
        coeffs = [1.0, 0.2, 0.1, 0.05]
        c, _contract = build_bnb_toeplitz_c_matrix_and_contract(
            coeffs=coeffs,
            bands=4,
            horizon=98,
        )
        gm = build_b_min_sep_gaussian_mixture(c_matrix=c, bands=4)
        self.assertEqual(gm.modes.shape[0], 4)
        self.assertEqual(gm.modes.shape[1], 100)

    def test_gaussian_mixture_requires_probs_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            GaussianMixture(
                modes=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                probs=torch.tensor([0.9, 0.2]),
            )

    def test_build_b_min_sep_mixture_expected_modes(self) -> None:
        c = torch.tensor(
            [
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                [4.0, 40.0, 5.0, 50.0, 6.0, 60.0],
            ]
        )
        gm = build_b_min_sep_gaussian_mixture(c_matrix=c, bands=3)
        expected_modes = torch.tensor(
            [
                [21.0, 54.0],
                [13.0, 46.0],
                [32.0, 65.0],
            ]
        )
        self.assertTrue(torch.allclose(gm.modes, expected_modes))
        self.assertTrue(
            torch.allclose(gm.probs, torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
        )

    def test_build_b_min_sep_mixture_can_separate_cycle_length_from_matrix_bandwidth(self) -> None:
        c = torch.tensor(
            [
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                [4.0, 40.0, 5.0, 50.0, 6.0, 60.0],
            ]
        )
        gm = build_b_min_sep_gaussian_mixture(
            c_matrix=c,
            bands=2,
            cycle_length=3,
        )
        expected_modes = torch.tensor(
            [
                [21.0, 54.0],
                [13.0, 46.0],
                [32.0, 65.0],
            ]
        )
        self.assertTrue(torch.allclose(gm.modes, expected_modes))
        self.assertTrue(
            torch.allclose(gm.probs, torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
        )

    def test_build_b_min_sep_reduced_dimensionality(self) -> None:
        # d > b so reduction should project to b-dimensional subspace.
        c = torch.arange(1, 16, dtype=torch.float32).reshape(5, 3)
        gm = build_b_min_sep_gaussian_mixture(
            c_matrix=c,
            bands=3,
            reduce_dimensionality=True,
        )
        self.assertEqual(gm.modes.shape[0], 3)
        self.assertEqual(gm.modes.shape[1], 3)

    def test_generate_mixture_samples_is_seed_reproducible(self) -> None:
        gm = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            probs=torch.tensor([0.3, 0.7]),
        )
        g1 = torch.Generator().manual_seed(12345)
        g2 = torch.Generator().manual_seed(12345)
        ids1, samples1 = generate_mixture_samples(
            gm=gm, sigma=0.5, num_samples=64, generator=g1
        )
        ids2, samples2 = generate_mixture_samples(
            gm=gm, sigma=0.5, num_samples=64, generator=g2
        )
        self.assertTrue(torch.equal(ids1, ids2))
        self.assertTrue(torch.allclose(samples1, samples2))

    def test_compute_llr_samples_requires_matching_dimension(self) -> None:
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            probs=torch.tensor([0.5, 0.5]),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0, 0.0]]),
            probs=torch.tensor([1.0]),
        )
        with self.assertRaisesRegex(ValueError, "same dimension"):
            compute_llr_samples(up_gm=up, lo_gm=lo, sigma=1.0, num_samples=8)

    def test_single_verify_upper_bound_dominates_empirical_delta(self) -> None:
        llr = torch.tensor([2.5, 1.2, 0.4, -0.2, 3.0], dtype=torch.float64)
        result = estimate_delta_upper_bound_single_verify_from_llr_samples(
            epsilon=0.5,
            llr_samples=llr,
            error_probability=1e-4,
        )
        empirical = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=0.5,
            llr_samples=llr,
        )
        self.assertIsInstance(result, SingleVerificationResult)
        self.assertGreaterEqual(result.upper_confidence_bound, empirical)
        self.assertGreaterEqual(result.upper_confidence_bound, result.delta_estimate)

    def test_single_verify_chunk_path_matches_sample_path(self) -> None:
        llr = torch.tensor([1.0, 0.7, -0.4, 2.1, 1.5, 0.1], dtype=torch.float64)
        sample_result = estimate_delta_upper_bound_single_verify_from_llr_samples(
            epsilon=0.3,
            llr_samples=llr,
            error_probability=1e-5,
        )
        chunk_result = estimate_delta_upper_bound_single_verify_from_llr_chunks(
            epsilon=0.3,
            llr_chunks=[llr[:2], llr[2:4], llr[4:]],
            error_probability=1e-5,
        )
        self.assertAlmostEqual(sample_result.delta_estimate, chunk_result.delta_estimate, places=12)
        self.assertAlmostEqual(
            sample_result.upper_confidence_bound,
            chunk_result.upper_confidence_bound,
            places=12,
        )

    def test_compute_llr_samples_is_seed_reproducible(self) -> None:
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            probs=torch.tensor([0.4, 0.6]),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [0.2, 0.0]]),
            probs=torch.tensor([0.4, 0.6]),
        )
        g1 = torch.Generator().manual_seed(2026)
        g2 = torch.Generator().manual_seed(2026)
        llr_1 = compute_llr_samples(
            up_gm=up,
            lo_gm=lo,
            sigma=0.8,
            num_samples=128,
            generator=g1,
        )
        llr_2 = compute_llr_samples(
            up_gm=up,
            lo_gm=lo,
            sigma=0.8,
            num_samples=128,
            generator=g2,
        )
        self.assertTrue(torch.allclose(llr_1, llr_2))

    def test_compute_llr_sample_chunks_are_seed_reproducible(self) -> None:
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            probs=torch.tensor([0.4, 0.6]),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [0.2, 0.0]]),
            probs=torch.tensor([0.4, 0.6]),
        )
        llr_chunks_1 = compute_llr_sample_chunks(
            up_gm=up,
            lo_gm=lo,
            sigma=0.8,
            num_samples=128,
            seed=2026,
            chunk_size=32,
            num_workers=0,
        )
        llr_chunks_2 = compute_llr_sample_chunks(
            up_gm=up,
            lo_gm=lo,
            sigma=0.8,
            num_samples=128,
            seed=2026,
            chunk_size=32,
            num_workers=0,
        )

        self.assertEqual(sum(int(chunk.numel()) for chunk in llr_chunks_1), 128)
        self.assertTrue(all(int(chunk.numel()) <= 32 for chunk in llr_chunks_1))
        self.assertEqual(len(llr_chunks_1), len(llr_chunks_2))
        for chunk_1, chunk_2 in zip(llr_chunks_1, llr_chunks_2):
            self.assertTrue(torch.allclose(chunk_1, chunk_2))

    def test_compute_llr_samples_matches_reference_notebook_formula(self) -> None:
        # Parity against examples/monte_carlo_accountant.py formulas:
        #   component_llrs = (points @ modes.T - ||modes||^2 / 2) / sigma^2
        #   log p(points) = logsumexp(component_llrs, b=probs)
        up = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [1.0, -0.5], [0.2, 1.3]], dtype=torch.float64),
            probs=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64),
        )
        lo = GaussianMixture(
            modes=torch.tensor([[0.0, 0.0], [0.5, -0.2]], dtype=torch.float64),
            probs=torch.tensor([0.4, 0.6], dtype=torch.float64),
        )
        sigma = 0.9
        num_samples = 1024
        seed = 4242

        g_impl = torch.Generator().manual_seed(seed)
        g_ref = torch.Generator().manual_seed(seed)

        llr_impl = compute_llr_samples(
            up_gm=up,
            lo_gm=lo,
            sigma=sigma,
            num_samples=num_samples,
            generator=g_impl,
        )

        # Reference sampling path used in the notebook.
        component_ids = torch.multinomial(
            up.probs.to(dtype=torch.float64),
            num_samples=num_samples,
            replacement=True,
            generator=g_ref,
        ).to(dtype=torch.long)
        points = up.modes[component_ids] + sigma * torch.randn(
            num_samples, up.modes.shape[1], generator=g_ref, dtype=up.modes.dtype
        )

        def ref_logpdf(points: torch.Tensor, gm: GaussianMixture, sigma: float) -> torch.Tensor:
            component_llrs = (
                points @ gm.modes.T - (gm.modes * gm.modes).sum(dim=1)[None, :] / 2.0
            ) / (sigma * sigma)
            return torch.logsumexp(component_llrs + torch.log(gm.probs)[None, :], dim=1)

        llr_ref = ref_logpdf(points, up, sigma) - ref_logpdf(points, lo, sigma)
        self.assertTrue(torch.allclose(llr_impl, llr_ref, atol=1e-10, rtol=1e-10))

    def test_hockey_stick_delta_is_in_unit_interval_and_eps_monotone(self) -> None:
        llr = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
        d1 = estimate_hockey_stick_delta_from_llr_samples(epsilon=0.1, llr_samples=llr)
        d2 = estimate_hockey_stick_delta_from_llr_samples(epsilon=0.9, llr_samples=llr)
        self.assertGreaterEqual(d1, 0.0)
        self.assertLessEqual(d1, 1.0)
        self.assertGreaterEqual(d2, 0.0)
        self.assertLessEqual(d2, 1.0)
        self.assertGreaterEqual(d1, d2)

    def test_find_sigma_binary_search_tracks_target_for_monotone_delta(self) -> None:
        target_delta = 0.2

        def delta_fn(sigma: float) -> float:
            # Monotone decreasing in sigma.
            return 1.0 / (1.0 + sigma)

        sigma = find_sigma_binary_search(
            delta_fn=delta_fn,
            target_delta=target_delta,
            sigma_low=1e-4,
            sigma_high=100.0,
            tolerance=1e-8,
            max_iterations=200,
        )
        self.assertAlmostEqual(delta_fn(sigma), target_delta, places=6)

    def test_estimate_epsilon_from_llr_samples_inverts_target_delta(self) -> None:
        llr = torch.tensor([-2.0, -0.5, 0.2, 1.0, 2.5, 3.0], dtype=torch.float64)
        true_epsilon = 0.7
        target_delta = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=true_epsilon, llr_samples=llr
        )
        estimated = estimate_epsilon_from_llr_samples(
            target_delta=target_delta,
            llr_samples=llr,
            tolerance=1e-8,
        )
        self.assertAlmostEqual(estimated, true_epsilon, places=6)

    def test_estimate_epsilon_from_llr_samples_returns_zero_when_feasible(self) -> None:
        llr = torch.tensor([0.1, 0.3, 1.2], dtype=torch.float64)
        delta_at_zero = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=0.0, llr_samples=llr
        )
        estimated = estimate_epsilon_from_llr_samples(
            target_delta=delta_at_zero + 1e-6,
            llr_samples=llr,
        )
        self.assertEqual(estimated, 0.0)

    def test_estimate_epsilon_from_llr_chunks_matches_sample_tensor_path(self) -> None:
        llr = torch.tensor([-2.0, -0.5, 0.2, 1.0, 2.5, 3.0], dtype=torch.float64)
        target_delta = estimate_hockey_stick_delta_from_llr_samples(
            epsilon=0.7,
            llr_samples=llr,
        )
        chunks = [llr[:2], llr[2:4], llr[4:]]
        eps_chunks = estimate_epsilon_from_llr_chunks(
            target_delta=target_delta,
            llr_chunks=chunks,
            tolerance=1e-8,
        )
        eps_full = estimate_epsilon_from_llr_samples(
            target_delta=target_delta,
            llr_samples=llr,
            tolerance=1e-8,
        )
        self.assertAlmostEqual(eps_chunks, eps_full, places=10)
        self.assertAlmostEqual(
            estimate_hockey_stick_delta_from_llr_chunks(epsilon=eps_chunks, llr_chunks=chunks),
            target_delta,
            delta=1e-8,
        )

    def test_estimate_b_min_sep_epsilon_is_seed_reproducible(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        kwargs = dict(
            c_matrix=c,
            bands=2,
            noise_multiplier=1.2,
            target_delta=0.2,
            num_samples=10_000,
            seed=7,
        )
        eps_1 = estimate_b_min_sep_epsilon_monte_carlo(**kwargs)
        eps_2 = estimate_b_min_sep_epsilon_monte_carlo(**kwargs)
        self.assertAlmostEqual(eps_1, eps_2, places=12)
        self.assertGreaterEqual(eps_1, 0.0)

    def test_estimate_b_min_sep_epsilon_chunked_matches_one_shot(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        kwargs = dict(
            c_matrix=c,
            bands=2,
            noise_multiplier=1.2,
            target_delta=0.2,
            num_samples=10_000,
            seed=7,
            tolerance=1e-6,
            max_iterations=200,
        )
        eps_full = estimate_b_min_sep_epsilon_monte_carlo(**kwargs)
        eps_chunked = estimate_b_min_sep_epsilon_monte_carlo(
            **kwargs,
            chunk_size=2_000,
            num_workers=0,
        )
        self.assertAlmostEqual(eps_full, eps_chunked, delta=0.02)

    def test_estimate_b_min_sep_epsilon_monotone_in_sigma_grid(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        target_delta = 0.2
        sigmas = [0.8, 1.2, 1.8]
        eps = []
        for sigma in sigmas:
            eps.append(
                estimate_b_min_sep_epsilon_monte_carlo(
                    c_matrix=c,
                    bands=2,
                    noise_multiplier=float(sigma),
                    target_delta=target_delta,
                    num_samples=20_000,
                    seed=1234,
                    tolerance=1e-4,
                    max_iterations=200,
                )
            )

        # MC estimate should respect monotonic trend up to tiny numerical slack.
        self.assertLessEqual(eps[1], eps[0] + 1e-3)
        self.assertLessEqual(eps[2], eps[1] + 1e-3)

    def test_estimate_epsilon_from_llr_samples_matches_delta_grid_targets(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        llr = sample_b_min_sep_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=25_000,
            seed=31415,
        )

        for target_delta in [0.3, 0.2, 0.1]:
            epsilon = estimate_epsilon_from_llr_samples(
                target_delta=target_delta,
                llr_samples=llr,
                tolerance=1e-5,
                max_iterations=300,
            )
            recovered_delta = estimate_hockey_stick_delta_from_llr_samples(
                epsilon=epsilon,
                llr_samples=llr,
            )
            self.assertAlmostEqual(recovered_delta, target_delta, delta=2e-3)

    def test_estimate_b_min_sep_epsilon_stability_extreme_sigma_and_long_horizon(self) -> None:
        # Long-horizon Toeplitz fixture to exercise numerical stability envelope.
        c = build_lower_toeplitz_c_matrix_from_coeffs(
            coeffs=[1.0, 0.4, 0.2, 0.1],
            horizon=64,
        )
        kwargs = dict(
            c_matrix=c,
            bands=4,
            target_delta=1e-5,
            num_samples=15_000,
            seed=2026,
            tolerance=1e-4,
            max_iterations=250,
        )

        eps_low_sigma = estimate_b_min_sep_epsilon_monte_carlo(
            noise_multiplier=0.35,
            **kwargs,
        )
        eps_high_sigma = estimate_b_min_sep_epsilon_monte_carlo(
            noise_multiplier=4.0,
            **kwargs,
        )

        self.assertTrue(torch.isfinite(torch.tensor(eps_low_sigma)))
        self.assertTrue(torch.isfinite(torch.tensor(eps_high_sigma)))
        self.assertGreaterEqual(eps_low_sigma, 0.0)
        self.assertGreaterEqual(eps_high_sigma, 0.0)
        self.assertLessEqual(eps_high_sigma, eps_low_sigma + 1e-2)

    def test_estimate_b_min_sep_epsilon_tiny_delta_is_finite(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        eps = estimate_b_min_sep_epsilon_monte_carlo(
            c_matrix=c,
            bands=2,
            noise_multiplier=1.8,
            target_delta=1e-7,
            num_samples=20_000,
            seed=424242,
            tolerance=1e-4,
            max_iterations=300,
        )
        self.assertTrue(torch.isfinite(torch.tensor(eps)))
        self.assertGreaterEqual(eps, 0.0)

    def test_verify_hockey_stick_delta_hoeffding_accepts_and_rejects(self) -> None:
        llr = torch.full((20_000,), 2.0, dtype=torch.float64)
        ok = verify_hockey_stick_delta_hoeffding(
            epsilon=1.0,
            llr_samples=llr,
            target_delta=0.7,
            confidence_alpha=0.5,
        )
        fail = verify_hockey_stick_delta_hoeffding(
            epsilon=1.0,
            llr_samples=llr,
            target_delta=0.1,
            confidence_alpha=0.5,
        )
        self.assertIsInstance(ok, DeltaVerificationResult)
        self.assertTrue(ok.accepted)
        self.assertFalse(fail.accepted)
        self.assertLessEqual(ok.delta_estimate, ok.upper_confidence_bound)

    def test_calibrate_sigma_evr_binary_search_finds_feasible_sigma(self) -> None:
        def llr_samples_fn(sigma: float) -> torch.Tensor:
            # Deterministic monotone setup: larger sigma -> smaller llr -> smaller delta.
            return torch.full((20_000,), 1.0 / sigma, dtype=torch.float64)

        sigma, verification = calibrate_sigma_evr_binary_search(
            llr_samples_fn=llr_samples_fn,
            target_epsilon=1.0,
            target_delta=0.1,
            confidence_alpha=0.5,
            sigma_low=0.1,
            sigma_high=3.0,
            tolerance=1e-4,
            max_iterations=80,
        )
        self.assertGreaterEqual(sigma, 0.1)
        self.assertLessEqual(sigma, 3.0)
        self.assertTrue(verification.accepted)

    def test_sample_b_min_sep_llr_is_seed_reproducible(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        llr_1 = sample_b_min_sep_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=5000,
            seed=42,
        )
        llr_2 = sample_b_min_sep_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=5000,
            seed=42,
        )
        self.assertTrue(torch.allclose(llr_1, llr_2))

    def test_sample_b_min_sep_llr_chunked_is_seed_reproducible(self) -> None:
        c = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        llr_chunked_1 = sample_b_min_sep_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=5000,
            seed=42,
            chunk_size=1024,
            num_workers=0,
        )
        llr_chunked_2 = sample_b_min_sep_llr(
            c_matrix=c,
            bands=2,
            sigma=1.1,
            num_samples=5000,
            seed=42,
            chunk_size=1024,
            num_workers=0,
        )
        self.assertTrue(torch.allclose(llr_chunked_1, llr_chunked_2))

    def test_get_bnb_base_delta_is_below_target_delta(self) -> None:
        base_delta = get_bnb_base_delta(num_samples=50_000, target_delta=0.2)
        self.assertGreater(base_delta, 0.0)
        self.assertLess(base_delta, 0.2)

    def test_sample_balls_in_bins_llr_chunks_are_seed_reproducible(self) -> None:
        coeffs = normalize_bnb_accountant_coeffs(coeffs=[1.0, 0.5])
        llr_chunks_1 = sample_balls_in_bins_llr_chunks(
            coeffs=coeffs,
            cycle_length=3,
            horizon=6,
            sigma=1.2,
            num_samples=128,
            seed=2026,
            chunk_size=32,
            num_workers=0,
            positive_sample=True,
        )
        llr_chunks_2 = sample_balls_in_bins_llr_chunks(
            coeffs=coeffs,
            cycle_length=3,
            horizon=6,
            sigma=1.2,
            num_samples=128,
            seed=2026,
            chunk_size=32,
            num_workers=0,
            positive_sample=True,
        )
        self.assertEqual(len(llr_chunks_1), len(llr_chunks_2))
        for chunk_1, chunk_2 in zip(llr_chunks_1, llr_chunks_2):
            self.assertTrue(torch.allclose(torch.as_tensor(chunk_1), torch.as_tensor(chunk_2)))

    def test_estimate_balls_in_bins_epsilon_chunked_matches_one_shot(self) -> None:
        coeffs = normalize_bnb_accountant_coeffs(coeffs=[1.0, 0.5])
        kwargs = dict(
            coeffs=coeffs,
            cycle_length=3,
            horizon=6,
            noise_multiplier=1.6,
            target_delta=0.2,
            num_samples=4_000,
            seed=99,
            tolerance=1e-4,
            max_iterations=80,
            num_workers=0,
        )
        eps_full = estimate_balls_in_bins_epsilon_monte_carlo(**kwargs)
        eps_chunked = estimate_balls_in_bins_epsilon_monte_carlo(
            **kwargs,
            chunk_size=1_000,
        )
        self.assertLess(abs(float(eps_full) - float(eps_chunked)), 0.15)

    def test_estimate_balls_in_bins_epsilon_optimistic_beats_legacy_high_memory_proxy(
        self,
    ) -> None:
        proxy = _representative_high_memory_bisr_proxy()

        t0 = time.perf_counter()
        legacy_epsilon = _legacy_balls_in_bins_epsilon_optimistic_baseline(
            coeffs=proxy["coeffs"],
            cycle_length=proxy["cycle_length"],
            horizon=proxy["horizon"],
            noise_multiplier=proxy["noise_multiplier"],
            target_delta=proxy["target_delta"],
            num_samples=proxy["num_samples"],
            seed=proxy["seed"],
            tolerance=proxy["tolerance"],
            max_iterations=proxy["max_iterations"],
            chunk_size=proxy["chunk_size"],
            num_workers=proxy["num_workers"],
        )
        legacy_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        fast_epsilon = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
            coeffs=proxy["coeffs"],
            cycle_length=proxy["cycle_length"],
            horizon=proxy["horizon"],
            noise_multiplier=proxy["noise_multiplier"],
            target_delta=proxy["target_delta"],
            num_samples=proxy["num_samples"],
            seed=proxy["seed"],
            tolerance=proxy["tolerance"],
            max_iterations=proxy["max_iterations"],
            chunk_size=proxy["chunk_size"],
            num_workers=proxy["num_workers"],
            backend="cpu",
            device="cpu",
            distributed_mode="none",
            distributed_dp_runtime=False,
        )
        fast_elapsed = time.perf_counter() - t0

        self.assertLess(abs(float(legacy_epsilon) - float(fast_epsilon)), 0.15)
        self.assertGreater(legacy_elapsed / max(fast_elapsed, 1e-9), 10.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BnB CUDA parity")
    def test_sample_balls_in_bins_llr_chunks_cuda_matches_cpu_seeded(self) -> None:
        coeffs = normalize_bnb_accountant_coeffs(coeffs=[1.0, 0.5])
        cpu_chunks = sample_balls_in_bins_llr_chunks(
            coeffs=coeffs,
            cycle_length=3,
            horizon=6,
            sigma=1.2,
            num_samples=128,
            seed=2026,
            chunk_size=32,
            num_workers=0,
            positive_sample=True,
            backend="cpu",
        )
        cuda_chunks = sample_balls_in_bins_llr_chunks(
            coeffs=coeffs,
            cycle_length=3,
            horizon=6,
            sigma=1.2,
            num_samples=128,
            seed=2026,
            chunk_size=32,
            num_workers=0,
            positive_sample=True,
            backend="cuda",
        )
        self.assertEqual(len(cpu_chunks), len(cuda_chunks))
        for cpu_chunk, cuda_chunk in zip(cpu_chunks, cuda_chunks):
            self.assertTrue(
                torch.allclose(
                    torch.as_tensor(cpu_chunk, dtype=torch.float64),
                    torch.as_tensor(cuda_chunk, dtype=torch.float64).cpu(),
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BnB CUDA parity")
    def test_estimate_balls_in_bins_epsilon_optimistic_cuda_matches_cpu_tolerant(self) -> None:
        coeffs = normalize_bnb_accountant_coeffs(coeffs=[1.0, 0.5])
        kwargs = dict(
            coeffs=coeffs,
            cycle_length=3,
            horizon=6,
            noise_multiplier=1.6,
            target_delta=0.2,
            num_samples=4_000,
            seed=99,
            tolerance=1e-4,
            max_iterations=80,
            chunk_size=1_000,
            num_workers=0,
        )
        eps_cpu = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
            **kwargs,
            backend="cpu",
        )
        eps_cuda = estimate_balls_in_bins_epsilon_monte_carlo_optimistic(
            **kwargs,
            backend="cuda",
        )
        self.assertLess(abs(float(eps_cpu) - float(eps_cuda)), 0.15)

    def test_make_bnb_calibration_report_schema(self) -> None:
        verification = DeltaVerificationResult(
            delta_estimate=0.11,
            upper_confidence_bound=0.12,
            confidence_alpha=1e-4,
            accepted=True,
        )
        report = make_bnb_calibration_report(
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=2.0,
            num_samples=10000,
            seed=7,
            bands=2,
            verification=verification,
            evr_confidence_alpha_total=1e-4,
            evr_num_checks=2,
            evr_per_check_alpha=5e-5,
            evr_pass_count=2,
        )
        self.assertIsInstance(report, BNBCalibrationReport)
        payload = report.to_dict()
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["bands"], 2)
        self.assertEqual(payload["num_samples"], 10000)
        self.assertTrue(payload["verification_passed"])
        self.assertEqual(payload["evr_num_checks"], 2)
        self.assertEqual(payload["evr_pass_count"], 2)
        self.assertEqual(payload["verification_contract"], "evr_union_bound_alpha_split_v1")
        self.assertGreaterEqual(payload["evr_composed_delta_upper_bound"], payload["target_delta"])
        expected = payload["target_delta"] + payload["evr_confidence_alpha_total"] * (
            1.0 - payload["target_delta"]
        )
        self.assertAlmostEqual(payload["evr_composed_delta_upper_bound"], expected, places=18)

    def test_split_confidence_alpha(self) -> None:
        alpha = split_confidence_alpha(total_confidence_alpha=1e-3, num_checks=4)
        self.assertAlmostEqual(alpha, 2.5e-4)

    def test_verify_evr_confidence_split(self) -> None:
        llr = torch.full((5000,), 2.0, dtype=torch.float64)
        worst, pass_count, per_alpha = verify_evr_confidence_split(
            llr_samples_seq=[llr, llr, llr],
            epsilon=1.0,
            target_delta=0.7,
            total_confidence_alpha=0.5,
        )
        self.assertTrue(worst.accepted)
        self.assertEqual(pass_count, 3)
        self.assertAlmostEqual(per_alpha, 0.5 / 3.0)

    def test_select_evr_candidate_ladder_chooses_first_accept(self) -> None:
        def llr_samples_seq_fn(sigma: float):
            # sigma<1.0 fails (large llr -> large hockey-stick delta),
            # sigma>=1.0 passes (llr=0 -> delta=0).
            if sigma < 1.0:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(3)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(3)]

        sigma, verification, pass_count, per_alpha = select_evr_candidate_ladder(
            candidate_sigmas=[0.6, 0.8, 1.0, 1.2],
            llr_samples_seq_fn=llr_samples_seq_fn,
            epsilon=1.0,
            target_delta=0.2,
            total_confidence_alpha=0.1,
        )
        self.assertEqual(sigma, 1.0)
        self.assertTrue(verification.accepted)
        self.assertEqual(pass_count, 3)
        self.assertAlmostEqual(per_alpha, 0.1 / (4.0 * 3.0))

    def test_select_evr_candidate_ladder_requires_strictly_increasing_order(self) -> None:
        def llr_samples_seq_fn(_sigma: float):
            return [torch.zeros((128,), dtype=torch.float64)]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_evr_candidate_ladder(
                candidate_sigmas=[1.0, 0.9, 1.1],
                llr_samples_seq_fn=llr_samples_seq_fn,
                epsilon=1.0,
                target_delta=0.2,
                total_confidence_alpha=0.1,
            )

    def test_select_evr_candidate_ladder_two_sided_chooses_first_accept(self) -> None:
        def llr_samples_seq_fn_forward(sigma: float):
            if sigma < 1.0:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(3)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(3)]

        def llr_samples_seq_fn_reverse(sigma: float):
            if sigma < 1.0:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(3)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(3)]

        sigma, verification, pass_count, per_alpha = select_evr_candidate_ladder_two_sided(
            candidate_sigmas=[0.6, 0.8, 1.0, 1.2],
            llr_samples_seq_fn_forward=llr_samples_seq_fn_forward,
            llr_samples_seq_fn_reverse=llr_samples_seq_fn_reverse,
            epsilon=1.0,
            target_delta=0.2,
            total_confidence_alpha=0.1,
        )
        self.assertEqual(sigma, 1.0)
        self.assertTrue(verification.accepted)
        self.assertEqual(pass_count, 6)  # 2 directions * 3 checks
        self.assertAlmostEqual(per_alpha, 0.1 / (4.0 * 2.0 * 3.0))

    def test_select_evr_candidate_ladder_two_sided_rejects_if_one_direction_fails(self) -> None:
        def llr_samples_seq_fn_forward(sigma: float):
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(2)]

        def llr_samples_seq_fn_reverse(sigma: float):
            if sigma < 1.2:
                return [torch.full((3000,), 10.0, dtype=torch.float64) for _ in range(2)]
            return [torch.zeros((3000,), dtype=torch.float64) for _ in range(2)]

        sigma, verification, pass_count, per_alpha = select_evr_candidate_ladder_two_sided(
            candidate_sigmas=[1.0, 1.1, 1.2],
            llr_samples_seq_fn_forward=llr_samples_seq_fn_forward,
            llr_samples_seq_fn_reverse=llr_samples_seq_fn_reverse,
            epsilon=1.0,
            target_delta=0.2,
            total_confidence_alpha=0.3,
        )
        self.assertEqual(sigma, 1.2)
        self.assertTrue(verification.accepted)
        self.assertEqual(pass_count, 4)  # 2 directions * 2 checks at accepted candidate
        self.assertAlmostEqual(per_alpha, 0.3 / (3.0 * 2.0 * 2.0))

    def test_parse_bnb_calibration_report_rejects_unsupported_version(self) -> None:
        payload = {"version": 1}
        with self.assertRaisesRegex(ValueError, "unsupported BNB calibration report version"):
            parse_bnb_calibration_report(payload)

    def test_parse_bnb_calibration_report_v2(self) -> None:
        payload_v2 = {
            "version": 2,
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "noise_multiplier": 2.0,
            "num_samples": 1000,
            "seed": 7,
            "bands": 2,
            "delta_estimate_at_target_epsilon": 0.1,
            "delta_upper_confidence_bound": 0.11,
            "confidence_alpha": 1e-4,
            "verification_passed": False,
            "evr_confidence_alpha_total": 1e-4,
            "evr_num_checks": 3,
            "evr_per_check_alpha": 1.0 / 30000.0,
            "evr_pass_count": 2,
        }
        parsed = parse_bnb_calibration_report(payload_v2)
        self.assertEqual(parsed.version, 2)
        self.assertEqual(parsed.evr_num_checks, 3)
        self.assertEqual(parsed.evr_pass_count, 2)
        self.assertEqual(parsed.verification_contract, "evr_union_bound_alpha_split_v1")

    def test_describe_bnb_calibration_report(self) -> None:
        payload_v2 = {
            "version": 2,
            "target_epsilon": 1.0,
            "target_delta": 1e-5,
            "noise_multiplier": 2.0,
            "num_samples": 1000,
            "seed": 7,
            "bands": 2,
            "delta_estimate_at_target_epsilon": 0.1,
            "delta_upper_confidence_bound": 0.11,
            "confidence_alpha": 1e-4,
            "verification_passed": False,
            "evr_confidence_alpha_total": 1e-4,
            "evr_num_checks": 3,
            "evr_per_check_alpha": 1.0 / 30000.0,
            "evr_pass_count": 2,
        }
        s = describe_bnb_calibration_report(payload_v2)
        self.assertIn("BNB calibration v2 [FAIL]", s)
        self.assertIn("checks=2/3", s)
        self.assertIn("guard_delta=", s)


if __name__ == "__main__":
    unittest.main()
