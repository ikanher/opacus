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

from __future__ import annotations

"""
Runtime/accountant adapter for balls-in-bins and BMinSep Monte Carlo accounting.

This module owns the accountant-facing boundary for amplified correlated
accounting. It resolves runtime matrix metadata, validates that metadata
against the selected sampling semantics, and delegates the actual Monte Carlo
estimation to `opacus.opacus.accountants.analysis.bnb`.

Canonically, this module owns the `bnb` accountant family only:
- `accountant='bnb'`
- `sampling_mode='balls_in_bins'`
- balls-in-bins / BMinSep paper contracts

It does not own repeated `k`-out-of-`t` random allocation. That family lives in
`opacus.opacus.accountants.random_allocation`.

Paper lineage:
- Balls-and-Bins (Chua et al., 2024) for the original one-shot balls-in-bins
  certification model;
- EVR (Wang et al., 2023) for the EVR-style confidence-splitting
  interpretation used by the current candidate-ladder calibration path.

Implementation lineage:
- the amplified parity and EVR-oriented helper structure is closely aligned
  with `google-deepmind/jax_privacy`.
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Tuple

from opacus.accountants.bnb_inputs import resolve_canonical_bnb_cycle_length
from opacus.accountants.analysis.bnb import (
    estimate_balls_in_bins_epsilon_monte_carlo,
    estimate_balls_in_bins_epsilon_monte_carlo_optimistic,
    estimate_b_min_sep_epsilon_monte_carlo,
    resolve_bnb_calibration_kwargs,
)
from opacus.accountants.analysis.random_allocation import (
    estimate_epsilon_random_allocation,
    resolve_random_allocation_accountant_inputs,
    resolve_random_allocation_gaussian_runtime_config,
)
from opacus.accountants.analysis.bnb_preflight import validate_bnb_runtime_consistency

from .accountant import IAccountant


def _timing_enabled() -> bool:
    return os.getenv("DEBUG_TIMING", "").strip().lower() not in ("", "0", "false", "no")


def _debug_timing(message: str) -> None:
    if _timing_enabled():
        print(f"[opacus.bnb_accountant] [timing] {message}", flush=True)


@contextmanager
def _timed(label: str):
    if not _timing_enabled():
        yield
        return
    start = time.perf_counter()
    _debug_timing(f"start {label}")
    try:
        yield
    finally:
        _debug_timing(f"done {label} elapsed={time.perf_counter() - start:.3f}s")


def resolve_bnb_b_min_sep_inputs(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    kwargs: Dict[str, Any],
) -> Tuple[Any, int, int, Dict[str, Any]]:
    """
    Resolve the matrix/metadata bundle consumed by the BNB accountant.

    For BLT amplified accounting, the `c_matrix` and contract are expected to
    come from the accountant-side forward-`c_col` bridge, not from the runtime
    noiser. The returned tuple is shared by both supported BNB consumers:
    `balls_in_bins` through the public BLT accountant route, and `b_min_sep`
    through the direct analysis-side Monte Carlo path.
    """
    state = mechanism_state if isinstance(mechanism_state, dict) else {}
    metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    c_matrix = kwargs.get(
        "bnb_c_matrix",
        kwargs.get("c_matrix", state.get("bnb_c_matrix", state.get("c_matrix"))),
    )

    metadata_bands = metadata.get("bands")
    explicit_bands = kwargs.get("bnb_bands")
    if explicit_bands is not None and metadata_bands is not None:
        if int(explicit_bands) != int(metadata_bands):
            raise ValueError(
                "bnb consistency check failed: sampling_semantics privacy_metadata['bands'] "
                f"({int(metadata_bands)}) != accounting bands ({int(explicit_bands)})"
            )

    bands = kwargs.get(
        "bnb_bands",
        kwargs.get(
            "bsr_bands",
            metadata.get("bands", state.get("bnb_bands", state.get("bsr_bands"))),
        ),
    )

    c_matrix_contract = kwargs.get(
        "bnb_c_matrix_contract",
        kwargs.get(
            "c_matrix_contract",
            state.get("bnb_c_matrix_contract", state.get("c_matrix_contract")),
        ),
    )

    if c_matrix is None or bands is None or c_matrix_contract is None:
        raise ValueError(
            "bnb calibration requires b_min_sep/balls_in_bins inputs: "
            "`bnb_c_matrix`, `bnb_bands`, and `bnb_c_matrix_contract`"
        )

    cycle_length = resolve_canonical_bnb_cycle_length(
        runtime_state=state,
        metadata=metadata,
        kwargs=kwargs,
        error_context="bnb calibration requires bnb cycle length or sampling bins metadata",
    )

    return c_matrix, int(bands), int(cycle_length), c_matrix_contract


def validate_bnb_accounting_runtime_consistency(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    c_matrix: Any,
    bands: int,
    c_matrix_contract: Dict[str, Any],
) -> None:
    """Validate that a runtime/accountant BLT or MF state matches the BNB contract."""
    validate_bnb_runtime_consistency(
        mechanism_state=mechanism_state,
        sampling_semantics=sampling_semantics,
        c_matrix=c_matrix,
        bands=int(bands),
        c_matrix_contract=c_matrix_contract,
        coeffs_error_prefix="bnb consistency check",
    )


def validate_bnb_sampling_policy(
    *,
    sampling_semantics,
    mechanism: str,
) -> None:
    """
    Validate that the chosen sampling mode is supported by the current BNB path.

    BLT participates here only through the amplified BNB bridge. This validation
    does not say that BLT has a standalone BNB proof artifact; it only checks
    the runtime/accountant contract supported by the current implementation.
    """
    if mechanism not in ("gaussian", "bandmf", "bsr", "bisr", "bandinvmf", "bifr", "blt") or sampling_semantics is None:
        return
    mode = sampling_semantics.sampling_mode
    if mode not in ("balls_in_bins", "b_min_sep"):
        raise ValueError(
            "bnb accountant path requires sampling_semantics in "
            "{'balls_in_bins', 'b_min_sep'}"
        )


class BNBAccountant(IAccountant):
    """
    Accountant adapter for BMinSep/Balls-in-Bins Monte Carlo accounting.

    The accountant resolves runtime matrix/sampling metadata, validates that
    metadata against the BMinSep contract, and then estimates epsilon from
    Monte Carlo privacy-loss samples for a target delta.

    Confidence-control logic lives in ``analysis.bnb``; this class wires those
    primitives into the Opacus accountant interface.

    Source:
    - Balls-and-Bins (Chua et al., 2024) for one-shot Monte Carlo
      certification;
    - EVR (Wang et al., 2023) for EVR-style candidate-ladder feasibility and
      confidence composition.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_builtin_b_min_sep_consistency(
        *,
        mechanism_state,
        sampling_semantics,
        c_matrix,
        bands: int,
        c_matrix_contract,
    ) -> None:
        """
        Validate that runtime matrix/sampler metadata obeys the BMinSep contract.

        This prevents using Monte Carlo calibration results with incompatible
        runtime wiring (for example, wrong bands or wrong matrix derivation).

        Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5 and Theorem 5.1.
        """
        validate_bnb_runtime_consistency(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
            c_matrix=c_matrix,
            bands=int(bands),
            c_matrix_contract=c_matrix_contract,
            coeffs_error_prefix="bnb consistency check",
        )

    def step(self, *, noise_multiplier: float, sample_rate: float):
        # `noise_multiplier` is Gaussian sigma; `sample_rate` is retained for API parity.
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))
        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(
        self,
        delta: float,
        *,
        mechanism_state=None,
        sampling_semantics=None,
        **kwargs,
    ) -> float:
        """
        Estimate epsilon via Monte Carlo BMinSep accounting.

        The method expects BMinSep/Balls-in-Bins sampling semantics together
        with explicit ``bnb_c_matrix`` metadata. It then:
        1. resolves calibration overrides,
        2. validates matrix/sampler consistency,
        3. delegates epsilon estimation to ``estimate_b_min_sep_epsilon_monte_carlo``.

        Source:
        - Balls-and-Bins (Chua et al., 2024) for one-shot Monte Carlo
          certification;
        - EVR (Wang et al., 2023) for the EVR-style feasibility path used by
          the current balls-in-bins calibration surface.
        """
        with _timed(f"get_epsilon delta={float(delta):.6g}"):
            if not self.history:
                return 0.0

            noise_multiplier, sample_rate, _ = self.history[0]
            for nm_i, sr_i, _steps_i in self.history:
                if nm_i != noise_multiplier or sr_i != sample_rate:
                    raise ValueError(
                        "bnb accountant currently expects constant "
                        "noise_multiplier and sample_rate across steps"
                    )

            state = mechanism_state if isinstance(mechanism_state, dict) else {}
            metadata = (
                sampling_semantics.privacy_metadata
                if sampling_semantics is not None
                else {}
            )
            sampling_mode = (
                sampling_semantics.sampling_mode
                if sampling_semantics is not None
                else None
            )
            _debug_timing(
                "get_epsilon state "
                f"sampling_mode={sampling_mode} mechanism={state.get('mechanism', state.get('name'))} "
                f"noise_multiplier={float(noise_multiplier):.6g} sample_rate={float(sample_rate):.6g}"
            )

        # `bands` is the min-separation/bin-width parameter in b-min-sep construction.
        c_matrix = kwargs.get("bnb_c_matrix", state.get("bnb_c_matrix"))
        bands = kwargs.get("bnb_bands", metadata.get("bands", state.get("bnb_bands")))
        cycle_length = kwargs.get(
            "bnb_cycle_length",
            metadata.get("bins", state.get("bnb_cycle_length", state.get("bnb_bins"))),
        )
        c_matrix_contract = kwargs.get(
            "bnb_c_matrix_contract",
            state.get("bnb_c_matrix_contract"),
        )

        persisted_kwargs = state.get("_bnb_accounting_kwargs", {})
        resolved_overrides = {}
        if isinstance(persisted_kwargs, dict):
            for key in (
                "bnb_calibration_mode",
                "bnb_num_samples",
                "bnb_seed",
                "bnb_reduce_dimensionality",
                "bnb_tolerance",
                "bnb_max_iterations",
                "bnb_chunk_size",
                "bnb_num_workers",
                "bnb_backend",
                "bnb_device",
                "bnb_distributed_mode",
                "bnb_distributed_dp_runtime",
            ):
                value = persisted_kwargs.get(key)
                if value is not None:
                    resolved_overrides[key] = value

        for key in (
            "bnb_calibration_mode",
            "bnb_num_samples",
            "bnb_seed",
            "bnb_reduce_dimensionality",
            "bnb_tolerance",
            "bnb_max_iterations",
            "bnb_chunk_size",
            "bnb_num_workers",
            "bnb_backend",
            "bnb_device",
            "bnb_distributed_mode",
            "bnb_distributed_dp_runtime",
        ):
            value = kwargs.get(key)
            if value is not None:
                resolved_overrides[key] = value

        with _timed("resolve_bnb_calibration_kwargs"):
            calibration_cfg = resolve_bnb_calibration_kwargs(
                overrides=resolved_overrides,
            )

        num_samples = int(calibration_cfg["bnb_num_samples"])
        seed = int(calibration_cfg["bnb_seed"])
        reduce_dimensionality = bool(calibration_cfg["bnb_reduce_dimensionality"])
        tolerance = float(calibration_cfg["bnb_tolerance"])
        max_iterations = int(calibration_cfg["bnb_max_iterations"])
        chunk_size = calibration_cfg["bnb_chunk_size"]
        num_workers = int(calibration_cfg["bnb_num_workers"])
        backend = str(calibration_cfg["bnb_backend"])
        device = calibration_cfg["bnb_device"]
        distributed_mode = calibration_cfg["bnb_distributed_mode"]
        distributed_dp_runtime = bool(calibration_cfg["bnb_distributed_dp_runtime"])
        calibration_mode = str(calibration_cfg["bnb_calibration_mode"])
        sigma_reuse_state = kwargs.get("bnb_sigma_reuse_state")
        accounting_backend = str(
            kwargs.get(
                "bnb_accounting_backend",
                persisted_kwargs.get("bnb_accounting_backend", "monte_carlo"),
            )
        )
        _debug_timing(
            "get_epsilon config "
            f"accounting_backend={accounting_backend} calibration_mode={calibration_mode} "
            f"num_samples={num_samples} seed={seed} tolerance={tolerance} "
            f"max_iterations={max_iterations} chunk_size={chunk_size} num_workers={num_workers} "
            f"backend={backend} device={device} distributed_mode={distributed_mode}"
        )
        if sampling_mode == "b_min_sep" and c_matrix is not None and bands is not None and c_matrix_contract is not None:
            if cycle_length is None:
                cycle_length = bands

            with _timed("validate_b_min_sep_consistency"):
                self._validate_builtin_b_min_sep_consistency(
                    mechanism_state=state,
                    sampling_semantics=sampling_semantics,
                    c_matrix=c_matrix,
                    bands=int(bands),
                    c_matrix_contract=c_matrix_contract,
                )
            with _timed("estimate_b_min_sep_epsilon_monte_carlo"):
                return float(
                    estimate_b_min_sep_epsilon_monte_carlo(
                        c_matrix=c_matrix,
                        bands=int(bands),
                        cycle_length=int(cycle_length),
                        noise_multiplier=float(noise_multiplier),
                        target_delta=float(delta),
                        num_samples=num_samples,
                        seed=seed,
                        reduce_dimensionality=reduce_dimensionality,
                        tolerance=float(tolerance),
                        max_iterations=max_iterations,
                        chunk_size=chunk_size,
                        num_workers=num_workers,
                    )
                )

        if sampling_mode == "balls_in_bins" and c_matrix is not None and bands is not None and c_matrix_contract is not None:
            if cycle_length is None:
                cycle_length = bands

            with _timed("validate_balls_in_bins_consistency"):
                self._validate_builtin_b_min_sep_consistency(
                    mechanism_state=state,
                    sampling_semantics=sampling_semantics,
                    c_matrix=c_matrix,
                    bands=int(bands),
                    c_matrix_contract=c_matrix_contract,
                )
            accountant_coeffs = kwargs.get(
                "bnb_accountant_coeffs",
                state.get("bnb_accountant_coeffs", state.get("coeffs")),
            )
            if accountant_coeffs is None:
                raise ValueError(
                    "balls_in_bins accounting requires accountant-side coefficients "
                    "via `bnb_accountant_coeffs` or `coeffs`"
                )

            horizon = int(c_matrix.shape[1])
            _debug_timing(
                "balls_in_bins contract "
                f"bands={int(bands)} cycle_length={int(cycle_length)} horizon={horizon} "
                f"coeff_len={len(accountant_coeffs)}"
            )
            if accounting_backend == "deterministic":
                with _timed("resolve_random_allocation_inputs"):
                    inputs = resolve_random_allocation_accountant_inputs(
                        mechanism=str(state.get("mechanism", state.get("name", "gaussian"))),
                        mechanism_state=state,
                        sampling_semantics=sampling_semantics,
                        kwargs={
                            "bnb_accountant_coeffs": accountant_coeffs,
                            "bnb_cycle_length": int(cycle_length),
                            "bnb_horizon": int(horizon),
                        },
                        noise_multiplier=float(noise_multiplier),
                    )
                with _timed("resolve_random_allocation_runtime_config"):
                    runtime_cfg = resolve_random_allocation_gaussian_runtime_config(
                        target_delta=float(delta),
                        loss_discretization=kwargs.get(
                            "random_allocation_loss_discretization",
                            persisted_kwargs.get("random_allocation_loss_discretization"),
                        ),
                        tail_truncation=kwargs.get(
                            "random_allocation_tail_truncation",
                            persisted_kwargs.get("random_allocation_tail_truncation"),
                        ),
                        max_grid_fft=kwargs.get(
                            "random_allocation_max_grid_fft",
                            persisted_kwargs.get("random_allocation_max_grid_fft"),
                        ),
                        max_grid_mult=kwargs.get(
                            "random_allocation_max_grid_mult",
                            persisted_kwargs.get("random_allocation_max_grid_mult"),
                        ),
                        convolution_method=kwargs.get(
                            "random_allocation_convolution_method",
                            persisted_kwargs.get("random_allocation_convolution_method"),
                        ),
                    )
                with _timed("estimate_epsilon_random_allocation"):
                    return float(
                        estimate_epsilon_random_allocation(
                            inputs=inputs,
                            target_delta=float(delta),
                            runtime_config=runtime_cfg,
                        )
                    )
            estimator = estimate_balls_in_bins_epsilon_monte_carlo
            if calibration_mode == "optimistic":
                estimator = estimate_balls_in_bins_epsilon_monte_carlo_optimistic
            with _timed(f"{estimator.__name__}"):
                return float(
                    estimator(
                        coeffs=accountant_coeffs,
                        cycle_length=int(cycle_length),
                        horizon=horizon,
                        noise_multiplier=float(noise_multiplier),
                        target_delta=float(delta),
                        num_samples=num_samples,
                        seed=seed,
                        tolerance=float(tolerance),
                        max_iterations=max_iterations,
                        chunk_size=chunk_size,
                        num_workers=num_workers,
                        backend=backend,
                        device=device,
                        distributed_mode=distributed_mode,
                        distributed_dp_runtime=distributed_dp_runtime,
                        sigma_reuse_state=sigma_reuse_state,
                    )
                )

        raise ValueError(
            "bnb accountant built-in calibration requires b_min_sep/balls_in_bins "
            "inputs (`c_matrix`, `bands`, `c_matrix_contract`, and "
            "sampling_mode in {'b_min_sep', 'balls_in_bins'})"
        )

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(_cls) -> str:
        return "bnb"
