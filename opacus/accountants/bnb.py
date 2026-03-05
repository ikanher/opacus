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

from opacus.accountants.analysis.bnb import (
    estimate_b_min_sep_epsilon_monte_carlo,
    resolve_bnb_calibration_kwargs,
)
from opacus.accountants.analysis.bnb_preflight import validate_bnb_runtime_consistency

from .accountant import IAccountant


class BNBAccountant(IAccountant):
    """
    Accountant adapter for BMinSep/Balls-in-Bins Monte Carlo accounting.

    The accountant resolves runtime matrix/sampling metadata, validates that
    metadata against the BMinSep contract, and then estimates epsilon from
    Monte Carlo privacy-loss samples for a target delta.

    Confidence-control logic lives in ``analysis.bnb``; this class wires those
    primitives into the Opacus accountant interface.

    Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5, Equations (2)-(4), Theorem 5.1.
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

        Source: BMinSep (Dong and Ganesh, 2025 draft), Section 5, Equations (2)-(4), Theorem 5.1.
        """
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

        # `bands` is the min-separation/bin-width parameter in b-min-sep construction.
        self._raise_on_legacy_aliases(
            source_name="kwargs",
            payload=kwargs,
            aliases={
                "c_matrix": "bnb_c_matrix",
                "bands": "bnb_bands",
                "c_matrix_contract": "bnb_c_matrix_contract",
            },
        )
        self._raise_on_legacy_aliases(
            source_name="mechanism_state",
            payload=state,
            aliases={
                "c_matrix": "bnb_c_matrix",
                "bands": "bnb_bands",
                "c_matrix_contract": "bnb_c_matrix_contract",
            },
        )
        c_matrix = kwargs.get("bnb_c_matrix", state.get("bnb_c_matrix"))
        bands = kwargs.get("bnb_bands", metadata.get("bands", state.get("bnb_bands")))
        c_matrix_contract = kwargs.get(
            "bnb_c_matrix_contract",
            state.get("bnb_c_matrix_contract"),
        )

        persisted_kwargs = state.get("_bnb_accounting_kwargs", {})
        resolved_overrides = {}
        if isinstance(persisted_kwargs, dict):
            for key in (
                "bnb_num_samples",
                "bnb_seed",
                "bnb_reduce_dimensionality",
                "bnb_tolerance",
                "bnb_max_iterations",
            ):
                value = persisted_kwargs.get(key)
                if value is not None:
                    resolved_overrides[key] = value

        for key in (
            "bnb_num_samples",
            "bnb_seed",
            "bnb_reduce_dimensionality",
            "bnb_tolerance",
            "bnb_max_iterations",
        ):
            value = kwargs.get(key)
            if value is not None:
                resolved_overrides[key] = value

        calibration_cfg = resolve_bnb_calibration_kwargs(
            overrides=resolved_overrides,
        )

        num_samples = int(calibration_cfg["bnb_num_samples"])
        seed = int(calibration_cfg["bnb_seed"])
        reduce_dimensionality = bool(calibration_cfg["bnb_reduce_dimensionality"])
        tolerance = float(calibration_cfg["bnb_tolerance"])
        max_iterations = int(calibration_cfg["bnb_max_iterations"])
        if (
            sampling_mode in ("b_min_sep", "balls_in_bins")
            and c_matrix is not None
            and bands is not None
            and c_matrix_contract is not None
        ):
            self._validate_builtin_b_min_sep_consistency(
                mechanism_state=state,
                sampling_semantics=sampling_semantics,
                c_matrix=c_matrix,
                bands=int(bands),
                c_matrix_contract=c_matrix_contract,
            )
            return float(
                estimate_b_min_sep_epsilon_monte_carlo(
                    c_matrix=c_matrix,
                    bands=int(bands),
                    noise_multiplier=float(noise_multiplier),
                    target_delta=float(delta),
                    num_samples=num_samples,
                    seed=seed,
                    reduce_dimensionality=reduce_dimensionality,
                    tolerance=float(tolerance),
                    max_iterations=max_iterations,
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
    def mechanism(cls) -> str:
        return "bnb"
    @staticmethod
    def _raise_on_legacy_aliases(*, source_name: str, payload: dict, aliases: dict) -> None:
        for legacy_name, canonical_name in aliases.items():
            if payload.get(legacy_name) is not None:
                raise ValueError(
                    f"{source_name} uses removed alias `{legacy_name}`; use `{canonical_name}`"
                )
