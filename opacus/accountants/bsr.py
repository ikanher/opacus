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

import math
from typing import Any, Dict, Optional

from opacus.accountants.analysis.bsr import (
    compute_bsr_mf_sensitivity_from_coeffs,
    compute_bsr_kappa_from_coeffs,
    bsr_fixed_batch_epsilon_upper_bound,
    bsr_cyclic_poisson_epsilon_upper_bound,
)

from .accountant import IAccountant


def resolve_bsr_mf_sensitivity_for_fixed_batch(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    sample_rate: Optional[float],
    kwargs: Dict[str, Any],
) -> float:
    metadata = (
        sampling_semantics.privacy_metadata
        if sampling_semantics is not None
        else {}
    )
    _raise_if_fixed_batch_has_cyclic_only_params(
        mechanism_state=mechanism_state,
        metadata=metadata,
        kwargs=kwargs,
    )

    (
        coeffs,
        max_participations,
        min_separation,
        sensitivity_steps,
        mf_sensitivity,
        explicit_mf_sensitivity_override,
    ) = BSRAccountant._resolve_fixed_batch_contract_inputs(
        state=mechanism_state,
        metadata=metadata,
        kwargs=kwargs,
        total_steps=int(steps),
        sample_rate=float(sample_rate) if sample_rate is not None else None,
    )

    return BSRAccountant._resolve_fixed_batch_mf_sensitivity(
        coeffs=coeffs,
        max_participations=max_participations,
        min_separation=min_separation,
        sensitivity_steps=sensitivity_steps,
        mf_sensitivity=mf_sensitivity,
        explicit_mf_sensitivity_override=explicit_mf_sensitivity_override,
    )


def resolve_bsr_sensitivity_scale_for_cyclic(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics,
    steps: int,
    kwargs: Dict[str, Any],
) -> float:
    metadata = (
        sampling_semantics.privacy_metadata
        if sampling_semantics is not None
        else {}
    )
    _raise_if_cyclic_has_fixed_batch_only_params(
        mechanism_state=mechanism_state,
        metadata=metadata,
        kwargs=kwargs,
    )

    bands = kwargs.get(
        "bsr_bands",
        metadata.get("bands", mechanism_state.get("bsr_bands")),
    )
    if bands is None:
        coeffs = mechanism_state.get("coeffs")
        bands = len(coeffs) if coeffs is not None else None

    if bands is None:
        raise ValueError(
            "cyclic_poisson bsr requires `bands` in sampling semantics metadata, "
            "`bsr_bands`, or derivable from coeffs"
        )

    return BSRAccountant._resolve_cyclic_sensitivity_scale(
        state=mechanism_state,
        metadata=metadata,
        kwargs=kwargs,
        total_steps=int(steps),
        bands=int(bands),
    )


def _raise_if_fixed_batch_has_cyclic_only_params(
    *,
    mechanism_state: Dict[str, Any],
    metadata: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> None:
    cyclic_only_params = []

    _raise_on_legacy_aliases(
        source_name="kwargs",
        payload=kwargs,
        aliases={"sensitivity_scale": "bsr_sensitivity_scale"},
    )

    _raise_on_legacy_aliases(
        source_name="privacy_metadata",
        payload=metadata,
        aliases={"sensitivity_scale": "bsr_sensitivity_scale"},
    )

    _raise_on_legacy_aliases(
        source_name="mechanism_state",
        payload=mechanism_state,
        aliases={"sensitivity_scale": "bsr_sensitivity_scale"},
    )

    if kwargs.get("bsr_sensitivity_scale") is not None:
        cyclic_only_params.append("bsr_sensitivity_scale")

    if metadata.get("bsr_sensitivity_scale") is not None:
        cyclic_only_params.append("privacy_metadata['bsr_sensitivity_scale']")

    if mechanism_state.get("bsr_sensitivity_scale") is not None:
        cyclic_only_params.append("mechanism_state['bsr_sensitivity_scale']")

    if cyclic_only_params:
        raise ValueError(
            "fixed-batch bsr accounting received cyclic-only parameters: "
            + ", ".join(cyclic_only_params)
        )


def _raise_if_cyclic_has_fixed_batch_only_params(
    *,
    mechanism_state: Dict[str, Any],
    metadata: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> None:
    fixed_only_params = []
    _raise_on_legacy_aliases(
        source_name="kwargs",
        payload=kwargs,
        aliases={
            "mf_sensitivity": "bsr_mf_sensitivity",
            "max_participations": "bsr_max_participations",
            "min_separation": "bsr_min_separation",
            "iterations_number": "bsr_iterations_number",
        },
    )
    _raise_on_legacy_aliases(
        source_name="privacy_metadata",
        payload=metadata,
        aliases={
            "mf_sensitivity": "bsr_mf_sensitivity",
            "max_participations": "bsr_max_participations",
            "min_separation": "bsr_min_separation",
            "iterations_number": "bsr_iterations_number",
        },
    )
    _raise_on_legacy_aliases(
        source_name="mechanism_state",
        payload=mechanism_state,
        aliases={
            "mf_sensitivity": "bsr_mf_sensitivity",
            "max_participations": "bsr_max_participations",
            "min_separation": "bsr_min_separation",
            "iterations_number": "bsr_iterations_number",
        },
    )
    if kwargs.get("bsr_mf_sensitivity") is not None:
        fixed_only_params.append("bsr_mf_sensitivity")

    if kwargs.get("bsr_max_participations") is not None:
        fixed_only_params.append("bsr_max_participations")

    if kwargs.get("bsr_min_separation") is not None:
        fixed_only_params.append("bsr_min_separation")

    if metadata.get("bsr_mf_sensitivity") is not None:
        fixed_only_params.append("privacy_metadata['bsr_mf_sensitivity']")

    if metadata.get("bsr_max_participations") is not None:
        fixed_only_params.append("privacy_metadata['bsr_max_participations']")

    if metadata.get("bsr_min_separation") is not None:
        fixed_only_params.append("privacy_metadata['bsr_min_separation']")

    if mechanism_state.get("bsr_mf_sensitivity") is not None:
        fixed_only_params.append("mechanism_state['bsr_mf_sensitivity']")

    if mechanism_state.get("bsr_max_participations") is not None:
        fixed_only_params.append("mechanism_state['bsr_max_participations']")

    if mechanism_state.get("bsr_min_separation") is not None:
        fixed_only_params.append("mechanism_state['bsr_min_separation']")

    if fixed_only_params:
        raise ValueError(
            "cyclic-poisson bandmf accounting received fixed-batch-only parameters: "
            + ", ".join(fixed_only_params)
        )


def _raise_on_legacy_aliases(*, source_name: str, payload: dict, aliases: dict) -> None:
    for legacy_name, canonical_name in aliases.items():
        if payload.get(legacy_name) is not None:
            raise ValueError(
                f"{source_name} uses removed alias `{legacy_name}`; use `{canonical_name}`"
            )


class BSRAccountant(IAccountant):
    """
    Accountant adapter for BSR-family mechanisms.

    This class is the runtime bridge between mechanism metadata and the two
    supported accounting reductions:
    1. fixed-batch BSR using matrix-factorization sensitivity ``S_{k,b}(C;T)``;
    2. cyclic-poisson BSR/BandMF using ``κ(T)`` plus sampled-Gaussian RDP composition.

    Source: BSR (Kalinin and Lampert, 2024), Section 3.2, Equation (10), Theorem 2; and
    BandMF (Choquette-Choo et al., 2023), Section 5 and Theorems 4 and 5.
    """

    def __init__(self):
        super().__init__()

    def step(self, *, noise_multiplier: float, sample_rate: float):
        # `noise_multiplier` is the runtime Gaussian multiplier (`sigma_ref`);
        # `sample_rate` is the per-step participation probability.
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

    @staticmethod
    def _resolve_constant_history(history) -> tuple[float, float, int]:
        """
        Collapse history under the current constant-parameter assumption.

        The implementation currently requires one constant
        ``(noise_multiplier, sample_rate)`` pair across all recorded segments.
        If that holds, we return the total horizon ``T`` for downstream
        sensitivity and composition logic.
        """
        noise_multiplier, sample_rate, _ = history[0]
        total_steps = 0
        for nm_i, sr_i, steps_i in history:
            if nm_i != noise_multiplier or sr_i != sample_rate:
                raise ValueError(
                    "bsr accountant currently expects constant "
                    "noise_multiplier and sample_rate across steps"
                )

            total_steps += int(steps_i)

        return float(noise_multiplier), float(sample_rate), int(total_steps)

    @staticmethod
    def _resolve_sampling_context(*, mechanism_state, sampling_semantics):
        """
        Normalize runtime context into ``state``, ``metadata``, and ``sampling_mode``.

        This helper centralizes defaulting behavior so both accounting branches
        consume a consistent view of mechanism metadata.
        """
        state = mechanism_state if isinstance(mechanism_state, dict) else {}
        metadata = (
            sampling_semantics.privacy_metadata
            if sampling_semantics is not None
            else {}
        )
        sampling_mode = (
            sampling_semantics.sampling_mode
            if sampling_semantics is not None
            else "torch_sampler"
        )

        return state, metadata, sampling_mode

    @staticmethod
    def _resolve_cyclic_sensitivity_scale(
        *,
        state: dict,
        metadata: dict,
        kwargs: dict,
        total_steps: int,
        bands: int,
    ) -> float:
        """
        Resolve the cyclic-path sensitivity normalization scale.

        Cyclic accounting expects ``bsr_sensitivity_scale`` (interpreted as
        ``κ(T)``). We prefer explicit metadata when provided; otherwise we
        derive ``κ(T)`` from Toeplitz coefficients over the resolved horizon.
        """
        explicit_scale = kwargs.get(
            "bsr_sensitivity_scale",
            metadata.get(
                "bsr_sensitivity_scale",
                state.get("bsr_sensitivity_scale"),
            ),
        )
        if explicit_scale is not None:
            sensitivity_scale = float(explicit_scale)
            if (not math.isfinite(sensitivity_scale)) or sensitivity_scale <= 0.0:
                raise ValueError("bsr_sensitivity_scale must be finite and > 0")

            return float(sensitivity_scale)

        coeffs = state.get("coeffs")
        if coeffs is None:
            raise ValueError(
                "cyclic-poisson bsr accounting requires either `bsr_sensitivity_scale` "
                "or `mechanism_state['coeffs']`"
            )

        scale_steps = kwargs.get(
            "bsr_iterations_number",
            metadata.get(
                "bsr_iterations_number",
                state.get("bsr_iterations_number"),
            ),
        )
        if scale_steps is None:
            scale_steps = total_steps

        scale_steps = int(scale_steps)
        if scale_steps < 1:
            raise ValueError("bsr_iterations_number must be >= 1")

        if scale_steps < bands:
            raise ValueError(
                "cyclic_poisson bsr requires steps >= bands; "
                f"got steps={scale_steps}, bands={bands}"
            )

        sensitivity_scale = float(
            compute_bsr_kappa_from_coeffs(
                coeffs=coeffs,
                steps=scale_steps,
            )
        )
        if (not math.isfinite(sensitivity_scale)) or sensitivity_scale <= 0.0:
            raise ValueError("resolved bsr_sensitivity_scale must be finite and > 0")

        return float(sensitivity_scale)

    @staticmethod
    def _get_epsilon_cyclic(
        *,
        delta: float,
        noise_multiplier: float,
        sample_rate: float,
        total_steps: int,
        state: dict,
        metadata: dict,
        kwargs: dict,
    ) -> float:
        """
        Compute epsilon for the cyclic-poisson branch.

        The branch-specific reduction is:
        ``q = b·p``, ``N_cycles = ⌈T / b⌉``,
        ``σ_eff = σ / κ(T)``.
        We then delegate to sampled-Gaussian/RDP composition.

        Source: BandMF (Choquette-Choo et al., 2023), Section 5 and Theorems 4 and 5.
        """
        bands = metadata.get("bands", None)
        if bands is None:
            raise ValueError(
                "cyclic_poisson sampling requires privacy_metadata['bands']"
            )

        bands = int(bands)
        if bands <= 0:
            raise ValueError("bands must be > 0")

        if total_steps < bands:
            raise ValueError(
                f"steps must be >= bands; got steps={total_steps}, bands={bands}"
            )

        sensitivity_scale = BSRAccountant._resolve_cyclic_sensitivity_scale(
            state=state,
            metadata=metadata,
            kwargs=kwargs,
            total_steps=total_steps,
            bands=bands,
        )

        return float(
            bsr_cyclic_poisson_epsilon_upper_bound(
                noise_multiplier=float(noise_multiplier) / float(sensitivity_scale),
                target_delta=float(delta),
                steps=int(total_steps),
                sample_rate=float(sample_rate),
                bands=int(bands),
            )
        )

    @staticmethod
    def _resolve_fixed_batch_contract_inputs(
        *,
        state: dict,
        metadata: dict,
        kwargs: dict,
        total_steps: int,
        sample_rate: float,
    ) -> tuple[object, int, int, int, object, bool]:
        """
        Resolve all fixed-batch sensitivity inputs from kwargs/state/metadata.

        This collects the parameter tuple needed by ``S_{k,b}(C;T)``-based
        accounting and applies branch defaults when fields are omitted.
        """
        mf_sensitivity = kwargs.get(
            "bsr_mf_sensitivity",
            metadata.get(
                "bsr_mf_sensitivity",
                state.get("bsr_mf_sensitivity"),
            ),
        )

        explicit_mf_sensitivity_override = "bsr_mf_sensitivity" in kwargs
        coeffs = state.get("coeffs")
        max_participations = kwargs.get(
            "bsr_max_participations",
            metadata.get(
                "bsr_max_participations",
                state.get("bsr_max_participations"),
            ),
        )

        min_separation = kwargs.get(
            "bsr_min_separation",
            metadata.get(
                "bsr_min_separation",
                state.get("bsr_min_separation"),
            ),
        )

        sensitivity_steps = kwargs.get(
            "bsr_iterations_number",
            metadata.get(
                "bsr_iterations_number",
                state.get("bsr_iterations_number"),
            ),
        )

        if sensitivity_steps is None:
            sensitivity_steps = total_steps

        sensitivity_steps = int(sensitivity_steps)
        if sensitivity_steps < 1:
            raise ValueError("bsr_iterations_number must be >= 1")

        if max_participations is None:
            max_participations = int(math.ceil(float(sample_rate) * float(sensitivity_steps)))
            max_participations = max(1, int(max_participations))

        if min_separation is None:
            min_separation = 1

        return (
            coeffs,
            int(max_participations),
            int(min_separation),
            int(sensitivity_steps),
            mf_sensitivity,
            bool(explicit_mf_sensitivity_override),
        )

    @staticmethod
    def _resolve_fixed_batch_mf_sensitivity(
        *,
        coeffs,
        max_participations: int,
        min_separation: int,
        sensitivity_steps: int,
        mf_sensitivity,
        explicit_mf_sensitivity_override: bool,
    ) -> float:
        """
        Resolve or validate fixed-batch MF sensitivity ``S_{k,b}(C;T)``.

        If explicit sensitivity is missing, derive it from coefficients. If an
        explicit override is present and coefficients are available, validate
        the override against the derived value.
        """
        if mf_sensitivity is None:
            if coeffs is None:
                raise ValueError(
                    "fixed-batch bsr accounting requires MF sensitivity or "
                    "enough data to derive it: "
                    "`coeffs`, `max_participations`, `min_separation`"
                )

            return float(
                compute_bsr_mf_sensitivity_from_coeffs(
                    coeffs=coeffs,
                    steps=sensitivity_steps,
                    max_participations=int(max_participations),
                    min_separation=int(min_separation),
                )
            )

        mf_sensitivity = float(mf_sensitivity)
        if not math.isfinite(mf_sensitivity) or mf_sensitivity <= 0.0:
            raise ValueError("bsr_mf_sensitivity must be finite and > 0")
        if (
            explicit_mf_sensitivity_override and coeffs is not None
        ):
            derived = float(
                compute_bsr_mf_sensitivity_from_coeffs(
                    coeffs=coeffs,
                    steps=sensitivity_steps,
                    max_participations=int(max_participations),
                    min_separation=int(min_separation),
                )
            )
            if not math.isfinite(derived) or derived <= 0.0:
                raise ValueError(
                    "derived bsr_mf_sensitivity must be finite and > 0 "
                    "when validating explicit bsr_mf_sensitivity"
                )

            if not math.isclose(
                mf_sensitivity, derived, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    "provided bsr_mf_sensitivity is inconsistent with "
                    "coeffs/max_participations/min_separation for the resolved "
                    "bsr_iterations_number"
                )

        return float(mf_sensitivity)

    @staticmethod
    def _get_epsilon_fixed_batch(
        *,
        delta: float,
        noise_multiplier: float,
        sample_rate: float,
        total_steps: int,
        state: dict,
        metadata: dict,
        kwargs: dict,
    ) -> float:
        """
        Compute epsilon for the fixed-batch branch.

        This path resolves ``S_{k,b}(C;T)`` first, then reduces to an effective
        Gaussian mechanism with ``sigma_eff = noise_multiplier / S_{k,b}(C;T)``
        and applies standard RDP-to-``(epsilon, delta)`` conversion.

        Source: BSR (Kalinin and Lampert, 2024), Section 3.2, Equation (10), Theorem 2.
        """
        (
            coeffs,
            max_participations,
            min_separation,
            sensitivity_steps,
            mf_sensitivity,
            explicit_mf_sensitivity_override,
        ) = BSRAccountant._resolve_fixed_batch_contract_inputs(
            state=state,
            metadata=metadata,
            kwargs=kwargs,
            total_steps=total_steps,
            sample_rate=sample_rate,
        )

        resolved_mf_sensitivity = BSRAccountant._resolve_fixed_batch_mf_sensitivity(
            coeffs=coeffs,
            max_participations=max_participations,
            min_separation=min_separation,
            sensitivity_steps=sensitivity_steps,
            mf_sensitivity=mf_sensitivity,
            explicit_mf_sensitivity_override=explicit_mf_sensitivity_override,
        )

        return float(
            bsr_fixed_batch_epsilon_upper_bound(
                noise_multiplier=float(noise_multiplier),
                target_delta=float(delta),
                mf_sensitivity=float(resolved_mf_sensitivity),
            )
        )

    def get_epsilon(
        self,
        delta: float,
        *,
        mechanism_state=None,
        sampling_semantics=None,
        **kwargs,
    ) -> float:
        """
        Compute epsilon using the branch selected by sampling semantics.

        - ``sampling_mode == "cyclic_poisson"``: cyclic contract accounting.
        - otherwise: fixed-batch BSR accounting.
        """
        if not self.history:
            return 0.0

        noise_multiplier, sample_rate, total_steps = self._resolve_constant_history(self.history)
        state, metadata, sampling_mode = self._resolve_sampling_context(
            mechanism_state=mechanism_state,
            sampling_semantics=sampling_semantics,
        )
        self._raise_on_legacy_aliases(
            source_name="kwargs",
            payload=kwargs,
            aliases={
                "sensitivity_scale": "bsr_sensitivity_scale",
                "mf_sensitivity": "bsr_mf_sensitivity",
                "max_participations": "bsr_max_participations",
                "min_separation": "bsr_min_separation",
                "iterations_number": "bsr_iterations_number",
            },
        )
        self._raise_on_legacy_aliases(
            source_name="privacy_metadata",
            payload=metadata,
            aliases={
                "sensitivity_scale": "bsr_sensitivity_scale",
                "mf_sensitivity": "bsr_mf_sensitivity",
                "max_participations": "bsr_max_participations",
                "min_separation": "bsr_min_separation",
                "iterations_number": "bsr_iterations_number",
            },
        )
        self._raise_on_legacy_aliases(
            source_name="mechanism_state",
            payload=state,
            aliases={
                "sensitivity_scale": "bsr_sensitivity_scale",
                "mf_sensitivity": "bsr_mf_sensitivity",
                "max_participations": "bsr_max_participations",
                "min_separation": "bsr_min_separation",
                "iterations_number": "bsr_iterations_number",
                "bands": "bsr_bands",
            },
        )

        if sampling_mode == "cyclic_poisson":
            return self._get_epsilon_cyclic(
                delta=delta,
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                total_steps=total_steps,
                state=state,
                metadata=metadata,
                kwargs=kwargs,
            )

        return self._get_epsilon_fixed_batch(
            delta=delta,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            total_steps=total_steps,
            state=state,
            metadata=metadata,
            kwargs=kwargs,
        )

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "bsr"

    @staticmethod
    def _raise_on_legacy_aliases(*, source_name: str, payload: dict, aliases: dict) -> None:
        for legacy_name, canonical_name in aliases.items():
            if payload.get(legacy_name) is not None:
                raise ValueError(
                    f"{source_name} uses removed alias `{legacy_name}`; use `{canonical_name}`"
                )
