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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Protocol


NoiseMechanismName = Literal["gaussian", "bandmf", "bsr", "bisr", "bandinvmf", "bifr", "blt"]
AccountingModeName = Literal[
    "standard_step_accountant",
    "bandmf_accountant",
    "bsr_accountant",
    "blt_accountant",
    "bnb_accountant",
    "random_allocation_accountant",
]
SamplingModeName = Literal[
    "poisson",
    "torch_sampler",
    "cyclic_poisson",
    "b_min_sep",
    "balls_in_bins",
    "k_out_of_t",
]


class MechanismStateSerializable(Protocol):
    """
    Contract for mechanism-specific state serialization.

    M0 contract only: implementations will be introduced in follow-up milestones.
    """

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


def resolve_accounting_mode_from_accountant(accountant: str) -> AccountingModeName:
    """
    Normalize user-facing accountant names to internal accounting_mode names.
    """
    accountant_to_mode = {
        "prv": "standard_step_accountant",
        "rdp": "standard_step_accountant",
        "gdp": "standard_step_accountant",
        "standard_step_accountant": "standard_step_accountant",
        "bandmf": "bandmf_accountant",
        "bandmf_accountant": "bandmf_accountant",
        "bsr": "bsr_accountant",
        "bisr": "bsr_accountant",
        "bifr": "bsr_accountant",
        "bandinvmf": "bsr_accountant",
        "bsr_accountant": "bsr_accountant",
        "blt": "blt_accountant",
        "blt_accountant": "blt_accountant",
        "bnb": "bnb_accountant",
        "bnb_accountant": "bnb_accountant",
        "random_allocation": "random_allocation_accountant",
        "random_allocation_accountant": "random_allocation_accountant",
    }

    normalized = accountant_to_mode.get(accountant)
    if normalized is None:
        supported = ", ".join(sorted(accountant_to_mode.keys()))
        raise ValueError(
            f"Unsupported accountant '{accountant}'. Supported values: {supported}"
        )

    return normalized


@dataclass(frozen=True)
class SamplingSemantics:
    sampling_mode: SamplingModeName
    privacy_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alias_map = {
            "balls_n_bins": "balls_in_bins",
            "balls-in-bins": "balls_in_bins",
            "balls_in_bins_sampler": "balls_in_bins",
            "balls-in-bins-sampler": "balls_in_bins",
            "k-out-of-t": "k_out_of_t",
        }
        normalized_sampling_mode = alias_map.get(self.sampling_mode, self.sampling_mode)
        object.__setattr__(self, "sampling_mode", normalized_sampling_mode)

        if self.sampling_mode not in (
            "poisson",
            "torch_sampler",
            "cyclic_poisson",
            "b_min_sep",
            "balls_in_bins",
            "k_out_of_t",
        ):
            raise ValueError(
                "sampling_mode must be one of "
                "{'poisson', 'torch_sampler', 'cyclic_poisson', 'b_min_sep', 'balls_in_bins', 'k_out_of_t'} "
                "(aliases: 'balls_n_bins', 'balls-in-bins', 'balls_in_bins_sampler', 'k-out-of-t')"
            )
        if self.sampling_mode == "k_out_of_t":
            num_steps = self.privacy_metadata.get("num_steps")
            num_selected = self.privacy_metadata.get("num_selected")
            if num_steps is None or num_selected is None:
                raise ValueError(
                    "k_out_of_t sampling requires privacy_metadata['num_steps'] and privacy_metadata['num_selected']"
                )
            num_steps = int(num_steps)
            num_selected = int(num_selected)
            if num_steps < 1 or num_selected < 1 or num_selected > num_steps:
                raise ValueError(
                    "invalid k-out-of-t contract: require 1 <= num_selected <= num_steps"
                )


@dataclass(frozen=True)
class NoiseMechanismConfig:
    mechanism: NoiseMechanismName = "gaussian"
    accounting_mode: AccountingModeName = "standard_step_accountant"
    mechanism_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mechanism = self.mechanism
        if mechanism not in ("gaussian", "bandmf", "bsr", "bisr", "bandinvmf", "bifr", "blt"):
            raise ValueError(
                "mechanism must be one of {'gaussian', 'bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr', 'blt'}"
            )

        accounting_mode = self.accounting_mode
        if mechanism == "blt" and accounting_mode == "standard_step_accountant":
            # Preserve legacy call sites while making the public BLT contract
            # explicit at the config surface.
            accounting_mode = "blt_accountant"
            object.__setattr__(self, "accounting_mode", accounting_mode)

        if accounting_mode not in (
            "standard_step_accountant",
            "bandmf_accountant",
            "bsr_accountant",
            "blt_accountant",
            "bnb_accountant",
            "random_allocation_accountant",
        ):
            raise ValueError(
                "accounting_mode must be one of "
                "{'standard_step_accountant', 'bandmf_accountant', 'bsr_accountant', 'blt_accountant', 'bnb_accountant', 'random_allocation_accountant'}"
            )

        if (
            mechanism == "bandmf"
            and accounting_mode not in ("bandmf_accountant", "bnb_accountant", "random_allocation_accountant")
        ):
            raise ValueError(
                "bandmf mechanism requires bandmf_accountant, bnb_accountant, or random_allocation_accountant "
                "for authoritative accounting"
            )

        if mechanism == "bifr" and accounting_mode not in ("bsr_accountant", "bnb_accountant"):
            raise ValueError(
                "bifr mechanism requires bsr_accountant or bnb_accountant"
            )

        if (
            mechanism == "bsr"
            and accounting_mode not in ("bsr_accountant", "bnb_accountant", "random_allocation_accountant")
        ):
            raise ValueError(
                "bsr mechanism requires bsr_accountant, bnb_accountant, or random_allocation_accountant "
                "for authoritative accounting"
            )

        if (
            mechanism == "bisr"
            and accounting_mode not in ("bsr_accountant", "bnb_accountant", "random_allocation_accountant")
        ):
            raise ValueError(
                "bisr mechanism requires bsr_accountant, bnb_accountant, or random_allocation_accountant "
                "for authoritative accounting"
            )

        if (
            mechanism == "bandinvmf"
            and accounting_mode not in ("bsr_accountant", "bnb_accountant", "random_allocation_accountant")
        ):
            raise ValueError(
                "bandinvmf mechanism requires bsr_accountant, bnb_accountant, or random_allocation_accountant "
                "for authoritative accounting"
            )

        if mechanism == "blt" and accounting_mode not in ("blt_accountant", "bnb_accountant"):
            raise ValueError("blt mechanism requires blt_accountant or bnb_accountant routing")
