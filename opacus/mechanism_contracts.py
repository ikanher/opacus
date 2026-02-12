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


NoiseMechanismName = Literal["gaussian", "bsr"]
AccountingModeName = Literal[
    "standard_step_accountant",
    "bsr_accountant",
]
SamplingModeName = Literal["poisson", "fixed_batch", "cyclic_poisson"]


class MechanismStateSerializable(Protocol):
    """
    Contract for mechanism-specific state serialization.

    M0 contract only: implementations will be introduced in follow-up milestones.
    """

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class SamplingSemantics:
    sampling_mode: SamplingModeName
    privacy_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoiseMechanismConfig:
    mechanism: NoiseMechanismName = "gaussian"
    accounting_mode: AccountingModeName = "standard_step_accountant"
    mechanism_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mechanism = self.mechanism
        if mechanism not in ("gaussian", "bsr"):
            raise ValueError("mechanism must be one of {'gaussian', 'bsr'}")

        if self.accounting_mode not in (
            "standard_step_accountant",
            "bsr_accountant",
        ):
            raise ValueError(
                "accounting_mode must be one of "
                "{'standard_step_accountant', 'bsr_accountant'}"
            )

        if (
            mechanism == "bsr"
            and self.accounting_mode != "bsr_accountant"
        ):
            raise ValueError(
                "bsr mechanism requires bsr_accountant "
                "for authoritative accounting"
            )
