"""
Runtime-only BLT accountant boundary.

This module owns the explicit honesty boundary for BLT runs that have runtime
support and checkpoint support but no authoritative epsilon accountant for the
current contract. It allows the privacy engine to attach an accountant-shaped
object without pretending that `get_epsilon` is meaningful.
"""

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

from collections import OrderedDict
from typing import Any, Mapping, TypeVar

from .accountant import IAccountant


T_state_dict = TypeVar("T_state_dict", bound=Mapping[str, Any])


class BLTRuntimeOnlyAccountant(IAccountant):
    """
    Runtime-only accountant boundary for BLT runs.

    This accountant participates in optimizer hook attachment and checkpointing
    without claiming that the run has an authoritative epsilon accountant.
    """

    _RUNTIME_ONLY_ERROR = (
        "BLT accountant support is unavailable for the current BLT contract; "
        "this run is runtime-only"
    )

    def __init__(self):
        """Initialize an empty runtime-only BLT event counter."""
        super().__init__()
        self._events_recorded = 0

    def step(self, *, noise_multiplier: float, sample_rate: float):
        """Record that one runtime-only BLT event occurred."""
        del noise_multiplier, sample_rate
        self._events_recorded += 1

    def get_epsilon(self, delta: float, *args, **kwargs) -> float:
        """Reject epsilon queries because this accountant is explicitly runtime-only."""
        del delta, args, kwargs
        raise ValueError(self._RUNTIME_ONLY_ERROR)

    def __len__(self) -> int:
        """Return the number of runtime-only BLT events that were recorded."""
        return int(self._events_recorded)

    @classmethod
    def mechanism(cls) -> str:
        """Return the router tag used for the runtime-only BLT accountant boundary."""
        return "blt_runtime_only"

    def state_dict(self, destination: T_state_dict = None) -> T_state_dict:
        """Serialize the runtime-only BLT accountant state for checkpointing."""
        if destination is None:
            destination = OrderedDict()
        destination["history"] = []
        destination["mechanism"] = self.mechanism()
        destination["events_recorded"] = int(self._events_recorded)
        destination["accounting_supported"] = False
        return destination

    def load_state_dict(self, state_dict: T_state_dict):
        """Restore the runtime-only BLT accountant event counter from a checkpoint."""
        super().load_state_dict(state_dict)
        self.history = []
        self._events_recorded = int(state_dict.get("events_recorded", 0))
