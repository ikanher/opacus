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

from importlib import import_module

from .accountant import IAccountant
from .registry import create_accountant, register_accountant


__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
    "PRVAccountant",
    "BandMFAccountant",
    "BLTAccountant",
    "BLTRuntimeOnlyAccountant",
    "BNBAccountant",
    "BSRAccountant",
    "RandomAllocationAccountant",
    "register_accountant",
    "create_accountant",
]


_LAZY_EXPORTS = {
    "GaussianAccountant": (".gdp", "GaussianAccountant"),
    "RDPAccountant": (".rdp", "RDPAccountant"),
    "PRVAccountant": (".prv", "PRVAccountant"),
    "BandMFAccountant": (".bandmf", "BandMFAccountant"),
    "BLTAccountant": (".blt", "BLTAccountant"),
    "BLTRuntimeOnlyAccountant": (".blt_runtime_only", "BLTRuntimeOnlyAccountant"),
    "BNBAccountant": (".bnb", "BNBAccountant"),
    "BSRAccountant": (".bsr", "BSRAccountant"),
    "RandomAllocationAccountant": (".random_allocation", "RandomAllocationAccountant"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
