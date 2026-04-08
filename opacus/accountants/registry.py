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
from typing import Dict, Type

from .accountant import IAccountant


_ACCOUNTANTS: Dict[str, Type[IAccountant] | tuple[str, str]] = {
    "rdp": (".rdp", "RDPAccountant"),
    "gdp": (".gdp", "GaussianAccountant"),
    "prv": (".prv", "PRVAccountant"),
    "bandmf": (".bandmf", "BandMFAccountant"),
    "bsr": (".bsr", "BSRAccountant"),
    "bnb": (".bnb", "BNBAccountant"),
    "random_allocation": (".random_allocation", "RandomAllocationAccountant"),
    "blt_runtime_only": (".blt_runtime_only", "BLTRuntimeOnlyAccountant"),
    "blt": (".blt", "BLTAccountant"),
}


def _resolve_accountant_class(
    accountant: Type[IAccountant] | tuple[str, str],
) -> Type[IAccountant]:
    if isinstance(accountant, tuple):
        module_name, attr_name = accountant
        module = import_module(module_name, __name__.rsplit(".", 1)[0])
        return getattr(module, attr_name)
    return accountant


def register_accountant(
    mechanism: str, accountant: Type[IAccountant], force: bool = False
):
    r"""
    Register a new accountant class to be used with a specified mechanism name.
    """
    if mechanism in _ACCOUNTANTS and not force:
        raise ValueError(f"Accountant for mechanism {mechanism} is already registered")

    _ACCOUNTANTS[mechanism] = accountant


def create_accountant(mechanism: str) -> IAccountant:
    r"""
    Create and return an accountant instance for the specified privacy mechanism.
    """
    if mechanism in _ACCOUNTANTS:
        accountant_cls = _resolve_accountant_class(_ACCOUNTANTS[mechanism])
        _ACCOUNTANTS[mechanism] = accountant_cls
        return accountant_cls()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")
