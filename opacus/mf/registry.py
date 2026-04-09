from __future__ import annotations

from opacus.mf.interfaces import MFFamilyEntry
from opacus.mf.blt_family import BLTFamily
from opacus.mf.bsr_family import BSRFamily
from opacus.mf.bifr_family import BIFRFamily


_BLT_ENTRY = MFFamilyEntry(
    family=BLTFamily(),
    supports_fixed_batch=True,
    supports_balls_in_bins=True,
    supports_optimization=True,
    needs_default_local_sampling_semantics=True,
    accounting_requires_context=True,
)

_BSR_ENTRY = MFFamilyEntry(
    family=BSRFamily("bsr"),
    supports_fixed_batch=True,
    supports_cyclic=True,
    supports_balls_in_bins=True,
    needs_default_local_sampling_semantics=True,
    accounting_requires_context=True,
)

_BISR_ENTRY = MFFamilyEntry(
    family=BSRFamily("bisr"),
    supports_fixed_batch=True,
    supports_cyclic=True,
    supports_balls_in_bins=True,
    needs_default_local_sampling_semantics=True,
    accounting_requires_context=True,
)

_BANDMF_ENTRY = MFFamilyEntry(
    family=BSRFamily("bandmf"),
    supports_fixed_batch=True,
    supports_cyclic=True,
    supports_balls_in_bins=True,
    needs_default_local_sampling_semantics=True,
    accounting_requires_context=True,
)

_BANDINVMF_ENTRY = MFFamilyEntry(
    family=BSRFamily("bandinvmf"),
    supports_fixed_batch=True,
    supports_cyclic=True,
    supports_balls_in_bins=True,
    needs_default_local_sampling_semantics=True,
    accounting_requires_context=True,
)

_BIFR_ENTRY = MFFamilyEntry(
    family=BIFRFamily(),
    supports_fixed_batch=True,
    supports_cyclic=False,
    supports_balls_in_bins=True,
    needs_default_local_sampling_semantics=True,
    accounting_requires_context=True,
)

_MF_FAMILY_REGISTRY = {
    "blt": _BLT_ENTRY,
    "bsr": _BSR_ENTRY,
    "bisr": _BISR_ENTRY,
    "bandmf": _BANDMF_ENTRY,
    "bandinvmf": _BANDINVMF_ENTRY,
    "bifr": _BIFR_ENTRY,
}


def get_mf_family_entry(mechanism: str) -> MFFamilyEntry | None:
    return _MF_FAMILY_REGISTRY.get(mechanism)


def mf_accounting_requires_context(accountant_mechanism: str) -> bool:
    entry = _MF_FAMILY_REGISTRY.get(accountant_mechanism)
    if entry is None:
        return False
    return bool(entry.accounting_requires_context)
