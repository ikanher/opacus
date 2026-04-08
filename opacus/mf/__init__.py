"""
Engine-facing matrix-factorization family provider layer.

This package owns family registration, capability dispatch, canonicalization,
and report/input translation for `PrivacyEngine`. It is not the owner of
runtime noise-mechanism implementations or accountant math.
"""

from .interfaces import (
    MFFamily,
    MFFamilyEntry,
    SupportsBallsInBins,
    SupportsCyclic,
    SupportsFixedBatch,
    SupportsOptimization,
)
from .registry import get_mf_family_entry, mf_accounting_requires_context
from .state import BLTFamilyState, BSRFamilyState
from .blt_family import BLTFamily
from .bsr_family import BSRFamily
from .bifr_family import BIFRFamily
from .report import (
    BLTReportSurface,
    BSRFamilyReportInputs,
    build_bsr_family_report_inputs,
    compute_blt_fixed_batch_report_surface,
    compute_bsr_family_cyclic_report_baseline,
    resolve_bsr_family_fixed_batch_report_sensitivity,
)

__all__ = [
    "MFFamily",
    "MFFamilyEntry",
    "SupportsBallsInBins",
    "SupportsCyclic",
    "SupportsFixedBatch",
    "SupportsOptimization",
    "BLTFamilyState",
    "BSRFamilyState",
    "get_mf_family_entry",
    "mf_accounting_requires_context",
    "BLTFamily",
    "BSRFamily",
    "BIFRFamily",
    "BLTReportSurface",
    "BSRFamilyReportInputs",
    "build_bsr_family_report_inputs",
    "compute_blt_fixed_batch_report_surface",
    "compute_bsr_family_cyclic_report_baseline",
    "resolve_bsr_family_fixed_batch_report_sensitivity",
]
