#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict


_BNB_CALIBRATION_PROFILES: Dict[str, Dict[str, Any]] = {
    # Conservative defaults for direct Opacus usage.
    "opacus_strict": {
        "bnb_num_samples": 100_000,
        "bnb_seed": 0,
        "bnb_reduce_dimensionality": False,
        "bnb_confidence_alpha": 1e-6,
        "bnb_require_evr_pass": False,
        "bnb_tolerance": 1e-7,
        "bnb_max_iterations": 1000,
    },
    # Faster defaults for DPDL training loops.
    "dpdl_fast": {
        "bnb_num_samples": 2_000,
        "bnb_seed": 0,
        "bnb_reduce_dimensionality": False,
        "bnb_confidence_alpha": 1e-3,
        "bnb_require_evr_pass": False,
        "bnb_tolerance": 1e-6,
        "bnb_max_iterations": 256,
    },
}


def resolve_bnb_calibration_kwargs(
    *,
    profile: str,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    defaults = _BNB_CALIBRATION_PROFILES.get(profile)
    if defaults is None:
        supported = ", ".join(sorted(_BNB_CALIBRATION_PROFILES.keys()))
        raise ValueError(
            f"Unsupported BNB calibration profile '{profile}'. Supported values: {supported}"
        )

    resolved = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if key in defaults and value is not None:
                resolved[key] = value

    return resolved
