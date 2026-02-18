#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict


_BNB_CALIBRATION_PROFILES: Dict[str, Dict[str, Any]] = {
    # Conservative defaults for direct Opacus usage.
    "opacus_strict": {
        "bnb_calibration_timeout_seconds": 10.0,
        "bnb_num_samples": 100_000,
        "bnb_seed": 0,
        "bnb_reduce_dimensionality": False,
        "bnb_confidence_alpha": 1e-6,
        "bnb_evr_num_checks": 3,
        "bnb_verify_both_directions": True,
        "bnb_require_evr_pass": True,
        "bnb_evr_use_candidate_ladder": True,
        "bnb_candidate_multipliers": (1.0, 1.1, 1.25, 1.5, 2.0),
        "bnb_tolerance": 1e-4,
        "bnb_max_iterations": 200,
    },
    # Faster defaults for DPDL training loops.
    "dpdl_fast": {
        "bnb_calibration_timeout_seconds": 10.0,
        "bnb_num_samples": 2_000,
        "bnb_seed": 0,
        "bnb_reduce_dimensionality": False,
        "bnb_confidence_alpha": 1e-3,
        "bnb_evr_num_checks": 1,
        "bnb_verify_both_directions": False,
        "bnb_require_evr_pass": True,
        "bnb_evr_use_candidate_ladder": True,
        "bnb_candidate_multipliers": (1.0, 1.1, 1.25, 1.5, 2.0),
        "bnb_tolerance": 1e-3,
        "bnb_max_iterations": 64,
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
