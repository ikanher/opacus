from __future__ import annotations

from typing import Any, Dict

import torch
from opacus.accountants.analysis.bnb import validate_bnb_c_matrix_contract
from opacus.mechanism_contracts import SamplingSemantics


def validate_bnb_runtime_consistency(
    *,
    mechanism_state: Dict[str, Any],
    sampling_semantics: SamplingSemantics | None,
    c_matrix: Any,
    bands: int,
    c_matrix_contract: Dict[str, Any],
    coeffs_error_prefix: str,
) -> None:
    state = mechanism_state if isinstance(mechanism_state, dict) else {}
    coeffs = state.get("bnb_accountant_coeffs", state.get("coeffs"))

    if coeffs is None or not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
        raise ValueError(f"{coeffs_error_prefix} requires non-empty `coeffs`")

    if int(bands) != len(coeffs):
        raise ValueError(
            "bnb consistency check failed: `bands` must match len(coeffs); "
            f"got bands={int(bands)} and len(coeffs)={len(coeffs)}"
        )

    metadata = (
        sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
    )

    metadata_bands = metadata.get("bands")
    if metadata_bands is not None and int(metadata_bands) != int(bands):
        raise ValueError(
            "bnb consistency check failed: sampling_semantics privacy_metadata['bands'] "
            f"({int(metadata_bands)}) != accounting bands ({int(bands)})"
        )

    if not torch.is_tensor(c_matrix):
        raise ValueError("bnb consistency check requires `c_matrix` to be a torch.Tensor")

    if c_matrix.ndim != 2:
        raise ValueError("bnb consistency check requires `c_matrix` with shape [d, m]")

    d, _m = c_matrix.shape
    if d < int(bands):
        raise ValueError(
            "bnb consistency check failed: c_matrix must have at least `bands` rows; "
            f"got rows={d}, bands={int(bands)}"
        )

    validate_bnb_c_matrix_contract(
        c_matrix=c_matrix,
        coeffs=coeffs,
        bands=int(bands),
        c_matrix_contract=c_matrix_contract,
    )
