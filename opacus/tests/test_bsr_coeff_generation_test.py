#!/usr/bin/env python3

from __future__ import annotations

import math

import pytest

from opacus.accountants.analysis.bsr import generate_bsr_coeffs_from_sgd_workload


@pytest.mark.parametrize(
    ("bands", "momentum", "weight_decay", "expected"),
    [
        (1, 0.0, 1.0, [1.0]),
        (2, 0.0, 1.0, [1.0, 0.5]),
        (3, 0.0, 1.0, [1.0, 0.5, 0.375]),
    ],
)
def test_generate_bsr_coeffs_known_cases(
    bands: int, momentum: float, weight_decay: float, expected: list[float]
) -> None:
    got = generate_bsr_coeffs_from_sgd_workload(
        bands=bands,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_generate_bsr_coeffs_no_weight_decay_maps_to_alpha_one() -> None:
    got = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=0.0,
    )
    assert got == pytest.approx([1.0, 0.5, 0.375, 0.3125], rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("bands", "momentum", "weight_decay", "err"),
    [
        (0, 0.0, 1.0, "bands must be >= 1"),
        (4, -1e-9, 1.0, "momentum must satisfy 0 <= momentum < 1"),
        (4, 1.0, 1.0, "momentum must satisfy 0 <= momentum < 1"),
        (4, 0.1, -1.0, "weight_decay must satisfy 0 < weight_decay <= 1"),
        (4, 0.1, 1.1, "weight_decay must satisfy 0 < weight_decay <= 1"),
        (4, 0.9, 0.8, "momentum <= effective weight decay"),
    ],
)
def test_generate_bsr_coeffs_validation(
    bands: int, momentum: float, weight_decay: float, err: str
) -> None:
    with pytest.raises(ValueError, match=err):
        generate_bsr_coeffs_from_sgd_workload(
            bands=bands,
            momentum=momentum,
            weight_decay=weight_decay,
        )


def test_generate_bsr_coeffs_are_finite() -> None:
    got = generate_bsr_coeffs_from_sgd_workload(
        bands=8,
        momentum=0.9,
        weight_decay=0.9999,
    )
    assert len(got) == 8
    assert all(math.isfinite(float(v)) for v in got)
