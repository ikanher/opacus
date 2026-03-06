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


def test_generate_bsr_coeffs_equal_roots_branch_is_geometric() -> None:
    # BSR-C01: alpha == beta branch follows geometric closed form alpha**j.
    got = generate_bsr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.9,
        weight_decay=0.9,
    )
    expected = [1.0, 0.9, 0.81, 0.729, 0.6561]
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_generate_bsr_coeffs_prefix_is_stable_across_band_truncation() -> None:
    # BSR-C01: band truncation is a pure suffix cut; shorter-band coeffs are
    # exactly the prefix of longer-band coeffs for the same optimizer workload.
    short = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.2,
        weight_decay=0.95,
    )
    long = generate_bsr_coeffs_from_sgd_workload(
        bands=10,
        momentum=0.2,
        weight_decay=0.95,
    )
    assert len(short) == 4
    assert len(long) == 10
    assert long[: len(short)] == pytest.approx(short, rel=0.0, abs=1e-12)


def test_generate_bsr_coeffs_nonnegative_and_nonincreasing_for_admissible_inputs() -> None:
    # BSR-C01: generated coefficients are nonnegative and nonincreasing on
    # admissible (momentum <= effective weight_decay) workloads.
    got = generate_bsr_coeffs_from_sgd_workload(
        bands=16,
        momentum=0.7,
        weight_decay=0.95,
    )
    assert all(v >= -1e-12 for v in got)
    for a, b in zip(got, got[1:]):
        assert b <= a + 1e-12
