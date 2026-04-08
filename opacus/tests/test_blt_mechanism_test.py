from __future__ import annotations

import numpy as np
import pytest
import torch

from opacus.noise_mechanisms import BufferedToeplitzNoiseMechanism
from opacus.accountants.analysis.blt import (
    BLTPairedParams,
    BLTParams,
    blt_materialize,
    blt_pair_from_theta_pair,
)


def _paired_blt() -> BLTPairedParams:
    return blt_pair_from_theta_pair(theta=[0.8, 0.3], theta_hat=[0.6, 0.1])


def test_streamed_inverse_matches_dense_paired_inverse_multiplication() -> None:
    pair = _paired_blt()
    mechanism = BufferedToeplitzNoiseMechanism(pair=pair, z_std=0.0)

    steps = 8
    dim = 5
    gen = torch.Generator().manual_seed(20260402)

    z_rows = [torch.randn(dim, generator=gen, dtype=torch.float64) for _ in range(steps)]
    u_rows = [mechanism._apply_inverse_stream(z) for z in z_rows]

    z_mat = torch.stack(z_rows, dim=0).numpy()
    u_stream = torch.stack(u_rows, dim=0).numpy()
    inverse = blt_materialize(pair.inverse, n=steps)

    assert np.allclose(u_stream, inverse @ z_mat, atol=1e-10, rtol=1e-8)


def test_state_roundtrip_replays_future_sequence() -> None:
    pair = _paired_blt()
    mechanism_a = BufferedToeplitzNoiseMechanism(pair=pair, z_std=0.0)
    mechanism_b = BufferedToeplitzNoiseMechanism(pair=pair, z_std=0.0)

    gen = torch.Generator().manual_seed(20260403)
    z_seq = [torch.randn(7, generator=gen, dtype=torch.float64) for _ in range(10)]

    first_outputs = [mechanism_a._apply_inverse_stream(z) for z in z_seq[:4]]
    assert len(first_outputs) == 4

    mechanism_b.load_state_dict(mechanism_a.state_dict())
    future_a = [mechanism_a._apply_inverse_stream(z) for z in z_seq[4:]]
    future_b = [mechanism_b._apply_inverse_stream(z) for z in z_seq[4:]]

    for a, b in zip(future_a, future_b):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-8)


def test_streamed_inverse_preserves_dtype() -> None:
    mechanism = BufferedToeplitzNoiseMechanism(pair=_paired_blt(), z_std=0.0)
    z = torch.randn(6, generator=torch.Generator().manual_seed(20260404), dtype=torch.float64)
    u = mechanism._apply_inverse_stream(z)
    assert u.dtype == torch.float64


def test_load_state_dict_rejects_mismatched_pair() -> None:
    pair = _paired_blt()
    other_pair = BLTPairedParams(
        forward=BLTParams(theta=[0.7, 0.2], omega=[0.2, -0.1]),
        inverse=BLTParams(theta=[0.5, 0.15], omega=[-0.4, 0.05]),
    ).canonicalized()
    mechanism = BufferedToeplitzNoiseMechanism(pair=pair, z_std=0.0)
    other = BufferedToeplitzNoiseMechanism(pair=other_pair, z_std=0.0)

    with pytest.raises(ValueError, match="mismatched BLT pair"):
        other.load_state_dict(mechanism.state_dict())
