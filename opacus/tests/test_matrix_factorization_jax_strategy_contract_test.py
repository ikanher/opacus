from __future__ import annotations

import numpy as np
import pytest

from opacus.privacy_engine import PrivacyEngine


pytest.importorskip("jax_privacy")

from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz


def _jax_strategy(*, steps: int, bands: int) -> list[float]:
    coeffs = jax_toeplitz.optimize_banded_toeplitz(n=int(steps), bands=int(bands))
    return [float(x) for x in np.asarray(coeffs)]


def _opacus_auto_strategy(*, steps: int, bands: int) -> list[float]:
    return PrivacyEngine._optimize_bsr_cyclic_coeffs(
        bands=int(bands),
        steps=int(steps),
    )


def test_id_band_strategy_matches_jax_reference() -> None:
    # Identity path (bands=1) should align exactly.
    got = _opacus_auto_strategy(steps=32, bands=1)
    expected = _jax_strategy(steps=32, bands=1)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_cyclic_bsr_auto_strategy_rejects_steps_below_bands() -> None:
    with pytest.raises(ValueError, match="steps >= bands"):
        _opacus_auto_strategy(steps=5, bands=8)


@pytest.mark.parametrize(
    "steps,bands",
    [
        (32, 8),
        (128, 8),
    ],
)
def test_cyclic_bsr_auto_strategy_matches_jax_optimizer(steps: int, bands: int) -> None:
    """
    Contract test for cyclic BSR replication:
    auto-generated correlated-noise strategy should match JAX Toeplitz optimizer output.
    """
    got = _opacus_auto_strategy(steps=steps, bands=bands)
    expected = _jax_strategy(steps=steps, bands=bands)
    tol = 1e-5
    assert got == pytest.approx(expected, rel=tol, abs=tol), (
        "Cyclic BSR strategy mismatch: DPDL auto strategy diverges from JAX "
        f"Toeplitz optimize_banded_toeplitz for steps={steps}, bands={bands}. "
        f"dpdl_head={got[:5]}, jax_head={expected[:5]}"
    )
