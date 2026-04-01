from __future__ import annotations

import math
import random

import pytest
import torch
from opacus.accountants.analysis.bsr import (
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
)
from opacus.optimizers import CorrelatedNoiseMechanism

pytest.importorskip("jax")
pytest.importorskip("jax_privacy")

import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz


def _sample_monotone_coeffs(rng: random.Random, *, bands: int) -> list[float]:
    coeffs = [1.0]
    prev = 1.0
    for _ in range(1, bands):
        # Positive non-increasing family (within Opacus default contract).
        prev *= rng.uniform(0.15, 0.95)
        coeffs.append(prev)
    return coeffs


def test_live_jax_bsr_sensitivity_parity_random_grid() -> None:
    rng = random.Random(20260303)
    for _ in range(30):
        bands = rng.randint(1, 8)
        coeffs = _sample_monotone_coeffs(rng, bands=bands)
        steps = rng.randint(max(2, bands), 80)
        min_sep = rng.randint(1, 8)
        max_participations = rng.randint(1, 10)

        opacus_value = compute_bsr_mf_sensitivity_from_coeffs(
            coeffs=coeffs,
            steps=steps,
            min_separation=min_sep,
            max_participations=max_participations,
        )
        jax_value = float(
            jnp.sqrt(
                jax_toeplitz.minsep_sensitivity_squared(
                    strategy_coef=jnp.asarray(coeffs, dtype=jnp.float32),
                    min_sep=min_sep,
                    max_participations=max_participations,
                    n=steps,
                )
            ).item()
        )

        assert math.isclose(opacus_value, jax_value, rel_tol=0.0, abs_tol=2e-6), (
            f"Mismatch for coeffs={coeffs}, steps={steps}, "
            f"min_sep={min_sep}, max_participations={max_participations}: "
            f"opacus={opacus_value}, jax={jax_value}"
        )


def test_live_jax_bsr_kappa_parity_random_grid() -> None:
    rng = random.Random(20260304)
    for _ in range(30):
        bands = rng.randint(1, 8)
        coeffs = _sample_monotone_coeffs(rng, bands=bands)
        steps = rng.randint(1, 80)

        opacus_value = compute_bsr_kappa_from_coeffs(coeffs=coeffs, steps=steps)
        jax_value = float(
            jnp.sqrt(
                jax_toeplitz.sensitivity_squared(
                    coef=jnp.asarray(coeffs, dtype=jnp.float32),
                    n=steps,
                )
            ).item()
        )

        assert math.isclose(opacus_value, jax_value, rel_tol=0.0, abs_tol=2e-6), (
            f"Mismatch for coeffs={coeffs}, steps={steps}: "
            f"opacus={opacus_value}, jax={jax_value}"
        )


def test_live_jax_toeplitz_solver_parity_random_grid() -> None:
    rng = random.Random(20260305)
    torch_gen = torch.Generator().manual_seed(20260305)

    for _ in range(12):
        bands = rng.randint(1, 6)
        coeffs = _sample_monotone_coeffs(rng, bands=bands)
        steps = rng.randint(max(2, bands), 48)
        dim = rng.randint(4, 32)

        z = torch.randn((steps, dim), generator=torch_gen, dtype=torch.float32)

        mechanism = CorrelatedNoiseMechanism(coeffs=coeffs, z_std=0.0)
        mechanism.reset_state()
        u_rows = []
        for t in range(steps):
            u_rows.append(mechanism._solve_correlated_noise(z[t]))
        u_opacus = torch.stack(u_rows, dim=0)

        z_cols = jnp.asarray(z.numpy().T, dtype=jnp.float32)  # [dim, steps]
        solve_one = lambda rhs: jax_toeplitz.solve_banded(
            jnp.asarray(coeffs, dtype=jnp.float32), rhs
        )
        u_cols = jax.vmap(solve_one)(z_cols)  # [dim, steps]
        u_jax = torch.from_numpy(u_cols.T.__array__()).to(dtype=u_opacus.dtype)

        assert torch.allclose(u_opacus, u_jax, atol=1e-5, rtol=1e-5), (
            f"Mismatch for coeffs={coeffs}, steps={steps}, dim={dim}"
        )
