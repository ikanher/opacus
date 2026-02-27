from __future__ import annotations

import math
import random

import dp_accounting
import pytest
import torch
from opacus.accountants.analysis.bsr import (
    bsr_cyclic_poisson_epsilon_upper_bound,
    compute_bsr_kappa_from_coeffs,
    compute_bsr_mf_sensitivity_from_coeffs,
)
from opacus.accountants.rdp import RDPAccountant
from opacus.optimizers import CorrelatedNoiseMechanism

pytest.importorskip("jax")
pytest.importorskip("jax_privacy")

import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz
from jax_privacy.experimental import accounting as jax_experimental_accounting


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


def test_live_jax_cyclic_accounting_prefix_trajectory_parity() -> None:
    rng = random.Random(20260306)
    delta = 1e-5
    orders = list(RDPAccountant.DEFAULT_ALPHAS)

    for _ in range(8):
        steps = rng.randint(8, 120)
        bands = rng.randint(1, 12)
        sample_rate = rng.uniform(0.002, min(0.2, 0.95 / bands))
        base_noise_multiplier = rng.uniform(0.6, 2.5)
        sensitivity_scale = rng.uniform(0.6, 2.4)
        effective_noise_multiplier = base_noise_multiplier / sensitivity_scale
        q = float(sample_rate) * float(bands)

        for prefix in range(1, steps + 1):
            opacus_eps = bsr_cyclic_poisson_epsilon_upper_bound(
                noise_multiplier=effective_noise_multiplier,
                target_delta=delta,
                steps=prefix,
                sample_rate=sample_rate,
                bands=bands,
                rdp_orders=orders,
            )

            event = jax_experimental_accounting.amplified_bandmf_event(
                noise_multiplier=effective_noise_multiplier,
                iterations=prefix,
                num_bands=bands,
                sampling_prob=q,
            )
            accountant = dp_accounting.rdp.RdpAccountant(orders=orders)
            jax_eps = float(accountant.compose(event).get_epsilon(target_delta=delta))

            assert math.isclose(opacus_eps, jax_eps, rel_tol=0.0, abs_tol=1e-12), (
                f"prefix mismatch for steps={steps}, prefix={prefix}, bands={bands}, "
                f"sample_rate={sample_rate}, q={q}, "
                f"effective_noise_multiplier={effective_noise_multiplier}: "
                f"opacus={opacus_eps}, jax={jax_eps}"
            )
