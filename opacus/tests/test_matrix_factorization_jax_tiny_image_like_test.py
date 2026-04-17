from __future__ import annotations

"""Tiny image-like JAX-linked parity tests.

Runtime constraints:
- synthetic dataset only (`n_samples=64`, `batch_size=8`)
- long-horizon training checks capped at 32 optimizer steps
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.accountants.analysis.bsr import compute_bsr_mf_sensitivity_from_coeffs
from opacus.optimizers import CorrelatedNoiseMechanism
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


pytest.importorskip("jax_privacy")


def _load_bsr_fixture() -> dict:
    path = Path(__file__).resolve().parent / "fixtures" / "bsr_jax_golden_values.json"
    return json.loads(path.read_text())


def _golden_sensitivity(*, coeffs: str, n: int, b: int, k: int) -> float:
    fixture = _load_bsr_fixture()
    for case in fixture["sensitivity_cases"]:
        if (
            case["coeffs"] == coeffs
            and int(case["n"]) == n
            and int(case["b"]) == b
            and int(case["k"]) == k
        ):
            return float(case["jax_sensitivity"])
    raise AssertionError(f"Missing sensitivity golden case: coeffs={coeffs} n={n} b={b} k={k}")


def _golden_kappa(*, coeffs: str, n: int) -> float:
    fixture = _load_bsr_fixture()
    for case in fixture["kappa_cases"]:
        if case["coeffs"] == coeffs and int(case["n"]) == n:
            return float(case["jax_kappa"])
    raise AssertionError(f"Missing kappa golden case: coeffs={coeffs} n={n}")


def _build_tiny_cifar_loader(*, n_samples: int = 64, batch_size: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(20260225)
    x = torch.randn(n_samples, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n_samples,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True)


def _build_tiny_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    )


def _run_short_training_loop(
    *,
    private_model: nn.Module,
    dp_optimizer: torch.optim.Optimizer,
    private_loader: DataLoader,
    steps: int,
) -> list[float]:
    step_count = 0
    losses: list[float] = []
    loader_iter = iter(private_loader)

    while step_count < steps:
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            # Continue past epoch boundaries so we can validate multi-epoch behavior.
            loader_iter = iter(private_loader)
            xb, yb = next(loader_iter)

        dp_optimizer.zero_grad()
        logits = private_model(xb)
        loss = F.cross_entropy(logits, yb)
        assert torch.isfinite(loss)
        loss.backward()
        dp_optimizer.step()
        losses.append(float(loss.detach()))

        mechanism = getattr(dp_optimizer, "noise_mechanism", None)
        if isinstance(mechanism, CorrelatedNoiseMechanism):
            if mechanism.last_flat_z is not None:
                assert torch.isfinite(mechanism.last_flat_z).all()
            if mechanism.last_flat_u is not None:
                assert torch.isfinite(mechanism.last_flat_u).all()

        step_count += 1

    assert step_count == steps
    assert losses
    return losses


def _run_non_private_training_loop(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    steps: int,
) -> None:
    step_count = 0
    loader_iter = iter(loader)
    while step_count < steps:
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            xb, yb = next(loader_iter)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        step_count += 1

    assert step_count == steps


def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)


def test_tiny_cifar_like_e2e_fixed_batch_resolves_jax_golden_sensitivity() -> None:
    coeffs = [1.0, 0.8, 0.4, 0.1]  # fixture key: "custom"
    total_steps = 16
    max_participations = 4
    min_separation = 2
    expected_sensitivity = _golden_sensitivity(
        coeffs="custom",
        n=total_steps,
        b=min_separation,
        k=max_participations,
    )

    model = _build_tiny_cnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _build_tiny_cifar_loader()

    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        total_steps=total_steps,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(11),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": coeffs,
                "max_participations": max_participations,
                "min_separation": min_separation,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert "mf_sensitivity" in state
    assert math.isclose(
        float(state["mf_sensitivity"]), expected_sensitivity, rel_tol=0.0, abs_tol=1e-6
    )

    losses = _run_short_training_loop(
        private_model=private_model,
        dp_optimizer=dp_optimizer,
        private_loader=private_loader,
        steps=32,
    )
    assert len(losses) == 32
    assert all(math.isfinite(x) for x in losses)


def test_tiny_cifar_like_multi_epoch_identity_bsr_matches_non_private_final_params() -> None:
    steps = 32
    lr = 0.01
    max_grad_norm = 1e9  # keep clipping inactive so DP/noise-free path matches SGD

    base = _build_tiny_cnn()
    non_private_model = _build_tiny_cnn()
    non_private_model.load_state_dict(base.state_dict())
    private_seed_model = _build_tiny_cnn()
    private_seed_model.load_state_dict(base.state_dict())

    non_private_optimizer = torch.optim.SGD(non_private_model.parameters(), lr=lr)
    private_optimizer = torch.optim.SGD(private_seed_model.parameters(), lr=lr)
    non_private_loader = _build_tiny_cifar_loader()
    private_loader_seed = _build_tiny_cifar_loader()

    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=private_seed_model,
        optimizer=private_optimizer,
        data_loader=private_loader_seed,
        noise_multiplier=0.0,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(123),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": [1.0], "z_std": 0.0},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    _run_non_private_training_loop(
        model=non_private_model,
        optimizer=non_private_optimizer,
        loader=non_private_loader,
        steps=steps,
    )
    _run_short_training_loop(
        private_model=private_model,
        dp_optimizer=dp_optimizer,
        private_loader=private_loader,
        steps=steps,
    )

    non_private_params = list(non_private_model.parameters())
    private_params = list(private_model.parameters())
    assert len(non_private_params) == len(private_params)
    for a, b in zip(non_private_params, private_params):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=1e-6)


def test_tiny_cifar_like_full_bsr_contract_with_clipping_noise_and_sensitivity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz

    coeffs = [1.0, 0.8, 0.4, 0.1]
    total_steps = 32
    max_participations = 4
    min_separation = 2
    max_grad_norm = 0.05  # force clipping on this synthetic setup
    batch_size = 8
    steps_to_run = 8

    expected_sensitivity = compute_bsr_mf_sensitivity_from_coeffs(
        coeffs=coeffs,
        steps=total_steps,
        max_participations=max_participations,
        min_separation=min_separation,
    )
    jax_sensitivity_sq = jax_toeplitz.minsep_sensitivity_squared(
        strategy_coef=jnp.asarray(coeffs, dtype=jnp.float32),
        min_sep=min_separation,
        max_participations=max_participations,
        n=total_steps,
    )
    expected_sensitivity_jax = float(jnp.sqrt(jax_sensitivity_sq).item())
    assert math.isclose(
        float(expected_sensitivity),
        expected_sensitivity_jax,
        rel_tol=0.0,
        abs_tol=1e-6,
    )

    model = _build_tiny_cnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _build_tiny_cifar_loader(batch_size=batch_size)

    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        total_steps=total_steps,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(20260301),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={
                "coeffs": coeffs,
                "max_participations": max_participations,
                "min_separation": min_separation,
            },
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert "mf_sensitivity" in state
    assert math.isclose(
        float(state["mf_sensitivity"]),
        float(expected_sensitivity),
        rel_tol=0.0,
        abs_tol=1e-6,
    )

    mechanism = getattr(dp_optimizer, "noise_mechanism", None)
    assert isinstance(mechanism, CorrelatedNoiseMechanism)
    assert tuple(mechanism.coeffs) == tuple(float(c) for c in coeffs)
    # Runtime calibration contract for mean-loss path:
    # z_std = noise_multiplier * max_grad_norm / expected_batch_size.
    expected_effective_z_std = (
        float(dp_optimizer.noise_multiplier)
        * float(max_grad_norm)
        / float(dp_optimizer.expected_batch_size)
    )
    got_effective_z_std = float(mechanism.z_std)
    assert math.isclose(got_effective_z_std, expected_effective_z_std, rel_tol=1e-6, abs_tol=1e-8)

    clipping_engaged = False
    z_rows: list[torch.Tensor] = []
    u_rows: list[torch.Tensor] = []
    loader_iter = iter(private_loader)
    for _ in range(steps_to_run):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(private_loader)
            xb, yb = next(loader_iter)

        dp_optimizer.zero_grad()
        loss = F.cross_entropy(private_model(xb), yb)
        assert torch.isfinite(loss)
        loss.backward()

        # Check whether clipping is actually active for this batch.
        grad_samples = dp_optimizer.grad_samples
        per_param_sq = [
            g.reshape(len(g), -1).norm(2, dim=-1).pow(2) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_sq, dim=0).sum(dim=0).sqrt()
        if bool((per_sample_norms > float(max_grad_norm) * 1.0001).any().item()):
            clipping_engaged = True

        dp_optimizer.step()

        assert mechanism.last_flat_z is not None
        assert mechanism.last_flat_u is not None
        z_rows.append(mechanism.last_flat_z.detach().clone())
        u_rows.append(mechanism.last_flat_u.detach().clone())
        flat_summed = _flatten_tensors([p.summed_grad for p in dp_optimizer.params])  # type: ignore[list-item]
        flat_grad = _flatten_tensors([p.grad for p in dp_optimizer.params])  # type: ignore[list-item]
        expected_flat_grad = (
            flat_summed + mechanism.last_flat_u.to(dtype=flat_summed.dtype)
        ) / float(dp_optimizer.expected_batch_size)
        assert torch.allclose(flat_grad, expected_flat_grad, atol=1e-6, rtol=1e-6)
        assert torch.isfinite(flat_grad).all()

    assert clipping_engaged
    z = torch.stack(z_rows, dim=0)
    u = torch.stack(u_rows, dim=0)
    assert torch.isfinite(z).all()
    assert torch.isfinite(u).all()

    # Live JAX parity on the exact generated BSR noise stream from the clipped/noisy run.
    z_cols = jnp.asarray(z.detach().cpu().numpy().T, dtype=jnp.float32)
    solve_one = lambda rhs: jax_toeplitz.solve_banded(
        jnp.asarray(coeffs, dtype=jnp.float32), rhs
    )
    u_cols = jax.vmap(solve_one)(z_cols)
    u_jax = u_cols.T
    u_jax_torch = torch.from_numpy(u_jax.__array__()).to(dtype=u.dtype)
    assert torch.allclose(u.cpu(), u_jax_torch, atol=1e-5, rtol=1e-5)


def test_tiny_cifar_like_e2e_cyclic_resolves_jax_golden_kappa_scale() -> None:
    coeffs = [1.0, 0.8, 0.4, 0.1]  # fixture key: "custom"
    total_steps = 16
    bands = 2
    expected_kappa = _golden_kappa(coeffs="custom", n=total_steps)

    model = _build_tiny_cnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _build_tiny_cifar_loader()

    pe = PrivacyEngine()
    private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=8.0,
        target_delta=1e-5,
        total_steps=total_steps,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(17),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": coeffs},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="cyclic_poisson",
            privacy_metadata={"bands": bands},
        ),
    )

    state = dp_optimizer.noise_mechanism_config.mechanism_state
    assert "sensitivity_scale" in state
    assert math.isclose(
        float(state["sensitivity_scale"]), expected_kappa, rel_tol=0.0, abs_tol=1e-6
    )
    assert "mf_sensitivity" not in state

    losses = _run_short_training_loop(
        private_model=private_model,
        dp_optimizer=dp_optimizer,
        private_loader=private_loader,
        steps=32,
    )
    assert len(losses) == 32
    assert all(math.isfinite(x) for x in losses)
