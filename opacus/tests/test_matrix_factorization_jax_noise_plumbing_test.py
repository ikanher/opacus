from __future__ import annotations

import pytest
import torch
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.optimizers import CorrelatedNoiseMechanism
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import opacus.optimizers.optimizer as optimizer_mod

pytest.importorskip("jax_privacy")


def _build_tiny_cifar_loader(*, n_samples: int = 32, batch_size: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(20260225)
    x = torch.randn(n_samples, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n_samples,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True)


def _build_tiny_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 10),
    )


def _set_zero_grad_samples(dp_optimizer, *, batch_size: int) -> None:
    for p in dp_optimizer.params:
        p.grad_sample = torch.zeros(
            (batch_size,) + tuple(p.shape),
            dtype=p.dtype,
            device=p.device,
        )


def _run_deterministic_noise_steps(
    *,
    monkeypatch,
    coeffs: list[float],
    steps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _build_tiny_cnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _build_tiny_cifar_loader()
    pe = PrivacyEngine()

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(99),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": coeffs, "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    del private_model
    del private_loader

    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, CorrelatedNoiseMechanism)

    shapes = [tuple(p.shape) for p in dp_optimizer.params]
    numels = [int(p.numel()) for p in dp_optimizer.params]
    total_dim = sum(numels)

    # Build deterministic z-stream per step, then split into per-parameter chunks
    # in the same order CorrelatedNoiseMechanism flattens params.
    z_flats = []
    deterministic_chunks = []
    for step in range(steps):
        base = torch.arange(total_dim, dtype=torch.float32) + float((step + 1) * 1000)
        z_flats.append(base.clone())
        offset = 0
        for shape, n in zip(shapes, numels):
            chunk = base[offset : offset + n].reshape(shape).clone()
            deterministic_chunks.append(chunk)
            offset += n

    def _deterministic_noise(*, std, reference, generator, secure_mode):
        del std, generator, secure_mode
        assert deterministic_chunks, "ran out of deterministic noise chunks"
        return deterministic_chunks.pop(0).to(device=reference.device, dtype=reference.dtype)

    monkeypatch.setattr(optimizer_mod, "_generate_noise", _deterministic_noise)

    z_rows = []
    u_rows = []
    batch_size = int(dp_optimizer.expected_batch_size)
    for _ in range(steps):
        dp_optimizer.zero_grad()
        _set_zero_grad_samples(dp_optimizer, batch_size=batch_size)
        dp_optimizer.step()

        assert mechanism.last_flat_z is not None
        assert mechanism.last_flat_u is not None
        z_rows.append(mechanism.last_flat_z.detach().clone())
        u_rows.append(mechanism.last_flat_u.detach().clone())

    assert mechanism.steps_with_noise == steps
    assert not deterministic_chunks

    z = torch.stack(z_rows, dim=0)
    u = torch.stack(u_rows, dim=0)

    expected_z = torch.stack(z_flats, dim=0).to(dtype=z.dtype, device=z.device)
    assert torch.allclose(z, expected_z, atol=0.0, rtol=0.0)
    return z, u


def _flatten_params(params: list[torch.Tensor]) -> torch.Tensor:
    if not params:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat([p.detach().reshape(-1).clone() for p in params], dim=0)


def _run_deterministic_noise_param_trajectory(
    *,
    monkeypatch,
    coeffs: list[float],
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    model = _build_tiny_cnn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _build_tiny_cifar_loader()
    pe = PrivacyEngine()

    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        noise_generator=torch.Generator().manual_seed(99),
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism="bsr",
            accounting_mode="bsr_accountant",
            mechanism_state={"coeffs": coeffs, "z_std": 0.01},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode="torch_sampler",
            privacy_metadata={},
        ),
    )
    del private_model
    del private_loader

    mechanism = dp_optimizer.noise_mechanism
    assert isinstance(mechanism, CorrelatedNoiseMechanism)

    params = list(dp_optimizer.params)
    shapes = [tuple(p.shape) for p in params]
    numels = [int(p.numel()) for p in params]
    total_dim = sum(numels)

    z_flats = []
    deterministic_chunks = []
    for step in range(steps):
        base = torch.arange(total_dim, dtype=torch.float32) + float((step + 1) * 1000)
        z_flats.append(base.clone())
        offset = 0
        for shape, n in zip(shapes, numels):
            chunk = base[offset : offset + n].reshape(shape).clone()
            deterministic_chunks.append(chunk)
            offset += n

    def _deterministic_noise(*, std, reference, generator, secure_mode):
        del std, generator, secure_mode
        assert deterministic_chunks, "ran out of deterministic noise chunks"
        return deterministic_chunks.pop(0).to(device=reference.device, dtype=reference.dtype)

    monkeypatch.setattr(optimizer_mod, "_generate_noise", _deterministic_noise)

    batch_size = int(dp_optimizer.expected_batch_size)
    init_flat = _flatten_params(params).to(dtype=torch.float64)
    u_rows = []
    flat_params = [init_flat.clone()]
    for _ in range(steps):
        dp_optimizer.zero_grad()
        _set_zero_grad_samples(dp_optimizer, batch_size=batch_size)
        dp_optimizer.step()

        assert mechanism.last_flat_u is not None
        u_t = mechanism.last_flat_u.detach().clone().to(dtype=torch.float64)
        u_rows.append(u_t)
        flat_params.append(_flatten_params(params).to(dtype=torch.float64))

    assert not deterministic_chunks
    z = torch.stack(z_flats, dim=0).to(dtype=torch.float64)
    u = torch.stack(u_rows, dim=0)
    trajectory = torch.stack(flat_params, dim=0)
    lr = float(dp_optimizer.original_optimizer.param_groups[0]["lr"])
    return z, u, trajectory, lr, batch_size


def test_e2e_bsr_noise_plumbing_matches_toeplitz_recurrence(monkeypatch) -> None:
    coeffs = [1.0, 0.5, 0.25]
    z, u = _run_deterministic_noise_steps(monkeypatch=monkeypatch, coeffs=coeffs, steps=4)

    # Toeplitz forward-substitution invariant: C u_t = z_t.
    for t in range(int(z.shape[0])):
        rhs = coeffs[0] * u[t]
        max_lag = min(t, len(coeffs) - 1)
        for lag in range(1, max_lag + 1):
            rhs = rhs + coeffs[lag] * u[t - lag]
        assert torch.allclose(rhs, z[t], atol=1e-6, rtol=1e-6)


def test_e2e_bsr_noise_plumbing_matches_jax_toeplitz_solver(monkeypatch) -> None:
    pytest.importorskip("jax")

    import jax
    import jax.numpy as jnp
    from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz

    coeffs = [1.0, 0.5, 0.25]
    z, u_opacus = _run_deterministic_noise_steps(
        monkeypatch=monkeypatch, coeffs=coeffs, steps=4
    )

    z_np = z.detach().cpu().numpy()
    # Solve C u = z for all flattened coordinates in one vectorized JAX call.
    z_cols = jnp.asarray(z_np.T, dtype=jnp.float32)  # [dim, steps]
    solve_one = lambda rhs: jax_toeplitz.solve_banded(
        jnp.asarray(coeffs, dtype=jnp.float32), rhs
    )
    u_cols = jax.vmap(solve_one)(z_cols)  # [dim, steps]
    u_jax = u_cols.T  # [steps, dim]
    u_jax_torch = torch.from_numpy(u_jax.__array__()).to(dtype=u_opacus.dtype)

    assert torch.allclose(u_opacus.cpu(), u_jax_torch, atol=1e-5, rtol=1e-5)


def test_e2e_bsr_noise_param_trajectory_matches_recorded_u(monkeypatch) -> None:
    coeffs = [1.0, 0.5, 0.25]
    _, u, trajectory, lr, batch_size = _run_deterministic_noise_param_trajectory(
        monkeypatch=monkeypatch,
        coeffs=coeffs,
        steps=32,
    )
    assert torch.isfinite(u).all()
    assert torch.isfinite(trajectory).all()

    for t in range(int(u.shape[0])):
        expected_next = trajectory[t] - (lr / float(batch_size)) * u[t]
        assert torch.allclose(trajectory[t + 1], expected_next, atol=1e-6, rtol=1e-6)


def test_e2e_bsr_noise_plumbing_matches_jax_toeplitz_solver_long_horizon(
    monkeypatch,
) -> None:
    pytest.importorskip("jax")

    import jax
    import jax.numpy as jnp
    from jax_privacy.matrix_factorization import toeplitz as jax_toeplitz

    coeffs = [1.0, 0.5, 0.25]
    z, u_opacus = _run_deterministic_noise_steps(
        monkeypatch=monkeypatch,
        coeffs=coeffs,
        steps=32,
    )
    assert torch.isfinite(z).all()
    assert torch.isfinite(u_opacus).all()

    z_np = z.detach().cpu().numpy()
    z_cols = jnp.asarray(z_np.T, dtype=jnp.float32)  # [dim, steps]
    solve_one = lambda rhs: jax_toeplitz.solve_banded(
        jnp.asarray(coeffs, dtype=jnp.float32), rhs
    )
    u_cols = jax.vmap(solve_one)(z_cols)  # [dim, steps]
    u_jax = u_cols.T  # [steps, dim]
    u_jax_torch = torch.from_numpy(u_jax.__array__()).to(dtype=u_opacus.dtype)

    assert torch.allclose(u_opacus.cpu(), u_jax_torch, atol=1e-5, rtol=1e-5)
