from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
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
) -> None:
    step_count = 0
    for xb, yb in private_loader:
        dp_optimizer.zero_grad()
        logits = private_model(xb)
        loss = F.cross_entropy(logits, yb)
        assert torch.isfinite(loss)
        loss.backward()
        dp_optimizer.step()
        step_count += 1
        if step_count >= steps:
            break
    assert step_count == steps


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

    _run_short_training_loop(
        private_model=private_model,
        dp_optimizer=dp_optimizer,
        private_loader=private_loader,
        steps=4,
    )


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

    _run_short_training_loop(
        private_model=private_model,
        dp_optimizer=dp_optimizer,
        private_loader=private_loader,
        steps=4,
    )
