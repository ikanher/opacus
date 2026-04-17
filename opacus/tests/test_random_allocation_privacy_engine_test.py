from __future__ import annotations

import io

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import NoiseMechanismConfig, PrivacyEngine, SamplingSemantics
from opacus.accountants import RandomAllocationAccountant, create_accountant
import opacus.privacy_engine as pe_mod


def _loader(*, n_samples: int = 40, in_dim: int = 4, n_classes: int = 3, batch_size: int = 8) -> DataLoader:
    gen = torch.Generator().manual_seed(20260327)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True)


def test_create_random_allocation_accountant() -> None:
    assert isinstance(create_accountant('random_allocation'), RandomAllocationAccountant)


def test_sampling_semantics_validates_k_out_of_t_contract() -> None:
    semantics = SamplingSemantics(sampling_mode='k-out-of-t', privacy_metadata={'num_steps': 5, 'num_selected': 2})
    assert semantics.sampling_mode == 'k_out_of_t'
    assert semantics.privacy_metadata['num_steps'] == 5
    with pytest.raises(ValueError, match='k_out_of_t sampling requires'):
        SamplingSemantics(sampling_mode='k_out_of_t', privacy_metadata={'num_steps': 5})
    with pytest.raises(ValueError, match='invalid k-out-of-t contract'):
        SamplingSemantics(sampling_mode='k_out_of_t', privacy_metadata={'num_steps': 2, 'num_selected': 3})


def test_privacy_engine_builds_gaussian_k_out_of_t_sampler_and_gets_finite_epsilon() -> None:
    pe = PrivacyEngine(accountant='random_allocation')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping='flat',
        grad_sample_mode='hooks',
        total_steps=50,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism='gaussian',
            accounting_mode='random_allocation_accountant',
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode='k_out_of_t',
            privacy_metadata={'num_steps': 5, 'num_selected': 2},
        ),
    )
    assert len(private_loader) == 5
    for _ in range(len(private_loader)):
        pe.accountant.step(noise_multiplier=1.0, sample_rate=0.4)
    epsilon = pe.get_epsilon(1e-5)
    assert float(epsilon) > 0.0
    assert dp_optimizer.accounting_mode == 'random_allocation_accountant'
    assert pe.noise_mechanism_config.mechanism_state['_random_allocation_accounting_kwargs'] is not None


def test_privacy_engine_k_out_of_t_persists_sampling_semantics_in_checkpoint() -> None:
    pe = PrivacyEngine(accountant='random_allocation')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    private_model, dp_optimizer, _ = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping='flat',
        grad_sample_mode='hooks',
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism='gaussian',
            accounting_mode='random_allocation_accountant',
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode='k_out_of_t',
            privacy_metadata={'num_steps': 5, 'num_selected': 2},
        ),
    )
    bio = io.BytesIO()
    pe.save_checkpoint(path=bio, module=private_model, optimizer=dp_optimizer)
    bio.seek(0)
    restored = PrivacyEngine(accountant='random_allocation')
    restored.load_checkpoint(path=bio, module=private_model, optimizer=dp_optimizer)
    assert restored.sampling_semantics.sampling_mode == 'k_out_of_t'
    assert restored.sampling_semantics.privacy_metadata['num_steps'] == 5
    assert restored.sampling_semantics.privacy_metadata['num_selected'] == 2


def test_privacy_engine_bnb_get_epsilon_uses_persisted_runtime_context() -> None:
    pe = PrivacyEngine(accountant='bnb')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    private_model, dp_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping='flat',
        grad_sample_mode='hooks',
        total_steps=20,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism='gaussian',
            accounting_mode='bnb_accountant',
            mechanism_state={},
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode='balls_in_bins',
            privacy_metadata={'bins': 5, 'bands': 1},
        ),
        bnb_calibration_mode='optimistic',
        bnb_num_samples=64,
        bnb_chunk_size=64,
    )
    for _ in range(len(private_loader)):
        pe.accountant.step(noise_multiplier=1.0, sample_rate=0.2)
    epsilon = pe.get_epsilon(1e-5)
    assert float(epsilon) > 0.0
    persisted = pe.noise_mechanism_config.mechanism_state['_bnb_accounting_kwargs']
    assert persisted['bnb_calibration_mode'] == 'optimistic'


@pytest.mark.parametrize(
    'mechanism,mechanism_state',
    [
        ('bsr', {'coeffs': [1.0, 0.2], 'bsr_bands': 2}),
        ('bisr', {'bisr_inv_coeffs': [1.0, -0.1], 'bsr_bands': 2}),
        ('bandmf', {'coeffs': [1.0, 0.2], 'bsr_bands': 2}),
        ('bandinvmf', {'bandinvmf_inv_coeffs': [1.0, -0.1], 'bsr_bands': 2}),
        ('bifr', {'bsr_bands': 2, 'bifr_frac': 0.5}),
        ('blt', {'theta': [0.8], 'theta_hat': [0.6], 'z_std': 0.03}),
    ],
)
def test_privacy_engine_supports_random_allocation_family_state_resolution(mechanism: str, mechanism_state: dict) -> None:
    pe = PrivacyEngine(accountant='random_allocation')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0, weight_decay=0.0)
    pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping='flat',
        grad_sample_mode='hooks',
        total_steps=20,
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism=mechanism,
            accounting_mode='random_allocation_accountant',
            mechanism_state=mechanism_state,
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode='k_out_of_t',
            privacy_metadata=(
                {'num_steps': 5, 'num_selected': 2}
                if mechanism == 'blt'
                else {'num_steps': 5, 'num_selected': 2, 'bands': 2}
            ),
        ),
    )
    state = pe.noise_mechanism_config.mechanism_state
    assert 'random_allocation_accountant_coeffs' in state
    assert len(state['random_allocation_accountant_coeffs']) > 0


@pytest.mark.parametrize(
    ("mechanism", "mechanism_state", "privacy_metadata", "expected_source"),
    [
        ("gaussian", {}, {"num_steps": 5, "num_selected": 2}, "raw_c_col"),
        ("bsr", {"coeffs": [1.0, 0.2], "bsr_bands": 2}, {"num_steps": 5, "num_selected": 2, "bands": 2}, "raw_c_col"),
        ("bandmf", {"coeffs": [1.0, 0.2], "bsr_bands": 2}, {"num_steps": 5, "num_selected": 2, "bands": 2}, "runtime_c_col"),
        ("bifr", {"bsr_bands": 2, "bifr_frac": 0.5}, {"num_steps": 5, "num_selected": 2, "bands": 2}, "abs_exact_factor_c_col"),
        ("blt", {"theta": [0.8], "theta_hat": [0.6], "z_std": 0.03}, {"num_steps": 5, "num_selected": 2}, "normalized_forward_c_col"),
    ],
)
def test_make_private_with_epsilon_supports_random_allocation_k_out_of_t_target_epsilon_route(
    monkeypatch: pytest.MonkeyPatch,
    mechanism: str,
    mechanism_state: dict,
    privacy_metadata: dict,
    expected_source: str,
) -> None:
    captured: dict[str, object] = {}

    def _fake_get_noise_multiplier(*, target_epsilon, target_delta, sample_rate, steps, accountant, **kwargs):
        captured.update(
            {
                "target_epsilon": float(target_epsilon),
                "target_delta": float(target_delta),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
                "accountant": str(accountant),
                **kwargs,
            }
        )
        return 1.0

    monkeypatch.setattr(pe_mod, "get_noise_multiplier", _fake_get_noise_multiplier)

    pe = PrivacyEngine(accountant='random_allocation')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    _private_model, dp_optimizer, private_loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        target_epsilon=3.0,
        target_delta=1e-5,
        total_steps=10,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping='flat',
        grad_sample_mode='hooks',
        noise_mechanism_config=NoiseMechanismConfig(
            mechanism=mechanism,
            accounting_mode='random_allocation_accountant',
            mechanism_state=mechanism_state,
        ),
        sampling_semantics=SamplingSemantics(
            sampling_mode='k_out_of_t',
            privacy_metadata=privacy_metadata,
        ),
    )

    assert captured["accountant"] == "random_allocation"
    assert captured["sampling_semantics"].sampling_mode == "k_out_of_t"
    assert captured["mechanism_state"]["random_allocation_accountant_coeffs_source"] == expected_source
    assert len(captured["mechanism_state"]["random_allocation_accountant_coeffs"]) > 0
    assert dp_optimizer.accounting_mode == "random_allocation_accountant"
    assert len(private_loader) == int(privacy_metadata["num_steps"])


def test_k_out_of_t_requires_random_allocation_routing() -> None:
    pe = PrivacyEngine(accountant='prv')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with pytest.raises(ValueError, match='random_allocation_accountant requires sampling_mode='):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping='flat',
            grad_sample_mode='hooks',
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism='gaussian',
                accounting_mode='random_allocation_accountant',
                mechanism_state={},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode='balls_in_bins',
                privacy_metadata={'bins': 5},
            ),
        )


def test_bnb_rejects_k_out_of_t_routing() -> None:
    pe = PrivacyEngine(accountant='bnb')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with pytest.raises(ValueError, match="balls_in_bins"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping='flat',
            grad_sample_mode='hooks',
            total_steps=20,
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism='gaussian',
                accounting_mode='bnb_accountant',
                mechanism_state={},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode='k_out_of_t',
                privacy_metadata={'num_steps': 5, 'num_selected': 2},
            ),
        )


@pytest.mark.parametrize("sampling_mode,privacy_metadata", [
    ('balls_in_bins', {'bins': 5}),
    ('b_min_sep', {'bands': 2}),
])
def test_random_allocation_rejects_balls_in_bins_family_sampling_modes(
    sampling_mode: str,
    privacy_metadata: dict,
) -> None:
    pe = PrivacyEngine(accountant='random_allocation')
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with pytest.raises(ValueError, match="random_allocation_accountant requires sampling_mode='k_out_of_t'|temporarily disabled"):
        pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping='flat',
            grad_sample_mode='hooks',
            noise_mechanism_config=NoiseMechanismConfig(
                mechanism='gaussian',
                accounting_mode='random_allocation_accountant',
                mechanism_state={},
            ),
            sampling_semantics=SamplingSemantics(
                sampling_mode=sampling_mode,
                privacy_metadata=privacy_metadata,
            ),
        )
