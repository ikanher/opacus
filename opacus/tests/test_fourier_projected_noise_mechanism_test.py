import copy
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import FourierClippingConfig, NoiseMechanismConfig, PrivacyEngine
from opacus.accountants.analysis.blt import BLTPairedParams, BLTParams
from opacus.noise_mechanisms import (
    BufferedToeplitzNoiseMechanism,
    CorrelatedNoiseMechanism,
    InverseBandNoiseMechanism,
)
from opacus.optimizers import (
    DPOptimizer,
    DistributedFourierDPOptimizer,
    FourierDPOptimizer,
    get_optimizer_class,
)


RUN_DISTRIBUTED = os.getenv("OPACUS_RUN_DISTRIBUTED_TESTS") == "1"


def _supports_local_gloo_process_group() -> bool:
    if not dist.is_available():
        return False

    with tempfile.NamedTemporaryFile(delete=False) as sync:
        init_method = f"file://{sync.name}"

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=0,
            world_size=1,
        )
        return True
    except Exception:
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


HAS_LOCAL_GLOO = _supports_local_gloo_process_group()


class RecordingNoiseMechanism:
    def __init__(self):
        self.calls = 0
        self.param_counts = []
        self.numels = []

    def add_noise(self, optimizer):
        self.calls += 1
        self.param_counts.append(len(optimizer.params))
        self.numels.append(int(optimizer.params[0].numel()))
        for p in optimizer.params:
            p.grad = p.summed_grad.clone().view_as(p)

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state_dict):
        self.calls = int(state_dict.get("calls", 0))


def _make_fourier_optimizer(
    parameter: nn.Parameter,
    *,
    config: FourierClippingConfig,
    max_grad_norm: float = 10_000.0,
    noise_mechanism=None,
    distributed: bool = False,
):
    base_optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer_cls = DistributedFourierDPOptimizer if distributed else FourierDPOptimizer
    kwargs = {}
    if noise_mechanism is not None:
        kwargs["noise_mechanism"] = noise_mechanism
    return optimizer_cls(
        optimizer=base_optimizer,
        noise_multiplier=0.0,
        max_grad_norm=max_grad_norm,
        expected_batch_size=2,
        loss_reduction="sum",
        fourier_clipping_config=config,
        **kwargs,
    )


def _clip_decode(parameter, grad_sample, *, config, max_grad_norm=10_000.0, noise_mechanism=None):
    optimizer = _make_fourier_optimizer(
        parameter,
        config=config,
        max_grad_norm=max_grad_norm,
        noise_mechanism=noise_mechanism,
    )
    parameter.grad_sample = grad_sample.clone()
    optimizer.clip_and_accumulate()
    optimizer.add_noise()
    return parameter.grad.detach().clone(), optimizer


def _loader(batch_size: int = 4):
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.5, -0.5, 1.0],
            [1.5, 0.5, -1.0],
            [0.25, 0.25, 0.25],
            [-0.25, 0.75, -0.5],
        ]
    )
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _blt_noise_mechanism() -> BufferedToeplitzNoiseMechanism:
    return BufferedToeplitzNoiseMechanism(
        pair=BLTPairedParams(
            forward=BLTParams(theta=[], omega=[]),
            inverse=BLTParams(theta=[], omega=[]),
        ),
        z_std=0.0,
    )


def _rank_grad_sample(rank: int) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 10.0 + float(rank)


def _single_process_fourier_reference(*, max_grad_norm: float = 10_000.0) -> torch.Tensor:
    parameter = nn.Parameter(torch.zeros(2, 3))
    optimizer = _make_fourier_optimizer(
        parameter,
        config=FourierClippingConfig(retain_count=2),
        max_grad_norm=max_grad_norm,
        noise_mechanism=RecordingNoiseMechanism(),
    )
    parameter.grad_sample = torch.cat([_rank_grad_sample(0), _rank_grad_sample(1)], dim=0)

    assert optimizer.pre_step() is True
    assert parameter.grad is not None
    return parameter.grad.detach().cpu()


def _set_rank_env(rank: int, world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")


def _worker_fourier_distributed_reference(
    rank: int,
    world_size: int,
    init_method: str,
    results_dir: str,
) -> None:
    _set_rank_env(rank, world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = nn.Parameter(torch.zeros(2, 3))
        mechanism = RecordingNoiseMechanism()
        optimizer = _make_fourier_optimizer(
            parameter,
            config=FourierClippingConfig(retain_count=2),
            max_grad_norm=0.5,
            noise_mechanism=mechanism,
            distributed=True,
        )
        parameter.grad_sample = _rank_grad_sample(rank)

        assert optimizer.pre_step() is True
        pre_reduce_has_grad = parameter.grad is not None and parameter.grad.shape == parameter.shape
        pre_reduce_finite = bool(torch.isfinite(parameter.grad).all())
        noiser_calls_match_rank = mechanism.calls == (1 if rank == 0 else 0)

        optimizer.reduce_gradients()
        reference = _single_process_fourier_reference(max_grad_norm=0.5)
        unclipped_reference = _single_process_fourier_reference(max_grad_norm=10_000.0)
        matches_reference = bool(torch.allclose(parameter.grad.cpu(), reference, atol=1e-6, rtol=1e-6))
        clipping_was_active = not bool(
            torch.allclose(reference, unclipped_reference, atol=1e-6, rtol=1e-6)
        )

        torch.save(
            {
                "rank": rank,
                "ok": bool(
                    pre_reduce_has_grad
                    and pre_reduce_finite
                    and noiser_calls_match_rank
                    and matches_reference
                    and clipping_was_active
                ),
            },
            Path(results_dir) / f"result_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _run_two_rank_fourier_worker(worker) -> list[tuple[int, bool]]:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.NamedTemporaryFile(delete=False) as sync:
            init_method = f"file://{sync.name}"

        procs = [
            ctx.Process(target=worker, args=(rank, 2, init_method, tmpdir))
            for rank in range(2)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=60)
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"distributed worker failed with exit code {proc.exitcode}"
                )

        results = []
        for rank in range(2):
            payload = torch.load(Path(tmpdir) / f"result_{rank}.pt")
            results.append((int(payload["rank"]), bool(payload["ok"])))
        results.sort(key=lambda x: x[0])
        return results


def test_get_optimizer_class_routes_fourier_clipping() -> None:
    assert get_optimizer_class("fourier", distributed=False, grad_sample_mode="hooks") is FourierDPOptimizer
    assert get_optimizer_class("fourier", distributed=True, grad_sample_mode="hooks") is DistributedFourierDPOptimizer
    assert get_optimizer_class("flat", distributed=False, grad_sample_mode="hooks") is DPOptimizer


@pytest.mark.parametrize("layout", ["layer_matrix_columns", "parameter_blockwise"])
def test_full_rank_decode_roundtrip_for_matrix_layouts(layout: str) -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    grad_sample = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 10.0
    config = FourierClippingConfig(layout=layout, retain_count=8, block_size=8)

    decoded, _ = _clip_decode(parameter, grad_sample, config=config)

    assert decoded.shape == parameter.shape
    assert torch.allclose(decoded, grad_sample.sum(dim=0), atol=1e-5, rtol=1e-5)


def test_matrix_column_projection_matches_manual_small_tensor() -> None:
    parameter = nn.Parameter(torch.zeros(3, 2))
    grad_sample = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.5], [2.5, -3.0], [4.0, -2.0]],
        ]
    )
    config = FourierClippingConfig(layout="layer_matrix_columns", retain_count=1)

    decoded, optimizer = _clip_decode(parameter, grad_sample, config=config)

    dct = optimizer._dct_matrix(size=2, device=grad_sample.device, dtype=grad_sample.dtype)
    q = dct.t()[:, :1]
    encoded = grad_sample @ q
    expected = encoded.sum(dim=0) @ q.t()
    assert torch.allclose(decoded, expected, atol=1e-6, rtol=1e-6)


def test_global_encoded_norm_controls_clipping() -> None:
    parameter = nn.Parameter(torch.zeros(2))
    grad_sample = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    config = FourierClippingConfig(layout="parameter_blockwise", retain_count=2, block_size=2)

    decoded, _ = _clip_decode(
        parameter,
        grad_sample,
        config=config,
        max_grad_norm=5.0,
    )

    assert torch.allclose(decoded, torch.tensor([6.0, 8.0]), atol=1e-6, rtol=1e-6)


def test_empty_fourier_batch_invokes_noiser_and_produces_parameter_grad() -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    mechanism = RecordingNoiseMechanism()
    optimizer = _make_fourier_optimizer(
        parameter,
        config=FourierClippingConfig(retain_count=2),
        noise_mechanism=mechanism,
    )
    parameter.grad_sample = torch.empty((0, 2, 3))

    assert optimizer.pre_step() is True

    assert mechanism.calls == 1
    assert mechanism.param_counts == [1]
    assert mechanism.numels == [6]
    assert parameter.grad is not None
    assert parameter.grad.shape == parameter.shape
    assert torch.allclose(parameter.grad, torch.zeros_like(parameter))


@pytest.mark.parametrize(
    "param_shape,grad_sample_shape,config,expected_encoded_numel,expected_fallbacks",
    [
        (
            (5,),
            (0, 5),
            FourierClippingConfig(layout="layer_matrix_columns", retain_count=2, block_size=4),
            4,
            ["parameter_blockwise"],
        ),
        (
            (5,),
            (0, 5),
            FourierClippingConfig(layout="parameter_blockwise", retain_count=2, block_size=4),
            4,
            [],
        ),
        (
            (2, 1, 2, 2),
            (0, 2, 1, 2, 2),
            FourierClippingConfig(layout="layer_matrix_columns", retain_count=2),
            8,
            [],
        ),
    ],
)
def test_empty_fourier_batch_handles_fallback_blockwise_and_conv(
    param_shape,
    grad_sample_shape,
    config,
    expected_encoded_numel,
    expected_fallbacks,
) -> None:
    parameter = nn.Parameter(torch.zeros(param_shape))
    mechanism = RecordingNoiseMechanism()
    optimizer = _make_fourier_optimizer(
        parameter,
        config=config,
        noise_mechanism=mechanism,
    )
    parameter.grad_sample = torch.empty(grad_sample_shape)

    assert optimizer.pre_step() is True

    assert mechanism.calls == 1
    assert mechanism.numels == [expected_encoded_numel]
    assert parameter.grad is not None
    assert parameter.grad.shape == parameter.shape
    assert torch.allclose(parameter.grad, torch.zeros_like(parameter))
    metadata = optimizer.state_dict()["_dp_fourier_clipping_metadata"]
    assert metadata["fallbacks"] == expected_fallbacks
    assert metadata["original_trainable_numel"] == parameter.numel()
    assert metadata["encoded_numel"] == expected_encoded_numel
    assert metadata["encoded_fraction_of_original"] == pytest.approx(
        expected_encoded_numel / parameter.numel()
    )
    assert metadata["compression_ratio_vs_original"] == pytest.approx(
        parameter.numel() / expected_encoded_numel
    )


def test_matrix_layout_records_blockwise_fallback_metadata() -> None:
    parameter = nn.Parameter(torch.zeros(5))
    grad_sample = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    config = FourierClippingConfig(
        layout="layer_matrix_columns",
        retain_count=4,
        block_size=4,
    )

    _, optimizer = _clip_decode(parameter, grad_sample, config=config)

    metadata = optimizer.state_dict()["_dp_fourier_clipping_metadata"]
    assert metadata["layout"] == "layer_matrix_columns"
    assert metadata["fallbacks"] == ["parameter_blockwise"]
    assert metadata["fallback_counts"] == {"parameter_blockwise": 1}
    assert metadata["plan_kind_counts"] == {"blockwise": 1}
    assert metadata["plan_entry_count"] == 1
    assert metadata["original_trainable_numel"] == 5
    assert metadata["encoded_numel"] == 8
    assert metadata["encoded_fraction_of_original"] == pytest.approx(8 / 5)
    assert metadata["compression_ratio_vs_original"] == pytest.approx(5 / 8)


def test_convolution_reshape_roundtrip_preserves_shape_dtype_and_values() -> None:
    parameter = nn.Parameter(torch.zeros(2, 1, 2, 2, dtype=torch.float64))
    grad_sample = torch.arange(16, dtype=torch.float64).reshape(2, 2, 1, 2, 2) / 7.0
    config = FourierClippingConfig(layout="layer_matrix_columns", retain_count=4)

    decoded, _ = _clip_decode(parameter, grad_sample, config=config)

    assert decoded.shape == parameter.shape
    assert decoded.dtype == torch.float64
    assert torch.allclose(decoded, grad_sample.sum(dim=0), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "noise_mechanism",
    [
        None,
        CorrelatedNoiseMechanism(coeffs=[1.0, 0.2], z_std=0.0),
        InverseBandNoiseMechanism(inverse_coeffs=[1.0, -0.1], z_std=0.0),
        BufferedToeplitzNoiseMechanism(
            pair=BLTPairedParams(
                forward=BLTParams(theta=[], omega=[]),
                inverse=BLTParams(theta=[], omega=[]),
            ),
            z_std=0.0,
        ),
    ],
)
def test_existing_noisers_operate_over_encoded_buffers(noise_mechanism) -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    grad_sample = torch.randn(2, 2, 3, generator=torch.Generator().manual_seed(7))
    config = FourierClippingConfig(layout="layer_matrix_columns", retain_count=3)

    decoded, optimizer = _clip_decode(
        parameter,
        grad_sample,
        config=config,
        noise_mechanism=noise_mechanism,
    )

    assert optimizer._encoded_param is not None
    assert optimizer._encoded_param.grad is not None
    assert decoded.shape == parameter.shape
    assert torch.isfinite(decoded).all()

def _manual_fourier_indices_for_matrix(optimizer, grad_sample: torch.Tensor, rank: int) -> torch.Tensor:
    matrix, _, _ = optimizer._matrix_view(grad_sample)
    matrix = matrix.to(optimizer._working_dtype(matrix.dtype))
    _, rows, cols = matrix.shape
    right = rows >= cols
    size = cols if right else rows
    dct = optimizer._dct_matrix(size=size, device=matrix.device, dtype=matrix.dtype)
    basis = dct.t()
    full_coeffs = matrix @ basis if right else torch.matmul(basis.t(), matrix)
    dim = 2 if right else 1
    reduce_dims = tuple(i for i in range(full_coeffs.ndim) if i != dim)
    scores = full_coeffs.detach().abs().sum(dim=reduce_dims)
    return torch.topk(scores, k=rank, largest=True, sorted=False).indices.sort().values


def _make_gradient_from_dct_coeffs(optimizer, coeffs: torch.Tensor, *, right: bool) -> torch.Tensor:
    size = coeffs.shape[-1] if right else coeffs.shape[-2]
    dct = optimizer._dct_matrix(size=size, device=coeffs.device, dtype=coeffs.dtype)
    basis = dct.t()
    if right:
        return coeffs @ basis.t()
    return torch.matmul(basis, coeffs)


def test_adaptive_topk_matrix_right_selects_scored_non_lowpass_indices() -> None:
    parameter = nn.Parameter(torch.zeros(4, 3))
    config = FourierClippingConfig(
        layout="layer_matrix_columns",
        mode="adaptive_topk_leaky",
        retain_count=2,
        allow_unaccounted_selection=True,
    )
    optimizer = _make_fourier_optimizer(parameter, config=config)
    coeffs = torch.zeros(2, 4, 3)
    coeffs[:, :, 1] = 10.0
    coeffs[:, :, 2] = -7.0
    coeffs[:, :, 0] = 0.25
    grad_sample = _make_gradient_from_dct_coeffs(optimizer, coeffs, right=True)

    _, optimizer = _clip_decode(parameter, grad_sample, config=config)

    entry = optimizer._encoded_plan[0]
    expected = _manual_fourier_indices_for_matrix(optimizer, grad_sample, rank=2)
    assert entry.kind == "matrix_right"
    assert entry.indices.cpu().tolist() == [1, 2]
    assert torch.equal(entry.indices.cpu(), expected.cpu())
    metadata = optimizer.state_dict()["_dp_fourier_clipping_metadata"]
    assert metadata["selected_indices"] == [
        {"kind": "matrix_right", "shape": [2], "numel": 2, "indices": [1, 2]}
    ]


def test_adaptive_topk_matrix_left_selects_scored_non_lowpass_indices() -> None:
    parameter = nn.Parameter(torch.zeros(3, 5))
    config = FourierClippingConfig(
        layout="layer_matrix_columns",
        mode="adaptive_topk_leaky",
        retain_count=1,
        allow_unaccounted_selection=True,
    )
    optimizer = _make_fourier_optimizer(parameter, config=config)
    coeffs = torch.zeros(2, 3, 5)
    coeffs[:, 2, :] = 9.0
    coeffs[:, 0, :] = 0.5
    grad_sample = _make_gradient_from_dct_coeffs(optimizer, coeffs, right=False)

    _, optimizer = _clip_decode(parameter, grad_sample, config=config)

    entry = optimizer._encoded_plan[0]
    expected = _manual_fourier_indices_for_matrix(optimizer, grad_sample, rank=1)
    assert entry.kind == "matrix_left"
    assert entry.indices.cpu().tolist() == [2]
    assert torch.equal(entry.indices.cpu(), expected.cpu())


def test_adaptive_topk_convolution_reshape_scores_matrix_view_and_preserves_shape() -> None:
    parameter = nn.Parameter(torch.zeros(6, 1, 1, 2))
    config = FourierClippingConfig(
        layout="layer_matrix_columns",
        mode="adaptive_topk_leaky",
        retain_count=1,
        allow_unaccounted_selection=True,
    )
    optimizer = _make_fourier_optimizer(parameter, config=config)
    coeffs = torch.zeros(2, 6, 2)
    coeffs[:, :, 1] = 11.0
    coeffs[:, :, 0] = 0.25
    matrix_grad = _make_gradient_from_dct_coeffs(optimizer, coeffs, right=True)
    grad_sample = matrix_grad.reshape(2, 6, 1, 1, 2)

    decoded, optimizer = _clip_decode(parameter, grad_sample, config=config)

    entry = optimizer._encoded_plan[0]
    expected = _manual_fourier_indices_for_matrix(optimizer, grad_sample, rank=1)
    assert entry.kind == "matrix_right"
    assert entry.conv_shape == torch.Size([6, 1, 1, 2])
    assert entry.indices.cpu().tolist() == [1]
    assert torch.equal(entry.indices.cpu(), expected.cpu())
    assert decoded.shape == parameter.shape


def test_adaptive_topk_blockwise_selects_indices_per_block() -> None:
    parameter = nn.Parameter(torch.zeros(8))
    config = FourierClippingConfig(
        layout="parameter_blockwise",
        mode="adaptive_topk_leaky",
        retain_count=1,
        block_size=4,
        allow_unaccounted_selection=True,
    )
    optimizer = _make_fourier_optimizer(parameter, config=config)
    dct = optimizer._dct_matrix(size=4, device=parameter.device, dtype=parameter.dtype)
    coeffs = torch.zeros(2, 2, 4)
    coeffs[:, 0, 3] = 8.0
    coeffs[:, 0, 0] = 0.5
    coeffs[:, 1, 2] = 6.0
    coeffs[:, 1, 1] = 0.25
    blocks = coeffs @ dct
    grad_sample = blocks.reshape(2, 8)

    _, optimizer = _clip_decode(parameter, grad_sample, config=config)

    entry = optimizer._encoded_plan[0]
    assert entry.kind == "blockwise"
    assert entry.indices.cpu().tolist() == [[3], [2]]
    metadata = optimizer.state_dict()["_dp_fourier_clipping_metadata"]
    assert metadata["selected_indices"] == [
        {"kind": "blockwise", "shape": [2, 1], "numel": 2, "indices": [[3], [2]]}
    ]


def test_adaptive_topk_metadata_marks_private_selection_and_leakage() -> None:
    parameter = nn.Parameter(torch.zeros(3, 4))
    grad_sample = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(3))
    config = FourierClippingConfig(
        layout="layer_matrix_columns",
        mode="adaptive_topk_leaky",
        retain_count=2,
        allow_unaccounted_selection=True,
    )

    _, optimizer = _clip_decode(parameter, grad_sample, config=config)

    metadata = optimizer.state_dict()["_dp_fourier_clipping_metadata"]
    assert metadata["privacy_claim_valid"] is False
    assert metadata["reported_epsilon_is_nominal"] is True
    assert metadata["reported_epsilon_excludes_selection"] is True
    assert metadata["fourier_selection_is_private"] is True
    assert metadata["allow_unaccounted_selection"] is True
    assert metadata["selection_metadata_includes_indices"] is True
    assert metadata["selected_indices"]


def test_fixed_lowpass_metadata_marks_public_selection() -> None:
    parameter = nn.Parameter(torch.zeros(3, 4))
    grad_sample = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(4))
    config = FourierClippingConfig(
        layout="layer_matrix_columns",
        mode="fixed_lowpass",
        retain_count=2,
    )

    _, optimizer = _clip_decode(parameter, grad_sample, config=config)

    metadata = optimizer.state_dict()["_dp_fourier_clipping_metadata"]
    assert metadata["reported_epsilon_excludes_selection"] is False
    assert metadata["fourier_selection_is_private"] is False
    assert metadata["selected_indices"] == [
        {"kind": "matrix_left", "shape": [2], "numel": 2, "indices": [0, 1]}
    ]


def test_zero_grad_clears_fourier_encoded_buffers() -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    config = FourierClippingConfig(retain_count=2)
    optimizer = _make_fourier_optimizer(parameter, config=config)
    parameter.grad_sample = torch.ones(2, 2, 3)

    optimizer.clip_and_accumulate()
    assert optimizer._encoded_param is not None
    assert optimizer._encoded_param.summed_grad is not None

    optimizer.zero_grad()
    assert optimizer._encoded_param.summed_grad is None
    assert optimizer._encoded_param.grad is None
    assert optimizer._encoded_plan == []


def test_noise_mechanism_config_rejects_fourier_mechanism() -> None:
    with pytest.raises(ValueError, match="clipping='fourier'"):
        NoiseMechanismConfig(mechanism="fourier")

    with pytest.raises(ImportError):
        from opacus.noise_mechanisms import FourierProjectedGaussianNoiseMechanism  # noqa: F401


def test_fourier_config_records_nonclaiming_nominal_metadata() -> None:
    config = FourierClippingConfig(retain_count=4)
    state = config.state_dict()

    assert state["layout"] == "layer_matrix_columns"
    assert state["privacy_claim_valid"] is False
    assert state["reported_epsilon_is_nominal"] is True
    assert state["reported_epsilon_excludes_selection"] is False


def test_adaptive_topk_requires_opt_in_and_records_selection_exclusion() -> None:
    with pytest.raises(ValueError, match="allow_unaccounted_selection"):
        FourierClippingConfig(mode="adaptive_topk_leaky", retain_count=2)

    config = FourierClippingConfig(
        mode="adaptive_topk_leaky",
        retain_count=2,
        allow_unaccounted_selection=True,
    )
    assert config.state_dict()["reported_epsilon_excludes_selection"] is True


def test_privacy_engine_wires_fourier_clipping_without_changing_base_mechanism() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine()
    config = FourierClippingConfig(retain_count=2)

    private_model, dp_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=10.0,
        poisson_sampling=False,
        clipping="fourier",
        fourier_clipping_config=config,
    )
    x, y = next(iter(private_loader))
    loss = F.cross_entropy(private_model(x), y)
    loss.backward()
    assert dp_optimizer.pre_step() is True

    assert isinstance(dp_optimizer, FourierDPOptimizer)
    assert engine.noise_mechanism_config.mechanism == "gaussian"
    assert engine.fourier_clipping_config == config
    assert dp_optimizer.fourier_clipping_config.state_dict()["privacy_claim_valid"] is False
    for p in dp_optimizer.params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_privacy_engine_rejects_fourier_config_without_fourier_clipping() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine()
    with pytest.raises(ValueError, match="only valid"):
        engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="flat",
            fourier_clipping_config=FourierClippingConfig(retain_count=2),
        )


def test_privacy_engine_rejects_list_max_grad_norm_for_fourier_make_private() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine()

    with pytest.raises(ValueError, match="single global encoded-space clipping norm"):
        engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=[1.0, 1.0],
            poisson_sampling=False,
            clipping="fourier",
            fourier_clipping_config=FourierClippingConfig(retain_count=2),
        )


def test_privacy_engine_rejects_list_max_grad_norm_for_fourier_make_private_with_epsilon() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine()

    with pytest.raises(ValueError, match="single global encoded-space clipping norm"):
        engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            target_epsilon=8.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=[1.0, 1.0],
            poisson_sampling=False,
            clipping="fourier",
            fourier_clipping_config=FourierClippingConfig(retain_count=2),
        )


def test_privacy_engine_rejects_adaptive_fourier_under_distributed(monkeypatch) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine()
    config = FourierClippingConfig(
        mode="adaptive_topk_leaky",
        retain_count=2,
        allow_unaccounted_selection=True,
    )
    monkeypatch.setattr(
        PrivacyEngine,
        "_resolve_distributed_runtime",
        staticmethod(lambda module: (True, False)),
    )

    with pytest.raises(ValueError, match="adaptive_topk_leaky"):
        engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=_loader(),
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="fourier",
            fourier_clipping_config=config,
        )


def test_checkpoint_roundtrip_preserves_fourier_config() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine()
    config = FourierClippingConfig(layout="parameter_blockwise", retain_count=2, block_size=4)
    private_model, dp_optimizer, _ = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=_loader(),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        clipping="fourier",
        fourier_clipping_config=config,
    )

    with tempfile.NamedTemporaryFile() as f:
        engine.save_checkpoint(path=f.name, module=private_model, optimizer=dp_optimizer)
        loaded_model = copy.deepcopy(private_model)
        loaded_engine = PrivacyEngine()
        loaded_engine.load_checkpoint(path=f.name, module=loaded_model)

    assert loaded_engine.fourier_clipping_config == config


def test_distributed_blt_fourier_checkpoint_save_rejects_nonzero_rank(monkeypatch) -> None:
    model = nn.Linear(3, 2)
    parameter = next(model.parameters())
    engine = PrivacyEngine()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    optimizer = _make_fourier_optimizer(
        parameter,
        config=FourierClippingConfig(retain_count=2),
        noise_mechanism=_blt_noise_mechanism(),
        distributed=True,
    )

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError, match="rank 0"):
            engine.save_checkpoint(path=f.name, module=model, optimizer=optimizer)


def test_distributed_fourier_rank_zero_noises_encoded_buffers(monkeypatch) -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    mechanism = RecordingNoiseMechanism()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    optimizer = _make_fourier_optimizer(
        parameter,
        config=FourierClippingConfig(retain_count=2),
        noise_mechanism=mechanism,
        distributed=True,
    )
    parameter.grad_sample = torch.ones(2, 2, 3)

    optimizer.clip_and_accumulate()
    optimizer.add_noise()

    assert mechanism.calls == 1
    assert mechanism.param_counts == [1]
    assert mechanism.numels == [6]
    assert parameter.grad is not None


def test_distributed_fourier_nonzero_rank_decodes_without_base_noiser(monkeypatch) -> None:
    parameter = nn.Parameter(torch.zeros(2, 3))
    mechanism = RecordingNoiseMechanism()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    optimizer = _make_fourier_optimizer(
        parameter,
        config=FourierClippingConfig(retain_count=2),
        noise_mechanism=mechanism,
        distributed=True,
    )
    parameter.grad_sample = torch.ones(2, 2, 3)

    optimizer.clip_and_accumulate()
    optimizer.add_noise()

    assert mechanism.calls == 0
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(
    not RUN_DISTRIBUTED or not HAS_LOCAL_GLOO,
    reason=(
        "requires OPACUS_RUN_DISTRIBUTED_TESTS=1 and local gloo process-group support"
    ),
)
def test_distributed_fourier_two_rank_reduces_decoded_gradients() -> None:
    results = _run_two_rank_fourier_worker(_worker_fourier_distributed_reference)
    assert results == [(0, True), (1, True)]
