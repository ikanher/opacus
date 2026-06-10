# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from opacus.mechanism_contracts import FourierClippingConfig
from opacus.noise_mechanisms import BufferedToeplitzNoiseMechanism, CorrelatedNoiseMechanism

from .optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed


@dataclass
class _PlanEntry:
    parameter: nn.Parameter
    original_shape: torch.Size
    encoded_shape: torch.Size
    kind: str
    working_dtype: torch.dtype
    original_dtype: torch.dtype
    indices: torch.Tensor
    matrix_shape: Optional[tuple[int, int]] = None
    conv_shape: Optional[torch.Size] = None
    pad: int = 0
    block_size: int = 0
    num_blocks: int = 0

    @property
    def encoded_numel(self) -> int:
        return int(math.prod(tuple(self.encoded_shape)))


class _EncodedOptimizerView:
    def __init__(self, owner: "FourierDPOptimizer") -> None:
        self._owner = owner

    @property
    def params(self):
        return [self._owner._encoded_param]

    @property
    def loss_reduction(self):
        return self._owner.loss_reduction

    @property
    def expected_batch_size(self):
        return self._owner.expected_batch_size

    @property
    def accumulated_iterations(self):
        return self._owner.accumulated_iterations

    @property
    def generator(self):
        return self._owner.generator

    @property
    def secure_mode(self):
        return self._owner.secure_mode

    @property
    def noise_multiplier(self):
        return self._owner.noise_multiplier

    @property
    def normalize_clipping(self):
        return self._owner.normalize_clipping

    @property
    def max_grad_norm(self):
        return self._owner.max_grad_norm


class FourierDPOptimizer(DPOptimizer):
    """
    DP optimizer that clips and noises retained Fourier/DCT coordinates.

    Fourier is a clipping/release-space transform, not a base noising
    mechanism. This optimizer encodes per-sample gradients, clips in the encoded
    space, exposes one synthetic encoded parameter to the selected base noiser,
    and decodes the noised encoded release back to normal parameter gradients.
    """

    _dct_cache: dict[tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        normalize_clipping: bool = False,
        fourier_clipping_config: FourierClippingConfig,
        **kwargs,
    ):
        if isinstance(max_grad_norm, (list, tuple)):
            raise ValueError(
                "Fourier clipping uses a single global encoded-space clipping norm; "
                "pass scalar max_grad_norm when clipping='fourier'"
            )
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
            **kwargs,
        )
        self.fourier_clipping_config = fourier_clipping_config
        self._encoded_param: Optional[nn.Parameter] = None
        self._encoded_plan: list[_PlanEntry] = []
        self._encoded_view = _EncodedOptimizerView(self)
        self._last_fourier_fallbacks: list[str] = []
        self.fourier_clipping_metadata = self._build_fourier_metadata()

    def _build_fourier_metadata(self) -> dict:
        metadata = dict(self.fourier_clipping_config.state_dict())
        original_trainable_numel = sum(int(p.numel()) for p in self.params if p.requires_grad)
        encoded_numel = (
            int(self._encoded_param.numel()) if self._encoded_param is not None else 0
        )
        kind_counts = Counter(entry.kind for entry in self._encoded_plan)
        fallback_counts = Counter(self._last_fourier_fallbacks)
        selected_indices, selected_total, selected_truncated = self._selected_indices_metadata()
        metadata["fallbacks"] = list(self._last_fourier_fallbacks)
        metadata["fallback_counts"] = dict(fallback_counts)
        metadata["plan_kind_counts"] = dict(kind_counts)
        metadata["plan_entry_count"] = int(len(self._encoded_plan))
        metadata["selected_indices"] = selected_indices
        metadata["selected_indices_total_numel"] = int(selected_total)
        metadata["selected_indices_truncated"] = bool(selected_truncated)
        metadata["selection_metadata_includes_indices"] = not bool(selected_truncated)
        metadata["fourier_selection_is_private"] = (
            self.fourier_clipping_config.mode == "adaptive_topk_leaky"
        )
        metadata["original_trainable_numel"] = int(original_trainable_numel)
        metadata["encoded_numel"] = int(encoded_numel)
        metadata["encoded_fraction_of_original"] = (
            float(encoded_numel) / float(original_trainable_numel)
            if original_trainable_numel > 0 and encoded_numel > 0
            else None
        )
        metadata["compression_ratio_vs_original"] = (
            float(original_trainable_numel) / float(encoded_numel)
            if original_trainable_numel > 0 and encoded_numel > 0
            else None
        )
        return metadata

    def _selected_indices_metadata(self, *, max_indices: int = 4096) -> tuple[list[dict], int, bool]:
        total = sum(int(entry.indices.numel()) for entry in self._encoded_plan)
        truncated = total > int(max_indices)
        payload = []
        for entry in self._encoded_plan:
            item = {
                "kind": entry.kind,
                "shape": list(entry.indices.shape),
                "numel": int(entry.indices.numel()),
            }
            if not truncated:
                item["indices"] = entry.indices.detach().cpu().tolist()
            payload.append(item)
        return payload, int(total), bool(truncated)

    @staticmethod
    def _working_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in (torch.float32, torch.float64):
            return dtype
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        raise ValueError(f"Fourier clipping requires floating gradients, got {dtype}")

    @classmethod
    def _dct_matrix(
        cls,
        *,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        size = int(size)
        key = (size, str(device.type), device.index, dtype)
        cached = cls._dct_cache.get(key)
        if cached is not None and cached.device == device:
            return cached

        n = torch.arange(size, device=device, dtype=dtype)
        k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
        matrix = torch.cos((math.pi / float(size)) * (n + 0.5) * k)
        matrix[0, :] *= math.sqrt(1.0 / float(size))
        if size > 1:
            matrix[1:, :] *= math.sqrt(2.0 / float(size))
        cls._dct_cache[key] = matrix
        return matrix

    def _retained_count(self, size: int) -> int:
        cfg = self.fourier_clipping_config
        size = int(size)
        if cfg.retain_count is not None:
            return max(1, min(size, int(cfg.retain_count)))
        assert cfg.retain_frac is not None
        return max(1, min(size, int(math.ceil(float(size) * float(cfg.retain_frac)))))

    def _fixed_indices(self, *, size: int, rank: int, device: torch.device) -> torch.Tensor:
        return torch.arange(rank, device=device, dtype=torch.long)

    def _topk_indices(self, coeffs: torch.Tensor, *, dim: int, rank: int) -> torch.Tensor:
        if rank >= coeffs.shape[dim]:
            return torch.arange(coeffs.shape[dim], device=coeffs.device, dtype=torch.long)
        reduce_dims = tuple(i for i in range(coeffs.ndim) if i != dim)
        scores = coeffs.detach().abs().sum(dim=reduce_dims)
        return torch.topk(scores, k=rank, largest=True, sorted=False).indices.sort().values

    def _matrix_view(self, grad_sample: torch.Tensor) -> tuple[torch.Tensor, str, Optional[torch.Size]]:
        if grad_sample.ndim == 3:
            return grad_sample, "matrix", None
        if grad_sample.ndim == 5:
            batch = grad_sample.shape[0]
            conv_shape = grad_sample.shape[1:]
            conv_width = int(math.prod(tuple(conv_shape[1:])))
            return grad_sample.reshape(batch, int(conv_shape[0]), conv_width), "conv", conv_shape
        raise ValueError("matrix view is supported only for 2D parameters and convolution kernels")

    def _build_matrix_entry(self, p: nn.Parameter, grad_sample: torch.Tensor) -> tuple[_PlanEntry, torch.Tensor]:
        matrix, kind, conv_shape = self._matrix_view(grad_sample)
        matrix = matrix.to(self._working_dtype(matrix.dtype))
        batch, rows, cols = matrix.shape
        right = rows >= cols
        size = cols if right else rows
        rank = self._retained_count(size)
        dct = self._dct_matrix(size=size, device=matrix.device, dtype=matrix.dtype)
        basis = dct.t()
        if self.fourier_clipping_config.mode == "fixed_lowpass":
            indices = self._fixed_indices(size=size, rank=rank, device=matrix.device)
        else:
            full_coeffs = matrix @ basis if right else torch.matmul(basis.t(), matrix)
            indices = self._topk_indices(full_coeffs, dim=2 if right else 1, rank=rank)
        q = basis[:, indices]
        encoded = matrix @ q if right else torch.matmul(q.t(), matrix)
        entry = _PlanEntry(
            parameter=p,
            original_shape=p.shape,
            encoded_shape=torch.Size(encoded.shape[1:]),
            kind="matrix_right" if right else "matrix_left",
            working_dtype=matrix.dtype,
            original_dtype=p.dtype,
            indices=indices.detach().clone(),
            matrix_shape=(int(rows), int(cols)),
            conv_shape=conv_shape,
        )
        return entry, encoded.reshape(batch, entry.encoded_numel)

    def _blockwise_plan_and_encode(self, p: nn.Parameter, grad_sample: torch.Tensor) -> tuple[_PlanEntry, torch.Tensor]:
        cfg = self.fourier_clipping_config
        original_dtype = p.dtype
        working_dtype = self._working_dtype(grad_sample.dtype)
        batch = grad_sample.shape[0]
        flat = grad_sample.reshape(batch, int(p.numel())).to(working_dtype)
        pad = (-int(flat.shape[1])) % int(cfg.block_size)
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        num_blocks = int(flat.shape[1]) // int(cfg.block_size)
        blocks = flat.reshape(batch, num_blocks, int(cfg.block_size))
        dct = self._dct_matrix(
            size=int(cfg.block_size),
            device=blocks.device,
            dtype=blocks.dtype,
        )
        coeffs = blocks @ dct.t()
        rank = self._retained_count(int(cfg.block_size))
        if cfg.mode == "fixed_lowpass":
            indices = self._fixed_indices(
                size=int(cfg.block_size),
                rank=rank,
                device=blocks.device,
            ).unsqueeze(0).expand(blocks.shape[1], rank)
        else:
            scores = coeffs.detach().abs().sum(dim=0)
            indices = torch.topk(scores, k=rank, dim=1, largest=True, sorted=False).indices.sort(dim=1).values
        gathered = coeffs.gather(2, indices.unsqueeze(0).expand(batch, -1, -1))
        entry = _PlanEntry(
            parameter=p,
            original_shape=p.shape,
            encoded_shape=torch.Size(gathered.shape[1:]),
            kind="blockwise",
            working_dtype=working_dtype,
            original_dtype=original_dtype,
            indices=indices.detach().clone(),
            pad=int(pad),
            block_size=int(cfg.block_size),
            num_blocks=int(blocks.shape[1]),
        )
        return entry, gathered.reshape(batch, entry.encoded_numel)

    def _build_entry_and_encode(self, p: nn.Parameter, grad_sample: torch.Tensor) -> tuple[_PlanEntry, torch.Tensor]:
        cfg = self.fourier_clipping_config
        if cfg.layout == "layer_matrix_columns":
            try:
                return self._build_matrix_entry(p, grad_sample)
            except ValueError:
                self._last_fourier_fallbacks.append("parameter_blockwise")
        return self._blockwise_plan_and_encode(p, grad_sample)

    def _same_plan_shape(self, plan: Iterable[_PlanEntry]) -> bool:
        old = self._encoded_plan
        new = list(plan)
        if len(old) != len(new):
            return False
        for a, b in zip(old, new):
            if a.parameter is not b.parameter:
                return False
            if a.kind != b.kind or a.encoded_shape != b.encoded_shape:
                return False
            if not torch.equal(a.indices.to(b.indices.device), b.indices):
                return False
        return True

    def clip_and_accumulate(self):
        grad_samples = self.grad_samples
        if not grad_samples:
            return
        entries: list[_PlanEntry] = []
        encoded_chunks: list[torch.Tensor] = []
        norm_terms: list[torch.Tensor] = []
        self._last_fourier_fallbacks = []
        target_device = grad_samples[0].device
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p).to(p.dtype)
            entry, encoded = self._build_entry_and_encode(p, grad_sample)
            encoded = encoded.to(target_device)
            entries.append(entry)
            encoded_chunks.append(encoded)
            norm_terms.append(encoded.square().sum(dim=1))

        per_sample_norms = torch.stack(norm_terms, dim=1).sum(dim=1).sqrt()
        clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)
        clipped_chunks = []
        for encoded in encoded_chunks:
            factor = clip_factor.to(encoded.device).to(encoded.dtype)
            clipped = torch.einsum("i,ij->j", factor, encoded)
            if self.normalize_clipping:
                clipped = clipped / self.max_grad_norm
            clipped_chunks.append(clipped)
        encoded_sum = torch.cat([chunk.reshape(-1) for chunk in clipped_chunks], dim=0)

        if self._encoded_param is not None and self._encoded_param.summed_grad is not None:
            if self.fourier_clipping_config.mode == "adaptive_topk_leaky":
                raise ValueError(
                    "adaptive_topk_leaky Fourier clipping does not support virtual-batch accumulation"
                )
            if not self._same_plan_shape(entries):
                raise ValueError("Fourier clipping encoded plan changed during accumulation")
            self._encoded_param.summed_grad = self._encoded_param.summed_grad + encoded_sum
        else:
            self._encoded_plan = entries
            self._encoded_param = nn.Parameter(torch.zeros_like(encoded_sum), requires_grad=False)
            self._encoded_param.summed_grad = encoded_sum
        self.fourier_clipping_metadata = self._build_fourier_metadata()

        for p in self.params:
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                _mark_as_processed(p.grad_sample)

    def _decode_blockwise(self, entry: _PlanEntry, encoded: torch.Tensor) -> torch.Tensor:
        coeffs = torch.zeros(
            (entry.num_blocks, entry.block_size),
            device=encoded.device,
            dtype=entry.working_dtype,
        )
        retained = encoded.reshape(entry.encoded_shape).to(entry.working_dtype)
        coeffs.scatter_(1, entry.indices.to(encoded.device), retained)
        dct = self._dct_matrix(
            size=entry.block_size,
            device=encoded.device,
            dtype=entry.working_dtype,
        )
        flat = (coeffs @ dct).reshape(-1)
        if entry.pad:
            flat = flat[:-entry.pad]
        return flat.reshape(entry.original_shape).to(entry.original_dtype)

    def _decode_matrix(self, entry: _PlanEntry, encoded: torch.Tensor) -> torch.Tensor:
        assert entry.matrix_shape is not None
        rows, cols = entry.matrix_shape
        size = cols if entry.kind == "matrix_right" else rows
        dct = self._dct_matrix(size=size, device=encoded.device, dtype=entry.working_dtype)
        q = dct.t()[:, entry.indices.to(encoded.device)]
        retained = encoded.reshape(entry.encoded_shape).to(entry.working_dtype)
        matrix = retained @ q.t() if entry.kind == "matrix_right" else q @ retained
        if entry.conv_shape is not None:
            matrix = matrix.reshape(entry.conv_shape)
        return matrix.reshape(entry.original_shape).to(entry.original_dtype)

    def _decode_encoded_grad(self) -> None:
        if self._encoded_param is None or self._encoded_param.grad is None:
            raise ValueError("Fourier encoded noiser did not produce an encoded grad")
        encoded_grad = self._encoded_param.grad.reshape(-1)
        offset = 0
        for entry in self._encoded_plan:
            next_offset = offset + entry.encoded_numel
            chunk = encoded_grad[offset:next_offset]
            if entry.kind == "blockwise":
                decoded = self._decode_blockwise(entry, chunk)
            else:
                decoded = self._decode_matrix(entry, chunk)
            entry.parameter.grad = decoded.view_as(entry.parameter)
            offset = next_offset
        if offset != int(encoded_grad.numel()):
            raise ValueError("unused Fourier encoded gradient tail during decode")

    def add_noise(self):
        if self._encoded_param is None:
            return
        self.noise_mechanism.add_noise(self._encoded_view)
        self._decode_encoded_grad()

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)
        if not self._is_last_step_skipped:
            if self._encoded_param is not None:
                self._encoded_param.summed_grad = None
                self._encoded_param.grad = None
            self._encoded_plan = []
            self._last_fourier_fallbacks = []
            self.fourier_clipping_metadata = self._build_fourier_metadata()

    def state_dict(self):
        state = dict(super().state_dict())
        state["_dp_fourier_clipping_config"] = self.fourier_clipping_config.state_dict()
        state["_dp_fourier_clipping_metadata"] = self._build_fourier_metadata()
        return state

    def load_state_dict(self, state_dict) -> None:
        cfg_state = state_dict.get("_dp_fourier_clipping_config")
        if cfg_state is not None:
            restored = FourierClippingConfig.from_state_dict(cfg_state)
            if restored != self.fourier_clipping_config:
                raise ValueError("cannot load Fourier checkpoint with mismatched clipping config")
        super().load_state_dict(state_dict)


class DistributedFourierDPOptimizer(FourierDPOptimizer):
    """DDP variant that adds encoded-space noise only on rank 0."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        if self._encoded_param is None:
            return
        if self.rank == 0:
            if isinstance(
                self.noise_mechanism,
                (BufferedToeplitzNoiseMechanism, CorrelatedNoiseMechanism),
            ):
                original_z_std = self.noise_mechanism.z_std
                self.noise_mechanism.z_std = original_z_std * float(self.world_size)
                try:
                    self.noise_mechanism.add_noise(self._encoded_view)
                finally:
                    self.noise_mechanism.z_std = original_z_std
            else:
                self.noise_mechanism.add_noise(self._encoded_view)
        else:
            self._encoded_param.grad = self._encoded_param.summed_grad.view_as(self._encoded_param)
        self._decode_encoded_grad()

    def reduce_gradients(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        return None

    def state_dict(self):
        state = dict(super().state_dict())
        state["_dp_distributed_saved_rank"] = int(self.rank)
        state["_dp_distributed_world_size"] = int(self.world_size)
        return state

    def load_state_dict(self, state_dict) -> None:
        saved_rank = state_dict.get("_dp_distributed_saved_rank")
        if saved_rank is not None and int(saved_rank) != 0:
            if isinstance(self.noise_mechanism, BufferedToeplitzNoiseMechanism):
                raise ValueError("distributed BLT checkpoint must be saved on rank 0")
            if isinstance(self.noise_mechanism, CorrelatedNoiseMechanism):
                raise ValueError("distributed correlated-noise checkpoint must be saved on rank 0")
        super().load_state_dict(state_dict)
