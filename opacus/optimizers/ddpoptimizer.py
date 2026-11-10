# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Callable, Optional

import torch
from opacus.noise_mechanisms.blt import BufferedToeplitzNoiseMechanism
from torch.optim import Optimizer

from .optimizer import CorrelatedNoiseMechanism, DPOptimizer, InverseBandNoiseMechanism


class DistributedDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` compatible with
    distributed data processing
    """

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
        **kwargs,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
            **kwargs,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        # Noise only gets added to the first worker
        if self.rank == 0:
            if isinstance(
                self.noise_mechanism,
                (BufferedToeplitzNoiseMechanism, CorrelatedNoiseMechanism),
            ):
                original_z_std = self.noise_mechanism.z_std
                self.noise_mechanism.z_std = original_z_std * float(self.world_size)
                try:
                    super().add_noise()
                finally:
                    self.noise_mechanism.z_std = original_z_std
            else:
                super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

    def state_dict(self):
        """Add saving rank/world size; correlated-noise resumes require rank 0.

        Save on rank 0 and distribute that payload to all restoring workers.
        """
        state = dict(super().state_dict())
        state["_dp_distributed_saved_rank"] = int(self.rank)
        state["_dp_distributed_world_size"] = int(self.world_size)

        return state

    def load_state_dict(self, state_dict) -> None:
        """Restore a rank-zero checkpoint without duplicating inverse history.

        Inverse noisers reject a changed recorded world size: rank-zero noise
        scales include world-size compensation before gradient reduction.
        Non-owners load optimizer state only and clear their noiser. This method
        performs no broadcast; the caller supplies the owner payload on all ranks.
        """
        saved_rank = state_dict.get("_dp_distributed_saved_rank")

        if saved_rank is not None and int(saved_rank) != 0:
            if isinstance(self.noise_mechanism, BufferedToeplitzNoiseMechanism):
                raise ValueError("distributed BLT checkpoint must be saved on rank 0")
            if isinstance(self.noise_mechanism, CorrelatedNoiseMechanism):
                raise ValueError(
                    "distributed correlated-noise checkpoint must be saved on rank 0"
                )

        if isinstance(self.noise_mechanism, InverseBandNoiseMechanism):
            saved_world = state_dict.get("_dp_distributed_world_size")
            if saved_world is not None and saved_world != self.world_size:
                raise ValueError("incompatible distributed checkpoint world size")

            if self.rank != 0:
                # Validate the owner payload but never install its history/RNG
                # on a worker that will not generate correlated noise.
                self.noise_mechanism.validate_state_dict(
                    state_dict["_dp_noise_mechanism_state"]
                )
                plain = {
                    k: v for k, v in state_dict.items() if not k.startswith("_dp_")
                }
                self.original_optimizer.load_state_dict(plain)
                self.noise_mechanism.reset_state()
                return

        super().load_state_dict(state_dict)

    def reduce_gradients(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        else:
            return None
