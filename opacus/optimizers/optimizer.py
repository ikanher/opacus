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

import logging
from collections import defaultdict
from typing import Callable, List, Optional

import torch
from opacus.noise_mechanisms import (
    BufferedToeplitzNoiseMechanism,
    CorrelatedNoiseMechanism,
    GaussianNoiseMechanism,
    InverseBandNoiseMechanism,
    NoiseMechanism,
)
from opacus.noise_mechanisms._utils import (
    check_processed_flag as _check_processed_flag,
    generate_noise as _generate_noise,
    mark_as_processed as _mark_as_processed,
)
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer


logger = logging.getLogger(__name__)
logger.disabled = True


class DPOptimizer(Optimizer):
    """
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add Gaussian noise.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have the ``grad_sample`` attribute.

    On a high level ``DPOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
    2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
    3) Aggregate clipped per sample gradients into ``p.grad``
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    5) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = DPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
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
        noise_mechanism: Optional[NoiseMechanism] = None,
        **kwargs,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
            normalize_clipping: Decouples the learning rate and max_grad_norm by normalizing the
                clipped gradients by 1/C as described in the paper "Unlocking High-Accuracy
                Differentially Private Image Classification through Scale" by De et al. (2022)
                - https://arxiv.org/pdf/2204.13650.pdf
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean"
            )

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode
        self.normalize_clipping = normalize_clipping
        self.noise_mechanism = noise_mechanism or GaussianNoiseMechanism()
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        for p in self.params:
            p.summed_grad = None

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.

        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.

        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated

        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute

        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list

        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per sample gradient not found. Are you using GradSampleModule?"
            )
        if p.grad_sample is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        return ret

    def signal_skip_step(self, do_skip=True):
        """
        Signals the optimizer to skip an optimization step and only perform clipping and
        per sample gradient accumulation.

        On every call of ``.step()`` optimizer will check the queue of skipped step
        signals. If non-empty and the latest flag is ``True``, optimizer will call
        ``self.clip_and_accumulate``, but won't proceed to adding noise and performing
        the actual optimization step.
        It also affects the behaviour of ``zero_grad()``. If the last step was skipped,
        optimizer will clear per sample gradients accumulated by
        ``self.clip_and_accumulate`` (``p.grad_sample``), but won't touch aggregated
        clipped gradients (``p.summed_grad``)

        Used by :class:`~opacus.utils.batch_memory_manager.BatchMemoryManager` to
        simulate large virtual batches with limited memory footprint.

        Args:
            do_skip: flag if next step should be skipped
        """
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self, pop_next=True):
        """
        Checks if next step should be skipped by the optimizer.
        This is for large Poisson batches that get split into smaller physical batches
        to fit on the device. Batches that do not correspond to the end of a Poisson
        batch or thus `skipped` as their gradient gets accumulated for one big step.
        """
        if self._step_skip_queue:
            if pop_next:
                return self._step_skip_queue.pop(0)
            else:
                return self._step_skip_queue[0]
        else:
            return False

    @property
    def params(self) -> List[nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self)

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        """
        Returns a flat list of per sample gradient tensors (one per parameter)
        """
        ret = []
        for p in self.params:
            ret.append(self._get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        """
        Returns number of batches currently accumulated and not yet processed.

        In other words ``accumulated_iterations`` tracks the number of forward/backward
        passed done in between two optimizer steps. The value would typically be 1,
        but there are possible exceptions.

        Used by privacy accountants to calculate real sampling rate.
        """
        vals = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )
            if isinstance(p.grad_sample, torch.Tensor):
                vals.append(1)
            elif isinstance(p.grad_sample, list):
                vals.append(len(p.grad_sample))
            else:
                raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        if len(set(vals)) > 1:
            raise ValueError(
                "Number of accumulated steps is inconsistent across parameters"
            )
        return vals[0]

    @property
    def param_groups(self) -> List[dict]:
        """
        Returns a list containing a dictionary of all parameters managed by the optimizer.
        """
        return self.original_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups: List[dict]):
        """
        Updates the param_groups of the optimizer.
        """
        self.original_optimizer.param_groups = param_groups

    @property
    def state(self) -> defaultdict:
        """
        Returns a dictionary holding current optimization state.
        """
        return self.original_optimizer.state

    @state.setter
    def state(self, state: defaultdict):
        """
        Updates the state of the optimizer.
        """
        self.original_optimizer.state = state

    @property
    def defaults(self) -> dict:
        """
        Returns a dictionary containing default values for optimization.
        """
        return self.original_optimizer.defaults

    @defaults.setter
    def defaults(self, defaults: dict):
        """
        Updates the defaults of the optimizer.
        """
        self.original_optimizer.defaults = defaults

    def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
        """
        Attaches a hook to be executed after gradient clipping/noising, but before the
        actual optimization step.

        Most commonly used for privacy accounting.

        Args:
            fn: hook function. Expected signature: ``foo(optim: DPOptimizer)``
        """
        self.step_hook = fn

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]

            if per_param_norms:
                target_device = per_param_norms[0].device
                per_param_norms = [norm.to(target_device) for norm in per_param_norms]

            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            # gradients should match the dtype of the optimizer parameters
            # for mixed precision, optimizer parameters are usually in FP32
            # lower precision grads will be cast up to FP32
            grad_sample = grad_sample.to(p.dtype)
            clip_factor_on_device = per_sample_clip_factor.to(grad_sample.device).to(
                p.dtype
            )
            grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

            if self.normalize_clipping:
                grad /= self.max_grad_norm

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        self.noise_mechanism.add_noise(self)

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        # The corner case when the optimizer has no trainable parameters.
        # Essentially the DPOptimizer act as a normal optimizer
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return True
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False
        self.add_noise()
        self.scale_grad()
        if self.step_hook:
            self.step_hook(self)
        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return None

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        state = dict(self.original_optimizer.state_dict())
        state["_dp_noise_mechanism_name"] = type(self.noise_mechanism).__name__
        state["_dp_noise_mechanism_state"] = dict(self.noise_mechanism.state_dict())
        if self.generator is not None and hasattr(self.generator, "get_state"):
            state["_dp_noise_generator_state"] = self.generator.get_state()
        return state

    def load_state_dict(self, state_dict) -> None:
        mechanism_state = state_dict.get("_dp_noise_mechanism_state")
        generator_state = state_dict.get("_dp_noise_generator_state")
        optimizer_state = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("_dp_noise_mechanism_")
            and k != "_dp_noise_generator_state"
            and not k.startswith("_dp_distributed_")
            and not k.startswith("_dp_fourier_")
        }
        self.original_optimizer.load_state_dict(optimizer_state)
        if mechanism_state is None and type(self.noise_mechanism) in (
            BufferedToeplitzNoiseMechanism,
            CorrelatedNoiseMechanism,
            InverseBandNoiseMechanism,
        ):
            missing_kind = (
                "blt"
                if isinstance(self.noise_mechanism, BufferedToeplitzNoiseMechanism)
                else
                "inverse-band"
                if isinstance(self.noise_mechanism, InverseBandNoiseMechanism)
                else "correlated"
            )
            raise ValueError(
                f"missing {missing_kind} noise mechanism state in optimizer checkpoint"
            )
        if mechanism_state is not None:
            self.noise_mechanism.load_state_dict(mechanism_state)
        if (
            generator_state is not None
            and self.generator is not None
            and hasattr(self.generator, "set_state")
        ):
            self.generator.set_state(generator_state)
