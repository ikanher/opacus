from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

from ._utils import check_processed_flag, generate_noise, mark_as_processed


class NoiseMechanismOptimizer(Protocol):
    """Minimal optimizer surface required by runtime noise mechanisms."""

    params: Sequence[Any]
    loss_reduction: str
    expected_batch_size: Optional[int]
    accumulated_iterations: int
    generator: Any
    secure_mode: bool
    noise_multiplier: float
    normalize_clipping: bool
    max_grad_norm: float


class NoiseMechanism:
    """
    Strategy interface for runtime gradient-noise addition.
    """

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        del state_dict


class GaussianNoiseMechanism(NoiseMechanism):
    """
    Default iid Gaussian mechanism matching historical DPOptimizer behavior.
    """

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        max_grad_norm = 1 if optimizer.normalize_clipping else optimizer.max_grad_norm

        for p in optimizer.params:
            check_processed_flag(p.summed_grad)

            noise = generate_noise(
                std=optimizer.noise_multiplier * max_grad_norm,
                reference=p.summed_grad,
                generator=optimizer.generator,
                secure_mode=optimizer.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            mark_as_processed(p.summed_grad)
