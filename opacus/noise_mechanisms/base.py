"""
Base runtime noise-mechanism interfaces.

This module defines the minimal optimizer contract consumed by runtime noisers
and the two simplest public runtime mechanisms:
- the abstract `NoiseMechanism` strategy surface
- the historical iid Gaussian DP-SGD implementation

These are runtime-only interfaces. They describe how clipped summed gradients
are perturbed before the optimizer step; they do not state privacy bounds or
accountant-side exactness by themselves.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

from ._utils import check_processed_flag, generate_noise, mark_as_processed


class NoiseMechanismOptimizer(Protocol):
    """
    Minimal optimizer surface required by runtime noise mechanisms.

    Attributes:
        params: Parameters whose `summed_grad` buffers will be noised.
        loss_reduction: Reduction convention used when per-sample gradients were
            accumulated. Runtime mechanisms use this to decide whether a
            per-step noise scale must be rescaled by the effective batch size.
        expected_batch_size: Logical batch size used when `loss_reduction` is
            not `"sum"`.
        accumulated_iterations: Number of microbatch accumulations folded into
            the current logical step.
        generator: Optional random generator used for reproducible noise draws.
        secure_mode: Whether to use the hardened Gaussian-sampling path in
            `_utils.generate_noise`.
        noise_multiplier: Runtime Gaussian multiplier for the iid mechanism.
        normalize_clipping: Whether clipping was normalized to unit norm before
            noising.
        max_grad_norm: Per-step clipping norm when normalization is disabled.
    """

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

    Implementations consume an optimizer-shaped object, read its clipped
    `summed_grad` buffers, and write noised `grad` buffers in place. This is an
    implementation-contract surface rather than an accountant surface.
    """

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        """Populate `p.grad` for each parameter using the mechanism's runtime rule."""
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Any]:
        """Return runtime-only checkpoint state for the mechanism."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore runtime-only checkpoint state for the mechanism."""
        del state_dict


class GaussianNoiseMechanism(NoiseMechanism):
    """
    Default iid Gaussian runtime matching historical `DPOptimizer` behavior.

    This is the standard DP-SGD noiser: at each step it adds iid Gaussian noise
    with standard deviation

        `noise_multiplier * effective_clip_norm`

    to each clipped summed-gradient buffer. The mechanism is runtime-exact for
    the configured Gaussian release rule, but privacy interpretation still
    belongs to the selected accountant outside this package.
    """

    def add_noise(self, optimizer: NoiseMechanismOptimizer) -> None:
        """Add iid Gaussian noise to each clipped summed-gradient buffer."""
        # The iid Gaussian runtime still needs the effective clip norm on the
        # summed-gradient scale seen by the optimizer step.
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
