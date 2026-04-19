from __future__ import annotations

"""
Optimizer-shape helpers shared by MF provider/accountant surfaces.
"""

import math

from torch import optim


def resolve_uniform_sgd_workload_from_optimizer(
    *,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    """
    Extract a uniform SGD workload `(momentum, weight_decay)` from an optimizer.

    MF analytical auto-coefficient generation currently assumes every parameter
    group shares the same momentum and weight decay. This helper enforces that
    contract before the family/accountant layers derive coefficients from the
    optimizer configuration.
    """
    momenta: list[float] = []
    decays: list[float] = []

    for group in optimizer.param_groups:
        momenta.append(float(group.get("momentum", 0.0)))
        decays.append(float(group.get("weight_decay", 0.0)))

    if len(momenta) == 0:
        raise ValueError("optimizer must contain at least one parameter group")

    m0 = float(momenta[0])
    d0 = float(decays[0])
    for m in momenta[1:]:
        if not math.isclose(float(m), m0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "bsr analytical auto-coeff generation requires uniform optimizer momentum "
                "across parameter groups"
            )

    for d in decays[1:]:
        if not math.isclose(float(d), d0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "bsr analytical auto-coeff generation requires uniform optimizer weight_decay "
                "across parameter groups"
            )

    return m0, d0
