"""
Runtime noise-mechanism package.

This package owns training-time noising strategies used by `DPOptimizer`.
It is intentionally separate from accountant math and from optimizer-search
utilities so matrix-factorization runtimes like BLT do not live under
`optimizers/` by accident.
"""

from .base import GaussianNoiseMechanism, NoiseMechanism
from .blt import BufferedToeplitzNoiseMechanism
from .correlated import CorrelatedNoiseMechanism, InverseBandNoiseMechanism

__all__ = [
    "NoiseMechanism",
    "GaussianNoiseMechanism",
    "CorrelatedNoiseMechanism",
    "InverseBandNoiseMechanism",
    "BufferedToeplitzNoiseMechanism",
]
