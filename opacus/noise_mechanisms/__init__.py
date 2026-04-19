"""
Runtime noise-mechanism package.

This package owns the training-time noising strategies consumed by
`DPOptimizer`. It is the runtime layer only: the classes here mutate clipped
summed gradients in place, while accountant math, calibration search, and paper
RMSE surfaces live elsewhere.

The main public families are:
- `GaussianNoiseMechanism`: the standard iid Gaussian DP-SGD runtime.
- `CorrelatedNoiseMechanism`: streamed lower-triangular Toeplitz noising for
  factor-side MF families such as BSR (Kalinin and Lampert, 2024) and Scaling
  BandMF (McKenna, 2025).
- `InverseBandNoiseMechanism`: inverse-side streamed recurrence for BISR
  (Kalinin et al., 2026) and related inverse-band surfaces.
- `BufferedToeplitzNoiseMechanism`: buffered BLT runtime noiser for BLT
  Practice (McMahan et al., 2024) parameter pairs.

All classes in this package are implementation-contract surfaces. They realize
runtime recurrences such as `C u = z` or its inverse-side counterpart, but they
do not by themselves prove privacy bounds or accountant exactness.

Canonical short names used by this package:
- `BSR (Kalinin and Lampert, 2024)`
- `BISR (Kalinin et al., 2026)`
- `Scaling BandMF (McKenna, 2025)`
- `BLT Practice (McMahan et al., 2024)`
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
