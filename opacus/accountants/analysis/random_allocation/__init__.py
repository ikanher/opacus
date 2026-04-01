"""Random-allocation analysis package for Opacus.

This package owns the production random-allocation analysis surface:

- `accountant.py` implements repeated `k`-out-of-`t` random-allocation PLD
  discretization, composition, and `ε(δ)` queries.
- `exact_laws.py` builds exact Gaussian neighboring-pair inputs used by the
  repeated accountant and the fixed-bin bridge.
- `initial_package.py` assembles deterministic initial packages and ambient
  quantitative-window packages above exact finite-mixture laws.
- `fixed_bin.py` builds the fixed-bin balls-in-bins bridge and calibrates the
  corresponding deterministic random-allocation route.

The package root exports the supported repeated-accountant surface and the
exact-law builders. Bridge- and initial-package-specific functionality lives in
explicit submodules to avoid import-time cycles with the accountant registry.

Paper short names used in this package:

- `RA`  — "Efficient Random Allocation for DP Accounting" (Feldman and
  Shenfeld, 2025)
- `PLD` — "Random Allocation PLD Accounting" (Feldman and Shenfeld, 2025)
- `BSR` — "Banded Square Root Mechanism" (Kalinin and Lampert, 2024)
- `MF`  — "MC-less Accounting for Matrix Factorization Mechanisms" (local
  project note, `latex/mc_less_accounting_full.tex`)
"""

from .accountant import (
    RandomAllocationAccountantInputs,
    RandomAllocationGaussianRuntimeConfig,
    build_gaussian_random_allocation_realization,
    estimate_epsilon_random_allocation,
    estimate_epsilon_range_random_allocation,
    resolve_random_allocation_accountant_inputs,
    resolve_random_allocation_gaussian_runtime_config,
)
from .exact_laws import (
    ExactLawMetadata,
    FiniteGaussianMixtureNeighboringPair,
    RealizableGaussianOneStepNeighboringPair,
    build_poisson_gaussian_mixture_neighboring_pair,
    build_product_gaussian_mixture_neighboring_pair,
    build_realizable_gaussian_one_step_neighboring_pair,
)


__all__ = [
    "ExactLawMetadata",
    "FiniteGaussianMixtureNeighboringPair",
    "RandomAllocationAccountantInputs",
    "RandomAllocationGaussianRuntimeConfig",
    "RealizableGaussianOneStepNeighboringPair",
    "build_gaussian_random_allocation_realization",
    "build_poisson_gaussian_mixture_neighboring_pair",
    "build_product_gaussian_mixture_neighboring_pair",
    "build_realizable_gaussian_one_step_neighboring_pair",
    "estimate_epsilon_random_allocation",
    "estimate_epsilon_range_random_allocation",
    "resolve_random_allocation_accountant_inputs",
    "resolve_random_allocation_gaussian_runtime_config",
]
