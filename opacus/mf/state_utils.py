from __future__ import annotations

from opacus.accountants.bnb_inputs import attach_accountant_coeff_surface


def attach_mf_accountant_coeff_surface(*args, **kwargs):
    return attach_accountant_coeff_surface(*args, **kwargs)


__all__ = ["attach_mf_accountant_coeff_surface"]
