import pytest

from opacus.mechanism_contracts import resolve_accounting_mode_from_accountant


@pytest.mark.parametrize("accountant", ["prv", "rdp", "gdp", "standard_step_accountant"])
def test_standard_accountants_map_to_standard_step_accounting_mode(accountant: str) -> None:
    assert (
        resolve_accounting_mode_from_accountant(accountant)
        == "standard_step_accountant"
    )


@pytest.mark.parametrize("accountant", ["bandmf", "bandmf_accountant"])
def test_bandmf_accountant_aliases_map_to_bandmf_accounting_mode(accountant: str) -> None:
    assert resolve_accounting_mode_from_accountant(accountant) == "bandmf_accountant"


@pytest.mark.parametrize("accountant", ["bsr", "bsr_accountant"])
def test_bsr_accountant_aliases_map_to_bsr_accounting_mode(accountant: str) -> None:
    assert resolve_accounting_mode_from_accountant(accountant) == "bsr_accountant"


@pytest.mark.parametrize("accountant", ["bandinvmf"])
def test_bandinvmf_accountant_alias_maps_to_bsr_accounting_mode(accountant: str) -> None:
    assert resolve_accounting_mode_from_accountant(accountant) == "bsr_accountant"


@pytest.mark.parametrize("accountant", ["bnb", "bnb_accountant"])
def test_bnb_accountant_aliases_map_to_bnb_accounting_mode(accountant: str) -> None:
    assert resolve_accounting_mode_from_accountant(accountant) == "bnb_accountant"


@pytest.mark.parametrize("accountant", ["random_allocation", "random_allocation_accountant"])
def test_random_allocation_accountant_aliases_map_to_random_allocation_accounting_mode(accountant: str) -> None:
    assert resolve_accounting_mode_from_accountant(accountant) == "random_allocation_accountant"


def test_unknown_accountant_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported accountant"):
        resolve_accounting_mode_from_accountant("unknown_accountant")
