from opacus.bnb_defaults import resolve_bnb_calibration_kwargs


def test_opacus_strict_profile_defaults() -> None:
    cfg = resolve_bnb_calibration_kwargs(profile="opacus_strict")
    assert cfg["bnb_num_samples"] == 100_000
    assert cfg["bnb_max_iterations"] == 200
    assert cfg["bnb_tolerance"] == 1e-4
    assert cfg["bnb_evr_num_checks"] == 3
    assert cfg["bnb_confidence_alpha"] == 1e-6
    assert cfg["bnb_verify_both_directions"] is True


def test_dpdl_fast_profile_defaults() -> None:
    cfg = resolve_bnb_calibration_kwargs(profile="dpdl_fast")
    assert cfg["bnb_num_samples"] == 2_000
    assert cfg["bnb_max_iterations"] == 64
    assert cfg["bnb_tolerance"] == 1e-3
    assert cfg["bnb_evr_num_checks"] == 1
    assert cfg["bnb_confidence_alpha"] == 1e-3
    assert cfg["bnb_verify_both_directions"] is False


def test_profile_overrides_take_precedence() -> None:
    cfg = resolve_bnb_calibration_kwargs(
        profile="dpdl_fast",
        overrides={
            "bnb_num_samples": 1234,
            "bnb_require_evr_pass": False,
        },
    )
    assert cfg["bnb_num_samples"] == 1234
    assert cfg["bnb_require_evr_pass"] is False
