from opacus.accountants.analysis.bnb import resolve_bnb_calibration_kwargs


def test_bnb_calibration_defaults_match_monte_carlo_reference() -> None:
    cfg = resolve_bnb_calibration_kwargs()
    assert cfg["bnb_num_samples"] == 500_000
    assert cfg["bnb_seed"] == 154
    assert cfg["bnb_max_iterations"] == 1000
    assert cfg["bnb_tolerance"] == 1e-7
    assert cfg["bnb_confidence_alpha"] == 1e-6
    assert cfg["bnb_chunk_size"] is None
    assert cfg["bnb_num_workers"] == 0


def test_overrides_take_precedence() -> None:
    cfg = resolve_bnb_calibration_kwargs(
        overrides={
            "bnb_num_samples": 1234,
            "bnb_require_evr_pass": False,
            "bnb_chunk_size": 100,
        },
    )
    assert cfg["bnb_num_samples"] == 1234
    assert cfg["bnb_require_evr_pass"] is False
    assert cfg["bnb_chunk_size"] == 100
