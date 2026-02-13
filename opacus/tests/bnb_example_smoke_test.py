#!/usr/bin/env python3

from __future__ import annotations

import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path


def test_bnb_calibration_status_example_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    example_path = repo_root / "examples" / "bnb_calibration_status.py"
    assert example_path.exists()

    out = io.StringIO()
    with redirect_stdout(out):
        runpy.run_path(str(example_path), run_name="__main__")

    text = out.getvalue()
    assert "BNB calibration v2" in text
    assert "version" in text
