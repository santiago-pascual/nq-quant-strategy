"""Small, opt-in validation harness for raw reconstruction parity."""

from __future__ import annotations

import os

import pytest

from src.research.raw_reconstruction import compare_hmm_window, reconstruct_mr_hmm_window


@pytest.mark.skipif(
    os.getenv("RUN_RAW_PARITY") != "1",
    reason="requires raw Databento bars and a one-window HMM fit",
)
def test_hmm_window_one_matches_frozen_oracle() -> None:
    report = compare_hmm_window(reconstruct_mr_hmm_window(window=1))
    assert report.passed, report
