from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import uuid

import pytest

from src.paper.causal_bootstrap_cli import _coverage_summary, _windows_memory_counters, main, validate_inputs


@pytest.fixture
def bootstrap_cli_scratch():
    path = Path(__file__).resolve().parent / f".tmp_bootstrap_cli_{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_coverage_summary_separates_known_closures_from_uncertain_absence():
    certificate = {
        "gap_classification_counts": {
            "verified_scheduled_closure": 3,
            "uncertain_uncertified_calendar": 2,
        },
        "missing_minute_classification_counts": {
            "scheduled_recurring_daily_maintenance": 20,
            "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown": 7,
        },
        "input": {"rows": 100, "first_utc": "a", "last_utc": "b"},
    }
    summary = _coverage_summary(certificate)
    assert summary["verified_exchange_closure_gap_spans"] == 3
    assert summary["uncertain_gap_spans"] == 2
    assert summary["recurring_closure_minutes"] == 20
    assert summary["uncertain_absent_open_or_exception_minutes"] == 7


def test_coverage_summary_separates_sparse_ohlcv_from_degraded_source_intervals():
    summary = _coverage_summary({
        "gap_classification_counts": {
            "sparse_trade_derived_ohlcv_absence": 10,
            "unresolved_source_quality_or_calendar": 2,
        },
        "missing_minute_classification_counts": {
            "trade_derived_ohlcv_minute_absence_not_required_by_grid_contract": 42,
            "degraded_source_day_gap_unresolved": 7,
        },
        "source_quality_metadata": {
            "degraded_dates_in_observed_range": ["2020-02-27", "2020-06-30"],
        },
        "input": {"rows": 100, "first_utc": "2019-05-05T22:03:00+00:00",
                  "last_utc": "2026-10-08T13:02:00+00:00"},
    })
    assert summary["sparse_trade_ohlcv_minutes_not_required_by_minute_grid"] == 42
    assert summary["degraded_source_day_gap_minutes"] == 7
    assert summary["degraded_source_date_count"] == 2
    assert summary["uncertain_absent_open_or_exception_minutes"] == 7
    assert summary["demonstrated_missing_required_observations"] == 0
    assert "cannot distinguish no-trade" in summary["interpretation"]


def test_current_style_uncertain_certificate_is_valid_observations_but_not_activation(bootstrap_cli_scratch):
    cert = {
        "schema_version": 2,
        "input": {
            "first_utc": "2019-05-05T22:03:00+00:00",
            "last_utc": "2026-10-08T13:02:00+00:00",
            "rows": 2_619_604,
            "duplicate_timestamps": 0,
            "nonmonotonic_transitions": 0,
        },
        "data_files": [],
        "calendar_snapshots": [{"review_status": "REVIEWED_TEST", "identity": "abc", "source_url": "test"}],
        "readiness": "NOT_FULLY_CERTIFIED",
        "gap_count": 5_937,
        "gap_classification_counts": {
            "verified_scheduled_closure": 3_435,
            "uncertain_uncertified_calendar": 2_502,
        },
        "missing_minute_classification_counts": {
            "scheduled_recurring_daily_maintenance": 1_238_690,
            "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown": 47_526,
        },
    }
    path = bootstrap_cli_scratch / "coverage.json"
    path.write_text(json.dumps(cert), encoding="utf-8")
    report = validate_inputs(path, check_files=False)
    assert report["status"] == "VALID"
    assert report["causal_computation_possible_on_observed_bars"] is True
    assert report["activation_coverage_verified"] is False
    assert report["activation_ready"] is False


def test_start_refuses_incomplete_coverage_without_loading_history(monkeypatch, bootstrap_cli_scratch, capsys):
    monkeypatch.setattr("src.paper.causal_bootstrap_cli.validate_inputs", lambda *_args, **_kwargs: {
        "status": "VALID", "activation_ready": False, "blockers": ["uncertain coverage"]
    })
    monkeypatch.setattr("src.paper.causal_bootstrap_cli._load_raw", lambda: (_ for _ in ()).throw(
        AssertionError("must fail before loading full history")
    ))
    code = main(["start", "--confirm-full-bootstrap", "--checkpoint", str(bootstrap_cli_scratch / "seed.json")])
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["started"] is False
    assert "activation-certified" in report["reason"]


def test_demonstrated_required_missing_observation_blocks_while_uncertainty_stays_separate(bootstrap_cli_scratch):
    cert = {
        "schema_version": 2,
        "input": {"first_utc": "2019-05-05T22:03:00+00:00", "last_utc": "2026-10-08T13:02:00+00:00",
                  "rows": 4, "duplicate_timestamps": 0, "nonmonotonic_transitions": 0},
        "data_files": [],
        "calendar_snapshots": [{"review_status": "REVIEWED_TEST", "identity": "abc", "source_url": "test"}],
        "readiness": "FULLY_CERTIFIED",
        "demonstrated_missing_required_observations": 1,
        "gap_classification_counts": {"verified_scheduled_closure": 1},
        "missing_minute_classification_counts": {"unverified_open_or_date_specific_exception_or_no_trade_capture_unknown": 2},
    }
    path = bootstrap_cli_scratch / "gap.json"
    path.write_text(json.dumps(cert), encoding="utf-8")
    report = validate_inputs(path, check_files=False)
    assert report["status"] == "OBSERVED_VALID_WITH_DEMONSTRATED_REQUIRED_GAPS"
    assert report["causal_computation_possible_on_observed_bars"] is False
    assert report["activation_ready"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows process counters are platform-specific")
def test_windows_peak_memory_counters_are_available_and_ordered():
    counters = _windows_memory_counters()
    assert counters is not None
    assert counters["peak_working_set_bytes"] >= counters["working_set_bytes"] > 0
