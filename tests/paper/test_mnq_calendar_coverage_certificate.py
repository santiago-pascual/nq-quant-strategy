from datetime import date

import pandas as pd

from scripts.build_mnq_calendar_coverage_certificate import (
    DEFAULT_DATA,
    _load_day_conditions,
    _utc_date_is_fully_scheduled_closed,
    classify_gap,
    classify_minute,
)
from src.paper.cme_calendar import CMECalendarSnapshot


def _reviewed_snapshot():
    return CMECalendarSnapshot(
        version="test-labor-day-v1",
        source="https://www.cmegroup.com/trading-hours.html",
        coverage_start=date(2026, 9, 7), coverage_end=date(2026, 9, 8),
        exceptions={"2026-09-07": {"session_type": "early_close",
                                   "rth_end": "13:00", "globex_close": "13:00"}},
    )


def test_gap_classifier_respects_reviewed_labor_day_and_uncovered_dates():
    snapshot = _reviewed_snapshot()
    snapshots = [snapshot]
    assert classify_minute(pd.Timestamp("2026-09-07T13:30:00Z"), snapshots) == "open_rth_minute_absent_trade_or_capture_unresolved"
    assert classify_minute(pd.Timestamp("2026-09-07T18:00:00Z"), snapshots) == "scheduled_holiday_or_early_close"
    assert classify_minute(pd.Timestamp("2026-09-08T21:30:00Z"), snapshots) == "scheduled_holiday_or_early_close"
    assert classify_minute(pd.Timestamp("2026-09-09T13:30:00Z"), snapshots) == "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"


def test_gap_classification_never_calls_absent_ohlcv_a_proven_missing_trade_bar():
    snapshot = _reviewed_snapshot()
    gap = classify_gap(pd.Timestamp("2026-09-06T21:59:00Z"),
                       pd.Timestamp("2026-09-06T22:02:00Z"), [snapshot])
    assert gap["classification"] == "unresolved_source_quality_or_calendar"
    assert gap["missing_minutes"] == 2
    assert gap["minute_class_counts"]["source_condition_unknown_gap_unresolved"] == 2


def test_available_day_sparse_trade_ohlcv_does_not_require_minute_grid_rows():
    gap = classify_gap(
        pd.Timestamp("2026-09-09T13:29:00Z"),
        pd.Timestamp("2026-09-09T13:32:00Z"), [],
        {"2026-09-09": "available"},
    )
    assert gap["missing_minutes"] == 2
    assert gap["classification"] == "sparse_trade_derived_ohlcv_absence"
    assert gap["minute_class_counts"] == {
        "trade_derived_ohlcv_minute_absence_not_required_by_grid_contract": 2
    }


def test_degraded_source_day_remains_unresolved_despite_sparse_bar_semantics():
    gap = classify_gap(
        pd.Timestamp("2025-11-28T13:29:00Z"),
        pd.Timestamp("2025-11-28T13:32:00Z"), [],
        {"2025-11-28": "degraded"},
    )
    assert gap["missing_minutes"] == 2
    assert gap["classification"] == "unresolved_source_quality_or_calendar"
    assert gap["minute_class_counts"] == {"degraded_source_day_gap_unresolved": 2}


def test_local_vendor_condition_metadata_is_manifest_hash_verified():
    conditions, provenance = _load_day_conditions(DEFAULT_DATA)
    assert provenance["status"] == "verified_hash"
    assert conditions["2020-02-27"] == "degraded"
    assert conditions["2024-09-18"] == "degraded"
    assert conditions["2020-02-26"] == "available"


def test_recurring_weekend_and_maintenance_closures_are_classified_without_holiday_inference():
    snapshots = []
    assert classify_minute(pd.Timestamp("2026-09-12T18:00:00Z"), snapshots) == "scheduled_recurring_weekend_closure"
    assert classify_minute(pd.Timestamp("2026-09-09T21:30:00Z"), snapshots) == "scheduled_recurring_daily_maintenance"
    # Sunday 17:00 Chicago is the weekly reopen; without a date-specific
    # schedule, this open/holiday-sensitive minute is not certified.
    assert classify_minute(pd.Timestamp("2026-09-13T22:00:00Z"), snapshots) == "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"


def test_recurring_equity_index_pause_applies_only_before_2021_rule_change():
    snapshots = []
    assert classify_minute(pd.Timestamp("2020-06-10T20:20:00Z"), snapshots) == "scheduled_recurring_equity_index_pause"
    assert classify_minute(pd.Timestamp("2022-06-08T20:20:00Z"), snapshots) == "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"


def test_recurring_maintenance_timezone_conversion_handles_dst_and_standard_time():
    snapshots = []
    # 16:30 local Chicago in summer and winter respectively.
    assert classify_minute(pd.Timestamp("2026-07-08T21:30:00Z"), snapshots) == "scheduled_recurring_daily_maintenance"
    assert classify_minute(pd.Timestamp("2026-01-07T22:30:00Z"), snapshots) == "scheduled_recurring_daily_maintenance"


def test_daily_maintenance_ends_at_1700_chicago_and_friday_enters_weekend():
    snapshots = []
    # Summer 2026: 16:30 CDT is maintenance; 17:00 CDT is the reopen.
    assert classify_minute(pd.Timestamp("2026-09-09T21:30:00Z"), snapshots) == "scheduled_recurring_daily_maintenance"
    assert classify_minute(pd.Timestamp("2026-09-09T22:00:00Z"), snapshots) == "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"
    # Friday 16:30 CDT is already the weekly closure, not daily maintenance.
    assert classify_minute(pd.Timestamp("2026-09-11T21:30:00Z"), snapshots) == "scheduled_recurring_weekend_closure"
    # The pre-2021 schedule ended its session at 16:15 CT.
    assert classify_minute(pd.Timestamp("2020-06-24T21:30:00Z"), snapshots) == "scheduled_recurring_daily_maintenance"
    assert classify_minute(pd.Timestamp("2020-06-24T22:00:00Z"), snapshots) == "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"
    assert classify_minute(pd.Timestamp("2020-06-26T21:30:00Z"), snapshots) == "scheduled_recurring_weekend_closure"


def test_vectorized_gap_classification_uses_the_daily_maintenance_end_boundary():
    gap = classify_gap(
        pd.Timestamp("2026-09-09T21:59:00Z"),
        pd.Timestamp("2026-09-09T22:02:00Z"), [],
        {"2026-09-09": "available"},
    )
    assert gap["minute_class_counts"] == {
        "trade_derived_ohlcv_minute_absence_not_required_by_grid_contract": 2
    }


def test_zero_row_degraded_utc_date_is_exempt_only_if_every_minute_is_closed():
    assert _utc_date_is_fully_scheduled_closed("2026-01-31", [])
    assert _utc_date_is_fully_scheduled_closed("2026-03-21", [])
    assert not _utc_date_is_fully_scheduled_closed("2026-03-16", [])


def test_uncovered_holiday_or_open_period_remains_explicitly_unverified():
    assert classify_minute(pd.Timestamp("2026-11-26T15:00:00Z"), []) == "unverified_open_or_date_specific_exception_or_no_trade_capture_unknown"
