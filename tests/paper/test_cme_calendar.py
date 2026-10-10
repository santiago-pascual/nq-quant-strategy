from datetime import date, datetime, time, timezone
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd
import pytest

from src.paper.cme_calendar import (
    CMECalendarSnapshot,
    CMETradingCalendar,
    CalendarUnavailable,
    CME_HOLIDAY_SOURCE,
)


def _review_validator_scratch() -> Path:
    """Use repository-local scratch because Windows pytest temp ACLs vary."""
    root = Path(__file__).resolve().parents[2] / "results" / "paper" / f"cme_review_validator_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


@pytest.fixture
def calendar():
    # Schedule fixture exercises the schema; production snapshots must be
    # populated from the CME source and versioned before they are activated.
    snapshot = CMECalendarSnapshot(
        version="fixture-cme-2024-v1",
        source=CME_HOLIDAY_SOURCE,
        coverage_start=date(2024, 3, 8),
        coverage_end=date(2024, 3, 31),
        exceptions={
            "2024-03-11": {
                "session_type": "early_close",
                "rth_end": "13:15",
                "globex_close": "13:15",
            },
            "2024-03-12": {"session_type": "closed"},
            "2024-03-13": {
                "session_type": "special",
                "rth_start": "09:30",
                "rth_end": "14:00",
                "globex_open": "18:00",
                "globex_close": "14:00",
            },
        },
    )
    return CMETradingCalendar(snapshot)


def test_regular_rth_and_final_bar_preserve_strategy_session(calendar):
    session = calendar.snapshot.session_for_rth_date(date(2024, 3, 8))
    assert session.session_type == "regular"
    assert session.rth_start.isoformat() == "2024-03-08T09:30:00-05:00"
    assert session.rth_end.isoformat() == "2024-03-08T16:00:00-05:00"
    assert session.final_rth_bar.isoformat() == "2024-03-08T15:59:00-05:00"
    assert calendar.is_final_rth_bar(pd.Timestamp("2024-03-08T20:59:00Z"))
    assert not calendar.is_final_rth_bar(pd.Timestamp("2024-03-08T21:00:00Z"))


def test_early_close_and_special_session_are_explicit(calendar):
    early = calendar.snapshot.session_for_rth_date(date(2024, 3, 11))
    assert early.session_type == "early_close"
    assert early.rth_end.isoformat() == "2024-03-11T13:15:00-04:00"
    assert early.final_rth_bar.isoformat() == "2024-03-11T13:14:00-04:00"
    assert calendar.is_final_rth_bar(pd.Timestamp("2024-03-11T17:14:00Z"))
    special = calendar.snapshot.session_for_rth_date(date(2024, 3, 13))
    assert special.session_type == "special"
    assert special.final_rth_bar.isoformat() == "2024-03-13T13:59:00-04:00"


def test_closed_holiday_weekend_and_uncovered_date(calendar):
    closed = calendar.snapshot.session_for_rth_date(date(2024, 3, 12))
    assert closed.session_type == "holiday_closed"
    assert closed.final_rth_bar is None
    assert calendar.snapshot.session_for_rth_date(date(2024, 3, 9)).session_type == "weekend"
    with pytest.raises(CalendarUnavailable, match="does not cover"):
        calendar.snapshot.session_for_rth_date(date(2024, 4, 1))


def test_globex_cross_midnight_dates_dst_weekend_and_maintenance(calendar):
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2024-03-10T22:00:00Z")) == date(2024, 3, 11)
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2024-03-12T22:00:00Z")) == date(2024, 3, 13)
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2024-03-08T22:00:00Z")) is None
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2024-03-09T20:00:00Z")) is None
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2024-03-11T21:30:00Z")) is None
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2024-03-14T20:20:00Z")) == date(2024, 3, 14)


def test_expected_globex_missing_minutes_respect_halt_and_maintenance(calendar):
    # 15:14 and 15:15 are both regular here; month-end pause is date-specific.
    missing = calendar.expected_missing_minutes(
        pd.Timestamp("2024-03-08T20:14:00Z"),
        pd.Timestamp("2024-03-08T20:17:00Z"),
    )
    assert missing == [pd.Timestamp("2024-03-08T20:15:00Z"), pd.Timestamp("2024-03-08T20:16:00Z")]


def test_equity_index_month_end_window_is_open_after_cme_pause_elimination():
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="fixture-month-end", source=CME_HOLIDAY_SOURCE,
        coverage_start=date(2024, 2, 1), coverage_end=date(2024, 2, 29),
    ))
    assert calendar.expected_globex_minute(pd.Timestamp("2024-02-28T21:20:00Z"))
    assert calendar.expected_globex_minute(pd.Timestamp("2024-02-29T21:14:00Z"))
    assert calendar.expected_globex_minute(pd.Timestamp("2024-02-29T21:15:00Z"))
    assert calendar.expected_globex_minute(pd.Timestamp("2024-02-29T21:30:00Z"))


def test_early_close_must_include_both_rth_and_globex_close():
    with pytest.raises(ValueError, match="globex_close"):
        CMECalendarSnapshot(
            version="fixture", source=CME_HOLIDAY_SOURCE,
            coverage_start=date(2024, 1, 1), coverage_end=date(2024, 12, 31),
            exceptions={"2024-07-03": {"session_type": "early_close", "rth_end": "13:15"}},
        )


def test_calendar_requires_timezone_aware_timestamps(calendar):
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.is_rth(datetime(2024, 3, 8, 15, 59))


def test_dst_boundary_uses_new_york_timezone_rules():
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="fixture-dst-2026", source=CME_HOLIDAY_SOURCE,
        coverage_start=date(2026, 10, 7), coverage_end=date(2026, 12, 31),
    ))
    before = calendar.snapshot.session_for_rth_date(date(2026, 10, 30))
    after = calendar.snapshot.session_for_rth_date(date(2026, 11, 2))
    assert before.rth_start.isoformat() == "2026-10-30T09:30:00-04:00"
    assert after.rth_start.isoformat() == "2026-11-02T09:30:00-05:00"


def test_installed_october_snapshot_covers_only_reviewed_dates_and_oct12_is_regular():
    from pathlib import Path

    snapshot_path = (
        Path(__file__).resolve().parents[2] / "src" / "paper" / "config"
        / "cme_mnq_calendar_2026-10-08_2026-10-31.json"
    )
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(snapshot_path))
    assert calendar.snapshot.coverage_start == date(2026, 10, 8)
    assert calendar.snapshot.coverage_end == date(2026, 10, 31)
    oct12 = calendar.snapshot.session_for_rth_date(date(2026, 10, 12))
    assert oct12.session_type == "regular"
    assert oct12.globex_close.isoformat() == "2026-10-12T17:00:00-04:00"
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2026-10-25T22:00:00Z")) == date(2026, 10, 26)
    assert calendar.snapshot.session_for_rth_date(date(2026, 10, 31)).session_type == "weekend"
    with pytest.raises(CalendarUnavailable, match="does not cover 2026-11-01"):
        calendar.snapshot.session_for_rth_date(date(2026, 11, 1))


def test_thanksgiving_and_christmas_overrides_are_read_from_snapshot_schema():
    # Schema-semantics fixture only; these values are NOT an official 2026 CME review.
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="fixture-holiday-overrides", source=CME_HOLIDAY_SOURCE,
        coverage_start=date(2026, 11, 25), coverage_end=date(2026, 12, 28),
        exceptions={
            "2026-11-26": {"session_type": "closed"},
            "2026-11-27": {"session_type": "early_close", "rth_end": "12:00", "globex_close": "12:00"},
            "2026-12-24": {"session_type": "early_close", "rth_end": "12:00", "globex_close": "12:00"},
            "2026-12-25": {"session_type": "closed"},
        },
    ))
    assert calendar.snapshot.session_for_rth_date(date(2026, 11, 26)).rth_end is None
    assert calendar.snapshot.session_for_rth_date(date(2026, 11, 27)).final_rth_bar.hour == 11
    assert calendar.snapshot.session_for_rth_date(date(2026, 12, 24)).final_rth_bar.hour == 11
    assert calendar.snapshot.session_for_rth_date(date(2026, 12, 25)).rth_end is None


def test_cme_review_validator_checks_product_coverage_and_identity():
    import json
    from scripts.validate_cme_snapshot import validate

    snapshot = CMECalendarSnapshot(
        version="fixture-reviewed", source=CME_HOLIDAY_SOURCE,
        coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31),
    )
    scratch = _review_validator_scratch()
    try:
        snapshot_path = scratch / "snapshot.json"
        review_path = scratch / "review.json"
        snapshot_path.write_text(json.dumps(snapshot.to_mapping()), encoding="utf-8")
        review_path.write_text(json.dumps({
            "source_url": CME_HOLIDAY_SOURCE, "product": "MNQ",
            "product_name": "MNQ Micro E-mini Nasdaq-100 Index Futures",
            "timezone": "America/New_York", "venue": "CME Globex", "product_view": "Futures",
            "source_timezone": "America/Chicago", "selection_method": "CME Full Calendar date/product selection",
            "coverage_start": "2026-10-08", "coverage_end": "2026-10-31",
            "reviewed_at_utc": "2026-10-08T10:11:43Z",
            "verified_schedule": {"october_12_2026": "regular"},
            "snapshot_identity": snapshot.identity,
        }), encoding="utf-8")
        assert validate(snapshot_path, review_path)["valid"] is True
        with pytest.raises(ValueError, match="identity"):
            review_path.write_text(json.dumps({"snapshot_identity": "wrong"}), encoding="utf-8")
            validate(snapshot_path, review_path)
    finally:
        shutil.rmtree(scratch)


def test_cme_review_validator_rejects_exceptions_outside_coverage():
    import json
    from src.paper.cme_calendar import CMECalendarSnapshot, CME_HOLIDAY_SOURCE
    from scripts.validate_cme_snapshot import validate

    snapshot = CMECalendarSnapshot(
        version="fixture-reviewed", source=CME_HOLIDAY_SOURCE,
        coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31),
        exceptions={"2028-01-01": {"session_type": "closed"}},
    )
    scratch = _review_validator_scratch()
    try:
        snapshot_path = scratch / "snapshot.json"
        review_path = scratch / "review.json"
        snapshot_path.write_text(json.dumps(snapshot.to_mapping()), encoding="utf-8")
        review_path.write_text(json.dumps({
            "source_url": CME_HOLIDAY_SOURCE, "product": "MNQ",
            "product_name": "MNQ Micro E-mini Nasdaq-100 Index Futures",
            "timezone": "America/New_York", "venue": "CME Globex", "product_view": "Futures",
            "source_timezone": "America/Chicago", "selection_method": "CME Full Calendar date/product selection",
            "coverage_start": "2026-10-08", "coverage_end": "2026-10-31",
            "reviewed_at_utc": "2026-10-08T10:11:43Z",
            "verified_schedule": {"october_12_2026": "regular"},
            "snapshot_identity": snapshot.identity,
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="outside declared coverage"):
            validate(snapshot_path, review_path)
    finally:
        shutil.rmtree(scratch)


def test_verified_labor_day_2026_schedule_closes_the_disputed_gap_and_converts_timezones():
    from pathlib import Path
    from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar

    root = Path(__file__).resolve().parents[2]
    snapshot_path = root / "src/paper/config/cme_mnq_calendar_2026-09-07_2026-09-08.json"
    review_path = root / "src/paper/config/cme_mnq_calendar_2026-09-07_2026-09-08.review.json"
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(snapshot_path))
    session = calendar.snapshot.session_for_rth_date(date(2026, 9, 7))

    assert session.session_type == "early_close"
    assert session.rth_end.isoformat() == "2026-09-07T13:00:00-04:00"
    assert session.globex_close.isoformat() == "2026-09-07T13:00:00-04:00"
    assert session.final_rth_bar.isoformat() == "2026-09-07T12:59:00-04:00"
    assert session.final_rth_bar.astimezone(timezone.utc).isoformat() == "2026-09-07T16:59:00+00:00"

    # A 16:59Z -> 22:00Z gap crosses the 13:00-18:00 EDT holiday halt and
    # ordinary 17:00-18:00 EDT maintenance; none of its minutes are expected.
    assert calendar.expected_missing_minutes(
        pd.Timestamp("2026-09-07T16:59:00Z"), pd.Timestamp("2026-09-07T22:00:00Z")
    ) == []
    # A true omission immediately before the halt remains an expected bar.
    assert calendar.expected_missing_minutes(
        pd.Timestamp("2026-09-07T16:58:00Z"), pd.Timestamp("2026-09-07T17:00:00Z")
    ) == [pd.Timestamp("2026-09-07T16:59:00Z")]
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2026-09-07T17:00:00Z")) is None
    assert calendar.trading_date_for_timestamp(pd.Timestamp("2026-09-07T22:00:00Z")) == date(2026, 9, 8)

    from scripts.validate_cme_snapshot import validate
    result = validate(snapshot_path, review_path)
    assert result["valid"] is True
    assert result["coverage_start"] == "2026-09-07"
    assert result["coverage_end"] == "2026-09-08"


def test_future_calendar_review_template_cannot_be_loaded_as_runtime_snapshot():
    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[2]
        / "src" / "paper" / "config"
        / "cme_mnq_calendar_2026-10-07_2027-12-31_REVIEW_TEMPLATE.json"
    )
    with pytest.raises(KeyError, match="coverage_start"):
        CMECalendarSnapshot.from_json(template)
