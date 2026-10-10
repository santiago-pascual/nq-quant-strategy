from datetime import date, datetime, timezone

import pytest

from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.ibkr_historical_poll import _poll_verdict, analyze_poll_snapshots, run_poll_test


def _bar(stamp, *, close=100.0, volume=3):
    return {"con_id": 815824267, "timestamp": stamp, "open": 100.0,
            "high": max(101.0, close), "low": 99.0, "close": close,
            "volume": volume, "arrival_utc": "2026-10-08T14:02:00+00:00"}


def _poll(index, epoch, bars, errors=None):
    return {"poll": index, "request_epoch": epoch,
            "arrival_utc": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            "bars": bars, "errors": errors or []}


def test_poll_analysis_excludes_forming_bar_and_deduplicates_overlapping_responses():
    now = 1791468120  # 2026-10-08 14:02 UTC
    first = _bar(now - 180)
    forming = _bar(now - 30)
    result = analyze_poll_snapshots([
        _poll(1, now, [first, forming]),
        _poll(2, now + 30, [first, _bar(now - 120)]),
    ], interval_seconds=30)

    assert result["completed_unique_bar_count"] == 2
    assert result["new_completed_bars_by_poll"] == [1, 1]
    assert result["new_bar_available_during_test"] is True
    assert result["duplicate_rows_across_responses"] == 1
    assert result["overlap_revision_count"] == 0
    assert result["bar_quality"]["usable"] is True


def test_poll_analysis_detects_revisions_gaps_and_provider_errors():
    now = 1791468120
    error = {"code": 10167, "message": "delayed data subscription message"}
    result = analyze_poll_snapshots([
        _poll(1, now, [_bar(now - 240), _bar(now - 120)]),
        _poll(2, now + 30, [_bar(now - 240, close=100.25), _bar(now - 120)], [error]),
    ], interval_seconds=30)

    assert result["overlap_revision_count"] == 1
    assert result["gap_count"] == 1
    assert result["provider_errors"] == [error]


def _reviewed_calendar():
    return CMETradingCalendar(CMECalendarSnapshot(
        version="reviewed-test", source="https://example.com/cme",
        coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31),
    ))


def _dated_bar(stamp: datetime, arrival="2026-10-10T00:10:00+00:00"):
    return {"con_id": 815824267, "timestamp": int(stamp.timestamp()), "open": 100.0,
            "high": 101.0, "low": 99.0, "close": 100.0, "volume": 3,
            "arrival_utc": arrival}


def test_poll_gap_in_reviewed_maintenance_is_classified_as_closure():
    result = analyze_poll_snapshots([_poll(1, 1791569400, [
        _dated_bar(datetime(2026, 10, 9, 20, 59, tzinfo=timezone.utc)),
        _dated_bar(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc)),
    ])], interval_seconds=30, calendar=_reviewed_calendar())

    assert result["gap_count"] == 1
    gap = result["timestamp_gaps"][0]
    assert gap["missing_minute_slots"] == 60
    assert gap["expected_open_minute_slots"] == 0
    assert gap["classification"] == "verified_scheduled_closure"


def test_poll_gap_during_reviewed_open_session_remains_unresolved():
    result = analyze_poll_snapshots([_poll(1, 1791569400, [
        _dated_bar(datetime(2026, 10, 9, 17, 0, tzinfo=timezone.utc)),
        _dated_bar(datetime(2026, 10, 9, 17, 2, tzinfo=timezone.utc)),
    ])], interval_seconds=30, calendar=_reviewed_calendar())

    gap = result["timestamp_gaps"][0]
    assert gap["missing_minute_slots"] == 1
    assert gap["expected_open_minute_slots"] == 1
    assert gap["classification"].startswith("expected_session_gap_unresolved")


def test_poll_gap_outside_calendar_coverage_fails_closed():
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="one-day-test", source="https://example.com/cme",
        coverage_start=date(2026, 10, 9), coverage_end=date(2026, 10, 9),
    ))
    result = analyze_poll_snapshots([_poll(1, 1791821400, [
        _dated_bar(datetime(2026, 10, 12, 14, 0, tzinfo=timezone.utc), "2026-10-12T15:00:00+00:00"),
        _dated_bar(datetime(2026, 10, 12, 14, 2, tzinfo=timezone.utc), "2026-10-12T15:00:00+00:00"),
    ])], interval_seconds=30, calendar=calendar)

    gap = result["timestamp_gaps"][0]
    assert gap["expected_open_minute_slots"] is None
    assert gap["classification"].startswith("uncertified_calendar_gap")


def test_pre_finalization_revision_is_usable_only_after_stability_gate():
    analysis = {"completed_unique_bar_count": 10, "invalid_rows": [],
                "overlap_revision_count": 1}
    finalized = {"finalized_bars": 9, "late_revisions": []}
    assert _poll_verdict("completed", analysis, finalized) == (
        "historical_poll_usable_after_revision_stability_gate")
    assert _poll_verdict("completed", analysis, None) == "historical_poll_not_validated"
    assert _poll_verdict("completed", analysis, {**finalized, "late_revisions": ["late"]}) == (
        "historical_poll_not_validated")


def test_poll_interval_obeys_identical_historical_request_pacing():
    with pytest.raises(ValueError, match="at least 15 seconds"):
        run_poll_test(interval_seconds=14)


def test_poll_count_is_bounded():
    with pytest.raises(ValueError, match="between 1 and 5"):
        run_poll_test(polls=6)
