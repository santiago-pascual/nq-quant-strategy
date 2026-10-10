from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.ibkr_delayed_market_data import (
    AppendOnlyFinalizationLedger,
    ContractScheduleUnavailable,
    DelayedBarFinalizer,
    FinalizationPolicy,
    IBKRContractSchedule,
    IBKRContractWindow,
    finalize_observation_sequence,
)
from src.paper.ibkr_observation_ledger import bar_value_hash


def _calendar():
    return CMETradingCalendar(CMECalendarSnapshot(
        version="test", source="https://example.com/cme", coverage_start=date(2026, 10, 8),
        coverage_end=date(2026, 10, 31),
    ))


def _schedule(start):
    return IBKRContractSchedule([IBKRContractWindow(
        con_id=815824267, local_symbol="MNQZ6", start_utc=start - timedelta(hours=1),
        end_utc=start + timedelta(hours=1),
    )])


def _bar(stamp, close=30983.25):
    return {"timestamp": int(stamp.timestamp()), "open": close, "high": close + 1,
            "low": close - 1, "close": close, "volume": 10}


def _finalizer(start):
    return DelayedBarFinalizer(
        contract_schedule=_schedule(start), calendar=_calendar(),
        policy=FinalizationPolicy(minimum_age_seconds=600, required_identical_observations=2,
                                  minimum_observation_spacing_seconds=30),
    )


def test_revision_resets_stability_and_only_latest_pre_finalization_value_is_delivered():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    f = _finalizer(start)
    assert f.observe(_bar(start), contract_id=815824267, local_symbol="MNQZ6",
                     observed_at=start + timedelta(minutes=11), poll_number=1) == "new_observation"
    revised = _bar(start, close=30975.75)
    assert f.observe(revised, contract_id=815824267, local_symbol="MNQZ6",
                     observed_at=start + timedelta(minutes=11, seconds=30), poll_number=2) == "revised_before_finalization"
    assert f.observe(revised, contract_id=815824267, local_symbol="MNQZ6",
                     observed_at=start + timedelta(minutes=12), poll_number=3) == "unchanged_observation"
    emitted = f.finalize_ready(now=start + timedelta(minutes=12))
    assert len(emitted) == 1
    assert emitted[0].bar.close == 30975.75
    assert emitted[0].exchange_bar_timestamp_utc == start
    assert emitted[0].first_observed_at_utc == start + timedelta(minutes=11)
    assert emitted[0].finalized_at_utc == start + timedelta(minutes=12)
    assert emitted[0].data_quality_status == "validated_stable_completed_calendar_covered"


def test_post_finalization_revision_is_recorded_but_never_redelivered():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    f = _finalizer(start)
    bar = _bar(start)
    for poll, seconds in ((1, 660), (2, 690)):
        f.observe(bar, contract_id=815824267, local_symbol="MNQZ6",
                  observed_at=start + timedelta(seconds=seconds), poll_number=poll)
    assert len(f.finalize_ready(now=start + timedelta(seconds=690))) == 1
    assert f.observe(_bar(start, close=30970), contract_id=815824267, local_symbol="MNQZ6",
                     observed_at=start + timedelta(seconds=720), poll_number=3) == "revision_after_delivery"
    assert f.late_revisions
    assert f.finalize_ready(now=start + timedelta(seconds=750)) == []


def test_missing_expected_cme_minute_blocks_later_bar_delivery():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    f = _finalizer(start)
    for stamp, poll in ((start, 1), (start + timedelta(minutes=2), 1)):
        for second, poll_num in ((660, poll), (690, poll + 1)):
            f.observe(_bar(stamp), contract_id=815824267, local_symbol="MNQZ6",
                      observed_at=stamp + timedelta(seconds=second), poll_number=poll_num)
    emitted = f.finalize_ready(now=start + timedelta(seconds=810))
    assert [row.exchange_bar_timestamp_utc for row in emitted] == [start]
    assert any(event["kind"] == "unresolved_expected_minute_gap" for event in f.quality_events)


def test_contract_schedule_fails_closed_for_gaps_and_wrong_contract():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    first = IBKRContractWindow(815824267, "MNQZ6", start - timedelta(hours=1), start)
    second = IBKRContractWindow(815824267, "MNQZ6", start + timedelta(minutes=1), start + timedelta(hours=1))
    with pytest.raises(ContractScheduleUnavailable, match="contiguous"):
        IBKRContractSchedule([first, second])
    f = _finalizer(start)
    with pytest.raises(ContractScheduleUnavailable, match="does not match"):
        f.observe(_bar(start), contract_id=42005282, local_symbol="MNQZ6",
                  observed_at=start + timedelta(minutes=11), poll_number=1)


def test_restart_cursor_prevents_duplicate_delivery_and_records_late_revision():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    contract = {"con_id": 815824267, "local_symbol": "MNQZ6"}
    original = _bar(start)
    key = (contract["con_id"], int(start.timestamp()))

    class PersistedFinalizations:
        rows = {key: bar_value_hash(original)}

        def append(self, item):
            raise AssertionError("an already delivered bar must not be appended again")

    f = DelayedBarFinalizer(contract_schedule=_schedule(start), calendar=_calendar(),
                            policy=FinalizationPolicy(), finalization_ledger=PersistedFinalizations())
    assert f.observe(original, contract_id=contract["con_id"], local_symbol=contract["local_symbol"],
                     observed_at=start + timedelta(minutes=12), poll_number=3) == "already_delivered"
    assert f.observe(_bar(start, close=30970), contract_id=contract["con_id"], local_symbol=contract["local_symbol"],
                     observed_at=start + timedelta(minutes=12, seconds=30), poll_number=4) == "revision_after_delivery"
    assert len(f.late_revisions) == 1


def test_rehydration_does_not_mislabel_pre_delivery_revision_as_late():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    path = Path("results/paper/ibkr_diagnostics") / f".test-finalized-{uuid4().hex}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _assert_rehydration_revision_classification(path, start)
    finally:
        path.unlink(missing_ok=True)


def _assert_rehydration_revision_classification(path, start):
    ledger = AppendOnlyFinalizationLedger(path)
    original = _bar(start)
    first = _finalizer(start)
    for poll, sec in ((1, 660), (2, 690)):
        first.observe(original, contract_id=815824267, local_symbol="MNQZ6",
                      observed_at=start + timedelta(seconds=sec), poll_number=poll)
    emitted = first.finalize_ready(now=start + timedelta(seconds=690))
    ledger.append(emitted[0])

    resumed = DelayedBarFinalizer(contract_schedule=_schedule(start), calendar=_calendar(),
        policy=FinalizationPolicy(), finalization_ledger=AppendOnlyFinalizationLedger(path))
    historical_revision = _bar(start, close=30970)
    assert resumed.observe(historical_revision, contract_id=815824267, local_symbol="MNQZ6",
        observed_at=start + timedelta(seconds=680), poll_number=3) == "observed_before_delivery_during_recovery"
    assert resumed.late_revisions == []
    post_delivery_revision = _bar(start, close=30965)
    assert resumed.observe(post_delivery_revision, contract_id=815824267, local_symbol="MNQZ6",
        observed_at=start + timedelta(seconds=720), poll_number=4) == "revision_after_delivery"
    assert len(resumed.late_revisions) == 1


def test_recorded_poll_sequence_finalizes_only_value_observed_stable_twice():
    start = datetime(2026, 10, 8, 23, 48, tzinfo=timezone.utc)
    epoch = int(start.timestamp())
    original = {"open": 30989.5, "high": 30989.5, "low": 30987.25,
                "close": 30989.5, "volume": 587.0}
    revised = {**original, "low": 30975.0, "close": 30975.75, "volume": 1571.0}
    records = []
    seq = 0
    for poll, seen, values in (
        (1, "2026-10-09T00:18:35+00:00", original),
        (2, "2026-10-09T00:19:06+00:00", revised),
        (3, "2026-10-09T00:19:36+00:00", revised),
    ):
        seq += 1
        records.append({"observation_sequence": seq, "poll_number": poll,
                        "bar_start_epoch_utc": epoch, "contract_id": 815824267,
                        "local_symbol": "MNQZ6", "observed_at_utc": seen, "ohlcv": values})
    schedule = _schedule(start)
    finalizer, emitted = finalize_observation_sequence(
        records, contract_schedule=schedule, calendar=_calendar(),
        policy=FinalizationPolicy(minimum_age_seconds=600, required_identical_observations=2,
                                  minimum_observation_spacing_seconds=30),
    )
    assert len(emitted) == 1
    assert emitted[0].bar.close == 30975.75
    assert emitted[0].first_observed_at_utc == datetime.fromisoformat(records[0]["observed_at_utc"])
    assert emitted[0].finalized_at_utc == datetime.fromisoformat(records[2]["observed_at_utc"])
