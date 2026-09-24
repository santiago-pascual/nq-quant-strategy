from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.paper.logger import (
    PaperEventLogger,
    PaperEventType,
)


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


# ============================================================================
# Basic event creation
# ============================================================================


def test_logger_creates_event(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    event = logger.append(
        PaperEventType.MARKET_DATA,
        {
            "symbol": "MNQ",
            "close": 20000.0,
        },
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    assert event.event_id
    assert event.sequence == 0
    assert event.event_type == PaperEventType.MARKET_DATA
    assert event.timestamp == ts("2026-09-23T14:30:00+00:00")
    assert event.payload["symbol"] == "MNQ"
    assert event.payload["close"] == 20000.0


def test_logger_persists_event(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    logger.append(
        PaperEventType.MARKET_DATA,
        {"close": 20000.0},
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    assert path.exists()

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1


# ============================================================================
# Ordering
# ============================================================================


def test_event_sequences_are_monotonic(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    first = logger.append(
        PaperEventType.MARKET_DATA,
        {"bar": 1},
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    second = logger.append(
        PaperEventType.MARKET_DATA,
        {"bar": 2},
        timestamp=ts("2026-09-23T14:31:00+00:00"),
    )

    third = logger.append(
        PaperEventType.STRATEGY_SIGNAL,
        {
            "strategy_name": "ORB",
            "signal": "LONG",
        },
        timestamp=ts("2026-09-23T14:32:00+00:00"),
    )

    assert first.sequence == 0
    assert second.sequence == 1
    assert third.sequence == 2

    assert len({first.event_id, second.event_id, third.event_id}) == 3


def test_logger_recovers_sequence_after_restart(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger_1 = PaperEventLogger(path)

    first = logger_1.append(
        PaperEventType.MARKET_DATA,
        {"bar": 1},
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    second = logger_1.append(
        PaperEventType.MARKET_DATA,
        {"bar": 2},
        timestamp=ts("2026-09-23T14:31:00+00:00"),
    )

    assert first.sequence == 0
    assert second.sequence == 1

    logger_2 = PaperEventLogger(path)

    third = logger_2.append(
        PaperEventType.MARKET_DATA,
        {"bar": 3},
        timestamp=ts("2026-09-23T14:32:00+00:00"),
    )

    assert third.sequence == 2


# ============================================================================
# Reading / replay
# ============================================================================


def test_read_all_reconstructs_events(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    logger.append(
        PaperEventType.MARKET_DATA,
        {
            "symbol": "MNQ",
            "close": 20000.0,
        },
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    logger.append(
        PaperEventType.STRATEGY_SIGNAL,
        {
            "strategy_name": "MRS2",
            "signal": "SHORT",
        },
        timestamp=ts("2026-09-23T14:31:00+00:00"),
    )

    events = logger.read_all()

    assert len(events) == 2
    assert events[0].sequence == 0
    assert events[1].sequence == 1

    assert events[0].event_type == PaperEventType.MARKET_DATA
    assert events[1].event_type == PaperEventType.STRATEGY_SIGNAL

    assert events[1].payload["strategy_name"] == "MRS2"


def test_last_event(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    assert logger.last_event() is None

    logger.append(
        PaperEventType.MARKET_DATA,
        {"bar": 1},
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    logger.append(
        PaperEventType.MARKET_DATA,
        {"bar": 2},
        timestamp=ts("2026-09-23T14:31:00+00:00"),
    )

    last = logger.last_event()

    assert last is not None
    assert last.sequence == 1
    assert last.payload["bar"] == 2


def test_count(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    assert logger.count() == 0

    for i in range(5):
        logger.append(
            PaperEventType.MARKET_DATA,
            {"bar": i},
            timestamp=ts(f"2026-09-23T14:{30 + i:02d}:00+00:00"),
        )

    assert logger.count() == 5


# ============================================================================
# Validation
# ============================================================================


def test_naive_timestamp_is_rejected(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    with pytest.raises(ValueError, match="timezone-aware"):
        logger.append(
            PaperEventType.MARKET_DATA,
            {"close": 20000.0},
            timestamp=datetime(2026, 9, 23, 14, 30),
        )


def test_invalid_event_type_is_rejected(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    with pytest.raises(TypeError):
        logger.append(
            "market_data",
            {"close": 20000.0},
            timestamp=ts("2026-09-23T14:30:00+00:00"),
        )


# ============================================================================
# Convenience APIs
# ============================================================================


def test_strategy_signal_helper(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    event = logger.log_strategy_signal(
        strategy_name="ORB",
        signal="LONG",
        timestamp=ts("2026-09-23T14:30:00+00:00"),
        reason="breakout",
    )

    assert event.event_type == PaperEventType.STRATEGY_SIGNAL
    assert event.payload == {
        "strategy_name": "ORB",
        "signal": "LONG",
        "reason": "breakout",
    }


def test_strategy_decision_helper(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    event = logger.log_strategy_decision(
        strategy_name="MRS2",
        action="ENTER",
        signal="SHORT",
        timestamp=ts("2026-09-23T14:30:00+00:00"),
        reason="zscore_extreme",
    )

    assert event.event_type == PaperEventType.STRATEGY_DECISION
    assert event.payload["strategy_name"] == "MRS2"
    assert event.payload["action"] == "ENTER"
    assert event.payload["signal"] == "SHORT"


def test_error_helper(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    event = logger.log_error(
        ValueError("bad fill"),
        timestamp=ts("2026-09-23T14:30:00+00:00"),
        context={
            "order_id": "abc",
        },
    )

    assert event.event_type == PaperEventType.ERROR
    assert event.payload["error_type"] == "ValueError"
    assert event.payload["message"] == "bad fill"
    assert event.payload["context"]["order_id"] == "abc"


# ============================================================================
# Corruption detection
# ============================================================================


def test_corrupted_event_log_is_detected(tmp_path):
    path = tmp_path / "paper.jsonl"

    path.write_text(
        '{"event_id":"1","sequence":0,"event_type":"market_data",'
        '"timestamp":"2026-09-23T14:30:00+00:00","payload":{}}\n'
        "THIS IS NOT JSON\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid event log"):
        PaperEventLogger(path)


def test_non_monotonic_event_log_is_detected(tmp_path):
    path = tmp_path / "paper.jsonl"

    path.write_text(
        '{"event_id":"1","sequence":1,"event_type":"market_data",'
        '"timestamp":"2026-09-23T14:30:00+00:00","payload":{}}\n'
        '{"event_id":"2","sequence":1,"event_type":"market_data",'
        '"timestamp":"2026-09-23T14:31:00+00:00","payload":{}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        PaperEventLogger(path)


# ============================================================================
# Reset
# ============================================================================


def test_clear_resets_logger(tmp_path):
    path = tmp_path / "paper.jsonl"

    logger = PaperEventLogger(path)

    logger.append(
        PaperEventType.MARKET_DATA,
        {"bar": 1},
        timestamp=ts("2026-09-23T14:30:00+00:00"),
    )

    assert logger.count() == 1

    logger.clear()

    assert logger.count() == 0

    event = logger.append(
        PaperEventType.MARKET_DATA,
        {"bar": 2},
        timestamp=ts("2026-09-23T14:31:00+00:00"),
    )

    assert event.sequence == 0
