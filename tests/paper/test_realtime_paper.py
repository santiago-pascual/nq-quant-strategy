from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import pandas as pd

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.context_adapter import PaperMarketContextAdapter
from src.paper.logger import PaperEvent, PaperEventLogger, PaperEventType
from src.paper.realtime_checkpoint import AtomicCheckpointStore, capture_engine_state, restore_engine_state
from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar, CME_HOLIDAY_SOURCE
from src.paper.realtime_market_data import (
    CanonicalBar, ReplayMarketDataSource, mark_replay_session_final_bars,
)
from src.paper.realtime_service import RealtimePaperConfig, RealtimePaperService
from src.paper.run_autonomous import build_real_paper_engine
from src.paper.shadow_replay import run_shadow_replay
from src.risk import RiskEngine, RiskLimits
from src.portfolio.conflict import PortfolioConflictEngine
from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.strategies.orb.strategy import ORBStrategy


def _bar(minute: int, *, day: int = 2) -> CanonicalBar:
    ts = datetime(2024, 1, day, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return CanonicalBar("MNQ", ts, 100.0, 100.5, 99.5, 100.0, 1000.0, provider="test")


def test_canonical_bar_requires_timezone_and_completed_finite_ohlcv():
    with pytest.raises(ValueError, match="timezone-aware"):
        CanonicalBar("MNQ", datetime(2024, 1, 2), 100, 101, 99, 100, 1)
    with pytest.raises(ValueError, match="completed"):
        CanonicalBar("MNQ", datetime(2024, 1, 2, tzinfo=timezone.utc), 100, 101, 99, 100, 1, completed=False)
    with pytest.raises(ValueError, match="finite"):
        CanonicalBar("MNQ", datetime(2024, 1, 2, tzinfo=timezone.utc), 100, 101, 99, float("nan"), 1)


def test_replay_source_filters_through_resume_timestamp():
    source = ReplayMarketDataSource([_bar(0), _bar(1), _bar(2)])
    source.start()
    source.subscribe(("MNQ",), after_timestamp=_bar(0).timestamp)
    assert [bar.timestamp for bar in source.bars()] == [_bar(1).timestamp, _bar(2).timestamp]


def test_supervisor_termination_requests_graceful_paper_stop(monkeypatch):
    import signal
    from src.paper.run_realtime_paper import _run_with_graceful_signals

    handlers = {}
    def set_handler(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous
    monkeypatch.setattr(signal, "signal", set_handler)
    requested = []

    class Service:
        def request_stop(self, *, reason):
            requested.append(reason)
        def run(self, *, restore):
            assert restore is False
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    _run_with_graceful_signals(Service(), restore=False)
    assert requested == [f"signal_{signal.SIGTERM}"]
    assert all(handlers[sig] is signal.SIG_DFL for sig in handlers)


def test_in_session_gap_is_backfilled_before_the_newer_bar(tmp_path):
    previous, missing, missing_next, newest = _bar(0), _bar(1), _bar(2), _bar(3)
    source = ReplayMarketDataSource([previous, newest], backfill_bars=[missing, missing_next])
    engine, adapter = build_real_paper_engine(tmp_path)
    engine.connect()
    service = RealtimePaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )

    assert service._process(previous)
    assert service._process(newest)
    assert service._last_bar.timestamp == newest.timestamp
    events = engine.logger.read_all()
    persisted = [event for event in events if event.event_type is PaperEventType.MARKET_DATA]
    assert [event.payload["timestamp"] for event in persisted] == [
        previous.timestamp.isoformat(), missing.timestamp.isoformat(),
        missing_next.timestamp.isoformat(), newest.timestamp.isoformat()
    ]
    types = [event.event_type for event in events]
    assert PaperEventType.MISSING_BAR in types
    assert PaperEventType.BACKFILL_STARTED in types
    assert PaperEventType.BACKFILL_COMPLETED in types


def test_incomplete_backfill_fails_closed_before_new_bar_reaches_engine(tmp_path):
    previous, newest = _bar(0), _bar(3)
    source = ReplayMarketDataSource([previous, newest], backfill_bars=[])
    engine, adapter = build_real_paper_engine(tmp_path)
    engine.connect()
    service = RealtimePaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    assert service._process(previous)
    with pytest.raises(RuntimeError, match="backfill failed closed"):
        service._process(newest)
    assert service._last_bar.timestamp == previous.timestamp
    assert not any(
        event.event_type is PaperEventType.MARKET_DATA
        and event.payload.get("timestamp") == newest.timestamp.isoformat()
        for event in engine.logger.read_all()
    )
    assert any(event.event_type is PaperEventType.BACKFILL_FAILED for event in engine.logger.read_all())


def test_contract_change_is_persisted_as_roll_event(tmp_path):
    first = CanonicalBar("MNQ", _bar(0).timestamp, 100, 101, 99, 100, 1,
                         contract_symbol="MNQU6")
    second = CanonicalBar("MNQ", _bar(1).timestamp, 100, 101, 99, 100, 1,
                          contract_symbol="MNQZ6")
    source = ReplayMarketDataSource([first, second])
    engine, adapter = build_real_paper_engine(tmp_path)
    engine.connect()
    service = RealtimePaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    assert service._process(first)
    assert service._process(second)
    roll = [event for event in engine.logger.read_all()
            if event.event_type is PaperEventType.CONTRACT_ROLL]
    assert len(roll) == 1
    assert roll[0].payload["previous_contract"] == "MNQU6"
    assert roll[0].payload["new_contract"] == "MNQZ6"


def test_realtime_service_marks_cme_early_close_bar_for_orb_exit(tmp_path):
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="fixture-cme-calendar",
        source=CME_HOLIDAY_SOURCE,
        coverage_start=datetime(2024, 1, 2).date(),
        coverage_end=datetime(2024, 1, 2).date(),
        exceptions={"2024-01-02": {
            "session_type": "early_close", "rth_end": "13:00", "globex_close": "13:00",
        }},
    ))
    bar = CanonicalBar("MNQ", datetime(2024, 1, 2, 17, 59, tzinfo=timezone.utc),
                       100, 100.5, 99.5, 100, 1000)
    engine, adapter = build_real_paper_engine(tmp_path)
    engine.connect()
    received = []
    original = engine.process_bar

    def capture(market_data):
        received.append(dict(market_data))
        return original(market_data)

    engine.process_bar = capture
    service = RealtimePaperService(
        source=ReplayMarketDataSource([bar]), engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path), calendar=calendar,
    )
    assert service._process(bar)
    assert received[0]["is_final_rth_bar"] is True
    assert received[0]["calendar_version"] == "fixture-cme-calendar"


def test_replay_source_normalizes_only_matching_vendor_root_symbol():
    row = {
        "symbol": "MNQ.v.0", "timestamp": _bar(0).timestamp,
        "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10,
    }
    source = ReplayMarketDataSource([row])
    source.start()
    source.subscribe(("MNQ",))
    assert next(source.bars()).symbol == "MNQ"

    wrong = ReplayMarketDataSource([{**row, "symbol": "NQ.v.0"}])
    wrong.start()
    wrong.subscribe(("MNQ",))
    with pytest.raises(ValueError, match="unexpected symbol"):
        next(wrong.bars())


def test_replay_marks_actual_final_rth_bar_per_new_york_session():
    import pandas as pd

    times = [
        "2024-07-02 19:59:00+00:00",  # 15:59 ET normal close
        "2024-07-03 16:59:00+00:00",  # 12:59 ET shortened session
        "2024-07-05 17:14:00+00:00",  # 13:14 ET shortened session
        "2024-07-05 19:59:00+00:00",  # later bar in same next session
    ]
    marked = mark_replay_session_final_bars(pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)}))
    assert marked["is_session_final"].tolist() == [True, True, False, True]


def test_realtime_service_rejects_duplicate_changes_and_out_of_order(tmp_path):
    engine, adapter = build_real_paper_engine(tmp_path)
    service = RealtimePaperService(
        source=ReplayMarketDataSource([]), engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    engine.connect()
    try:
        assert service._process(_bar(0))
        assert not service._process(_bar(0))
        assert service._bars_processed == 1
        with pytest.raises(ValueError, match="duplicate timestamp"):
            service._process(CanonicalBar("MNQ", _bar(0).timestamp, 100, 101, 99, 100.25, 1000))
        with pytest.raises(ValueError, match="out-of-order"):
            service._process(_bar(-1))
        events = engine.logger.read_all()
        assert any(event.event_type is PaperEventType.DUPLICATE_BAR for event in events)
        assert any(event.event_type is PaperEventType.OUT_OF_ORDER_BAR for event in events)
    finally:
        engine.disconnect()


def test_realtime_service_fails_closed_when_missing_rth_minute_cannot_be_backfilled(tmp_path):
    engine, adapter = build_real_paper_engine(tmp_path)
    service = RealtimePaperService(
        source=ReplayMarketDataSource([]), engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    engine.connect()
    try:
        service._process(_bar(0))
        with pytest.raises(RuntimeError, match="backfill failed closed"):
            service._process(_bar(2))
        assert any(e.event_type is PaperEventType.MISSING_BAR for e in engine.logger.read_all())
        assert any(e.event_type is PaperEventType.BACKFILL_FAILED for e in engine.logger.read_all())
        assert service._last_bar.timestamp == _bar(0).timestamp
    finally:
        engine.disconnect()


def test_candidate_accounting_distinguishes_risk_and_conflict_rejection(tmp_path):
    engine, adapter = build_real_paper_engine(tmp_path)
    service = RealtimePaperService(
        source=ReplayMarketDataSource([]), engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    timestamp = _bar(0).timestamp

    def event(kind, payload, sequence):
        return PaperEvent(str(sequence), sequence, kind, timestamp, payload)

    service._observe_event(event(PaperEventType.STRATEGY_DECISION, {
        "strategy_name": "MRL1", "action": "enter", "signal": "long",
    }, 1))
    service._observe_event(event(PaperEventType.RISK_DECISION, {
        "strategy_name": "MRL1", "approved": True, "risk_per_contract": 125,
    }, 2))
    service._observe_event(event(PaperEventType.ERROR, {
        "message": "Portfolio conflict rejected entry: max positions",
        "context": {"strategy_name": "MRL1"},
    }, 3))
    metrics = service._metrics["MRL1"]
    assert metrics["candidates"] == 1
    assert metrics["risk_approved"] == 1
    assert metrics["accepted"] == 0
    assert metrics["conflict_rejected"] == 1
    assert metrics["rejected"] == 1


def test_scheduled_refit_with_insufficient_rows_fails_closed_before_bar(tmp_path):
    engine, adapter = build_real_paper_engine(tmp_path)
    stream = adapter.context._raw_hmm_provider.mr
    stream.model = SimpleNamespace(artifact_hash="dummy-for-fail-closed-test")
    stream.next_refit_timestamp = pd.Timestamp("2024-01-02 14:30:00+00:00")
    service = RealtimePaperService(
        source=ReplayMarketDataSource([]), engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    with pytest.raises(RuntimeError, match="insufficient valid training rows"):
        service._process(_bar(0))
    assert engine.last_timestamp is None
    assert any(event.event_type is PaperEventType.HMM_REFIT_FAILED for event in engine.logger.read_all())


def test_watchdog_records_stale_and_reconnect_transitions(tmp_path):
    class HealthSource(ReplayMarketDataSource):
        def __init__(self):
            super().__init__([])
            self.connected = True
            self.is_stale = True

        def health(self):
            return {"connected": self.connected, "stale": self.is_stale}

    source = HealthSource()
    engine, adapter = build_real_paper_engine(tmp_path)
    service = RealtimePaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path, stale_after_seconds=10),
    )
    service._last_feed_walltime = time.time()
    worker = threading.Thread(target=service._watch_feed)
    worker.start()

    def wait_for(event_type):
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if any(event.event_type is event_type for event in engine.logger.read_all()):
                return
            time.sleep(0.05)
        pytest.fail(f"watchdog did not emit {event_type.value}")

    wait_for(PaperEventType.FEED_STALE)
    source.is_stale = False
    wait_for(PaperEventType.FEED_RECOVERED)
    source.connected = False
    wait_for(PaperEventType.FEED_DISCONNECTED)
    source.connected = True
    wait_for(PaperEventType.FEED_CONNECTED)
    service._stop_requested.set()
    worker.join(timeout=2)
    events = engine.logger.read_all()
    types = {event.event_type for event in events}
    assert PaperEventType.FEED_STALE in types
    assert PaperEventType.FEED_RECOVERED in types
    assert PaperEventType.FEED_DISCONNECTED in types
    assert PaperEventType.FEED_CONNECTED in types


def test_status_replace_retries_a_transient_windows_file_lock(monkeypatch, tmp_path):
    import src.paper.realtime_service as realtime_service

    engine, adapter = build_real_paper_engine(tmp_path)
    service = RealtimePaperService(
        source=ReplayMarketDataSource([]), engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=tmp_path),
    )
    actual_replace = realtime_service.os.replace
    attempts = {"count": 0}

    def transient_lock(source, target):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("temporary reader lock")
        return actual_replace(source, target)

    monkeypatch.setattr(realtime_service.os, "replace", transient_lock)
    service._write_status()
    assert attempts["count"] == 2
    assert service.status_path.exists()


def _run_service(output: str, bars, *, resume=False):
    from pathlib import Path
    output_dir = Path(output)
    engine, adapter = build_real_paper_engine(output_dir, recover_trailing_event=resume)
    source = ReplayMarketDataSource(bars)
    service = RealtimePaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=output_dir, checkpoint_every_bars=2),
        run_id="restart-test",
    )
    service.run(restore=resume)
    return engine.logger.read_all()


def test_realtime_checkpoint_restart_matches_uninterrupted_bar_stream(tmp_path):
    bars = [_bar(i) for i in range(6)]
    baseline_dir = tmp_path / "baseline"
    baseline_events = _run_service(str(baseline_dir), bars)

    resumed_dir = tmp_path / "resumed"
    first_events = _run_service(str(resumed_dir), bars[:3])
    assert (resumed_dir / "paper_checkpoint.json").exists()
    second_events = _run_service(str(resumed_dir), bars[3:], resume=True)

    def signature(events):
        return [
            (event.event_type.value, event.timestamp, {
                key: value for key, value in event.payload.items()
                if key not in {"_idempotency_key", "run_id", "severity"}
            })
            for event in events
            if event.event_type in {
                PaperEventType.MARKET_DATA, PaperEventType.HMM_STATE,
                PaperEventType.STRATEGY_DECISION, PaperEventType.RISK_DECISION,
                PaperEventType.FILL, PaperEventType.POSITION_OPENED,
                PaperEventType.POSITION_CLOSED,
            }
        ]
    assert signature(baseline_events) == signature(second_events)
    assert len([e for e in second_events if e.event_type is PaperEventType.MARKET_DATA]) == 6
    assert any(e.event_type is PaperEventType.SYSTEM_RECOVERED for e in second_events)


def test_shadow_replay_recomputes_the_recorded_canonical_stream(tmp_path):
    bars = [_bar(i) for i in range(5)]
    _run_service(str(tmp_path / "baseline"), bars)
    report = run_shadow_replay(tmp_path / "baseline")
    assert report["status"] == "GREEN"
    assert report["bars_replayed"] == len(bars)
    assert report["deterministic_events_expected"] == report["deterministic_events_actual"]


def test_realtime_cli_runs_historical_stream_and_daily_shadow(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    from src.paper import run_realtime_paper

    history_rows = []
    ordinal = 0
    for day in (2, 3):
        start = datetime(2024, 1, day, 14, 30, tzinfo=timezone.utc)
        for minute in range(300):
            close = 100.0 + ordinal * 0.002 + float(np.sin(ordinal / 11.0)) * 0.1
            history_rows.append({
                "timestamp": start + timedelta(minutes=minute),
                "open": close - 0.02, "high": close + 0.5,
                "low": close - 0.5, "close": close, "volume": 1000.0,
            })
            ordinal += 1

    stream_rows = []
    for day, count in ((4, 390), (5, 390)):
        start = datetime(2024, 1, day, 14, 30, tzinfo=timezone.utc)
        for minute in range(count):
            if minute < 30:
                high, low, close = 101.0, 99.0, 100.0
            elif minute == 30:
                high, low, close = 102.0, 100.0, 101.0
            elif minute == 31:
                high, low, close = 108.0, 101.0, 107.0
            else:
                high, low, close = 101.0, 99.0, 100.0
            stream_rows.append({
                "timestamp": start + timedelta(minutes=minute),
                "open": close, "high": high, "low": low,
                "close": close, "volume": 1000.0,
            })
    all_rows = pd.DataFrame(history_rows + stream_rows)
    monkeypatch.setattr(run_realtime_paper, "load_canonical_raw_mnq", lambda: all_rows.copy())
    output = tmp_path / "cli"
    args = [
        "--command", "run", "--mode", "PAPER",
        "--replay-start", "2024-01-04T14:30:00Z",
        "--replay-end", "2024-01-05T20:59:00Z",
        "--output-dir", str(output), "--checkpoint-every-bars", "100",
        "--cost-config", str(Path(__file__).parents[2] / "src" / "paper" / "config" / "topstepx_mnq_fees_2026-07.json"),
    ]
    assert run_realtime_paper.main(args) == 0
    with (output / "events.jsonl").open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    market_events = [row for row in events if row["event_type"] == "market_data"]
    assert len(market_events) == 780
    assert sum(row["event_type"] == "paper_order_filled" for row in events) >= 1
    assert all({"run_id", "severity", "symbol"}.issubset(row["payload"]) for row in events)
    assert (output / "paper_checkpoint.json").exists()
    assert (output / "bootstrap_checkpoint.json").exists()
    assert (output / "hmm_refits.jsonl").exists()
    assert (output / "paper_analytics.sqlite3").exists()
    assert (output / "daily_reports").exists()
    refits = [json.loads(line) for line in (output / "hmm_refits.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert any(row.get("stream") == "MR" and row.get("phase") == "causal_bootstrap" for row in refits)
    assert run_realtime_paper.main([
        "--command", "shadow-report", "--output-dir", str(output),
    ]) == 0
    assert json.loads((output / "daily_shadow_parity.json").read_text(encoding="utf-8"))["status"] == "GREEN"
    assert run_realtime_paper.main([
        "--command", "verify-checkpoint", "--output-dir", str(output),
        "--cost-config", str(Path(__file__).parents[2] / "src" / "paper" / "config" / "topstepx_mnq_fees_2026-07.json"),
    ]) == 0
    assert run_realtime_paper.main([
        "--command", "status", "--output-dir", str(output),
    ]) == 0


def test_atomic_checkpoint_uses_backup_when_current_file_is_corrupt(tmp_path):
    store = AtomicCheckpointStore(tmp_path / "state.json")
    store.save({"generation": 1})
    store.save({"generation": 2})
    store.path.write_text("{partial", encoding="utf-8")
    assert store.load() == {"generation": 1}


def test_logger_idempotency_prevents_duplicate_bar_events_after_restart(tmp_path):
    path = tmp_path / "events.jsonl"
    ts = _bar(0).timestamp
    first = PaperEventLogger(path)
    first.set_event_context(run_id="run-1", symbol="MNQ", severity="INFO")
    first.set_idempotency_scope(f"run-1:{ts.isoformat()}")
    event = first.append(PaperEventType.PAPER_POSITION_OPENED, {"strategy_name": "ORB"}, timestamp=ts)

    seen = []
    restarted = PaperEventLogger(path)
    restarted.subscribe(seen.append)
    restarted.set_event_context(run_id="run-1", symbol="MNQ", severity="INFO")
    restarted.set_idempotency_scope(f"run-1:{ts.isoformat()}")
    replayed = restarted.append(PaperEventType.PAPER_POSITION_OPENED, {"strategy_name": "ORB"}, timestamp=ts)
    assert replayed.event_id == event.event_id
    assert len(restarted.read_all()) == 1
    assert seen == [event]


def test_realtime_logger_repairs_only_an_incomplete_trailing_record(tmp_path):
    path = tmp_path / "recover.jsonl"
    logger = PaperEventLogger(path)
    logger.append(PaperEventType.SYSTEM_STARTED, {"mode": "PAPER"})
    with path.open("ab") as handle:
        handle.write(b'{"sequence":1,"payload":')
    recovered = PaperEventLogger(path, recover_trailing_partial=True)
    assert recovered.count() == 1
    recovered.append(PaperEventType.SYSTEM_STOPPED, {"mode": "PAPER"})
    assert recovered.count() == 2


def _orb_engine(path):
    return PaperTradingEngine(
        strategies=[ORBStrategy()], execution=ExecutionEngine(),
        risk=RiskEngine(RiskLimits(
            risk_per_trade=500, max_total_risk=10_000, max_daily_loss=5_000,
            max_concurrent_positions=1, max_daily_trades=10, max_contracts=1,
        )),
        conflict=PortfolioConflictEngine(max_concurrent_positions=1),
        broker=InMemoryBrokerAdapter(), logger=PaperEventLogger(path),
        config=PaperEngineConfig(initial_equity=50_000, point_value=2,
            commission_per_contract=0, automatic_simulated_fills=True),
    )


def test_open_orb_position_and_risk_state_restore(tmp_path):
    engine = _orb_engine(tmp_path / "open.jsonl")
    engine.connect()
    start = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
    for minute in range(31):
        if minute < 30:
            high, low, close = 101.0, 99.0, 100.0
        else:
            high, low, close = 102.0, 100.0, 101.5
        engine.process_bar({
            "timestamp": start + timedelta(minutes=minute), "open": close,
            "high": high, "low": low, "close": close, "volume": 1000.0,
        })
    assert engine.execution.get_positions()
    state = capture_engine_state(engine)
    restored = _orb_engine(tmp_path / "restored.jsonl")
    restore_engine_state(restored, state)
    assert len(restored.execution.get_positions()) == 1
    assert restored.risk.open_risk == engine.risk.open_risk
    assert restored.risk.daily_trade_count == engine.risk.daily_trade_count
    assert restored.strategies[0].in_trade
