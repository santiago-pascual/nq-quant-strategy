from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import shutil
from uuid import uuid4

import pytest

from src.paper.ibkr_paper_recovery import AppendOnlyPaperDeliveryLedger


def _root() -> Path:
    return Path(__file__).parents[2] / "results" / "paper" / f"paper_delivery_journal_test_{uuid4().hex}"


def _bar(minute: int = 0) -> dict:
    stamp = datetime(2026, 10, 9, 0, minute, tzinfo=timezone.utc)
    return {
        "contract_id": 815824267,
        "local_symbol": "MNQZ6",
        "expiry": "20261218",
        "exchange_bar_timestamp_utc": stamp.isoformat(),
        "exchange_bar_end_utc": (stamp + timedelta(minutes=1)).isoformat(),
        "first_observed_at_utc": (stamp + timedelta(minutes=11)).isoformat(),
        "finalized_at_utc": (stamp + timedelta(minutes=12)).isoformat(),
        "provider": "ibkr_tws_delayed_historical",
        "data_quality_status": "validated_stable_completed_calendar_covered",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    }


def _ledger(path: Path, calendar_identity: str = "reviewed-calendar-test-v1") -> AppendOnlyPaperDeliveryLedger:
    return AppendOnlyPaperDeliveryLedger(
        path, contract_id=815824267, local_symbol="MNQZ6", expiry="20261218",
        calendar_identity=calendar_identity,
    )


def test_pending_finalized_bar_survives_restart_until_paper_checkpoint_ack():
    root = _root()
    try:
        path = root / "delivery.jsonl"
        original = _ledger(path)
        row = _bar()
        assert original.stage(row, recovered=True)
        assert not original.stage(row, recovered=True)

        restarted = _ledger(path)
        pending = restarted.pending_after(None)
        assert len(pending) == 1
        assert pending[0]["provenance"] == "RECOVERED_PAPER"
        assert pending[0]["bar_value_sha256"]

        # A crash after Paper checkpoint replacement but before feed-cursor ack.
        assert restarted.reconcile_checkpoint(
            row["exchange_bar_timestamp_utc"], "a" * 64
        ) == 1
        after_restart = _ledger(path)
        assert after_restart.pending_after(row["exchange_bar_timestamp_utc"]) == []
        with pytest.raises(ValueError, match="older than an acknowledged"):
            after_restart.pending_after(None)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_conflicting_recovered_bar_version_is_not_silently_substituted():
    root = _root()
    try:
        path = root / "delivery.jsonl"
        journal = _ledger(path)
        row = _bar()
        journal.stage(row, recovered=False)
        revised = dict(row, close=100.75, high=101.0)
        with pytest.raises(RuntimeError, match="preserve it and audit the revision"):
            journal.stage(revised, recovered=True)
        restored = _ledger(path)
        assert restored.pending_after(None)[0]["ohlcv"]["close"] == 100.5
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_contract_calendar_quality_and_ohlcv_fail_closed():
    root = _root()
    try:
        path = root / "delivery.jsonl"
        journal = _ledger(path)
        with pytest.raises(ValueError, match="contract ID"):
            journal.stage(dict(_bar(), contract_id=1), recovered=True)
        with pytest.raises(ValueError, match="calendar-covered"):
            journal.stage(dict(_bar(), data_quality_status="unverified"), recovered=True)
        with pytest.raises(ValueError, match="invalid OHLCV"):
            journal.stage(dict(_bar(), low=102.0), recovered=True)
        journal.stage(_bar(), recovered=True)
        with pytest.raises(ValueError, match="contract/calendar/schema identity"):
            AppendOnlyPaperDeliveryLedger(
                path, contract_id=815824267, local_symbol="MNQZ6", expiry="20261218",
                calendar_identity="different-calendar",
            )
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_known_r3_delivery_identity_resumes_without_rewriting_history():
    root = _root()
    try:
        path = root / "delivery.jsonl"
        journal = _ledger(path)
        assert journal.stage(_bar(), recovered=True)
        journal.acknowledge_through(_bar()["exchange_bar_timestamp_utc"], "a" * 64)

        # Model the r3 journal created by the earlier source revision. This is
        # test-fixture setup only; production recovery never edits prior rows.
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in records:
            row["ledger_identity"]["recovery_module_sha256"] = (
                "4d2f4c5c4bc4ba1ecf47d819329da7c4320b50b84709162e0f65cf6bfac4dcf9"
            )
            row.pop("writer_recovery_module_sha256", None)
        historical_bytes = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records) + "\n"
        path.write_text(historical_bytes, encoding="utf-8")

        resumed = _ledger(path)
        assert resumed.compatibility_mode is True
        assert resumed.identity["recovery_module_sha256"] == (
            "4d2f4c5c4bc4ba1ecf47d819329da7c4320b50b84709162e0f65cf6bfac4dcf9"
        )
        assert resumed.pending_after(_bar()["exchange_bar_timestamp_utc"]) == []

        next_bar = _bar(minute=1)
        assert resumed.stage(next_bar, recovered=True)
        appended = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        assert appended["ledger_identity"] == resumed.identity
        assert appended["writer_recovery_module_sha256"] == resumed.writer_module_sha256
        assert path.read_text(encoding="utf-8").startswith(historical_bytes)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_unknown_legacy_delivery_identity_is_rejected():
    root = _root()
    try:
        path = root / "delivery.jsonl"
        journal = _ledger(path)
        journal.stage(_bar(), recovered=True)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["ledger_identity"]["recovery_module_sha256"] = "f" * 64
        path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="contract/calendar/schema identity"):
            _ledger(path)
    finally:
        if root.exists():
            shutil.rmtree(root)


def _calendar_and_schedule():
    from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
    from src.paper.ibkr_delayed_market_data import IBKRContractSchedule, IBKRContractWindow

    project = Path(__file__).parents[2]
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(
        project / "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json"
    ))
    schedule = IBKRContractSchedule([IBKRContractWindow(
        con_id=815824267, local_symbol="MNQZ6",
        start_utc=datetime(2026, 10, 8, tzinfo=timezone.utc),
        end_utc=datetime(2026, 11, 1, tzinfo=timezone.utc),
    )])
    return calendar, schedule


def _prepare_finalized_input(root: Path, timestamps: list[datetime], *, ohlcv_rows=None):
    from src.paper.ibkr_delayed_market_data import AppendOnlyFinalizationLedger, FinalizedIBKRBar
    from src.paper.realtime_market_data import CanonicalBar

    calendar, schedule = _calendar_and_schedule()
    finalization = AppendOnlyFinalizationLedger(root / "ibkr_finalized.jsonl")
    delivery = AppendOnlyPaperDeliveryLedger(
        root / "paper_delivery.jsonl", contract_id=815824267, local_symbol="MNQZ6",
        expiry="20261218", calendar_identity=calendar.snapshot.identity,
    )
    for index, stamp in enumerate(timestamps):
        values = (ohlcv_rows[index] if ohlcv_rows is not None else {
            "open": 100 + index * .25, "high": 101 + index * .25,
            "low": 99 + index * .25, "close": 100.5 + index * .25,
            "volume": 10,
        })
        bar = CanonicalBar(
            symbol="MNQ", timestamp=stamp, **values,
            provider="ibkr_tws_delayed_historical", source_id="815824267",
            contract_symbol="MNQZ6",
        )
        finalization.append(FinalizedIBKRBar(
            bar=bar, contract_id=815824267, exchange_bar_timestamp_utc=stamp,
            first_observed_at_utc=stamp + timedelta(minutes=11),
            finalized_at_utc=stamp + timedelta(minutes=12),
            data_quality_status="validated_stable_completed_calendar_covered",
            stabilization_observations=2,
        ))
    return calendar, schedule, finalization, delivery


def test_finalized_market_source_orders_catchup_and_preserves_provenance():
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource

    root = _root()
    start = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(
            root, [start + timedelta(minutes=i) for i in range(3)]
        )
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        source.subscribe(("MNQ",), after_timestamp=None)
        bars = list(source.bars())
        assert [bar.timestamp for bar in bars] == [start + timedelta(minutes=i) for i in range(3)]
        assert all(bar.source_id.endswith("RECOVERED_PAPER") for bar in bars)
        assert source.audit_metadata(bars[0].timestamp)["first_observed_at_utc"]
        assert len(delivery.pending_after(None)) == 3
        assert source.health()["backlog_bars"] == 3
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_source_reconciles_acknowledged_cursor_and_marks_bootstrap_catchup_recovered():
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource

    root = _root()
    start = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(
            root, [start + timedelta(minutes=i) for i in range(3)]
        )
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
            recovery_status=lambda: {"catchup_active": True},
            is_recovered_bar=lambda _epoch: True,
        )
        source.start()
        source.subscribe(("MNQ",), after_timestamp=start - timedelta(minutes=1))
        emitted = list(source.bars())
        assert len(emitted) == 3
        assert all(bar.source_id.endswith("RECOVERED_PAPER") for bar in emitted)

        # The source's initial activation cursor becomes stale as acknowledgments
        # advance. Re-staging must use the durable ack cursor, not reject rollback.
        delivery.acknowledge_through(start, "a" * 64)
        source._stage_finalized_rows(recovered=False)
        assert len(delivery.pending_after(start)) == 2
        assert source.health()["mode"] == "RECOVERING"
        assert delivery._bars[int(start.timestamp())]["provenance"] == "RECOVERED_PAPER"
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_finalized_market_source_fails_closed_on_calendar_gap_and_unapproved_contract():
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource

    root = _root()
    start = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(
            root, [start, start + timedelta(minutes=2)]
        )
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        with pytest.raises(RuntimeError, match="open-session bars missing"):
            source.subscribe(("MNQ",), after_timestamp=None)

        # The contract schedule is explicit: a bar after the approved window fails.
        outside = start.replace(day=1)
        with pytest.raises(Exception):
            schedule.resolve(outside)
    finally:
        if root.exists():
            shutil.rmtree(root)


def _run_mechanics_service(output: Path, source, delivery, *, resume: bool,
                           enable_strategies: bool = False,
                           run_id: str = "recovery-mechanics-test",
                           context_state: dict | None = None):
    from src.paper.costs import PaperCostPolicy
    from src.paper.ibkr_paper_recovery import AcknowledgedDelayedPaperService
    from src.paper.realtime_service import RealtimePaperConfig
    from src.paper.run_autonomous import build_real_paper_engine

    project = Path(__file__).parents[2]
    costs = PaperCostPolicy.from_json(project / "src/paper/config/topstepx_mnq_fees_2026-07.json")
    engine, adapter = build_real_paper_engine(
        output, initial_equity=50_000,
        commission_per_contract=costs.commission_per_contract_side,
        exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
        regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
        recover_trailing_event=resume,
    )
    if context_state is not None and not resume:
        adapter.context.load_state_dict(context_state)
    if not enable_strategies:
        # Existing mechanics-only cases isolate journal protocol behavior.
        engine.strategies = []
    service = AcknowledgedDelayedPaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=output,
                                   checkpoint_every_bars=1),
        calendar=_calendar_and_schedule()[0], cost_policy=costs,
        delivery_ledger=delivery, run_id=run_id,
    )
    service.run(restore=resume)
    return engine.logger.read_all()


def _fitted_context_seed():
    """Fit production causal MR/S2R streams on deterministic OHLCV input."""
    import numpy as np
    import pandas as pd
    from zoneinfo import ZoneInfo
    from datetime import time

    from src.paper.bootstrap_artifacts import build_replay_causal_features
    from src.paper.market_context import CausalMarketContext

    ny = ZoneInfo("America/New_York")
    dates = pd.bdate_range("2024-07-09", "2026-10-08")
    rows = []
    ordinal = 0
    for date in dates:
        local = datetime.combine(date.date(), time(9, 30), ny)
        start = pd.Timestamp(local).tz_convert("UTC")
        for offset in range(3):
            close = (20_000.0 + ordinal * 0.001
                     + 2.0 * np.sin(ordinal / 17.0)
                     + 0.7 * np.sin(ordinal / 5.0))
            rows.append({
                "timestamp": start + pd.Timedelta(minutes=offset),
                "symbol": "MNQ.v.0", "instrument_id": 42005282,
                "open": close - 0.25, "high": close + 0.75,
                "low": close - 0.75, "close": close,
                "volume": float(100 + ordinal % 73),
            })
            ordinal += 1
    raw = pd.DataFrame(rows)
    features = build_replay_causal_features(raw)
    context = CausalMarketContext()
    context.bootstrap_causal_history(raw, precomputed_features=features)
    state = context.state_dict()
    provider = state["hmm_provider"]
    assert provider["mr"]["fit_count"] > 0 and provider["mr"]["model"] is not None
    assert provider["s2r"]["fit_count"] > 0 and provider["s2r"]["model"] is not None
    # This fixture's real 2-year-anchor/quarterly schedule makes an S2R refit
    # due on the first live bar below; no schedule fields are edited.
    assert pd.Timestamp(provider["s2r"]["next_refit_timestamp"]) == pd.Timestamp(
        "2026-10-09T13:30:00Z"
    )
    assert provider["s2r"]["fit_end_exclusive"] < provider["s2r"]["next_refit_timestamp"]
    return state


def _without_wallclock_metrics(value):
    """Drop diagnostic timings while keeping every semantic HMM field exact."""
    if isinstance(value, dict):
        return {
            key: _without_wallclock_metrics(item)
            for key, item in value.items()
            if not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [_without_wallclock_metrics(item) for item in value]
    return value


def test_strategy_enabled_engine_restart_matches_uninterrupted_paper_state():
    """Exercise the real configured strategies through a short delayed-bar restart."""
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource
    from src.paper.realtime_checkpoint import AtomicCheckpointStore

    root = _root()
    baseline, resumed = root / "strategy_baseline", root / "strategy_resumed"
    start = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=i) for i in range(12)]
    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(baseline, times)
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        source.subscribe(("MNQ",), after_timestamp=None)
        baseline_events = _run_mechanics_service(
            baseline, source, delivery, resume=False, enable_strategies=True
        )
        expected = AtomicCheckpointStore(baseline / "paper_checkpoint.json").load()
        assert set(expected["engine"]["strategies"]) == {"MRL1", "MRS2", "S2R", "ORB"}

        calendar, schedule, finalization, delivery = _prepare_finalized_input(resumed, times)
        class PrefixSource(IBKRFinalizedLedgerSource):
            def bars(self):
                for index, bar in enumerate(super().bars()):
                    yield bar
                    if index == 5:
                        return

        first = PrefixSource(finalization_ledger=finalization,
                             delivery_ledger=delivery,
                             contract_schedule=schedule, calendar=calendar)
        first.start()
        first.subscribe(("MNQ",), after_timestamp=None)
        _run_mechanics_service(resumed, first, delivery, resume=False,
                               enable_strategies=True)
        middle = AtomicCheckpointStore(resumed / "paper_checkpoint.json").load()
        assert middle["runtime"]["bars_processed"] == 6

        second_delivery = _ledger(resumed / "paper_delivery.jsonl",
                                  calendar.snapshot.identity)
        second = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=second_delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        second.start()
        second.subscribe(("MNQ",), after_timestamp=times[5])
        recovered_events = _run_mechanics_service(
            resumed, second, second_delivery, resume=True, enable_strategies=True
        )
        actual = AtomicCheckpointStore(resumed / "paper_checkpoint.json").load()
        assert actual["engine"] == expected["engine"]
        # Refit/forward-filter timing diagnostics are intentionally wall-clock
        # measurements. Compare all fitted parameters, posterior/state,
        # history, schedule, hashes and counters exactly.
        expected_context = _without_wallclock_metrics(expected["context"])
        actual_context = _without_wallclock_metrics(actual["context"])
        assert actual_context == expected_context
        for stream_name in ("mr", "s2r"):
            expected_stream = expected_context["hmm_provider"][stream_name]
            actual_stream = actual_context["hmm_provider"][stream_name]
            assert actual_stream["model"] == expected_stream["model"]
            assert actual_stream["model_identity_hash"] == expected_stream["model_identity_hash"]
            assert actual_stream["log_posterior"] == expected_stream["log_posterior"]
            assert actual_stream["last_emitted_state"] == expected_stream["last_emitted_state"]
        assert actual["runtime"]["last_bar"] == expected["runtime"]["last_bar"]
        assert actual["runtime"]["bars_processed"] == len(times)
        important = {"market_data", "hmm_state", "strategy_decision", "risk_decision",
                     "fill", "position_opened", "position_closed"}
        event_projection = lambda rows: [
            (event.event_type.value, event.timestamp, event.payload)
            for event in rows if event.event_type.value in important
        ]
        assert event_projection(recovered_events) == event_projection(baseline_events)
        assert second_delivery.pending_after(times[-1]) == []
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_strategy_generated_orb_fill_survives_paper_restart_with_open_position():
    """Use the real configured strategies and ORB entry/fill lifecycle across restart."""
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource
    from src.paper.realtime_checkpoint import AtomicCheckpointStore

    root = _root()
    baseline, resumed = root / "orb_baseline", root / "orb_resumed"
    start = datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc)  # 09:30 New York
    times = [start + timedelta(minutes=i) for i in range(33)]
    ohlcv = []
    for index in range(len(times)):
        if index < 30:
            ohlcv.append({"open": 100.0, "high": 101.0, "low": 99.0,
                          "close": 100.0, "volume": 100.0})
        elif index == 30:
            ohlcv.append({"open": 101.0, "high": 102.0, "low": 100.0,
                          "close": 101.5, "volume": 100.0})
        else:
            ohlcv.append({"open": 101.5, "high": 102.0, "low": 101.0,
                          "close": 101.5, "volume": 100.0})

    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(
            baseline, times, ohlcv_rows=ohlcv
        )
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        source.subscribe(("MNQ",), after_timestamp=None)
        baseline_events = _run_mechanics_service(
            baseline, source, delivery, resume=False, enable_strategies=True
        )
        baseline_state = AtomicCheckpointStore(baseline / "paper_checkpoint.json").load()
        important = {"strategy_decision", "order_created", "fill", "position_opened"}
        baseline_projection = [
            (event.event_type.value, event.timestamp, event.payload)
            for event in baseline_events if event.event_type.value in important
        ]
        assert any(kind == "strategy_decision" and payload.get("strategy_name") == "ORB"
                   and payload.get("action") == "enter"
                   for kind, _stamp, payload in baseline_projection)
        assert any(kind == "order_created" and payload.get("strategy_name") == "ORB"
                   for kind, _stamp, payload in baseline_projection)
        assert any(kind == "fill" and payload.get("strategy_name") == "ORB"
                   for kind, _stamp, payload in baseline_projection)
        assert any(position.get("fields", {}).get("strategy_name") == "ORB"
                   for position in baseline_state["engine"]["execution"]["positions"].values())

        calendar, schedule, finalization, delivery = _prepare_finalized_input(
            resumed, times, ohlcv_rows=ohlcv
        )

        class PrefixSource(IBKRFinalizedLedgerSource):
            def bars(self):
                for index, bar in enumerate(super().bars()):
                    yield bar
                    if index == 30:
                        return

        prefix = PrefixSource(finalization_ledger=finalization,
                              delivery_ledger=delivery,
                              contract_schedule=schedule, calendar=calendar)
        prefix.start()
        prefix.subscribe(("MNQ",), after_timestamp=None)
        _run_mechanics_service(resumed, prefix, delivery, resume=False,
                               enable_strategies=True)
        partial = AtomicCheckpointStore(resumed / "paper_checkpoint.json").load()
        assert partial["runtime"]["bars_processed"] == 31
        assert any(position.get("fields", {}).get("strategy_name") == "ORB"
                   for position in partial["engine"]["execution"]["positions"].values())

        delivery_after_restart = _ledger(resumed / "paper_delivery.jsonl",
                                         calendar.snapshot.identity)
        resumed_source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery_after_restart,
            contract_schedule=schedule, calendar=calendar,
        )
        resumed_source.start()
        resumed_source.subscribe(("MNQ",), after_timestamp=times[30])
        recovered_events = _run_mechanics_service(
            resumed, resumed_source, delivery_after_restart,
            resume=True, enable_strategies=True,
        )
        recovered_state = AtomicCheckpointStore(resumed / "paper_checkpoint.json").load()
        assert recovered_state["engine"] == baseline_state["engine"]
        assert recovered_state["context"] == baseline_state["context"]
        assert recovered_state["runtime"]["last_bar"] == baseline_state["runtime"]["last_bar"]
        assert delivery_after_restart.pending_after(times[-1]) == []

        recovered_projection = [
            (event.event_type.value, event.timestamp, event.payload)
            for event in recovered_events if event.event_type.value in important
        ]
        assert recovered_projection == baseline_projection
        assert sum(kind == "fill" for kind, *_ in recovered_projection) == 1
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_real_strategy_pending_entry_order_survives_checkpoint_and_restores_idempotently():
    """A real ORB entry can be checkpointed before its simulated fill settles."""
    from src.paper.realtime_checkpoint import (
        AtomicCheckpointStore, _encode, capture_engine_state, restore_engine_state,
    )
    from src.paper.run_autonomous import build_real_paper_engine

    root = _root()
    baseline_dir, recovered_dir = root / "pending_baseline", root / "pending_recovered"
    start = datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=index) for index in range(33)]
    rows = [
        ({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0}
         if index < 30 else
         {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 100.0}
         if index == 30 else
         {"open": 101.5, "high": 102.0, "low": 101.0, "close": 101.5, "volume": 100.0})
        for index in range(len(times))
    ]

    def engine_at_pending_order(path: Path):
        engine, adapter = build_real_paper_engine(path, initial_equity=50_000)
        adapter.context.load_state_dict(_fitted_context_seed())
        engine._automatic_simulated_fills = False
        engine.connect()
        for stamp, values in zip(times[:31], rows[:31]):
            engine.process_bar({"symbol": "MNQ", "timestamp": stamp, **values})
        orders = [order for order in engine.broker._orders.values()
                  if order.request.strategy_name == "ORB"]
        assert len(orders) == 1
        assert orders[0].status.value == "submitted"
        assert engine.position("ORB") is None
        return engine, adapter

    try:
        baseline, baseline_adapter = engine_at_pending_order(baseline_dir)
        baseline_payload = _encode(capture_engine_state(baseline))
        context_payload = baseline_adapter.context.state_dict()
        checkpoint = AtomicCheckpointStore(recovered_dir / "paper_checkpoint.json")
        checkpoint.save({"engine_state": baseline_payload})

        recovered, recovered_adapter = build_real_paper_engine(recovered_dir, initial_equity=50_000)
        recovered_adapter.context.load_state_dict(context_payload)
        recovered.connect()
        persisted = checkpoint.load()["engine_state"]
        restore_engine_state(recovered, persisted)
        assert _encode(capture_engine_state(recovered)) == baseline_payload
        assert set(recovered._pending_strategy_orders["ORB"]) == set(
            baseline._pending_strategy_orders["ORB"]
        )

        # Settle the same simulated order after restoration on both paths.
        # This exercises the existing broker/Paper fill path, not an IBKR order.
        for engine in (baseline, recovered):
            engine.process_bar({"symbol": "MNQ", "timestamp": times[31], **rows[31]})
            order_id = next(iter(engine._pending_strategy_orders["ORB"]))
            broker_order = engine.broker.get_order(order_id)
            assert broker_order is not None
            fill = engine.broker.process_fill(
                order_id, broker_order.request.quantity, price=102.0
            )
            engine.process_broker_fill(fill, timestamp=times[31])
            engine.process_bar({"symbol": "MNQ", "timestamp": times[32], **rows[32]})
        recovered_state = _encode(capture_engine_state(recovered))
        baseline_state = _encode(capture_engine_state(baseline))
        assert recovered_state["last_market_data"] == baseline_state["last_market_data"]
        assert recovered_state == baseline_state
        assert recovered.execution.get_positions() == baseline.execution.get_positions()
        assert recovered.broker._fills == baseline.broker._fills
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_hard_crash_after_real_orb_fill_replays_bar_without_duplicate_execution():
    """Abruptly terminate after an actual ORB fill, then compare full state/log."""
    from src.paper.ibkr_delayed_market_data import AppendOnlyFinalizationLedger
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource
    from src.paper.realtime_checkpoint import AtomicCheckpointStore

    root = _root()
    baseline, resumed = root / "orb_crash_baseline", root / "orb_crash_resumed"
    start = datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc)
    # Crash after the entry fill, then catch up more than one hour of already
    # finalized market bars and exercise the original stop lifecycle.
    times = [start + timedelta(minutes=i) for i in range(93)]
    ohlcv = [
        ({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0}
        if index < 30 else
        {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 100.0}
        if index == 30 else
        {"open": 101.5, "high": 102.0, "low": 98.0, "close": 99.0, "volume": 100.0}
        if index == 92 else
        {"open": 101.5, "high": 102.0, "low": 101.0, "close": 101.5, "volume": 100.0})
        for index in range(len(times))
    ]
    try:
        root.mkdir(parents=True, exist_ok=True)
        causal_seed = _fitted_context_seed()
        seed_path = root / "causal_context_seed.json"
        seed_path.write_text(json.dumps(causal_seed, sort_keys=True, default=str), encoding="utf-8")
        calendar, schedule, _finalization, delivery = _prepare_finalized_input(
            baseline, times, ohlcv_rows=ohlcv
        )
        baseline_finalization = AppendOnlyFinalizationLedger(baseline / "ibkr_finalized.jsonl")
        baseline_source = IBKRFinalizedLedgerSource(
            finalization_ledger=baseline_finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        baseline_source.start()
        baseline_source.subscribe(("MNQ",), after_timestamp=None)
        baseline_events = _run_mechanics_service(
            baseline, baseline_source, delivery, resume=False, enable_strategies=True,
            run_id="orb-hard-crash-test", context_state=causal_seed,
        )
        expected = AtomicCheckpointStore(baseline / "paper_checkpoint.json").load()
        expected_hmm = expected["context"]["hmm_provider"]
        assert expected_hmm["mr"]["fit_count"] >= 1
        assert expected_hmm["s2r"]["fit_count"] >= 2  # initial + scheduled Oct 9 refit
        assert expected_hmm["s2r"]["refit_events"][-1]["fit_timestamp"] == times[0].isoformat()
        recovery_event_types = {
            "market_data", "hmm_state", "strategy_decision", "risk_request",
            "risk_decision", "candidate_created", "candidate_rejected",
            "order_created", "order_submitted", "fill", "position_opened",
            "position_updated", "position_closed",
        }
        expected_projection = [
            (event.payload.get("_idempotency_key"), event.event_type.value,
             event.timestamp, event.payload)
            for event in baseline_events
            if event.event_type.value in recovery_event_types
        ]
        assert all(identity for identity, *_ in expected_projection)
        assert sum(kind == "fill" and payload.get("strategy_name") == "ORB"
                   for _key, kind, _stamp, payload in expected_projection) == 2  # entry + stop exit
        assert any(kind == "position_closed" and payload.get("strategy_name") == "ORB"
                   for _key, kind, _stamp, payload in expected_projection)

        calendar, schedule, _finalization, _delivery = _prepare_finalized_input(
            resumed, times, ohlcv_rows=ohlcv
        )
        child = textwrap.dedent(r'''
            import os, sys, runpy
            import json
            from pathlib import Path
            from datetime import datetime, timedelta, timezone
            from src.paper.ibkr_delayed_market_data import AppendOnlyFinalizationLedger
            from src.paper.ibkr_paper_recovery import AppendOnlyPaperDeliveryLedger, IBKRFinalizedLedgerSource, AcknowledgedDelayedPaperService
            from src.paper.realtime_service import RealtimePaperConfig
            from src.paper.run_autonomous import build_real_paper_engine
            from src.paper.costs import PaperCostPolicy
            output = Path(sys.argv[1])
            helpers = runpy.run_path(sys.argv[2])
            start = datetime(2026, 10, 9, 13, 30, tzinfo=timezone.utc)
            times = [start + timedelta(minutes=i) for i in range(93)]
            ohlcv = [
                ({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0}
                 if index < 30 else
                 {"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 100.0}
                 if index == 30 else
                 {"open": 101.5, "high": 102.0, "low": 98.0, "close": 99.0, "volume": 100.0}
                 if index == 92 else
                 {"open": 101.5, "high": 102.0, "low": 101.0, "close": 101.5, "volume": 100.0})
                for index in range(len(times))
            ]
            calendar, schedule, _finalization, _delivery = helpers["_prepare_finalized_input"](
                output, times, ohlcv_rows=ohlcv
            )
            finalization = AppendOnlyFinalizationLedger(output / "ibkr_finalized.jsonl")
            delivery = AppendOnlyPaperDeliveryLedger(
                output / "paper_delivery.jsonl", contract_id=815824267,
                local_symbol="MNQZ6", expiry="20261218",
                calendar_identity=calendar.snapshot.identity,
            )
            source = IBKRFinalizedLedgerSource(
                finalization_ledger=finalization, delivery_ledger=delivery,
                contract_schedule=schedule, calendar=calendar,
            )
            source.start()
            source.subscribe(("MNQ",), after_timestamp=None)
            costs = PaperCostPolicy.from_json(Path(sys.argv[3]))
            engine, adapter = build_real_paper_engine(
                output, initial_equity=50000,
                commission_per_contract=costs.commission_per_contract_side,
                exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
                regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
                recover_trailing_event=True,
            )
            adapter.context.load_state_dict(json.loads(Path(sys.argv[4]).read_text(encoding="utf-8")))
            original_process_bar = engine.process_bar
            def process_then_crash(market_data, **kwargs):
                result = original_process_bar(market_data, **kwargs)
                if market_data["timestamp"] == times[30]:
                    os._exit(75)
                return result
            engine.process_bar = process_then_crash
            service = AcknowledgedDelayedPaperService(
                source=source, engine=engine, context_adapter=adapter,
                config=RealtimePaperConfig(mode="PAPER", output_dir=output, checkpoint_every_bars=1),
                calendar=calendar, cost_policy=costs, delivery_ledger=delivery,
                run_id="orb-hard-crash-test",
            )
            service.run(restore=False)
        ''')
        project = Path(__file__).parents[2]
        costs = project / "src/paper/config/topstepx_mnq_fees_2026-07.json"
        outcome = subprocess.run(
                [sys.executable, "-c", child, str(resumed), str(Path(__file__).resolve()),
                 str(costs), str(seed_path)],
            cwd=project, capture_output=True, text=True, timeout=90,
        )
        assert outcome.returncode == 75, outcome.stdout + outcome.stderr
        with (resumed / "events.jsonl").open("r", encoding="utf-8") as handle:
            crash_rows = [json.loads(line) for line in handle if line.strip()]
        boundary_identity_before = [
            (row["event_id"], row["sequence"], row["payload"].get("_idempotency_key"))
            for row in crash_rows
            if row["timestamp"] == times[30].isoformat()
            and row["payload"].get("_idempotency_key")
        ]
        assert boundary_identity_before

        finalization = AppendOnlyFinalizationLedger(resumed / "ibkr_finalized.jsonl")
        delivery = _ledger(resumed / "paper_delivery.jsonl", calendar.snapshot.identity)
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        checkpoint = AtomicCheckpointStore(resumed / "paper_checkpoint.json")
        after = datetime.fromisoformat(checkpoint.load()["runtime"]["last_bar"]["timestamp"])
        source.subscribe(("MNQ",), after_timestamp=after)
        recovered_events = _run_mechanics_service(
            resumed, source, delivery, resume=True, enable_strategies=True,
            run_id="orb-hard-crash-test",
        )
        actual = checkpoint.load()
        assert actual["engine"] == expected["engine"]
        expected_context = _without_wallclock_metrics(expected["context"])
        actual_context = _without_wallclock_metrics(actual["context"])
        assert actual_context == expected_context
        for stream_name in ("mr", "s2r"):
            expected_stream = expected_context["hmm_provider"][stream_name]
            actual_stream = actual_context["hmm_provider"][stream_name]
            assert actual_stream["model"] == expected_stream["model"]
            assert actual_stream["model_identity_hash"] == expected_stream["model_identity_hash"]
            assert actual_stream["log_posterior"] == expected_stream["log_posterior"]
            assert actual_stream["last_emitted_state"] == expected_stream["last_emitted_state"]
        assert actual["runtime"]["last_bar"] == expected["runtime"]["last_bar"]
        projection = [
            (event.payload.get("_idempotency_key"), event.event_type.value,
             event.timestamp, event.payload)
            for event in recovered_events
            if event.event_type.value in recovery_event_types
        ]
        assert projection == expected_projection
        assert sum(kind == "fill" and payload.get("strategy_name") == "ORB"
                   for _key, kind, _stamp, payload in projection) == 2
        assert len(projection) == len(expected_projection)
        with (resumed / "events.jsonl").open("r", encoding="utf-8") as handle:
            recovered_rows = [json.loads(line) for line in handle if line.strip()]
        crash_prefix_identity = [
            (row["event_id"], row["sequence"], row["event_type"], row["timestamp"], row["payload"])
            for row in crash_rows
        ]
        final_prefix_identity = [
            (row["event_id"], row["sequence"], row["event_type"], row["timestamp"], row["payload"])
            for row in recovered_rows[:len(crash_rows)]
        ]
        assert final_prefix_identity == crash_prefix_identity
        boundary_identity_after = [
            (row["event_id"], row["sequence"], row["payload"].get("_idempotency_key"))
            for row in recovered_rows
            if row["timestamp"] == times[30].isoformat()
            and row["payload"].get("_idempotency_key")
        ]
        # Events already persisted before the crash keep the same identity;
        # service-level projections are emitted once when recovery completes
        # the bar because the crash happened immediately after engine.process_bar.
        assert boundary_identity_after[:len(boundary_identity_before)] == boundary_identity_before
        new_boundary_rows = [
            row for row in recovered_rows
            if row["timestamp"] == times[30].isoformat()
            and row["payload"].get("_idempotency_key")
            and row["event_id"] not in {item[0] for item in boundary_identity_before}
        ]
        orb_boundary_types = [
            row["event_type"] for row in new_boundary_rows
            if row["payload"].get("strategy_name") == "ORB"
        ]
        for required_event in (
            "candidate_created", "paper_order_created", "paper_order_filled",
            "paper_position_opened",
        ):
            assert orb_boundary_types.count(required_event) == 1
        assert "checkpoint_saved" in [row["event_type"] for row in new_boundary_rows]
        keys = [row["payload"]["_idempotency_key"] for row in recovered_rows
                if row["payload"].get("_idempotency_key")]
        assert len(keys) == len(set(keys))
        baseline_boundary = [
            (event.event_type.value, event.timestamp.isoformat(),
             {key: value for key, value in event.payload.items()
              if key not in {"_idempotency_key", "checkpoint_path"}})
            for event in baseline_events if event.timestamp == times[30]
        ]
        recovered_boundary = [
            (row["event_type"], row["timestamp"],
             {key: value for key, value in row["payload"].items()
              if key not in {"_idempotency_key", "checkpoint_path"}})
            for row in recovered_rows if row["timestamp"] == times[30].isoformat()
        ]
        assert recovered_boundary == baseline_boundary
        assert delivery.pending_after(times[-1]) == []
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_acknowledged_paper_replay_matches_uninterrupted_engine_state():
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource
    from src.paper.realtime_checkpoint import AtomicCheckpointStore

    root = _root()
    baseline = root / "baseline"
    resumed = root / "resumed"
    start = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=i) for i in range(6)]
    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(baseline, times)
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        source.subscribe(("MNQ",), after_timestamp=None)
        baseline_events = _run_mechanics_service(baseline, source, delivery, resume=False)
        baseline_state = AtomicCheckpointStore(baseline / "paper_checkpoint.json").load()

        calendar, schedule, finalization, delivery = _prepare_finalized_input(resumed, times)

        class PrefixSource(IBKRFinalizedLedgerSource):
            def bars(self):
                for index, bar in enumerate(super().bars()):
                    yield bar
                    if index == 2:
                        return

        first_source = PrefixSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        first_source.start()
        first_source.subscribe(("MNQ",), after_timestamp=None)
        _run_mechanics_service(resumed, first_source, delivery, resume=False)
        middle = AtomicCheckpointStore(resumed / "paper_checkpoint.json").load()
        assert middle["runtime"]["last_bar"]["timestamp"] == times[2].isoformat()
        assert len(delivery.pending_after(times[2])) == 3

        second_source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization,
            delivery_ledger=_ledger(resumed / "paper_delivery.jsonl", calendar.snapshot.identity),
            contract_schedule=schedule, calendar=calendar,
        )
        second_source.start()
        second_source.subscribe(("MNQ",), after_timestamp=times[2])
        resumed_events = _run_mechanics_service(
            resumed, second_source, second_source.delivery_ledger, resume=True
        )
        resumed_state = AtomicCheckpointStore(resumed / "paper_checkpoint.json").load()

        assert resumed_state["context"] == baseline_state["context"]
        assert resumed_state["engine"] == baseline_state["engine"]
        assert resumed_state["runtime"]["last_bar"] == baseline_state["runtime"]["last_bar"]
        assert resumed_state["runtime"]["bars_processed"] == len(times)
        assert second_source.delivery_ledger.pending_after(times[-1]) == []
        event_types = {"market_data", "hmm_state", "strategy_decision", "risk_decision",
                       "fill", "position_opened", "position_closed"}
        baseline_market = [e for e in baseline_events if e.event_type.value in event_types]
        resumed_market = [e for e in resumed_events if e.event_type.value in event_types]
        assert [(e.event_type.value, e.timestamp, e.payload) for e in baseline_market] == [
            (e.event_type.value, e.timestamp, e.payload) for e in resumed_market
        ]
    finally:
            if root.exists():
                shutil.rmtree(root)


@pytest.mark.parametrize(
    "crash_point",
    ["before_process", "during_process", "after_checkpoint_before_ack", "after_ack"],
)
def test_hard_process_crash_recovers_to_uninterrupted_state(crash_point: str):
    """Use a child process exit (no finally/close) at each durable handoff edge."""
    from src.paper.ibkr_delayed_market_data import AppendOnlyFinalizationLedger
    from src.paper.ibkr_paper_recovery import IBKRFinalizedLedgerSource
    from src.paper.realtime_checkpoint import AtomicCheckpointStore

    root = _root()
    baseline, resumed = root / "baseline", root / "resumed"
    start = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=i) for i in range(6)]
    try:
        calendar, schedule, finalization, delivery = _prepare_finalized_input(baseline, times)
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        source.subscribe(("MNQ",), after_timestamp=None)
        baseline_events = _run_mechanics_service(baseline, source, delivery, resume=False)
        expected = AtomicCheckpointStore(baseline / "paper_checkpoint.json").load()

        calendar, schedule, finalization, delivery = _prepare_finalized_input(resumed, times)
        child = textwrap.dedent(r'''
            import os, sys, runpy
            from pathlib import Path
            from src.paper.ibkr_paper_recovery import AppendOnlyPaperDeliveryLedger, IBKRFinalizedLedgerSource, AcknowledgedDelayedPaperService
            from src.paper.realtime_service import RealtimePaperConfig, RealtimePaperService
            from src.paper.run_autonomous import build_real_paper_engine
            from src.paper.costs import PaperCostPolicy
            from src.paper.ibkr_delayed_market_data import AppendOnlyFinalizationLedger
            output = Path(sys.argv[1])
            crash_point = sys.argv[2]
            helpers = runpy.run_path(sys.argv[3])
            calendar, schedule = helpers["_calendar_and_schedule"]()
            finalization = AppendOnlyFinalizationLedger(output / "ibkr_finalized.jsonl")
            delivery = AppendOnlyPaperDeliveryLedger(
                output / "paper_delivery.jsonl", contract_id=815824267,
                local_symbol="MNQZ6", expiry="20261218",
                calendar_identity=calendar.snapshot.identity,
            )
            source = IBKRFinalizedLedgerSource(
                finalization_ledger=finalization, delivery_ledger=delivery,
                contract_schedule=schedule, calendar=calendar,
            )
            costs = PaperCostPolicy.from_json(Path(sys.argv[4]))
            engine, adapter = build_real_paper_engine(
                output, initial_equity=50000,
                commission_per_contract=costs.commission_per_contract_side,
                exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
                regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
            )
            engine.strategies = []
            class FaultService(AcknowledgedDelayedPaperService):
                killed = False
                def _process(self, bar):
                    if crash_point == "before_process" and not self.killed:
                        self.killed = True
                        os._exit(71)
                    if crash_point == "during_process" and not self.killed:
                        self.killed = True
                        original = self.analytics.record_account_snapshot
                        def crash_after_state_update(**kwargs):
                            original(**kwargs)
                            os._exit(72)
                        self.analytics.record_account_snapshot = crash_after_state_update
                    return super()._process(bar)
                def save_checkpoint(self):
                    if crash_point == "after_checkpoint_before_ack" and not self.killed and self._last_bar is not None:
                        self.killed = True
                        with self.logger.path.open("ab") as handle:
                            handle.flush()
                            os.fsync(handle.fileno())
                        from dataclasses import replace
                        committed = self._last_bar
                        self._last_bar = replace(committed, provider_timestamp=None)
                        try:
                            RealtimePaperService.save_checkpoint(self)
                        finally:
                            self._last_bar = committed
                        os._exit(73)
                    if crash_point == "after_ack" and not self.killed and self._last_bar is not None:
                        self.killed = True
                        super().save_checkpoint()
                        os._exit(74)
                    return super().save_checkpoint()
            service = FaultService(
                source=source, engine=engine, context_adapter=adapter,
                config=RealtimePaperConfig(mode="PAPER", output_dir=output, checkpoint_every_bars=1),
                calendar=calendar, cost_policy=costs, delivery_ledger=delivery,
                    run_id="recovery-mechanics-test",
            )
            service.run(restore=False)
        ''')
        project = Path(__file__).parents[2]
        costs = project / "src/paper/config/topstepx_mnq_fees_2026-07.json"
        outcome = subprocess.run(
            [sys.executable, "-c", child, str(resumed), crash_point,
             str(Path(__file__).resolve()), str(costs)],
            cwd=project, capture_output=True, text=True, timeout=90,
        )
        assert outcome.returncode in {71, 72, 73, 74}, outcome.stdout + outcome.stderr

        calendar, schedule = _calendar_and_schedule()
        finalization = AppendOnlyFinalizationLedger(resumed / "ibkr_finalized.jsonl")
        delivery = _ledger(resumed / "paper_delivery.jsonl", calendar.snapshot.identity)
        source = IBKRFinalizedLedgerSource(
            finalization_ledger=finalization, delivery_ledger=delivery,
            contract_schedule=schedule, calendar=calendar,
        )
        source.start()
        last_checkpoint = AtomicCheckpointStore(resumed / "paper_checkpoint.json")
        checkpoint_timestamp = None
        if last_checkpoint.path.exists():
            checkpoint_timestamp = datetime.fromisoformat(
                last_checkpoint.load()["runtime"]["last_bar"]["timestamp"]
            )
        source.subscribe(("MNQ",), after_timestamp=checkpoint_timestamp)
        resumed_events = _run_mechanics_service(resumed, source, delivery, resume=True)
        actual = last_checkpoint.load()
        assert actual["context"] == expected["context"]
        assert actual["engine"] == expected["engine"]
        assert actual["runtime"]["last_bar"] == expected["runtime"]["last_bar"]
        assert actual["runtime"]["bars_processed"] == len(times)
        assert delivery.pending_after(times[-1]) == []
        market_data = lambda rows: [
            (event.event_type.value, event.timestamp, event.payload)
            for event in rows if event.event_type.value == "market_data"
        ]
        assert market_data(resumed_events) == market_data(baseline_events)
        assert len(market_data(resumed_events)) == len(times)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_acquisition_finalization_paper_checkpoint_ack_and_restart_fixture():
    """Bounded local fixture traverses acquisition through Paper and restart."""
    from src.paper.ibkr_paper_runner import build_cursor_aware_pipeline
    from src.paper.ibkr_observation_ledger import load_observation_ledger
    from src.paper.realtime_checkpoint import AtomicCheckpointStore

    root = _root()
    start = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=i) for i in range(3)]
    clock_value = [datetime(2026, 10, 9, 1, 0, tzinfo=timezone.utc)]
    request_count = [0]
    try:
        calendar, _schedule = _calendar_and_schedule()

        def request_historical(_request, _request_id):
            request_count[0] += 1
            arrival = clock_value[0].isoformat()
            return {
                "completed": True, "connected": True, "errors": [],
                "rows": [{
                    "con_id": 815824267,
                    "timestamp": int(stamp.timestamp()),
                    "open": 100 + i * .25, "high": 101 + i * .25,
                    "low": 99 + i * .25, "close": 100.5 + i * .25,
                    "volume": 10,
                    "arrival_utc": arrival,
                } for i, stamp in enumerate(timestamps)],
            }

        pipeline = build_cursor_aware_pipeline(
            output_dir=root,
            calendar=calendar,
            request_historical=request_historical,
            clock=lambda: clock_value[0],
        )
        source = pipeline["source"]
        controller = pipeline["controller"]
        pipeline["source"].refresh_callback = None

        # Two acquisition cycles are spaced by 31 seconds to satisfy the
        # existing historical-data pacing policy. Trigger them through the
        # source's refresh callback, without sleeping in the fixture.
        refresh_calls = [0]

        def refresh(paper_cursor):
            if controller.cycles >= 2:
                source.stop()
                return
            controller.refresh(paper_cursor)
            refresh_calls[0] += 1
            clock_value[0] += timedelta(seconds=31)
            source._wake.set()

        source.refresh_callback = refresh
        source._wake.set()

        from src.paper.costs import PaperCostPolicy
        from src.paper.ibkr_paper_recovery import AcknowledgedDelayedPaperService
        from src.paper.realtime_service import RealtimePaperConfig
        from src.paper.run_autonomous import build_real_paper_engine

        project = Path(__file__).parents[2]
        costs = PaperCostPolicy.from_json(project / "src/paper/config/topstepx_mnq_fees_2026-07.json")
        engine, adapter = build_real_paper_engine(
            root, initial_equity=50_000,
            commission_per_contract=costs.commission_per_contract_side,
            exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
            regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
        )
        engine.strategies = []

        class StopAfterFixture(AcknowledgedDelayedPaperService):
            def _process(self, bar):
                result = super()._process(bar)
                if self._bars_processed >= len(timestamps):
                    source._wake.set()
                return result

        service = StopAfterFixture(
            source=source, engine=engine, context_adapter=adapter,
            config=RealtimePaperConfig(mode="PAPER", output_dir=root, checkpoint_every_bars=1),
            calendar=calendar, cost_policy=costs,
            delivery_ledger=pipeline["delivery_ledger"], run_id="fixture-e2e",
        )
        service.run(restore=False)
        first_events = engine.logger.read_all()

        observation_rows = load_observation_ledger(root / "ibkr_observations.jsonl")
        finalized_rows = pipeline["finalization_ledger"].records
        committed = pipeline["delivery_ledger"].committed_timestamps
        checkpoint = AtomicCheckpointStore(root / "paper_checkpoint.json").load()
        assert len(observation_rows) == 9  # 3 bars across two discovery + one confirm response
        assert len(finalized_rows) == len(timestamps)
        assert len(committed) == len(timestamps)
        assert checkpoint["runtime"]["bars_processed"] == len(timestamps)
        assert checkpoint["runtime"]["last_bar"]["timestamp"] == timestamps[-1].isoformat()
        assert request_count[0] == 3
        assert controller.finalized == len(timestamps)
        assert refresh_calls[0] == 2

        # Reopen all journals and Paper state, then verify an empty catch-up
        # does not redeliver the committed exchange timestamps.
        restarted = build_cursor_aware_pipeline(
            output_dir=root, calendar=calendar,
            request_historical=lambda *_args: {
                "completed": True, "connected": True, "rows": [], "errors": []
            },
            clock=lambda: clock_value[0],
        )
        restarted_source = restarted["source"]
        restarted_source.refresh_callback = lambda _cursor: restarted_source.stop()
        restarted_source._wake.set()
        engine2, adapter2 = build_real_paper_engine(
            root, initial_equity=50_000,
            commission_per_contract=costs.commission_per_contract_side,
            exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
            regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
            recover_trailing_event=True,
        )
        engine2.strategies = []
        restarted_service = AcknowledgedDelayedPaperService(
            source=restarted_source, engine=engine2, context_adapter=adapter2,
            config=RealtimePaperConfig(mode="PAPER", output_dir=root, checkpoint_every_bars=1),
            calendar=calendar, cost_policy=costs,
            delivery_ledger=restarted["delivery_ledger"], run_id="fixture-e2e",
        )
        restarted_service.run(restore=True)
        second_events = engine2.logger.read_all()
        market_data_events = [e for e in second_events if e.event_type.value == "market_data"]
        assert len(market_data_events) == len(timestamps)
        assert [e.timestamp for e in market_data_events] == timestamps
        assert restarted["delivery_ledger"].committed_timestamps == committed
        assert restarted["delivery_ledger"].pending_after(timestamps[-1]) == []
        assert len([e for e in first_events if e.event_type.value == "market_data"]) == len(timestamps)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_bootstrap_backfill_cursor_is_durable_and_resumes_before_recent_discovery():
    from src.paper.ibkr_paper_runner import build_cursor_aware_pipeline
    from src.paper.ibkr_observation_ledger import load_observation_ledger

    root = _root()
    first = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    now = [datetime(2026, 10, 9, 0, 30, tzinfo=timezone.utc)]
    values = {"open": 20000.0, "high": 20001.0, "low": 19999.0,
              "close": 20000.5, "volume": 12.0}
    expected = [first, first + timedelta(minutes=1)]
    seen_requests = []

    def response(request, _request_id):
        seen_requests.append(request)
        if request.kind in {"recent_discovery", "bootstrap_backfill"}:
            rows = [
                {"con_id": 815824267, "timestamp": int(stamp.timestamp()), **values,
                 "arrival_utc": now[0].isoformat()} for stamp in expected
            ]
        else:
            rows = [{"con_id": 815824267, "timestamp": int(stamp.timestamp()), **values,
                     "arrival_utc": now[0].isoformat()} for stamp in expected
                    if request.kind == "bootstrap_backfill"
                    or int(stamp.timestamp()) in request.covered_pending_starts]
        return {"completed": True, "connected": True, "rows": rows, "errors": []}

    try:
        calendar, _ = _calendar_and_schedule()
        pipeline = build_cursor_aware_pipeline(
            output_dir=root, calendar=calendar, request_historical=response,
            clock=lambda: now[0],
        )
        source = pipeline["source"]
        source.bootstrap_after_timestamp = first - timedelta(minutes=1)
        source.subscribe(("MNQ",))
        first_cycle = pipeline["controller"].refresh(source._cursor)
        assert first_cycle["requests"][0]["kind"] == "recent_discovery"
        assert pipeline["cursor_store"].cursor.bootstrap_backfill_target_end_epoch_utc == int(expected[-1].timestamp()) + 60
        expected_next = int(expected[0].timestamp())
        assert pipeline["cursor_store"].cursor.bootstrap_backfill_next_start_epoch_utc == expected_next
        persisted_next = pipeline["cursor_store"].cursor.bootstrap_backfill_next_start_epoch_utc
        assert len(load_observation_ledger(root / "ibkr_observations.jsonl")) == len(expected)

        # Simulated process restart restores both pending observations and the
        # acquisition cursor; confirmation happens without repeating the
        # activation-to-now backfill window.
        now[0] += timedelta(seconds=31)
        resumed = build_cursor_aware_pipeline(
            output_dir=root, calendar=calendar, request_historical=response,
            clock=lambda: now[0],
        )
        resumed_source = resumed["source"]
        resumed_source.bootstrap_after_timestamp = first - timedelta(minutes=1)
        resumed_source.subscribe(("MNQ",))
        resumed_cycle = resumed["controller"].refresh(resumed_source._cursor)
        assert resumed_cycle["requests"][0]["kind"] == "bootstrap_backfill"
        assert resumed_cycle["requests"][1]["kind"] == "pending_confirmation"
        assert resumed["cursor_store"].cursor.bootstrap_backfill_next_start_epoch_utc == int(expected[-1].timestamp()) + 60
        assert resumed["cursor_store"].cursor.bootstrap_backfill_complete is True
        assert len(resumed["finalization_ledger"].records) == len(expected)
        assert [request.kind for request in seen_requests] == [
            "recent_discovery", "bootstrap_backfill", "pending_confirmation"
        ]
        assert len(load_observation_ledger(root / "ibkr_observations.jsonl")) == 3 * len(expected)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_ibkr_backfill_records_bounded_three_minute_leading_overlap():
    """IBKR's observed 20:30 row is retained for a 20:33–21:03 request."""
    from src.paper.ibkr_paper_runner import build_cursor_aware_pipeline
    from src.paper.ibkr_observation_ledger import load_observation_ledger

    root = _root()
    now = [datetime(2026, 10, 8, 21, 14, tzinfo=timezone.utc)]
    request_rows = [
        {"con_id": 815824267, "timestamp": int((datetime(2026, 10, 8, 20, 30, tzinfo=timezone.utc)
                                                    + timedelta(minutes=i)).timestamp()),
         "open": 20000.0, "high": 20001.0, "low": 19999.0,
         "close": 20000.5, "volume": 12.0, "arrival_utc": now[0].isoformat()}
        for i in range(30)
    ]

    def response(request, _request_id):
        assert request.kind == "bootstrap_backfill"
        assert request.end_time_utc == datetime(2026, 10, 8, 21, 3, tzinfo=timezone.utc)
        return {"completed": True, "connected": True, "rows": request_rows, "errors": []}

    try:
        calendar, _ = _calendar_and_schedule()
        pipeline = build_cursor_aware_pipeline(
            output_dir=root, calendar=calendar, request_historical=response, clock=lambda: now[0],
        )
        cursor = pipeline["cursor_store"].cursor
        cursor.bootstrap_backfill_complete = False
        cursor.bootstrap_backfill_next_start_epoch_utc = int(
            datetime(2026, 10, 8, 20, 33, tzinfo=timezone.utc).timestamp()
        )
        cursor.bootstrap_backfill_target_end_epoch_utc = int(
            datetime(2026, 10, 8, 21, 3, tzinfo=timezone.utc).timestamp()
        )
        pipeline["cursor_store"].save()

        result = pipeline["controller"].refresh()
        assert result["requests"][0]["rows"] == 30
        assert result["requests"][0]["leading_overlap_rows"] == 3
        assert result["requests"][0]["exclusive_end_rows_deferred"] == 0
        ledger = load_observation_ledger(root / "ibkr_observations.jsonl")
        assert [row["bar_start_epoch_utc"] for row in ledger] == [
            item["timestamp"] for item in request_rows
        ]
        # Advance only through observations actually returned. The request
        # window ended at 21:03, but the last observed bar starts at 20:59,
        # so the next cursor is 21:00; the verified maintenance closure is
        # skipped separately on the next refresh.
        assert pipeline["cursor_store"].cursor.bootstrap_backfill_next_start_epoch_utc == int(
            datetime(2026, 10, 8, 21, 0, tzinfo=timezone.utc).timestamp()
        )
        assert pipeline["cursor_store"].cursor.oldest_pending_bar_start_epoch_utc == request_rows[0]["timestamp"]
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_ibkr_backfill_rejects_overlap_beyond_five_minutes_and_defers_exclusive_end():
    from src.paper.ibkr_paper_runner import build_cursor_aware_pipeline
    from src.paper.ibkr_observation_ledger import load_observation_ledger

    root = _root()
    now = [datetime(2026, 10, 8, 21, 14, tzinfo=timezone.utc)]
    start = datetime(2026, 10, 8, 20, 33, tzinfo=timezone.utc)
    end = datetime(2026, 10, 8, 21, 3, tzinfo=timezone.utc)
    base = {"con_id": 815824267, "open": 20000.0, "high": 20001.0,
            "low": 19999.0, "close": 20000.5, "volume": 12.0,
            "arrival_utc": now[0].isoformat()}
    mode = {"far": True}

    def response(request, _request_id):
        if mode["far"]:
            stamp = start - timedelta(minutes=6)
            rows = [{**base, "timestamp": int(stamp.timestamp())}]
        else:
            rows = [{**base, "timestamp": int(end.timestamp())}]
        return {"completed": True, "connected": True, "rows": rows, "errors": []}

    try:
        calendar, _ = _calendar_and_schedule()
        pipeline = build_cursor_aware_pipeline(
            output_dir=root, calendar=calendar, request_historical=response, clock=lambda: now[0],
        )
        cursor = pipeline["cursor_store"].cursor
        cursor.bootstrap_backfill_complete = False
        cursor.bootstrap_backfill_next_start_epoch_utc = int(start.timestamp())
        cursor.bootstrap_backfill_target_end_epoch_utc = int(end.timestamp())
        pipeline["cursor_store"].save()
        import pytest
        with pytest.raises(RuntimeError, match="exceeded bounded request overlap"):
            pipeline["controller"].refresh()
        assert not (root / "ibkr_observations.jsonl").exists()

        mode["far"] = False
        now[0] += timedelta(seconds=31)
        base["arrival_utc"] = now[0].isoformat()
        result = pipeline["controller"].refresh()
        assert result["requests"][0]["exclusive_end_rows_deferred"] == 1
        observed = load_observation_ledger(root / "ibkr_observations.jsonl")
        assert len(observed) == 1
        assert observed[0]["eligible_for_finalization"] is False
        assert pipeline["finalization_ledger"].records == {}
    finally:
        if root.exists():
            shutil.rmtree(root)
