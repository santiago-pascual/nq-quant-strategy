"""PAPER-only realtime entrypoint; defaults to deterministic replay source."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import sys

import pandas as pd

from src.paper.autonomous_runner import load_canonical_raw_mnq
from src.paper.context_adapter import PaperMarketContextAdapter
from src.paper.market_context import build_causal_context_features
from src.paper.realtime_market_data import ReplayMarketDataSource, mark_replay_session_final_bars
from src.paper.realtime_service import RealtimePaperConfig, RealtimePaperService
from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.costs import PaperCostPolicy
from src.paper.analytics import PaperAnalyticsReader
from src.paper.run_autonomous import build_real_paper_engine


def _run_with_graceful_signals(service: RealtimePaperService, *, restore: bool) -> None:
    previous_handlers = {}

    def _request_graceful_stop(signum, _frame):
        service.request_stop(reason=f"signal_{signum}")

    handled_signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        handled_signals.append(signal.SIGTERM)
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)
    try:
        for signum in handled_signals:
            previous_handlers[signum] = signal.signal(signum, _request_graceful_stop)
        service.run(restore=restore)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _utc(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MNQ realtime PAPER (no live orders).")
    parser.add_argument("--command", choices=("run", "validate", "stop", "status", "events", "verify-checkpoint", "shadow-report", "daily-report", "analytics", "backup-database", "api"), default="run")
    parser.add_argument("--mode", choices=("PAPER",), help="Safety mode; PAPER is the only supported mode.")
    parser.add_argument("--replay-start", help="First completed bar timestamp, UTC or ISO aware.")
    parser.add_argument("--replay-end", help="Optional final timestamp for deterministic replay.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Restore the latest valid checkpoint and continue after its last bar.")
    parser.add_argument("--checkpoint-every-bars", type=int, default=500)
    parser.add_argument("--queue-size", type=int, default=2000)
    parser.add_argument("--initial-equity", type=float, default=50_000.0)
    parser.add_argument("--price-offset", type=float, default=0.0)
    parser.add_argument("--count", type=int, default=25, help="Number of recent events to display.")
    parser.add_argument("--calendar-snapshot", type=Path, help="Reviewed CME schedule snapshot JSON.")
    parser.add_argument("--cost-config", type=Path, help="Explicit versioned realtime Paper cost policy JSON.")
    parser.add_argument("--date", help="New York session date for --command daily-report (YYYY-MM-DD).")
    parser.add_argument("--backup-path", type=Path, help="Destination for a consistent SQLite analytics backup.")
    parser.add_argument("--api-host", default="127.0.0.1", help="Monitoring API bind host; API always requires a bearer token.")
    parser.add_argument("--api-port", type=int, default=8765, help="Monitoring API TCP port.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "api":
        import os
        from src.paper.monitoring_api import create_monitoring_server
        token = os.environ.get("PAPER_MONITORING_TOKEN", "")
        server = create_monitoring_server(args.output_dir, token=token,
                                          host=args.api_host, port=args.api_port)
        print(f"Read-only Paper monitoring API listening on {args.api_host}:{server.server_port}", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    if args.command == "stop":
        status_path = args.output_dir / "status.json"
        if not status_path.exists():
            raise SystemExit("No active PAPER session status was found; no stop request was written")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = status.get("system", {}).get("state")
        if state not in {"RUNNING", "PAUSED_REFIT", "DEGRADED"}:
            raise SystemExit(f"PAPER session is not active (state={state}); no stop request was written")
        temporary = args.output_dir / "stop.request.tmp"
        temporary.write_text("stop\n", encoding="utf-8")
        temporary.replace(args.output_dir / "stop.request")
        print(f"Graceful stop requested for PAPER session in {args.output_dir}")
        return 0
    if args.command == "status":
        path = args.output_dir / "status.json"
        if not path.exists():
            raise SystemExit(f"No status snapshot exists: {path}")
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.command in {"analytics", "daily-report"}:
        db_path = args.output_dir / "paper_analytics.sqlite3"
        if not db_path.exists():
            raise SystemExit(f"Paper analytics database is missing: {db_path}")
        reader = PaperAnalyticsReader(str(db_path))
        try:
            if args.command == "analytics":
                report = {"status":reader.status(),"positions":reader.positions(),
                          "portfolio_statistics":reader.portfolio_statistics(),
                          "strategy_statistics":reader.strategy_statistics(),
                          "performance_by_exit_day":reader.performance_series("day"),
                          "performance_by_exit_week":reader.performance_series("week"),
                          "performance_by_exit_month":reader.performance_series("month"),
                          "recent_trades":reader.recent_trades(20),"hmm":reader.hmm_state(),
                          "risk":reader.risk_status(),"events":reader.event_timeline(50),
                          "daily_reports":reader.daily_reports(),"shadow_parity":reader.shadow_parity()}
            else:
                from datetime import date
                if not args.date:
                    raise SystemExit("--date YYYY-MM-DD is required for --command daily-report")
                from src.paper.analytics_db import PaperAnalyticsStore
                store = PaperAnalyticsStore(db_path)
                report = store.write_daily_report(date.fromisoformat(args.date), args.output_dir)
                store.close()
        finally:
            reader.close()
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "backup-database":
        db_path = args.output_dir / "paper_analytics.sqlite3"
        if not db_path.exists():
            raise SystemExit(f"Paper analytics database is missing: {db_path}")
        if not args.backup_path:
            raise SystemExit("--backup-path is required for --command backup-database")
        from src.paper.analytics_db import PaperAnalyticsStore
        store = PaperAnalyticsStore(db_path)
        try:
            backup_path = store.backup_to(args.backup_path)
            print(json.dumps({"backup": str(backup_path), "integrity_check": "ok"}, indent=2))
        finally:
            store.close()
        return 0
    if args.command == "events":
        path = args.output_dir / "events.jsonl"
        if not path.exists():
            raise SystemExit(f"No event log exists: {path}")
        tail = deque(maxlen=max(1, args.count))
        with path.open("r", encoding="utf-8") as handle:
            tail.extend(json.loads(line) for line in handle if line.strip())
        for row in tail:
            print(json.dumps(row, sort_keys=True))
        return 0
    if args.command == "verify-checkpoint":
        from src.paper.realtime_checkpoint import AtomicCheckpointStore, runtime_identity
        payload = AtomicCheckpointStore(args.output_dir / "paper_checkpoint.json").load()
        if payload.get("mode") != "PAPER" or payload.get("system", {}).get("runtime_identity") != runtime_identity():
            raise SystemExit("Checkpoint is not valid for the current PAPER runtime")
        expected_calendar = (
            CMECalendarSnapshot.from_json(args.calendar_snapshot).identity
            if args.calendar_snapshot else None
        )
        if payload.get("system", {}).get("calendar_identity") != expected_calendar:
            raise SystemExit("Checkpoint CME calendar snapshot differs from the requested runtime calendar")
        expected_cost = PaperCostPolicy.from_json(args.cost_config).identity if args.cost_config else None
        if payload.get("system", {}).get("cost_profile_identity") != expected_cost:
            raise SystemExit("Checkpoint cost profile differs from the requested runtime profile")
        last = payload.get("runtime", {}).get("last_bar")
        print(json.dumps({"valid": True, "mode": "PAPER", "last_processed_bar": last.get("timestamp") if last else None}, indent=2))
        return 0
    if args.command == "shadow-report":
        from src.paper.shadow_replay import run_shadow_replay
        report = run_shadow_replay(
            args.output_dir, initial_equity=args.initial_equity,
            price_offset=args.price_offset,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["status"] == "GREEN" else 2

    if args.command == "validate":
        if args.mode != "PAPER":
            raise SystemExit("--mode PAPER is required; no live trading mode exists")
        if args.replay_start:
            start = _utc(args.replay_start)
            end = _utc(args.replay_end)
            if end is not None and end < start:
                raise SystemExit("--replay-end must be at or after --replay-start")
        cost_info = {"configured":False}
        if args.cost_config:
            policy=PaperCostPolicy.from_json(args.cost_config)
            cost_info={"configured":True,"profile_id":policy.profile_id,
                       "identity":policy.identity,
                       "round_turn_per_contract":policy.round_turn_per_contract,
                       "slippage_ticks":policy.artificial_slippage_ticks}
        calendar_info = {"configured": False, "version": None, "source": None}
        if args.calendar_snapshot:
            calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(args.calendar_snapshot))
            calendar_info = {"configured": True, "version": calendar.version,
                             "source": calendar.snapshot.source,
                             "coverage_start": calendar.snapshot.coverage_start.isoformat(),
                             "coverage_end": calendar.snapshot.coverage_end.isoformat()}
        print(json.dumps({
            "valid": True, "mode": "PAPER", "source": "deterministic replay",
            "real_market_data_provider": "not configured",
            "cme_calendar": calendar_info,
            "cost_policy": cost_info,
            "output_dir": str(args.output_dir),
        }, indent=2))
        return 0

    if args.mode != "PAPER":
        raise SystemExit("--mode PAPER is required; no live trading mode exists")
    if not args.replay_start:
        raise SystemExit("--replay-start is required for the deterministic replay command")
    if not args.cost_config:
        raise SystemExit("--cost-config is required; realtime Paper never assumes fee values")
    cost_policy=PaperCostPolicy.from_json(args.cost_config)
    if cost_policy.symbol != "MNQ":
        raise SystemExit(f"Cost profile is for {cost_policy.symbol}, expected MNQ")
    if cost_policy.artificial_slippage_ticks != 0:
        raise SystemExit("Realtime Paper requires artificial_slippage_ticks=0")
    if args.price_offset != 0:
        raise SystemExit("Realtime Paper artificial price offset must be zero")
    start = _utc(args.replay_start)
    end = _utc(args.replay_end)
    if end is not None and end < start:
        raise SystemExit("--replay-end must be at or after --replay-start")
    print("MODE: PAPER — simulated fills only; no real-money order routing.", flush=True)

    output_dir = args.output_dir
    events_path = output_dir / "events.jsonl"
    checkpoint_path = output_dir / "paper_checkpoint.json"
    if not args.resume and (events_path.exists() or checkpoint_path.exists()):
        raise SystemExit("Output directory already contains a Paper run; choose a new directory or use --resume")
    raw = load_canonical_raw_mnq()
    # The available internal validation data is frozen at this timestamp.
    raw = raw.loc[raw["timestamp"] <= pd.Timestamp("2026-08-26 23:59:00+00:00")].copy()
    raw = mark_replay_session_final_bars(raw)
    if end is not None:
        raw = raw.loc[raw["timestamp"] <= end].copy()
    history = raw.loc[raw["timestamp"] < start].copy()
    stream_frame = raw.loc[raw["timestamp"] >= start].copy()
    if history.empty or stream_frame.empty:
        raise SystemExit("The selected replay requires prior warmup history and at least one stream bar.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cost_profile.json").write_text(
        json.dumps(cost_policy.__dict__,indent=2),encoding="utf-8")
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(args.calendar_snapshot)) if args.calendar_snapshot else None
    engine, context_adapter = build_real_paper_engine(
        output_dir,
        initial_equity=args.initial_equity,
        commission_per_contract=cost_policy.commission_per_contract_side,
        exchange_fee_per_contract=cost_policy.exchange_fee_per_contract_side,
        regulatory_fee_per_contract=cost_policy.regulatory_fee_per_contract_side,
        price_offset=0.0,
        recover_trailing_event=args.resume,
    )
    if not args.resume:
        # Causal historical bootstrap emits no Paper strategy/order events.
        features = build_causal_context_features(history)
        directional_ready = features["log_return"].rolling(30).count() == 30
        features.loc[~directional_ready, "close_location_30"] = float("nan")
        context_adapter.context.bootstrap_causal_history(
            history, precomputed_features=features
        )
    if args.resume:
        from src.paper.realtime_checkpoint import AtomicCheckpointStore
        payload = AtomicCheckpointStore(checkpoint_path).load()
        last = pd.Timestamp(payload["runtime"]["last_bar"]["timestamp"])
        stream_frame = stream_frame.loc[stream_frame["timestamp"] > last].copy()
    source = ReplayMarketDataSource(stream_frame.to_dict(orient="records"))
    config = RealtimePaperConfig(
        mode=args.mode,
        output_dir=output_dir,
        checkpoint_every_bars=args.checkpoint_every_bars,
        feed_queue_size=args.queue_size,
    )
    service = RealtimePaperService(
        source=source,
        engine=engine,
        context_adapter=context_adapter,
        config=config,
        calendar=calendar,
        cost_policy=cost_policy,
    )
    # The service handles the active bar, checkpoint, and report flush.
    _run_with_graceful_signals(service, restore=args.resume)
    print(f"PAPER status: {output_dir / 'status.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
