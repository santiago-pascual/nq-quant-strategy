"""Realtime orchestration for the existing causal Paper engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from queue import Queue
import threading
import time
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
import numpy as np

from src.paper.logger import PaperEvent, PaperEventLogger, PaperEventType
from src.paper.realtime_checkpoint import (
    AtomicCheckpointStore,
    capture_engine_state,
    restore_engine_state,
    runtime_identity,
)
from src.paper.cme_calendar import CMETradingCalendar, CalendarUnavailable
from src.paper.realtime_market_data import CanonicalBar, MarketDataSource
from src.paper.analytics_db import PaperAnalyticsStore
from src.paper.realtime_checkpoint import engine_fingerprint
from src.paper.single_writer import PaperWriterLock
from src.paper.systemd_notify import notify_systemd


@dataclass(frozen=True)
class RealtimePaperConfig:
    mode: str
    output_dir: Path
    checkpoint_every_bars: int = 500
    feed_queue_size: int = 2000
    stale_after_seconds: float = 90.0
    symbol: str = "MNQ"

    def __post_init__(self) -> None:
        if self.mode != "PAPER":
            raise ValueError("Realtime runner supports PAPER mode only")
        if self.checkpoint_every_bars <= 0 or self.feed_queue_size <= 0:
            raise ValueError("checkpoint cadence and feed queue size must be positive")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")


_SENTINEL = object()


class RealtimePaperService:
    """Consume completed canonical bars through the validated Paper engine.

    The source thread may continue receiving while a scheduled causal HMM fit
    pauses strategy evaluation. A bounded queue applies backpressure; overflow
    is never silently dropped.
    """

    def __init__(self, *, source: MarketDataSource, engine: Any,
                 context_adapter: Any, config: RealtimePaperConfig,
                 run_id: str | None = None,
                 calendar: CMETradingCalendar | None = None,
                 cost_policy: Any | None = None) -> None:
        self.source = source
        self.engine = engine
        self.context_adapter = context_adapter
        self.calendar = calendar
        self.cost_policy = cost_policy
        if getattr(source, "name", "") != "deterministic_replay":
            if cost_policy is None:
                raise ValueError("non-replay Paper sources require an explicit fee profile")
            if (cost_policy.symbol != config.symbol or cost_policy.artificial_slippage_ticks != 0
                    or engine.config.price_offset != 0):
                raise ValueError("live Paper requires a matching symbol fee profile and zero artificial slippage")
            if not (
                abs(engine.config.commission_per_contract-cost_policy.commission_per_contract_side) < 1e-12
                and abs(engine.config.exchange_fee_per_contract-cost_policy.exchange_fee_per_contract_side) < 1e-12
                and abs(engine.config.regulatory_fee_per_contract-cost_policy.regulatory_fee_per_contract_side) < 1e-12
            ):
                raise ValueError("engine fill costs differ from the explicit Paper cost profile")
        self.config = config
        self.run_id = run_id or str(uuid4())
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.config.output_dir / "status.json"
        self.checkpoint_store = AtomicCheckpointStore(
            self.config.output_dir / "paper_checkpoint.json"
        )
        self.bootstrap_store = AtomicCheckpointStore(
            self.config.output_dir / "bootstrap_checkpoint.json"
        )
        self.refit_ledger_path = self.config.output_dir / "hmm_refits.jsonl"
        self._queue: Queue[Any] = Queue(maxsize=config.feed_queue_size)
        self._stop_requested = threading.Event()
        self._producer: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._state = "STOPPED"
        self._started_monotonic: float | None = None
        self._last_bar: CanonicalBar | None = None
        self._last_bar_digest: str | None = None
        self._bars_processed = 0
        self._last_processing_seconds: float | None = None
        self._max_processing_seconds = 0.0
        self._last_feed_walltime: float | None = None
        self._last_feed_connected: bool | None = None
        self._feed_stale = False
        self._bar_events: list[PaperEvent] | None = None
        self._metrics: dict[str, dict[str, float]] = defaultdict(
            lambda: {"candidates": 0, "accepted": 0, "rejected": 0,
                     "risk_approved": 0, "risk_rejected": 0,
                     "conflict_rejected": 0, "execution_rejected": 0,
                     "closed_trades": 0, "cumulative_r": 0.0,
                     "wins": 0, "gross_win_r": 0.0, "gross_loss_r": 0.0,
                     "peak_equity": 0.0, "max_drawdown": 0.0}
        )
        self._risk_per_contract: dict[str, float] = {}
        self._daily_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"candidates": 0, "accepted": 0, "rejected": 0,
                     "risk_approved": 0, "risk_rejected": 0,
                     "conflict_rejected": 0, "execution_rejected": 0}
        )
        self._bar_candidate_pending: set[str] = set()
        self._daily_date: str | None = None
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._previous_equity = engine.account_equity
        self._equity_peak = engine.account_equity
        self._portfolio_max_drawdown = 0.0
        self._lock = threading.RLock()
        self._status_lock = threading.Lock()
        self.logger.set_event_context(run_id=self.run_id, severity="INFO", symbol=self.config.symbol)
        engine.logger.subscribe(self._observe_event)
        self.analytics = PaperAnalyticsStore(
            self.config.output_dir / "paper_analytics.sqlite3",
            point_value=engine.config.point_value, cost_policy=cost_policy, calendar=calendar,
        )
        self.analytics.backfill_jsonl(self.logger.read_all())
        self.logger.subscribe(self.analytics.ingest_event)
        self.analytics.record_configuration(
            config_kind="paper_engine", version="paper-runtime-v1",
            identity=engine_fingerprint(engine),
            source={"engine":engine.config.__dict__,"risk_limits":engine.risk.limits.__dict__,
                    "strategies":[strategy.name for strategy in engine.strategies],
                    "runtime_identity":runtime_identity()},
        )
        if calendar is not None:
            self.analytics.record_configuration(
                config_kind="cme_calendar",version=calendar.version,
                identity=calendar.snapshot.identity,source=calendar.snapshot.to_mapping(),
            )
        if cost_policy is not None:
            self.analytics.record_configuration(
                config_kind="paper_cost_policy", version=cost_policy.profile_id,
                identity=cost_policy.identity, source=cost_policy.__dict__,
            )

    @property
    def logger(self) -> PaperEventLogger:
        return self.engine.logger

    @staticmethod
    def _digest(bar: CanonicalBar) -> str:
        body = json.dumps({
            "symbol": bar.symbol, "timestamp": bar.timestamp.isoformat(),
            "open": bar.open, "high": bar.high, "low": bar.low,
            "close": bar.close, "volume": bar.volume,
            "contract_symbol":bar.contract_symbol,"bid":bar.bid,"ask":bar.ask,
            "provider_timestamp":bar.provider_timestamp.isoformat() if bar.provider_timestamp else None,
        }, sort_keys=True, separators=(",", ":"))
        return sha256(body.encode()).hexdigest()

    def restore(self) -> None:
        """Restore a validated checkpoint before connecting the Paper engine."""
        payload = self.checkpoint_store.load()
        if payload.get("mode") != "PAPER" or payload.get("symbol") != self.config.symbol:
            raise ValueError("Checkpoint mode or symbol differs from runtime config")
        if payload.get("system", {}).get("runtime_identity") != runtime_identity():
            raise ValueError("Checkpoint code or numerical dependency identity differs")
        expected_calendar = self.calendar.snapshot.identity if self.calendar else None
        if payload.get("system", {}).get("calendar_identity") != expected_calendar:
            raise ValueError("Checkpoint CME calendar snapshot differs from runtime")
        if payload.get("system", {}).get("cost_profile_identity") != (self.cost_policy.identity if self.cost_policy else None):
            raise ValueError("Checkpoint Paper cost profile differs from runtime")
        self.context_adapter.context.load_state_dict(payload["context"])
        restore_engine_state(self.engine, payload["engine"])
        runtime = payload["runtime"]
        self.run_id = str(runtime["run_id"])
        self._last_bar = CanonicalBar.from_mapping(runtime["last_bar"]) if runtime.get("last_bar") else None
        self._last_bar_digest = str(runtime["last_bar_digest"]) if runtime.get("last_bar_digest") else None
        self._bars_processed = int(runtime["bars_processed"])
        self._metrics = defaultdict(lambda: {"candidates": 0, "accepted": 0,
            "rejected": 0, "risk_approved": 0, "risk_rejected": 0,
            "conflict_rejected": 0, "execution_rejected": 0,
            "closed_trades": 0, "cumulative_r": 0.0,
            "wins": 0, "gross_win_r": 0.0, "gross_loss_r": 0.0,
            "peak_equity": 0.0, "max_drawdown": 0.0}, runtime.get("metrics", {}))
        self._risk_per_contract = dict(runtime.get("risk_per_contract", {}))
        self._equity_peak = float(runtime.get("equity_peak", self.engine.account_equity))
        self._portfolio_max_drawdown = float(runtime.get("portfolio_max_drawdown", 0.0))
        self._daily_date = runtime.get("daily_date")
        self._daily_counts = defaultdict(
            lambda: {"candidates": 0, "accepted": 0, "rejected": 0,
                     "risk_approved": 0, "risk_rejected": 0,
                     "conflict_rejected": 0, "execution_rejected": 0},
            runtime.get("daily_counts", {}),
        )
        self.logger.set_event_context(run_id=self.run_id, severity="INFO", symbol=self.config.symbol)
        self.logger.append(PaperEventType.SYSTEM_RECOVERED, {
            "run_id": self.run_id, "severity": "INFO",
            "symbol": self.config.symbol,
            "last_processed_bar": self._last_bar.timestamp.isoformat(),
        }, timestamp=datetime.now(timezone.utc))
        self._write_status()

    def save_bootstrap_checkpoint(self) -> None:
        """Persist the pre-stream causal history seed for restart and shadow replay."""
        payload = {
            "mode": "PAPER", "symbol": self.config.symbol,
            "system": {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_identity": runtime_identity(),
                "calendar_identity": self.calendar.snapshot.identity if self.calendar else None,
                "calendar_snapshot": self.calendar.snapshot.to_mapping() if self.calendar else None,
                "cost_profile_id": self.cost_policy.profile_id if self.cost_policy else None,
                "cost_profile_identity": self.cost_policy.identity if self.cost_policy else None,
            },
            "context": self.context_adapter.context.state_dict(),
            "engine": capture_engine_state(self.engine),
            "runtime": {
                "run_id": self.run_id, "last_bar": None,
                "last_bar_digest": None, "bars_processed": 0,
                "metrics": {}, "risk_per_contract": {},
                "equity_peak": self.engine.account_equity,
                "portfolio_max_drawdown": 0.0, "daily_date": None,
                "daily_counts": {},
            },
        }
        self.bootstrap_store.save(payload)
        self._record_bootstrap_refits()

    def _record_bootstrap_refits(self) -> None:
        provider = self.context_adapter.context._raw_hmm_provider
        recorded: set[tuple[str, int]] = set()
        if self.refit_ledger_path.exists():
            with self.refit_ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    recorded.add((str(row.get("stream")), int(row.get("model_version", 0))))
        for name, stream in (("MR", provider.mr), ("S2R", provider.s2r)):
            for fit in stream.refit_events:
                identity = (name, int(fit.get("model_version", 0)))
                if identity in recorded:
                    continue
                row = {
                    **fit, "stream": name, "phase": "causal_bootstrap",
                    "model_hash": fit.get("new_model_hash"),
                    "wall_duration_seconds": fit.get("fit_duration_seconds"),
                    "raw_state_at_activation": fit.get("first_raw_state"),
                }
                self._append_refit_ledger(row)
                recorded.add(identity)

    def _observe_event(self, event: PaperEvent) -> None:
        if self._bar_events is not None:
            self._bar_events.append(event)
        payload = event.payload
        context = payload.get("context", {})
        strategy = payload.get("strategy_name")
        if not strategy and isinstance(context, Mapping):
            strategy = context.get("strategy_name")
        if not strategy:
            return
        metric = self._metrics[strategy]
        if event.event_type is PaperEventType.STRATEGY_DECISION:
            if payload.get("action", "").lower() == "enter":
                metric["candidates"] += 1
                self._daily_counts[strategy]["candidates"] += 1
                self._bar_candidate_pending.add(strategy)
        elif event.event_type is PaperEventType.RISK_DECISION:
            if payload.get("approved"):
                metric["risk_approved"] += 1
                self._daily_counts[strategy]["risk_approved"] += 1
                self._risk_per_contract[strategy] = float(payload.get("risk_per_contract", 0.0))
            else:
                metric["risk_rejected"] += 1
                metric["rejected"] += 1
                self._daily_counts[strategy]["risk_rejected"] += 1
                self._daily_counts[strategy]["rejected"] += 1
                self._bar_candidate_pending.discard(strategy)
        elif event.event_type is PaperEventType.ORDER_CREATED:
            if strategy in self._bar_candidate_pending:
                metric["accepted"] += 1
                self._daily_counts[strategy]["accepted"] += 1
                self._bar_candidate_pending.discard(strategy)
        elif event.event_type is PaperEventType.ERROR:
            message = str(payload.get("message", ""))
            if strategy in self._bar_candidate_pending and "Portfolio conflict rejected entry" in message:
                metric["conflict_rejected"] += 1
                metric["rejected"] += 1
                self._daily_counts[strategy]["conflict_rejected"] += 1
                self._daily_counts[strategy]["rejected"] += 1
                self._bar_candidate_pending.discard(strategy)
        elif event.event_type is PaperEventType.ORDER_REJECTED:
            metric["execution_rejected"] += 1
            metric["rejected"] += 1
            self._daily_counts[strategy]["execution_rejected"] += 1
            self._daily_counts[strategy]["rejected"] += 1
            self._bar_candidate_pending.discard(strategy)
        elif event.event_type is PaperEventType.POSITION_CLOSED:
            risk = self._risk_per_contract.get(strategy, 0.0)
            quantity = int(payload.get("quantity", 0))
            entry, exit_ = float(payload.get("entry_price", 0)), float(payload.get("exit_price", 0))
            side = str(payload.get("side", "")).lower()
            points = exit_ - entry if side in {"long", "strategysignal.long"} else entry - exit_
            pnl = points * quantity * float(self.engine.config.point_value)
            net_pnl = float(payload.get("net_pnl", pnl))
            r = payload.get("realized_r")
            if r is None:
                total_costs = float(payload.get("total_costs", 0.0))
                r = (pnl-total_costs) / (risk * quantity) if risk > 0 and quantity else 0.0
            metric["closed_trades"] += 1
            metric["cumulative_r"] += r
            metric["gross_pnl"] = metric.get("gross_pnl", 0.0) + pnl
            metric["net_pnl"] = metric.get("net_pnl", 0.0) + net_pnl
            metric["total_costs"] = metric.get("total_costs", 0.0) + float(payload.get("total_costs", 0.0))
            metric["peak_equity"] = max(metric["peak_equity"], metric["cumulative_r"])
            metric["max_drawdown"] = min(
                metric["max_drawdown"],
                metric["cumulative_r"] - metric["peak_equity"],
            )
            if r > 0:
                metric["wins"] += 1
                metric["gross_win_r"] += r
            elif r < 0:
                metric["gross_loss_r"] += abs(r)

    def _emit(self, event_type: PaperEventType, payload: Mapping[str, Any], *,
              timestamp: datetime | None = None, severity: str = "INFO") -> None:
        document = {"run_id": self.run_id, "severity": severity,
                    "symbol": self.config.symbol, **dict(payload)}
        self.logger.append(event_type, document, timestamp=timestamp or datetime.now(timezone.utc))

    def _put(self, value: Any) -> bool:
        while not self._stop_requested.is_set() or value is _SENTINEL:
            try:
                self._queue.put(value, timeout=0.25)
                return True
            except Exception:
                if value is _SENTINEL:
                    continue
                health = self.source.health()
                if not health.get("connected", True):
                    try:
                        self._queue.put(RuntimeError("market data source disconnected"), timeout=0.25)
                    except Exception:
                        pass
                    return False
        return False

    def _produce(self) -> None:
        try:
            for bar in self.source.bars():
                self._last_feed_walltime = time.time()
                if not self._put(bar):
                    break
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(_SENTINEL)

    def _refit_due(self, bar: CanonicalBar) -> list[tuple[str, Any]]:
        provider = self.context_adapter.context._raw_hmm_provider
        due: list[tuple[str, Any]] = []
        for name, stream in (("MR", provider.mr), ("S2R", provider.s2r)):
            scheduled = stream.next_refit_timestamp is not None and pd.Timestamp(bar.timestamp) >= stream.next_refit_timestamp
            initial = stream.model is None and (
                stream.schedule == "expanding"
                or (
                    stream.initial_fit_timestamp is not None
                    and stream._initial_fit_due(pd.Timestamp(bar.timestamp))
                )
            )
            if scheduled or initial:
                if scheduled and stream.model is not None:
                    start, stop = stream._training_bounds(pd.Timestamp(bar.timestamp))
                    matrix = stream._training_matrix(start, stop)
                    valid_rows = int(np.isfinite(matrix).all(axis=1).sum())
                    if valid_rows < stream.config.min_train_valid:
                        self._state = "ERROR"
                        self._emit(PaperEventType.HMM_REFIT_FAILED, {
                            "stream": name,
                            "error_type": "InsufficientTrainingRows",
                            "valid_training_rows": valid_rows,
                            "required_training_rows": stream.config.min_train_valid,
                            "first_live_timestamp": bar.timestamp.isoformat(),
                            "training_end_exclusive": bar.timestamp.isoformat(),
                            "message": "scheduled refit cannot be completed; refusing to continue with the stale model",
                        }, timestamp=bar.timestamp, severity="CRITICAL")
                        self._write_status()
                        raise RuntimeError(f"{name} scheduled refit has insufficient valid training rows")
                if stream._history.size >= stream.config.min_train_valid:
                    due.append((name, stream))
        return due

    def _append_refit_ledger(self, row: Mapping[str, Any]) -> None:
        with self.refit_ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _process(self, bar: CanonicalBar) -> bool:
        if bar.symbol != self.config.symbol:
            self._emit(PaperEventType.SYSTEM_WARNING, {"message": "unexpected_symbol", "received_symbol": bar.symbol}, severity="WARNING")
            return False
        digest = self._digest(bar)
        if self._last_bar is not None:
            if bar.timestamp == self._last_bar.timestamp:
                self.logger.set_idempotency_scope(f"{self.run_id}:{bar.timestamp.isoformat()}")
                self._emit(PaperEventType.DUPLICATE_BAR, {"timestamp": bar.timestamp.isoformat(), "same_payload": digest == self._last_bar_digest}, timestamp=bar.timestamp)
                self.logger.set_idempotency_scope(None)
                if digest != self._last_bar_digest:
                    raise ValueError("duplicate timestamp arrived with different OHLCV")
                return False
            if bar.timestamp < self._last_bar.timestamp:
                self.logger.set_idempotency_scope(f"{self.run_id}:{bar.timestamp.isoformat()}")
                self._emit(PaperEventType.OUT_OF_ORDER_BAR, {"timestamp": bar.timestamp.isoformat(), "last_processed": self._last_bar.timestamp.isoformat()}, timestamp=bar.timestamp, severity="ERROR")
                self.logger.set_idempotency_scope(None)
                raise ValueError("out-of-order market data bar")
            previous_contract = self._last_bar.contract_symbol
            if previous_contract and bar.contract_symbol and previous_contract != bar.contract_symbol:
                self._emit(PaperEventType.CONTRACT_ROLL, {
                    "previous_contract": previous_contract,
                    "new_contract": bar.contract_symbol,
                    "timestamp": bar.timestamp.isoformat(),
                    "provider": bar.provider,
                }, timestamp=bar.timestamp, severity="INFO")
            self._backfill_gap_before(bar)

        self.logger.set_idempotency_scope(f"{self.run_id}:{bar.timestamp.isoformat()}")

        local_day = pd.Timestamp(bar.timestamp).tz_convert("America/New_York").date().isoformat()
        if self._daily_date != local_day:
            self._daily_date = local_day
            self._daily_counts = defaultdict(
                lambda: {"candidates": 0, "accepted": 0, "rejected": 0,
                         "risk_approved": 0, "risk_rejected": 0,
                         "conflict_rejected": 0, "execution_rejected": 0}
            )

        source_health = dict(self.source.health())
        if source_health.get("stale"):
            self._state = "DEGRADED"
            self._emit(PaperEventType.FEED_STALE, {"health": source_health}, timestamp=bar.timestamp, severity="WARNING")
        market_data = bar.to_market_data()
        local = pd.Timestamp(bar.timestamp).tz_convert("America/New_York")
        if self.calendar is not None:
            try:
                in_rth = self.calendar.is_rth(bar.timestamp)
                session_final = self.calendar.is_final_rth_bar(bar.timestamp) if in_rth else False
                market_data["calendar_source"] = self.calendar.snapshot.source
                market_data["calendar_version"] = self.calendar.version
            except CalendarUnavailable as exc:
                self._state = "ERROR"
                self._emit(PaperEventType.SYSTEM_CRITICAL, {
                    "message": "authoritative CME calendar coverage unavailable",
                    "error": str(exc), "timestamp": bar.timestamp.isoformat(),
                }, timestamp=bar.timestamp, severity="CRITICAL")
                self._write_status()
                self.logger.set_idempotency_scope(None)
                raise RuntimeError("CME calendar unavailable; refusing strategy processing") from exc
            market_data["is_final_rth_bar"] = session_final
        else:
            is_normal_close = local.hour * 60 + local.minute == 15 * 60 + 59
            if (self.source.name != "deterministic_replay" and bar.is_session_final is None
                    and 9 * 60 + 30 <= local.hour * 60 + local.minute < 16 * 60):
                self._state = "ERROR"
                self._emit(PaperEventType.SYSTEM_CRITICAL, {
                    "message": "live bar has no authoritative session-final marker or CME calendar",
                    "timestamp": bar.timestamp.isoformat(),
                }, timestamp=bar.timestamp, severity="CRITICAL")
                self._write_status()
                self.logger.set_idempotency_scope(None)
                raise RuntimeError("live RTH processing requires CME calendar session metadata")
            market_data["is_final_rth_bar"] = bool(
                bar.is_session_final if bar.is_session_final is not None else is_normal_close
            )
        due = self._refit_due(bar)
        before = {name: len(stream.refit_events) for name, stream in due}
        before_hash = {name: stream.model_hash for name, stream in due}
        if due:
            self._state = "PAUSED_REFIT"
            for name, stream in due:
                start, stop = stream._training_bounds(pd.Timestamp(bar.timestamp))
                last_train = (pd.Timestamp(stream._training_timestamps(start, stop)[-1], tz="UTC").isoformat() if stop > start else None)
                self._emit(PaperEventType.HMM_REFIT_STARTED, {
                    "stream": name, "previous_model_hash": before_hash[name],
                    "training_observations": stop - start,
                    "training_start": pd.Timestamp(stream._history.timestamps_ns[start], tz="UTC").isoformat() if stop > start else None,
                    "training_last_timestamp": last_train,
                    "first_live_timestamp": bar.timestamp.isoformat(),
                    "activation_rule": "fit from rows strictly before activation bar; pause strategy evaluation until fit completes",
                }, timestamp=bar.timestamp)
            self._write_status()

        started = time.perf_counter()
        self._bar_events = []
        self._bar_candidate_pending.clear()
        try:
            step = self.engine.process_bar(market_data)
        except BaseException as exc:
            if due:
                self._state = "ERROR"
                self._errors.append(str(exc))
                self._emit(PaperEventType.HMM_REFIT_FAILED, {
                    "streams": [name for name, _ in due],
                    "error_type": type(exc).__name__, "message": str(exc),
                }, timestamp=bar.timestamp, severity="ERROR")
            else:
                self._state = "ERROR"
                self._errors.append(str(exc))
                self._emit(PaperEventType.SYSTEM_ERROR, {"error_type": type(exc).__name__, "message": str(exc)}, timestamp=bar.timestamp, severity="ERROR")
            self._bar_events = None
            raise
        elapsed = time.perf_counter() - started
        self._last_processing_seconds = elapsed
        self._max_processing_seconds = max(self._max_processing_seconds, elapsed)
        bar_events = self._bar_events or []
        self._bar_events = None
        for event in bar_events:
            if event.event_type is PaperEventType.STRATEGY_DECISION and event.payload.get("action", "").lower() == "enter":
                self._emit(PaperEventType.CANDIDATE_CREATED, {
                    "strategy_name": event.payload.get("strategy_name"),
                    "signal": event.payload.get("signal"),
                    "reason": event.payload.get("reason"),
                }, timestamp=bar.timestamp)
            if event.event_type is PaperEventType.RISK_DECISION and not event.payload.get("approved", False):
                self._emit(PaperEventType.RISK_REJECTION, event.payload, timestamp=bar.timestamp, severity="WARNING")
                self._emit(PaperEventType.CANDIDATE_REJECTED, {
                    "strategy_name": event.payload.get("strategy_name"),
                    "reason": event.payload.get("reason", "risk_rejected"),
                    "risk_decision": event.payload,
                }, timestamp=bar.timestamp, severity="WARNING")
            if event.event_type is PaperEventType.ERROR:
                context = event.payload.get("context", {})
                strategy_name = context.get("strategy_name") if isinstance(context, Mapping) else None
                if strategy_name and "Portfolio conflict rejected entry" in str(event.payload.get("message", "")):
                    self._emit(PaperEventType.CANDIDATE_REJECTED, {
                        "strategy_name": strategy_name,
                        "reason": str(event.payload.get("message")),
                        "rejection_stage": "portfolio_conflict",
                    }, timestamp=bar.timestamp, severity="WARNING")
            if event.event_type is PaperEventType.ORDER_REJECTED:
                self._emit(PaperEventType.CANDIDATE_REJECTED, {
                    "strategy_name": event.payload.get("strategy_name"),
                    "reason": event.payload.get("reason", "order_rejected"),
                    "rejection_stage": "paper_execution",
                }, timestamp=bar.timestamp, severity="WARNING")
            if event.event_type is PaperEventType.ORDER_CREATED:
                self._emit(PaperEventType.PAPER_ORDER_CREATED, event.payload, timestamp=bar.timestamp)
            if event.event_type is PaperEventType.FILL:
                self._emit(PaperEventType.PAPER_ORDER_FILLED, event.payload, timestamp=bar.timestamp)
            if event.event_type is PaperEventType.POSITION_OPENED:
                self._emit(PaperEventType.PAPER_POSITION_OPENED, event.payload, timestamp=bar.timestamp)
            if event.event_type is PaperEventType.POSITION_CLOSED:
                self._emit(PaperEventType.PAPER_POSITION_CLOSED, event.payload, timestamp=bar.timestamp)
        for name, stream in due:
            new_events = stream.refit_events[before[name]:]
            if new_events:
                fit = new_events[-1]
                row = {**fit, "stream": name, "wall_duration_seconds": elapsed,
                       "model_hash": stream.model_hash, "raw_state_at_activation": fit.get("first_raw_state")}
                self._append_refit_ledger(row)
                self._emit(PaperEventType.HMM_REFIT_COMPLETED, row, timestamp=bar.timestamp)
        enriched = self.engine.last_market_data or {}
        for stream_name, state_key, posterior_key, version, model_hash in (
            ("MR", "hmm_state", "hmm_posterior", "mr_model_version", "mr_model_hash"),
            ("S2R", "s2r_hmm_state", "s2r_hmm_posterior", "s2r_model_version", "s2r_model_hash"),
        ):
            if enriched.get(state_key) is not None:
                posterior = enriched.get(posterior_key)
                if hasattr(posterior, "tolist"):
                    posterior = posterior.tolist()
                self._emit(PaperEventType.HMM_STATE, {
                    "stream": stream_name, "raw_state": enriched.get(state_key),
                    "posterior": posterior,
                    "model_version": enriched.get(version), "model_hash": enriched.get(model_hash),
                }, timestamp=bar.timestamp)
        self._state = "RUNNING"
        self._last_bar = bar
        self._last_bar_digest = digest
        self._bars_processed += 1
        self._equity_peak = max(self._equity_peak, self.engine.account_equity)
        self._portfolio_max_drawdown = min(
            self._portfolio_max_drawdown, self.engine.account_equity - self._equity_peak
        )
        open_positions = []
        unrealized = gross_exposure = 0.0
        mark = float(bar.close)
        for position in self.engine.execution.get_positions():
            side = str(position.side.value).lower()
            signed = mark - float(position.entry_price)
            if side in {"short", "strategysignal.short"}:
                signed = -signed
            position_unrealized = signed * int(position.quantity) * float(self.engine.config.point_value)
            unrealized += position_unrealized
            gross_exposure += abs(mark * int(position.quantity) * float(self.engine.config.point_value))
            open_positions.append({"strategy":position.strategy_name,"side":side,
                                   "quantity":int(position.quantity),"entry_price":float(position.entry_price),
                                   "mark_price":mark,"unrealized_pnl":position_unrealized})
        equity = float(self.engine.account_equity) + unrealized
        self._equity_peak = max(self._equity_peak, equity)
        drawdown = equity - self._equity_peak
        self.analytics.record_account_snapshot(
            timestamp=bar.timestamp, balance=float(self.engine.account_equity), equity=equity,
            realized_pnl=float(self.engine.realized_pnl)-float(self.engine.total_costs),
            unrealized_pnl=unrealized,
            cumulative_net_r=sum(float(metric["cumulative_r"]) for metric in self._metrics.values()),
            open_risk=float(self.engine.risk.open_risk), gross_exposure=gross_exposure,
            drawdown=drawdown, open_positions=open_positions,
        )
        self.analytics.update_closed_trade_account(timestamp=bar.timestamp,
            balance=float(self.engine.account_equity), equity=equity, drawdown=drawdown)
        should_checkpoint = (
            self._bars_processed % self.config.checkpoint_every_bars == 0
            or bool(step.submitted_orders)
            or any(event.event_type in {PaperEventType.POSITION_OPENED, PaperEventType.POSITION_CLOSED} for event in bar_events)
            or bool(due)
        )
        if should_checkpoint:
            self.save_checkpoint()
        self._write_status()
        self.logger.set_idempotency_scope(None)
        return True

    def _backfill_gap_before(self, bar: CanonicalBar) -> None:
        """Backfill expected open-market minutes before evaluating a newer bar."""
        previous = self._last_bar
        if previous is None:
            return
        if self.calendar is not None:
            try:
                missing = self.calendar.snapshot  # coverage checked as we enumerate
                del missing
                expected = self.calendar.expected_missing_minutes(previous.timestamp, bar.timestamp)
            except CalendarUnavailable as exc:
                self._state = "ERROR"
                self._emit(PaperEventType.SYSTEM_CRITICAL, {
                    "message": "calendar unavailable while validating a feed gap",
                    "error": str(exc), "from": previous.timestamp.isoformat(),
                    "to": bar.timestamp.isoformat(),
                }, timestamp=bar.timestamp, severity="CRITICAL")
                self._write_status()
                raise RuntimeError("CME calendar unavailable during feed-gap validation") from exc
        else:
            before = pd.Timestamp(previous.timestamp).tz_convert("America/New_York")
            after = pd.Timestamp(bar.timestamp).tz_convert("America/New_York")
            if before.date() != after.date():
                return
            expected = []
            cursor = pd.Timestamp(previous.timestamp).tz_convert("UTC").floor("min") + pd.Timedelta(minutes=1)
            end = pd.Timestamp(bar.timestamp).tz_convert("UTC")
            while cursor < end:
                local_cursor = cursor.tz_convert("America/New_York")
                minute = local_cursor.hour * 60 + local_cursor.minute
                if local_cursor.date() == before.date() and 9 * 60 + 30 <= minute < 16 * 60:
                    expected.append(cursor)
                cursor += pd.Timedelta(minutes=1)
        if not expected:
            return

        self._state = "DEGRADED"
        self._emit(PaperEventType.MISSING_BAR, {
            "from": previous.timestamp.isoformat(),
            "to": bar.timestamp.isoformat(),
            "missing_minute_count": len(expected),
            "expected_timestamps": [stamp.isoformat() for stamp in expected[:50]],
        }, timestamp=bar.timestamp, severity="WARNING")
        self._emit(PaperEventType.BACKFILL_STARTED, {
            "after_timestamp": previous.timestamp.isoformat(),
            "before_timestamp": bar.timestamp.isoformat(),
            "expected_missing_count": len(expected),
            "expected_timestamps": [stamp.isoformat() for stamp in expected[:50]],
        }, timestamp=bar.timestamp, severity="WARNING")
        backfill = getattr(self.source, "backfill", None)
        if not callable(backfill):
            self._fail_backfill(bar, "market-data source does not support historical catch-up")
        try:
            recovered = list(backfill(
                after_timestamp=previous.timestamp,
                before_timestamp=bar.timestamp,
            ))
            recovered_bars = [item if isinstance(item, CanonicalBar) else CanonicalBar.from_mapping(item) for item in recovered]
            stamps = [pd.Timestamp(item.timestamp).tz_convert("UTC") for item in recovered_bars]
            if any(stamp <= pd.Timestamp(previous.timestamp).tz_convert("UTC") or stamp >= pd.Timestamp(bar.timestamp).tz_convert("UTC") for stamp in stamps):
                self._fail_backfill(bar, "backfill returned a bar outside the requested exclusive interval")
            if any(left >= right for left, right in zip(stamps, stamps[1:])):
                self._fail_backfill(bar, "backfill bars are duplicate or out of order")
            recovered_set = set(stamps)
            missing = [stamp for stamp in expected if stamp not in recovered_set]
            if missing:
                self._fail_backfill(bar, f"backfill omitted {len(missing)} expected open-market bars")
            for recovered_bar in recovered_bars:
                if recovered_bar.symbol != self.config.symbol:
                    self._fail_backfill(bar, f"backfill returned unexpected symbol {recovered_bar.symbol!r}")
                processed = self._process(recovered_bar)
                if not processed or self._last_bar is None or self._last_bar.timestamp != recovered_bar.timestamp:
                    self._fail_backfill(bar, "a backfilled bar was not committed in sequence")
            if self._last_bar is None or self._last_bar.timestamp >= bar.timestamp:
                self._fail_backfill(bar, "backfill advanced beyond the pending live bar")
            self._emit(PaperEventType.BACKFILL_COMPLETED, {
                "recovered_bar_count": len(recovered_bars),
                "before_timestamp": bar.timestamp.isoformat(),
            }, timestamp=bar.timestamp)
            self._state = "RUNNING"
        except BaseException as exc:
            if self._state != "ERROR":
                self._fail_backfill(bar, f"backfill failed: {type(exc).__name__}: {exc}")
            raise

    def _fail_backfill(self, bar: CanonicalBar, reason: str) -> None:
        self._state = "ERROR"
        self._emit(PaperEventType.BACKFILL_FAILED, {
            "reason": reason,
            "last_processed_timestamp": self._last_bar.timestamp.isoformat() if self._last_bar else None,
            "blocked_bar_timestamp": bar.timestamp.isoformat(),
        }, timestamp=bar.timestamp, severity="CRITICAL")
        self._write_status()
        raise RuntimeError(f"market-data backfill failed closed: {reason}")

    def save_checkpoint(self) -> None:
        if self._last_bar is None:
            return
        payload = {
            "mode": "PAPER", "symbol": self.config.symbol,
            "system": {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_identity": runtime_identity(),
                "calendar_identity": self.calendar.snapshot.identity if self.calendar else None,
                "calendar_snapshot": self.calendar.snapshot.to_mapping() if self.calendar else None,
                "cost_profile_id": self.cost_policy.profile_id if self.cost_policy else None,
                "cost_profile_identity": self.cost_policy.identity if self.cost_policy else None,
            },
            "context": self.context_adapter.context.state_dict(),
            "engine": capture_engine_state(self.engine),
            "runtime": {
                "run_id": self.run_id,
                "last_bar": {**self._last_bar.to_market_data(), "timestamp": self._last_bar.timestamp.isoformat()},
                "last_bar_digest": self._last_bar_digest,
                "bars_processed": self._bars_processed,
                "metrics": dict(self._metrics),
                "risk_per_contract": self._risk_per_contract,
                "equity_peak": self._equity_peak,
                "portfolio_max_drawdown": self._portfolio_max_drawdown,
                "daily_date": self._daily_date,
                "daily_counts": dict(self._daily_counts),
            },
        }
        try:
            self.checkpoint_store.save(payload)
            self._emit(PaperEventType.CHECKPOINT_SAVED, {
                "checkpoint_path": str(self.checkpoint_store.path),
                "last_processed_bar": self._last_bar.timestamp.isoformat(),
                "bars_processed": self._bars_processed,
            }, timestamp=self._last_bar.timestamp)
        except Exception as exc:
            self._errors.append(f"checkpoint: {exc}")
            self._emit(PaperEventType.CHECKPOINT_FAILED, {
                "error_type": type(exc).__name__, "message": str(exc),
            }, timestamp=self._last_bar.timestamp, severity="CRITICAL")
            self._state = "DEGRADED"
            raise

    def run(self, *, restore: bool = False) -> None:
        """Run while holding exclusive process ownership for this output directory."""
        with PaperWriterLock(self.config.output_dir / "paper_writer.lock"):
            self._run_owned(restore=restore)

    def _run_owned(self, *, restore: bool = False) -> None:
        if restore:
            self.restore()
        elif not self.bootstrap_store.path.exists():
            self.save_bootstrap_checkpoint()
        self.engine.connect()
        try:
            self.source.start()
            self.source.subscribe(
                (self.config.symbol,),
                after_timestamp=self._last_bar.timestamp if self._last_bar is not None else None,
            )
            initial_health = dict(self.source.health())
            if not initial_health.get("connected", True):
                raise RuntimeError(f"Market data source is not connected: {initial_health}")
        except BaseException:
            self.engine.disconnect()
            raise
        self._started_monotonic = time.monotonic()
        self._state = "RUNNING"
        self._emit(PaperEventType.SYSTEM_STARTED, {
            "mode": "PAPER", "source": self.source.name,
            "run_id": self.run_id, "symbol": self.config.symbol,
        })
        self._emit(PaperEventType.FEED_CONNECTED, {"source": self.source.name, "health": initial_health})
        self._last_feed_connected = bool(initial_health.get("connected", True))
        self._feed_stale = bool(initial_health.get("stale", False))
        self._producer = threading.Thread(target=self._produce, name="paper-market-feed", daemon=True)
        self._producer.start()
        self._watchdog = threading.Thread(target=self._watch_feed, name="paper-feed-watchdog", daemon=True)
        self._watchdog.start()
        notify_systemd(f"READY=1\nSTATUS=PAPER processing; source={self.source.name}")
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                self._process(item)
        except KeyboardInterrupt:
            self.request_stop(reason="keyboard_interrupt")
        except BaseException as exc:
            self._state = "ERROR"
            self._errors.append(str(exc))
            self._emit(PaperEventType.SYSTEM_ERROR, {
                "error_type": type(exc).__name__, "message": str(exc),
            }, severity="ERROR")
            raise
        finally:
            notify_systemd("STOPPING=1\nSTATUS=PAPER stopping after checkpoint flush")
            self._stop_requested.set()
            self.source.stop()
            if self._producer is not None:
                self._producer.join(timeout=5.0)
            if self._watchdog is not None:
                self._watchdog.join(timeout=2.0)
            self.engine.disconnect()
            if self._last_bar is not None:
                self.save_checkpoint()
            if self._state not in {"ERROR", "DEGRADED"}:
                self._state = "STOPPED"
            self._emit(PaperEventType.SYSTEM_STOPPED, {
                "mode": "PAPER", "last_processed_bar": self._last_bar.timestamp.isoformat() if self._last_bar else None,
                "state": self._state,
            })
            self._write_status()
            try:
                self.analytics.generate_daily_reports(self.config.output_dir)
            except Exception as exc:
                self._warnings.append(f"daily report generation failed: {type(exc).__name__}: {exc}")
                self._write_status()
            finally:
                self.analytics.close()

    def request_stop(self, *, reason: str = "requested") -> None:
        self._stop_requested.set()
        self.source.stop()
        self._emit(PaperEventType.SYSTEM_WARNING, {"message": "stop_requested", "reason": reason}, severity="WARNING")

    def _watch_feed(self) -> None:
        while not self._stop_requested.wait(1.0):
            last = self._last_bar.timestamp.isoformat() if self._last_bar else None
            stop_request = self.config.output_dir / "stop.request"
            if stop_request.exists():
                stop_request.unlink(missing_ok=True)
                self.request_stop(reason="stop_file")
                return
            health = dict(self.source.health())
            connected = bool(health.get("connected", True))
            stale = bool(health.get("stale", False))
            if self._last_feed_walltime is not None and self._queue.empty() and self._state != "PAUSED_REFIT":
                stale = stale or time.time() - self._last_feed_walltime > self.config.stale_after_seconds
            if connected != self._last_feed_connected:
                self._emit(
                    PaperEventType.FEED_CONNECTED if connected else PaperEventType.FEED_DISCONNECTED,
                    {"source": self.source.name, "health": health},
                    severity="INFO" if connected else "WARNING",
                )
                if connected and self._last_feed_connected is False:
                    self._emit(PaperEventType.FEED_RECOVERED, {"source": self.source.name, "health": health})
                self._last_feed_connected = connected
            if stale != self._feed_stale:
                self._feed_stale = stale
                if stale:
                    self._state = "DEGRADED"
                    self._emit(PaperEventType.FEED_STALE, {"source": self.source.name, "health": health}, severity="WARNING")
                else:
                    self._emit(PaperEventType.FEED_RECOVERED, {"source": self.source.name, "health": health})
                    if self._state == "DEGRADED":
                        self._state = "RUNNING"
            self._write_status()
            notify_systemd(f"WATCHDOG=1\nSTATUS=PAPER state={self._state}; last_bar={last or 'none'}")

    def _write_status(self) -> None:
        with self._status_lock:
            self._write_status_snapshot()

    def _write_status_snapshot(self) -> None:
        context = self.context_adapter.context
        provider = context._raw_hmm_provider
        last = self._last_bar.timestamp.isoformat() if self._last_bar else None
        now = time.time()
        stale = self._feed_stale
        if stale and self._state == "RUNNING":
            self._state = "DEGRADED"
        position_rows = []
        mark = float((self.engine.last_market_data or {}).get("close", 0.0) or 0.0)
        unrealized = 0.0
        for position in self.engine.execution.get_positions():
            side = str(position.side.value).lower()
            signed = mark-float(position.entry_price)
            if side in {"short", "strategysignal.short"}:
                signed = -signed
            position_unrealized = signed*int(position.quantity)*float(self.engine.config.point_value)
            unrealized += position_unrealized
            position_rows.append({"strategy": position.strategy_name, "side": position.side.value,
                                  "quantity": position.quantity, "entry_price": position.entry_price,
                                  "entry_timestamp": position.entry_timestamp.isoformat(),
                                  "mark_price": mark or None,"unrealized_pnl":position_unrealized})
        strategies = {}
        for strategy in self.engine.strategies:
            metrics = dict(self._metrics[strategy.name])
            closed = metrics["closed_trades"]
            strategies[strategy.name] = {
                **metrics,
                "expectancy_r": metrics["cumulative_r"] / closed if closed else 0.0,
                "profit_factor_r": metrics["gross_win_r"] / metrics["gross_loss_r"] if metrics["gross_loss_r"] else None,
                "win_rate": metrics["wins"] / closed if closed else 0.0,
                "has_open_position": self.engine.has_position(strategy.name),
                "candidates_today": self._daily_counts[strategy.name]["candidates"],
                "accepted_today": self._daily_counts[strategy.name]["accepted"],
                "rejected_today": self._daily_counts[strategy.name]["rejected"],
            }
        disk = shutil.disk_usage(self.config.output_dir)
        wal_path = self.config.output_dir / "paper_analytics.sqlite3-wal"
        snapshot = {
            "mode": "PAPER", "run_id": self.run_id,
            "system": {"state": self._state, "uptime_seconds": time.monotonic() - self._started_monotonic if self._started_monotonic else 0.0,
                       "last_bar": last, "feed_stale": stale, "feed_health": dict(self.source.health()),
                       "queue_depth": self._queue.qsize(),
                       "last_bar_processing_seconds": self._last_processing_seconds,
                       "max_bar_processing_seconds": self._max_processing_seconds,
                       "bars_processed": self._bars_processed,
                       "output_disk_bytes": {"total": disk.total, "free": disk.free,
                                             "used": disk.used},
                       "analytics_database_bytes": (self.config.output_dir / "paper_analytics.sqlite3").stat().st_size
                           if (self.config.output_dir / "paper_analytics.sqlite3").exists() else 0,
                       "analytics_wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
                       "last_checkpoint": self.checkpoint_store.path.stat().st_mtime if self.checkpoint_store.path.exists() else None,
                       "cme_calendar": ({"source": self.calendar.snapshot.source,
                                         "version": self.calendar.version,
                                         "identity": self.calendar.snapshot.identity}
                                        if self.calendar else {"configured": False}),
                       "analytics_database": str(self.config.output_dir / "paper_analytics.sqlite3"),
                       "cost_profile_id": self.cost_policy.profile_id if self.cost_policy else None,
                       "errors": list(self._errors), "warnings": list(self._warnings)},
            "portfolio": {"open_positions": position_rows, "balance": self.engine.account_equity,
                          "equity": self.engine.account_equity + unrealized,
                          "gross_realized_pnl": self.engine.realized_pnl,
                          "net_realized_pnl": self.engine.realized_pnl-self.engine.total_costs,
                          "commission": self.engine.commissions,
                          "exchange_fees": self.engine.exchange_fees,
                          "regulatory_fees": self.engine.regulatory_fees,
                          "total_costs": self.engine.total_costs,
                          "unrealized_pnl": unrealized,
                          "daily_pnl": self.engine.risk.daily_realized_pnl,
                          "open_risk": self.engine.risk.open_risk,
                          "drawdown": self.engine.account_equity + unrealized - self._equity_peak,
                          "max_drawdown": self._portfolio_max_drawdown,
                          "realized_r": sum(float(m["cumulative_r"]) for m in self._metrics.values()),
                          "unrealized_r": None},
            "strategies": strategies,
            "hmm": {
                "mr": {"raw_state": (self.engine.last_market_data or {}).get("hmm_state"),
                       "posterior": (self.engine.last_market_data or {}).get("hmm_posterior"),
                       "model_version": provider.mr.fit_count, "model_hash": provider.mr.model_hash,
                       "last_fit": provider.mr.fit_timestamp.isoformat() if provider.mr.fit_timestamp is not None else None,
                       "next_refit": provider.mr.next_refit_timestamp.isoformat() if provider.mr.next_refit_timestamp is not None else None},
                "s2r": {"raw_state": (self.engine.last_market_data or {}).get("s2r_hmm_state"),
                        "posterior": (self.engine.last_market_data or {}).get("s2r_hmm_posterior"),
                        "model_version": provider.s2r.fit_count, "model_hash": provider.s2r.model_hash,
                        "last_fit": provider.s2r.fit_timestamp.isoformat() if provider.s2r.fit_timestamp is not None else None,
                        "next_refit": provider.s2r.next_refit_timestamp.isoformat() if provider.s2r.next_refit_timestamp is not None else None},
            },
        }
        temporary = self.status_path.with_name(f".{self.status_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(snapshot, handle, default=str, indent=2, allow_nan=True)
            handle.flush()
            os.fsync(handle.fileno())
        last_error = None
        for attempt in range(5):
            try:
                os.replace(temporary, self.status_path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.02 * (attempt + 1))
        if last_error is not None:
            temporary.unlink(missing_ok=True)
            self._warnings.append(f"status snapshot replace deferred: {last_error}")
            self._emit(PaperEventType.SYSTEM_WARNING, {
                "message": "status_snapshot_replace_failed",
                "error_type": type(last_error).__name__,
            }, severity="WARNING")


__all__ = ["RealtimePaperConfig", "RealtimePaperService"]
