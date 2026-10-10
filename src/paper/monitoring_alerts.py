"""Read-only, durable Paper alert evaluation.

This module never calls the Paper engine or a broker. It evaluates persisted
status/events and emits an independent alert journal consumed by Telegram.
Thresholds here are monitoring thresholds only; risk policy remains authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Mapping, Protocol
from zoneinfo import ZoneInfo

from src.paper.atomic_io import atomic_replace_with_retry
from src.paper.cme_calendar import CMETradingCalendar, CalendarUnavailable

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


def _readonly_sqlite_uri(path: Path) -> str:
    normalized = str(path.resolve()).replace("\\", "/")
    return "file:" + normalized + "?mode=ro"


class LiveBrokerMonitor(Protocol):
    """Future read-only reconciliation contract; intentionally not activated."""
    enabled: bool
    def positions(self) -> list[Mapping[str, Any]]: ...
    def orders(self) -> list[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class AlertConfig:
    symbol: str = "MNQ"
    backlog_warning_bars: int = 60
    # These execution thresholds stay unset until a non-recovered live-like
    # Paper sample supports them. Historical catch-up fills are not evidence.
    slippage_warning_ticks: float | None = None
    slippage_critical_ticks: float | None = None
    execution_latency_warning_ms: float | None = None
    acquisition_latency_warning_seconds: float | None = None
    provider_delay_seconds: float = 600.0
    finalization_min_age_seconds: float = 600.0
    finalization_confirmation_spacing_seconds: float = 30.0
    provider_stall_warning_seconds: float = 300.0
    provider_stall_critical_seconds: float = 900.0
    processing_stall_warning_seconds: float = 300.0
    processing_stall_critical_seconds: float = 900.0
    stall_debounce_seconds: float = 60.0
    drawdown_warning_usd: float | None = None
    drawdown_critical_usd: float | None = None

    def __post_init__(self) -> None:
        numbers = (self.provider_delay_seconds, self.finalization_min_age_seconds,
                   self.finalization_confirmation_spacing_seconds,
                   self.provider_stall_warning_seconds, self.provider_stall_critical_seconds,
                   self.processing_stall_warning_seconds, self.processing_stall_critical_seconds,
                   self.stall_debounce_seconds)
        if self.backlog_warning_bars < 0 or any(not math.isfinite(x) or x <= 0 for x in numbers):
            raise ValueError("monitoring thresholds must be positive finite values")
        optional_positive = (self.slippage_warning_ticks, self.slippage_critical_ticks,
                             self.execution_latency_warning_ms, self.acquisition_latency_warning_seconds)
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in optional_positive):
            raise ValueError("optional monitoring thresholds must be positive finite values")
        if (self.slippage_warning_ticks is not None and self.slippage_critical_ticks is not None
                and self.slippage_warning_ticks > self.slippage_critical_ticks):
            raise ValueError("slippage warning threshold must not exceed critical threshold")
        if self.provider_stall_warning_seconds > self.provider_stall_critical_seconds:
            raise ValueError("provider stall warning must not exceed critical threshold")
        if self.processing_stall_warning_seconds > self.processing_stall_critical_seconds:
            raise ValueError("processing stall warning must not exceed critical threshold")
        if any(value is not None and (not math.isfinite(value) or value <= 0)
               for value in (self.drawdown_warning_usd, self.drawdown_critical_usd)):
            raise ValueError("optional drawdown monitoring thresholds must be positive and finite")
        if (self.drawdown_warning_usd is not None and self.drawdown_critical_usd is not None
                and self.drawdown_warning_usd > self.drawdown_critical_usd):
            raise ValueError("drawdown warning threshold must not exceed critical threshold")

    @classmethod
    def from_environment(cls) -> "AlertConfig":
        def optional_float(name: str) -> float | None:
            raw = os.environ.get(name)
            return None if raw is None or not raw.strip() else float(raw)
        return cls(
            backlog_warning_bars=int(os.environ.get("MNQ_ALERT_BACKLOG_BARS", "60")),
            slippage_warning_ticks=optional_float("MNQ_ALERT_SLIPPAGE_WARNING_TICKS"),
            slippage_critical_ticks=optional_float("MNQ_ALERT_SLIPPAGE_CRITICAL_TICKS"),
            execution_latency_warning_ms=optional_float("MNQ_ALERT_EXECUTION_LATENCY_WARNING_MS"),
            acquisition_latency_warning_seconds=optional_float("MNQ_ALERT_ACQUISITION_LATENCY_WARNING_SECONDS"),
            provider_delay_seconds=float(os.environ.get("MNQ_IBKR_DELAY_SECONDS", "600")),
            finalization_min_age_seconds=float(os.environ.get("MNQ_BAR_FINALIZATION_MIN_AGE_SECONDS", "600")),
            finalization_confirmation_spacing_seconds=float(os.environ.get("MNQ_BAR_CONFIRMATION_SPACING_SECONDS", "30")),
            provider_stall_warning_seconds=float(os.environ.get("MNQ_ALERT_PROVIDER_STALL_WARNING_SECONDS", "300")),
            provider_stall_critical_seconds=float(os.environ.get("MNQ_ALERT_PROVIDER_STALL_CRITICAL_SECONDS", "900")),
            processing_stall_warning_seconds=float(os.environ.get("MNQ_ALERT_PROCESSING_STALL_WARNING_SECONDS", "300")),
            processing_stall_critical_seconds=float(os.environ.get("MNQ_ALERT_PROCESSING_STALL_CRITICAL_SECONDS", "900")),
            stall_debounce_seconds=float(os.environ.get("MNQ_ALERT_STALL_DEBOUNCE_SECONDS", "60")),
            drawdown_warning_usd=optional_float("MNQ_ALERT_DRAWDOWN_WARNING_USD"),
            drawdown_critical_usd=optional_float("MNQ_ALERT_DRAWDOWN_CRITICAL_USD"),
        )


def _utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def evaluate_cme_market_state(calendar: CMETradingCalendar | None, timestamp: datetime) -> dict[str, Any]:
    """Resolve the schedule state without touching the feed or advancing cursors.

    The reviewed snapshot's coverage is mandatory even for a familiar weekly
    pattern. Out-of-coverage dates therefore remain UNKNOWN.
    """
    if calendar is None:
        return {"state": "MARKET_UNKNOWN", "reason": "reviewed_calendar_unavailable",
                "calendar_version": None, "calendar_identity": None}
    if timestamp.tzinfo is None:
        raise ValueError("market-state timestamp must be timezone-aware")
    utc = timestamp.astimezone(UTC)
    local = utc.astimezone(NEW_YORK)
    snap = calendar.snapshot
    identity = snap.identity
    try:
        local_date = local.date()
        if not snap.coverage_start <= local_date <= snap.coverage_end:
            raise CalendarUnavailable(f"calendar does not cover local date {local_date}")
        minute = local.hour * 60 + local.minute
        maintenance_start = snap.maintenance_start.hour * 60 + snap.maintenance_start.minute
        maintenance_end = snap.maintenance_end.hour * 60 + snap.maintenance_end.minute

        if local.weekday() == 5:
            state, reason = "MARKET_CLOSED", "cme_weekend_closure"
        elif local.weekday() == 4 and minute >= snap.globex_close.hour * 60 + snap.globex_close.minute:
            state, reason = "MARKET_CLOSED", "cme_weekend_closure_after_friday_close"
        elif minute >= maintenance_start and minute < maintenance_end and local.weekday() in {6, 0, 1, 2, 3}:
            state, reason = "MARKET_BREAK", "reviewed_daily_or_weekly_maintenance_break"
        else:
            trade_date = calendar.trading_date_for_timestamp(utc)
            if trade_date is not None:
                # trading_date_for_timestamp has already enforced the snapshot
                # coverage and any date-specific closure/session exception.
                state, reason = "MARKET_OPEN", "timestamp_inside_reviewed_globex_session"
            else:
                # A date-specific full holiday closure must be distinguishable
                # from an ordinary out-of-session period.
                candidate = local.date()
                if minute >= snap.globex_open.hour * 60 + snap.globex_open.minute:
                    candidate = candidate + timedelta(days=1)
                    while candidate.weekday() >= 5:
                        candidate += timedelta(days=1)
                    if candidate > snap.coverage_end:
                        raise CalendarUnavailable(f"calendar does not cover next trade date {candidate}")
                session = snap.session_for_rth_date(candidate)
                exception = snap.exceptions.get(candidate.isoformat(), {})
                state = "MARKET_CLOSED"
                reason = "outside_reviewed_globex_session"
                if session.session_type == "holiday_closed":
                    reason = "reviewed_holiday_closure"
                for start_text, end_text in exception.get("closed_intervals", []):
                    start = datetime.combine(candidate, datetime.strptime(start_text, "%H:%M").time(), tzinfo=NEW_YORK)
                    end = datetime.combine(candidate, datetime.strptime(end_text, "%H:%M").time(), tzinfo=NEW_YORK)
                    if start <= local < end:
                        state, reason = "MARKET_BREAK", "reviewed_date_specific_closed_interval"
                        break
        return {"state": state, "reason": reason, "evaluated_at_utc": utc.isoformat(),
                "local_time": local.isoformat(), "calendar_version": snap.version,
                "calendar_identity": identity, "source": snap.source}
    except (CalendarUnavailable, ValueError, KeyError) as exc:
        return {"state": "MARKET_UNKNOWN", "reason": f"calendar_unresolved:{type(exc).__name__}",
                "evaluated_at_utc": utc.isoformat(), "local_time": local.isoformat(),
                "calendar_version": snap.version, "calendar_identity": identity, "source": snap.source}
def make_alert(*, alert_type: str, severity: str, expected: Any = None, observed: Any = None,
               threshold: Any = None, strategy: str = "SYSTEM", details: str = "",
               run_id: str = "unavailable", event_id: str | None = None,
               status: str = "ACTIVE", symbol: str = "MNQ", detected_at: datetime | None = None) -> dict[str, Any]:
    stamp = (detected_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    key_source = f"{run_id}|{alert_type}|{strategy}|{event_id or ''}"
    return {
        "schema_version": 1,
        "event_id": hashlib.sha256(key_source.encode()).hexdigest(),
        "alert_id": hashlib.sha256(f"{run_id}|{alert_type}|{strategy}".encode()).hexdigest(),
        "alert_type": alert_type,
        "severity": severity,
        "environment": "PAPER",
        "strategy": strategy,
        "symbol": symbol,
        "timestamp_utc": stamp,
        "account_run_id": run_id,
        "expected": expected,
        "observed": observed,
        "threshold": threshold,
        "status": status,
        "details": details,
        "source_event_id": event_id,
        "source": "persisted_paper_state",
    }


class AlertStateStore:
    """Atomic alert state plus append-only transition ledger."""
    def __init__(self, state_path: str | Path, events_path: str | Path) -> None:
        self.state_path, self.events_path = Path(state_path), Path(events_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (self.state.get("schema_version") != 1 or
                    not {"active", "history", "event_offset", "event_ids", "capabilities"}.issubset(self.state)):
                raise ValueError("alert state schema is incompatible")
        else:
            self.state = {"schema_version": 1, "active": {}, "history": [], "event_offset": 0,
                          "event_ids": [], "alert_generations": {}, "capabilities": {}}
            self._save()
        self.state.setdefault("alert_generations", {})

    def _save(self) -> None:
        fd, name = tempfile.mkstemp(prefix=self.state_path.name + ".", suffix=".tmp", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.state, f, sort_keys=True, separators=(",", ":"), default=str)
                f.flush(); os.fsync(f.fileno())
            atomic_replace_with_retry(name, self.state_path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def _append(self, alert: Mapping[str, Any]) -> None:
        envelope = {"event_id": alert["event_id"], "event_type": "monitoring_alert",
                    "timestamp": alert["timestamp_utc"], "payload": dict(alert)}
        line = json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n"); f.flush(); os.fsync(f.fileno())

    def observe_conditions(self, alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Update monitored conditions; absent conditions transition to RECOVERED."""
        current = {a["alert_id"]: a for a in alerts}
        emitted: list[dict[str, Any]] = []
        for key, old in list(self.state["active"].items()):
            if old.get("_auto_recover") is True and key not in current:
                generation = int(old.get("_generation", 1))
                recovered = {**old, "event_id": hashlib.sha256((old["alert_id"] + f"|RECOVERED|{generation}").encode()).hexdigest(),
                             "timestamp_utc": datetime.now(UTC).isoformat(), "status": "RECOVERED",
                             "details": "Condition cleared or no longer applies under the current persisted state and verified CME session."}
                self._append(recovered); self.state["history"].append(recovered); emitted.append(recovered)
                del self.state["active"][key]
        for key, alert in current.items():
            old = self.state["active"].get(key)
            generation = int(old.get("_generation", 1)) if old else int(self.state["alert_generations"].get(key, 0)) + 1
            # A changing lag/age measurement belongs in the live state and
            # dashboard, not in a new Telegram message every polling cycle.
            # Notify once on activation, severity escalation, or threshold change.
            if old is None or old.get("severity") != alert.get("severity") or old.get("threshold") != alert.get("threshold"):
                self.state["alert_generations"][key] = generation
                transition = {**alert, "event_id": hashlib.sha256((alert["alert_id"] + f"|ACTIVE|{generation}|" + json.dumps([alert.get("observed"), alert.get("threshold"), alert.get("details")], sort_keys=True)).encode()).hexdigest()}
                self._append(transition); self.state["history"].append(transition); emitted.append(transition)
            self.state["active"][key] = {**alert, "_generation": generation, "_auto_recover": True}
        self.state["history"] = self.state["history"][-1000:]
        self._save()
        return emitted

    def record_event_alert(self, alert: dict[str, Any]) -> bool:
        """Persist a one-shot invariant/order incident, deduplicated by source event."""
        ids = set(self.state.get("event_ids", []))
        if alert["event_id"] in ids:
            return False
        self._append(alert); self.state["history"].append(alert)
        self.state["active"][alert["alert_id"]] = {**alert, "_auto_recover": False}
        self.state["history"] = self.state["history"][-1000:]
        self.state["event_ids"] = (self.state.get("event_ids", []) + [alert["event_id"]])[-5000:]
        self._save()
        return True

    def acknowledge(self, alert_id: str, *, resolved: bool = False) -> bool:
        old = self.state["active"].get(alert_id)
        if old is None:
            return False
        changed = {**old, "event_id": hashlib.sha256((old["event_id"] + ("|RESOLVED" if resolved else "|ACK")).encode()).hexdigest(),
                   "timestamp_utc": datetime.now(UTC).isoformat(), "status": "RESOLVED" if resolved else "ACKNOWLEDGED"}
        self._append(changed); self.state["history"].append(changed)
        if resolved:
            del self.state["active"][alert_id]
        else:
            self.state["active"][alert_id] = changed
        self._save()
        return True


def _read_risk_limits(db_path: Path) -> dict[str, Any] | None:
    if not db_path.is_file(): return None
    try:
        uri = _readonly_sqlite_uri(db_path)
        with sqlite3.connect(uri, uri=True, timeout=0.15) as db:
            row = db.execute("SELECT source_json FROM configuration_versions WHERE config_kind='paper_engine' ORDER BY rowid DESC LIMIT 1").fetchone()
            if not row: return None
            config = json.loads(row[0]); limits = config.get("risk_limits")
            return limits if isinstance(limits, dict) else None
    except (OSError, sqlite3.Error, json.JSONDecodeError):
        return None


def evaluate_status(status: Mapping[str, Any], *, config: AlertConfig, risk_limits: Mapping[str, Any] | None,
                    now: datetime | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Evaluate only persisted facts. Unknown inputs are capability gaps, not healthy."""
    now = now or datetime.now(UTC)
    run_id = str(status.get("run_id") or "unavailable")
    system = status.get("system") if isinstance(status.get("system"), Mapping) else {}
    portfolio = status.get("portfolio") if isinstance(status.get("portfolio"), Mapping) else {}
    feed = system.get("feed_health") if isinstance(system.get("feed_health"), Mapping) else {}
    alerts: list[dict[str, Any]] = []
    market = system.get("market_state") if isinstance(system.get("market_state"), Mapping) else {}
    market_state = str(market.get("state", "MARKET_UNKNOWN"))
    market_open = market_state == "MARKET_OPEN"
    caps = {"feed": "AVAILABLE" if feed else "UNAVAILABLE",
            "market_calendar": "AVAILABLE" if market_state != "MARKET_UNKNOWN" else "UNAVAILABLE",
            "market_state": market_state,
            "feed_stall_market_hours": "AVAILABLE" if market_state != "MARKET_UNKNOWN" else "UNAVAILABLE_MARKET_SESSION",
            "provider_stall": "AVAILABLE" if market_open and _utc(feed.get("latest_available_bar")) else "UNAVAILABLE_NO_OPEN_SESSION_FRONTIER",
            "paper_processing_stall": "AVAILABLE" if market_open and _utc(feed.get("latest_available_bar")) and _utc(feed.get("last_processed_bar")) else "UNAVAILABLE_NO_COMPARABLE_FRONTIER",
            "daily_loss": "UNAVAILABLE",
            "drawdown": "UNAVAILABLE", "position_reconciliation": "UNAVAILABLE",
            "latency": "UNAVAILABLE_THRESHOLD_UNCONFIGURED", "live_broker_reconciliation": "DISABLED"}
    def add(name: str, severity: str, expected: Any, observed: Any, threshold: Any, details: str,
            strategy: str = "SYSTEM") -> None:
        alerts.append(make_alert(alert_type=name, severity=severity, expected=expected, observed=observed,
                                 threshold=threshold, details=details, strategy=strategy, run_id=run_id))
    state = str(system.get("state", "UNAVAILABLE")).upper()
    if state in {"ERROR", "FAILED"}:
        add("Paper Engine error state", "CRITICAL", "RUNNING", state, None,
            "Persisted engine state reports an error. The independent process watchdog separately determines whether the process terminated.")
    if feed:
        reconnect_state = str(feed.get("connection_state", "")).upper()
        outage_active = bool(feed.get("incident_id")) or reconnect_state in {"DISCONNECTED", "RECONNECTING"}
        if feed.get("connected") is False or feed.get("provider_connected") is False:
            outage_active = True
        if outage_active:
            add("Data feed disconnected", "CRITICAL", "provider connected and Paper caught up",
                {"connection_state": reconnect_state or feed.get("mode"),
                 "provider_connected": feed.get("provider_connected"),
                 "retry_count": feed.get("retry_count"),
                 "next_retry_at_utc": feed.get("next_retry_at_utc")}, None,
                f"IBKR connectivity incident is active; CME state is {market_state}. "
                f"Provider error: {feed.get('last_error') or 'unavailable'}. "
                "A TWS maintenance cause is not inferred from connectivity codes.")
        elif feed.get("last_error"):
            add("Data feed recovery failed", "CRITICAL", "no current acquisition error", str(feed.get("last_error")), None,
                f"Persisted IBKR acquisition/recovery error; CME state is {market_state}. A TWS maintenance cause is not inferred from connectivity code 1100.")
        backlog = feed.get("backlog_bars")
        if isinstance(backlog, (int, float)) and backlog > config.backlog_warning_bars:
            add("Feed backlog excessive", "WARNING", f"<= {config.backlog_warning_bars} backlog bars", backlog,
                config.backlog_warning_bars, "Delayed Paper backlog exceeds monitoring threshold.")
        if market_open:
            latest = _utc(feed.get("latest_available_bar"))
            if latest:
                age = max(0.0, (now.astimezone(UTC) - latest).total_seconds())
                # Provider delay and finalizer minimum age overlap; use max,
                # then add one poll/confirmation spacing. They are not summed.
                baseline = max(config.provider_delay_seconds, config.finalization_min_age_seconds)
                expected_frontier = now.astimezone(UTC).timestamp() - baseline - config.finalization_confirmation_spacing_seconds
                provider_lag = max(0.0, expected_frontier - latest.timestamp())
                if provider_lag >= config.provider_stall_critical_seconds:
                    severity = "CRITICAL"
                elif provider_lag >= config.provider_stall_warning_seconds:
                    severity = "WARNING"
                else:
                    severity = None
                if severity:
                    add("Provider data stalled", severity, f"bar at or after {datetime.fromtimestamp(expected_frontier, UTC).isoformat()} minus tolerance",
                        {"last_provider_bar_utc": latest.isoformat(), "bar_age_seconds": round(age, 1),
                         "provider_lag_beyond_expected_frontier_seconds": round(provider_lag, 1)},
                        config.provider_stall_critical_seconds if severity == "CRITICAL" else config.provider_stall_warning_seconds,
                        f"Confirmed exchange-bar frontier lag while market is open. Expected frontier is based on {baseline:g}s delayed/finalization age plus {config.finalization_confirmation_spacing_seconds:g}s confirmation spacing; severity uses persistence tolerance.")
            latest = _utc(feed.get("latest_available_bar")); processed = _utc(feed.get("last_processed_bar"))
            if latest and processed:
                lag = max(0.0, (latest - processed).total_seconds())
                backlog = feed.get("backlog_bars")
                if lag >= config.processing_stall_critical_seconds:
                    severity = "CRITICAL"
                elif lag >= config.processing_stall_warning_seconds or (isinstance(backlog, (int, float)) and backlog > 0):
                    severity = "WARNING"
                else:
                    severity = None
                if severity:
                    add("Paper processing stalled", severity, "committed bar near provider frontier",
                        {"latest_provider_bar_utc": latest.isoformat(), "last_committed_bar_utc": processed.isoformat(),
                         "lag_seconds": round(lag, 1), "backlog_bars": backlog},
                        config.processing_stall_critical_seconds if severity == "CRITICAL" else config.processing_stall_warning_seconds,
                        "Provider bars are available but the durable Paper cursor trails them; distinct from provider delay.")
            elif feed.get("stale") is True:
                add("Provider data stalled", "WARNING", "fresh finalized bar while market open", "stale",
                    config.provider_stall_warning_seconds, "Feed explicitly marked stale during a calendar-confirmed open session.")
    processing_seconds = system.get("last_bar_processing_seconds")
    if config.execution_latency_warning_ms is not None and isinstance(processing_seconds, (int, float)) and processing_seconds * 1000 > config.execution_latency_warning_ms:
        add("Execution latency abnormal", "WARNING", f"<= {config.execution_latency_warning_ms} ms/bar",
            f"{processing_seconds * 1000:.1f} ms Paper processing", config.execution_latency_warning_ms,
            "Persisted per-bar Paper processing time; excludes broker submission and fill latency.")
    daily_pnl = portfolio.get("daily_pnl")
    loss_limit = (risk_limits or {}).get("max_daily_loss")
    if isinstance(daily_pnl, (int, float)) and isinstance(loss_limit, (int, float)) and loss_limit > 0:
        caps["daily_loss"] = "AVAILABLE"
        if daily_pnl <= -loss_limit:
            add("Daily loss limit reached", "CRITICAL", f"> -{loss_limit} USD", daily_pnl, -loss_limit,
                "Existing Paper risk-policy daily loss limit is reached or breached.")
    drawdown = portfolio.get("drawdown")
    if isinstance(drawdown, (int, float)):
        caps["drawdown"] = "AVAILABLE" if config.drawdown_warning_usd is not None or config.drawdown_critical_usd is not None else "DATA_AVAILABLE_THRESHOLD_UNCONFIGURED"
        amount = abs(float(drawdown))
        if config.drawdown_critical_usd is not None and amount >= config.drawdown_critical_usd:
            add("Drawdown beyond expected threshold", "CRITICAL", f"< {config.drawdown_critical_usd} USD from Paper HWM", drawdown,
                config.drawdown_critical_usd, "Monitoring-only threshold; not a funded-account rule.")
        elif config.drawdown_warning_usd is not None and amount >= config.drawdown_warning_usd:
            add("Drawdown beyond expected threshold", "WARNING", f"< {config.drawdown_warning_usd} USD from Paper HWM", drawdown,
                config.drawdown_warning_usd, "Monitoring-only threshold; not a funded-account rule.")
    positions = portfolio.get("open_positions")
    if isinstance(positions, list):
        caps["position_reconciliation"] = "AVAILABLE_PENDING_LEDGER_COMPARE"
    return alerts, caps


class PaperAlertMonitor:
    """Sidecar monitor. Reads status, JSONL, and SQLite with read-only access."""
    EVENT_ALERTS = {
        "paper_order_rejected": ("Simulated order rejected", "WARNING"),
        "order_rejected": ("Simulated order rejected", "WARNING"),
        "execution_rejected": ("Simulated order rejected", "WARNING"),
        "risk_limit_exceeded": ("Risk limit reached", "CRITICAL"),
        "trading_halted": ("Paper trading halted", "CRITICAL"),
        "account_floor_breach": ("Paper account floor breached", "CRITICAL"),
        "risk_warning": ("Paper risk policy warning", "WARNING"),
        "risk_rejection": ("Paper risk gate rejected entry", "WARNING"),
        "feed_disconnected": ("Data feed disconnected", "CRITICAL"),
        "ibkr_disconnected": ("Data feed disconnected", "CRITICAL"),
        "feed_gap": ("Data feed gap", "CRITICAL"),
        "unresolved_expected_minute_gap": ("Data feed gap", "CRITICAL"),
        "data_gap": ("Data feed gap", "CRITICAL"),
        "missing_bar": ("Data feed gap", "CRITICAL"),
        "out_of_order_bar": ("Data feed timestamp anomaly", "CRITICAL"),
        "backfill_failed": ("Data feed recovery failed", "CRITICAL"),
        "system_error": ("Paper system error", "CRITICAL"),
        "checkpoint_recovery_failed": ("Paper recovery failed", "CRITICAL"),
        "recovery_failed": ("Paper recovery failed", "CRITICAL"),
        "hmm_refit_failed": ("Causal HMM refit failed", "CRITICAL"),
        "drawdown_threshold_exceeded": ("Drawdown beyond expected threshold", "CRITICAL"),
        "account_floor_warning": ("Paper account floor warning", "WARNING"),
        "account_floor_breach": ("Paper account floor breached", "CRITICAL"),
        "position_size_mismatch": ("Position size mismatch", "CRITICAL"),
        "unexpected_position": ("Unexpected position", "CRITICAL"),
        "strategy_invariant_violation": ("Strategy generated unexpected signal", "CRITICAL"),
        "out_of_session_signal": ("Strategy generated unexpected signal", "CRITICAL"),
        "invalid_strategy_input": ("Strategy generated unexpected signal", "CRITICAL"),
    }
    STRUCTURED_EVENT_TYPES = frozenset(EVENT_ALERTS)
    CRITICAL_RECOVERY_EVENT_TYPES = frozenset({"feed_disconnected", "ibkr_disconnected", "backfill_failed",
                                               "system_error", "checkpoint_recovery_failed", "recovery_failed",
                                               "hmm_refit_failed", "out_of_order_bar"})

    def __init__(self, *, run_dir: str | Path, state_path: str | Path, events_path: str | Path,
                 config: AlertConfig | None = None, calendar: CMETradingCalendar | None = None) -> None:
        self.run_dir = Path(run_dir)
        fresh = not Path(state_path).exists()
        self.state = AlertStateStore(state_path, events_path)
        self.config = config or AlertConfig.from_environment()
        self.calendar = calendar
        # On first installation, start at the existing Paper journal tail to
        # prevent historical catch-up from generating a notification storm.
        source = self.run_dir / "events.jsonl"
        if fresh and source.is_file():
            size = source.stat().st_size
            with source.open("rb") as f:
                f.seek(max(0, size - 1)); last = f.read(1)
            self.state.state["event_offset"] = size
            self.state.state["discard_partial_until_newline"] = bool(size and last != b"\n")
            self.state._save()

    def _read_new_events(self, max_rows: int = 1000) -> tuple[list[dict[str, Any]], int]:
        source = self.run_dir / "events.jsonl"
        if not source.is_file(): return [], int(self.state.state.get("event_offset", 0))
        offset = int(self.state.state.get("event_offset", 0))
        size = source.stat().st_size
        if size < offset: raise RuntimeError("Paper event log shrank below alert-monitor cursor")
        rows = []
        with source.open("rb") as f:
            f.seek(offset)
            if self.state.state.pop("discard_partial_until_newline", False):
                partial = f.readline()
                if not partial.endswith(b"\n"):
                    self.state.state["discard_partial_until_newline"] = True
                    self.state._save()
                    return [], offset
                offset = f.tell()
            for _ in range(max_rows):
                start = f.tell(); line = f.readline()
                if not line or not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                rows.append(event)
            end_offset = f.tell()
        return rows, end_offset

    def _event_alerts(self) -> list[dict[str, Any]]:
        out = []
        rows, end_offset = self._read_new_events()
        for event in rows:
            kind = str(event.get("event_type", "")).lower()
            payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
            mapping = self.EVENT_ALERTS.get(kind)
            if mapping:
                title, severity = mapping
                # Historical recovery events are incidents from reconstructed history, not new notifications.
                recovered = str(payload.get("market_data_provenance", "")).upper() == "RECOVERED_PAPER"
                if recovered and kind not in self.CRITICAL_RECOVERY_EVENT_TYPES:
                    mapping = None
                if mapping and kind == "risk_limit_exceeded" and "daily" in str(payload.get("reason", "")).lower():
                    title = "Daily loss limit reached"
            if mapping:
                expected = payload.get("expected_quantity", payload.get("expected"))
                observed = payload.get("actual_quantity", payload.get("observed"))
                alert = make_alert(alert_type=title, severity=severity, expected=expected, observed=observed,
                                   threshold=payload.get("threshold"), strategy=str(payload.get("strategy_name") or payload.get("strategy") or "SYSTEM"),
                                   symbol=str(payload.get("symbol") or "MNQ"), run_id=str(payload.get("run_id") or self.run_dir.name),
                                   event_id=str(event.get("event_id")), details=str(payload.get("message") or payload.get("reason") or kind))
                active_condition = any(
                    row.get("_auto_recover") is True and row.get("alert_type") == title
                    and row.get("strategy") == alert["strategy"]
                    for row in self.state.state["active"].values()
                )
                if active_condition:
                    # The persisted status condition already generated the
                    # actionable notification. Consume this matching source
                    # event without issuing a duplicate incident message.
                    ids = self.state.state.setdefault("event_ids", [])
                    if alert["event_id"] not in ids:
                        ids.append(alert["event_id"]); self.state.state["event_ids"] = ids[-5000:]
                        self.state._save()
                elif self.state.record_event_alert(alert):
                    out.append(alert)
            if str(payload.get("market_data_provenance", "")).upper() != "RECOVERED_PAPER" and kind in {"paper_order_filled", "fill", "order_filled"}:
                alert = self._slippage_alert(event, payload)
                if alert and self.state.record_event_alert(alert): out.append(alert)
            latency = self._latency_alert(event, payload)
            if latency and self.state.record_event_alert(latency): out.append(latency)
            if kind in {"risk_limit_exceeded", "trading_halted", "account_floor_breach", "risk_warning", "risk_rejection"} and kind not in self.EVENT_ALERTS:
                alert = make_alert(alert_type="Daily loss limit reached" if "daily" in str(payload.get("reason", "")).lower() else "Risk policy event",
                    severity="CRITICAL" if kind in {"trading_halted", "account_floor_breach", "risk_limit_exceeded"} else "WARNING",
                    expected=payload.get("expected"), observed=payload.get("observed", payload.get("daily_pnl")),
                    threshold=payload.get("threshold"), strategy=str(payload.get("strategy_name") or "SYSTEM"),
                    run_id=str(payload.get("run_id") or self.run_dir.name), event_id=str(event.get("event_id")),
                    details=str(payload.get("message") or payload.get("reason") or kind))
                if self.state.record_event_alert(alert): out.append(alert)
        if rows:
            # Advance only after incident records have been durably journaled.
            self.state.state["event_offset"] = end_offset
            self.state._save()
        return out

    def _slippage_alert(self, event: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.config.slippage_warning_ticks is None:
            return None
        reference = payload.get("reference_price")
        fill = payload.get("price", payload.get("fill_price"))
        ticks = payload.get("artificial_slippage_ticks")
        if reference is None and payload.get("broker_order_id"):
            db_path = self.run_dir / "paper_analytics.sqlite3"
            try:
                with sqlite3.connect(_readonly_sqlite_uri(db_path), uri=True, timeout=0.15) as db:
                    row = db.execute("SELECT reference_price FROM orders WHERE order_id=? AND reference_price IS NOT NULL ORDER BY rowid DESC LIMIT 1",
                                     (str(payload["broker_order_id"]),)).fetchone()
                reference = row[0] if row else None
            except (sqlite3.Error, OSError):
                reference = None
        if reference is not None and fill is not None:
            try: ticks = abs(float(fill) - float(reference)) / 0.25
            except (TypeError, ValueError): return None
        if not isinstance(ticks, (float, int)) or ticks <= self.config.slippage_warning_ticks:
            return None
        severity = "CRITICAL" if self.config.slippage_critical_ticks is not None and ticks >= self.config.slippage_critical_ticks else "WARNING"
        return make_alert(alert_type="Slippage > threshold", severity=severity, expected=f"<= {self.config.slippage_warning_ticks} ticks",
            observed=f"{ticks:g} modeled ticks", threshold=self.config.slippage_warning_ticks, strategy=str(payload.get("strategy_name") or "SYSTEM"),
            run_id=str(payload.get("run_id") or self.run_dir.name), event_id=str(event.get("event_id")),
            details="SIMULATED execution slippage; not an observed broker fill.")

    def _latency_alert(self, event: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if str(payload.get("market_data_provenance", "")).upper() == "RECOVERED_PAPER": return None
        raw_ms = payload.get("signal_to_order_latency_ms", payload.get("execution_processing_latency_ms"))
        if self.config.execution_latency_warning_ms is not None and isinstance(raw_ms, (int, float)) and raw_ms > self.config.execution_latency_warning_ms:
            return make_alert(alert_type="Execution latency abnormal", severity="WARNING", expected=f"<= {self.config.execution_latency_warning_ms} ms",
                observed=f"{raw_ms:g} ms processing latency", threshold=self.config.execution_latency_warning_ms,
                strategy=str(payload.get("strategy_name") or "SYSTEM"), run_id=str(payload.get("run_id") or self.run_dir.name),
                event_id=str(event.get("event_id")), details="Persisted Paper processing latency; no broker submission latency is measured.")
        observed = _utc(payload.get("market_data_first_observed_at_utc")); finalized = _utc(payload.get("market_data_finalized_at_utc"))
        if self.config.acquisition_latency_warning_seconds is not None and observed and finalized:
            delay = (finalized - observed).total_seconds()
            if delay > self.config.acquisition_latency_warning_seconds:
                return make_alert(alert_type="Data acquisition latency abnormal", severity="WARNING", expected=f"<= {self.config.acquisition_latency_warning_seconds}s",
                    observed=f"{delay:.1f}s acquisition-to-finalization", threshold=self.config.acquisition_latency_warning_seconds,
                    strategy="SYSTEM", run_id=str(payload.get("run_id") or self.run_dir.name), event_id=str(event.get("event_id")),
                    details="Observed acquisition/finalization delay; distinct from Paper processing and broker latency.")
        return None

    def _position_reconciliation(self, status: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        db_path = self.run_dir / "paper_analytics.sqlite3"
        positions = (status.get("portfolio") or {}).get("open_positions")
        if not isinstance(positions, list) or not db_path.is_file(): return None
        try:
            uri = _readonly_sqlite_uri(db_path)
            with sqlite3.connect(uri, uri=True, timeout=0.15) as db:
                rows = db.execute("SELECT strategy,contract,direction,quantity FROM trades WHERE status='OPEN'").fetchall()
                filled = db.execute("SELECT order_id,quantity,strategy,payload_json FROM orders WHERE lower(status) IN ('paper_order_filled','order_filled') ORDER BY rowid DESC LIMIT 100").fetchall()
                fills = {str(order_id): int(quantity or 0) for order_id, quantity in db.execute(
                    "SELECT order_id,COALESCE(SUM(quantity),0) FROM fills GROUP BY order_id").fetchall()}
            def canon(rows):
                result = {}
                for row in rows:
                    if isinstance(row, sqlite3.Row):
                        strategy, contract, side, qty = row["strategy"], row["contract"], row["direction"], row["quantity"]
                    elif isinstance(row, Mapping):
                        strategy = row.get("strategy") or row.get("strategy_name"); contract = row.get("contract") or row.get("contract_symbol")
                        side = row.get("direction") or row.get("side"); qty = row.get("quantity")
                    else: strategy, contract, side, qty = row
                    if not strategy or not qty: continue
                    key = f"{strategy}|{contract or 'MNQZ6'}|{str(side or '').lower()}"
                    result[key] = result.get(key, 0) + int(qty)
                return result
            expected, observed = canon(rows), canon(positions)
            alerts = []
            if expected != observed and set(expected) != set(observed):
                alerts.append(make_alert(alert_type="Unexpected position", severity="CRITICAL", expected=expected, observed=observed,
                    details="Paper open-position snapshot differs from the durable open-trade ledger.", run_id=str(status.get("run_id") or self.run_dir.name)))
            elif expected != observed:
                for key in expected:
                    if expected[key] != observed[key]:
                        key_strategy, key_contract, key_side = key.split("|", 2)
                        alerts.append(make_alert(alert_type="Position size mismatch", severity="CRITICAL", expected=expected[key], observed=observed[key],
                            strategy=key_strategy, details=f"Persisted position quantity differs for {key_contract} {key_side}.", run_id=str(status.get("run_id") or self.run_dir.name)))
            for order_id, intended, strategy, payload_json in filled:
                actual = fills.get(str(order_id), 0)
                if intended is not None and int(intended) != actual:
                    detail = json.loads(payload_json or "{}")
                    alerts.append(make_alert(alert_type="Position size mismatch", severity="CRITICAL",
                        expected=int(intended), observed=actual, strategy=strategy or str(detail.get("strategy_name") or "SYSTEM"),
                        details=f"Simulated order {order_id} is marked filled but cumulative persisted fill quantity differs.",
                        run_id=str(status.get("run_id") or self.run_dir.name), event_id=f"order:{order_id}:size"))
            return alerts
        except (sqlite3.Error, OSError, ValueError, TypeError):
            return None

    def poll_once(self, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("alert polling timestamp must be timezone-aware")
        status_path = self.run_dir / "status.json"
        if status_path.is_file():
            try: status = json.loads(status_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError): status = {}
        else: status = {}
        system = status.get("system") if isinstance(status.get("system"), dict) else {}
        market_state = evaluate_cme_market_state(self.calendar, now)
        system["market_state"] = market_state
        feed = system.get("feed_health") if isinstance(system.get("feed_health"), dict) else {}
        feed["market_open"] = market_state["state"] == "MARKET_OPEN"
        system["feed_health"] = feed
        status["system"] = system
        # Version-1 monitor snapshots used to mislabel an engine ERROR state as
        # unexpected process termination. Retire that classification only when
        # the current persisted status itself records ERROR/FAILED; the separate
        # watchdog remains authoritative for actual process disappearance.
        if str(system.get("state", "")).upper() in {"ERROR", "FAILED"}:
            for alert_id, old in list(self.state.state["active"].items()):
                if (old.get("alert_type") == "Unexpected Paper Engine termination"
                        and str(old.get("observed", "")).upper() in {"ERROR", "FAILED"}
                        and old.get("source_event_id") is None):
                    self.state.acknowledge(alert_id, resolved=True)
        risk_limits = _read_risk_limits(self.run_dir / "paper_analytics.sqlite3")
        conditions, capabilities = evaluate_status(status, config=self.config, risk_limits=risk_limits, now=now)
        # Stall alerts persist for one minute before notification. A direct API
        # disconnect/error remains immediate and is never mislabeled as a stall.
        pending = self.state.state.setdefault("stall_candidates", {})
        stall_ids = {a["alert_id"] for a in conditions if a["alert_type"] in {"Provider data stalled", "Paper processing stalled"}}
        for key in list(pending):
            if key not in stall_ids:
                del pending[key]
        debounced = []
        for alert in conditions:
            if alert["alert_type"] not in {"Provider data stalled", "Paper processing stalled"}:
                debounced.append(alert); continue
            first = pending.setdefault(alert["alert_id"], now.astimezone(UTC).isoformat())
            elapsed = (now.astimezone(UTC) - (_utc(first) or now.astimezone(UTC))).total_seconds()
            alert["details"] += f" Debounce: {max(0.0, elapsed):.0f}/{self.config.stall_debounce_seconds:.0f}s."
            if elapsed >= self.config.stall_debounce_seconds:
                debounced.append(alert)
        conditions = debounced
        position_alerts = self._position_reconciliation(status)
        if position_alerts is None:
            capabilities["position_reconciliation"] = "UNAVAILABLE_OR_READ_FAILED"
        else:
            capabilities["position_reconciliation"] = "AVAILABLE"
            conditions.extend(position_alerts)
        emitted = self.state.observe_conditions(conditions)
        self.state.state["capabilities"] = capabilities
        emitted.extend(self._event_alerts())
        self.state.state["last_evaluated_utc"] = now.astimezone(UTC).isoformat()
        self.state.state["market_state"] = market_state
        acquisition_connected = feed.get("connected")
        provider_connected = feed.get("provider_connected")
        if isinstance(acquisition_connected, bool) and isinstance(provider_connected, bool) and acquisition_connected != provider_connected:
            connection = "CONNECTION_FIELDS_CONFLICT"
        elif acquisition_connected is False or provider_connected is False:
            connection = "DISCONNECTED"
        elif acquisition_connected is True and provider_connected is True:
            connection = "CONNECTED"
        else:
            connection = "UNKNOWN"
        allowance = max(self.config.provider_delay_seconds, self.config.finalization_min_age_seconds) + self.config.finalization_confirmation_spacing_seconds
        frontier = now.astimezone(UTC).timestamp() - allowance
        frontier -= frontier % 60
        latest_bar = _utc(feed.get("latest_available_bar"))
        self.state.state["feed_assessment"] = {
            "connection": connection,
            "acquisition_connected": acquisition_connected,
            "provider_connected": provider_connected,
            "schedule_state": market_state["state"],
            "connection_error_code_1100": bool(feed.get("last_error") and "1100" in str(feed.get("last_error"))),
            "operator_reported_maintenance": os.environ.get("MNQ_TWS_MAINTENANCE_NOTE"),
            "last_provider_bar_utc": latest_bar.isoformat() if latest_bar else None,
            "last_committed_bar_utc": feed.get("last_processed_bar") or system.get("last_bar"),
            "latest_observed_bar_epoch_utc": feed.get("latest_observed_bar_epoch_utc"),
            "acquisition_cursor_epoch_utc": feed.get("backfill_next_start_epoch_utc"),
            "backlog_bars": feed.get("backlog_bars"),
            "last_error": feed.get("last_error"),
            "expected_provider_frontier_utc": datetime.fromtimestamp(frontier, UTC).isoformat() if market_state["state"] == "MARKET_OPEN" else None,
            "provider_bar_age_seconds": round(max(0.0, (now.astimezone(UTC) - latest_bar).total_seconds()), 1) if latest_bar else None,
            "delay_budget_seconds": self.config.provider_delay_seconds,
            "finalization_min_age_seconds": self.config.finalization_min_age_seconds,
            "confirmation_spacing_seconds": self.config.finalization_confirmation_spacing_seconds,
            "expected_frontier_lag_seconds": max(self.config.provider_delay_seconds, self.config.finalization_min_age_seconds) + self.config.finalization_confirmation_spacing_seconds,
        }
        self.state.state["thresholds"] = monitoring_thresholds(self.config, risk_limits)
        self.state.state["capabilities"].update({
            "order_rejection": "EVENT_DRIVEN_SIMULATED_ONLY",
            "slippage": "UNCONFIGURED_NO_NONRECOVERED_SAMPLE" if self.config.slippage_warning_ticks is None else "SIMULATED_ONLY_THRESHOLD_CONFIGURED",
            "position_size_reconciliation": "EVENT_AND_LEDGER_LIMITED",
            "strategy_invariant_monitoring": "EXPLICIT_DIAGNOSTIC_EVENTS_ONLY",
            "execution_latency": "UNCONFIGURED_NO_NONRECOVERED_DISTRIBUTION" if self.config.execution_latency_warning_ms is None else "PERSISTED_PAPER_LATENCY_THRESHOLD_CONFIGURED",
            "acquisition_latency": "UNCONFIGURED_NO_NONRECOVERED_DISTRIBUTION" if self.config.acquisition_latency_warning_seconds is None else "PERSISTED_FINALIZATION_LATENCY_THRESHOLD_CONFIGURED",
            "calendar_roll_reminders": "REVIEWED_SCHEDULE_ONLY",
            "broker_order_status": "LIVE_ADAPTER_DISABLED",
            "broker_fill_slippage": "LIVE_ADAPTER_DISABLED",
            "broker_position_reconciliation": "LIVE_ADAPTER_DISABLED",
        })
        self.state._save()
        return {"emitted": emitted, "state": self.state.state}


def monitoring_thresholds(config: AlertConfig, risk_limits: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expose provenance for each threshold; unset metrics stay visibly unset."""
    return {
        "daily_loss_limit_usd": {"value": (risk_limits or {}).get("max_daily_loss"),
                                 "source": "persisted Paper risk policy", "kind": "existing hard engine limit"},
        "provider_delay_seconds": {"value": config.provider_delay_seconds, "source": "approved delayed IBKR operating mode", "kind": "monitoring baseline"},
        "finalization_min_age_seconds": {"value": config.finalization_min_age_seconds, "source": "IBKR FinalizationPolicy", "kind": "finalization rule"},
        "confirmation_spacing_seconds": {"value": config.finalization_confirmation_spacing_seconds, "source": "IBKR FinalizationPolicy", "kind": "finalization rule"},
        "provider_stall_warning_seconds_beyond_expected_frontier": {"value": config.provider_stall_warning_seconds, "source": "MNQ_ALERT_PROVIDER_STALL_WARNING_SECONDS", "kind": "monitoring-only; default 300s"},
        "provider_stall_critical_seconds_beyond_expected_frontier": {"value": config.provider_stall_critical_seconds, "source": "MNQ_ALERT_PROVIDER_STALL_CRITICAL_SECONDS", "kind": "monitoring-only; default 900s"},
        "processing_stall_warning_seconds": {"value": config.processing_stall_warning_seconds, "source": "MNQ_ALERT_PROCESSING_STALL_WARNING_SECONDS", "kind": "monitoring-only; default 300s"},
        "processing_stall_critical_seconds": {"value": config.processing_stall_critical_seconds, "source": "MNQ_ALERT_PROCESSING_STALL_CRITICAL_SECONDS", "kind": "monitoring-only; default 900s"},
        "stall_debounce_seconds": {"value": config.stall_debounce_seconds, "source": "MNQ_ALERT_STALL_DEBOUNCE_SECONDS", "kind": "persistence debounce; default 60s"},
        "slippage_warning_ticks": {"value": config.slippage_warning_ticks, "source": "MNQ_ALERT_SLIPPAGE_WARNING_TICKS", "kind": "unconfigured until non-recovered sample exists"},
        "slippage_critical_ticks": {"value": config.slippage_critical_ticks, "source": "MNQ_ALERT_SLIPPAGE_CRITICAL_TICKS", "kind": "unconfigured until non-recovered sample exists"},
        "execution_latency_warning_ms": {"value": config.execution_latency_warning_ms, "source": "MNQ_ALERT_EXECUTION_LATENCY_WARNING_MS", "kind": "unconfigured until non-recovered timing distribution exists"},
        "acquisition_latency_warning_seconds": {"value": config.acquisition_latency_warning_seconds, "source": "MNQ_ALERT_ACQUISITION_LATENCY_WARNING_SECONDS", "kind": "unconfigured until non-recovered timing distribution exists"},
        "drawdown_warning_usd": {"value": config.drawdown_warning_usd, "source": "MNQ_ALERT_DRAWDOWN_WARNING_USD", "kind": "unconfigured; no separate account drawdown threshold"},
        "drawdown_critical_usd": {"value": config.drawdown_critical_usd, "source": "MNQ_ALERT_DRAWDOWN_CRITICAL_USD", "kind": "unconfigured; no separate account drawdown threshold"},
    }
