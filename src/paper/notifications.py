"""Read-only, persistent notifications for delayed Paper event JSONL.

Delivery is at-least-once across an ambiguous network failure: Telegram does
not expose an idempotency key, so a process crash after Telegram accepts a
message but before the local journal fsync can result in a duplicate. Trading
state and the Paper engine are never modified by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.paper.atomic_io import atomic_replace_with_retry

UTC = timezone.utc


@dataclass(frozen=True)
class NotificationPreferences:
    trading: bool = True
    account: bool = True
    system: bool = True
    minimum_severity: str = "INFO"
    minimum_interval_seconds: float = 3.0
    recovered_trade_summary: bool = False
    significant_equity_change_usd: float = 500.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_interval_seconds) or self.minimum_interval_seconds < 0:
            raise ValueError("notification minimum interval must be finite and non-negative")
        if not math.isfinite(self.significant_equity_change_usd) or self.significant_equity_change_usd < 0:
            raise ValueError("equity alert threshold must be finite and non-negative")

    @classmethod
    def from_environment(cls) -> "NotificationPreferences":
        def enabled(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}
        severity = os.environ.get("MNQ_NOTIFY_MIN_SEVERITY", "INFO").strip().upper()
        if severity not in {"INFO", "WARNING", "CRITICAL"}:
            raise ValueError("MNQ_NOTIFY_MIN_SEVERITY must be INFO, WARNING or CRITICAL")
        return cls(
            trading=enabled("MNQ_NOTIFY_TRADING", True),
            account=enabled("MNQ_NOTIFY_ACCOUNT", True),
            system=enabled("MNQ_NOTIFY_SYSTEM", True),
            minimum_severity=severity,
            minimum_interval_seconds=float(os.environ.get("MNQ_NOTIFY_MIN_INTERVAL_SECONDS", "3")),
            recovered_trade_summary=enabled("MNQ_NOTIFY_RECOVERED_SUMMARY", False),
            significant_equity_change_usd=float(os.environ.get("MNQ_NOTIFY_EQUITY_CHANGE_USD", "500")),
        )


_LEVEL = {"INFO": 10, "WARNING": 20, "CRITICAL": 30}


def format_event(event: Mapping[str, Any], preferences: NotificationPreferences) -> tuple[str, str] | None:
    """Map persisted Paper events to concise messages; ignore recovered trades."""
    event_type = str(event.get("event_type", "")).lower()
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    if event_type == "monitoring_alert":
        status = str(payload.get("status", "ACTIVE")).upper()
        severity = str(payload.get("severity", "WARNING")).upper()
        if status in {"RECOVERED", "RESOLVED"}:
            severity = "INFO"
        if _LEVEL.get(severity, 10) < _LEVEL.get(preferences.minimum_severity, 10):
            return None
        name = str(payload.get("alert_type", "Paper monitoring alert"))
        icon = "✅" if status in {"RECOVERED", "RESOLVED"} else "🚨" if severity == "CRITICAL" else "⚠️"
        lines = [f"{icon} {name}", f"Environment: {payload.get('environment', 'PAPER')}",
                 f"Strategy: {payload.get('strategy', 'SYSTEM')}", f"Symbol: {payload.get('symbol', 'MNQ')}",
                 f"Detected: {payload.get('timestamp_utc', event.get('timestamp', 'unavailable'))}",
                 f"Expected: {payload.get('expected') if payload.get('expected') is not None else 'unavailable'}",
                 f"Observed: {payload.get('observed') if payload.get('observed') is not None else 'unavailable'}"]
        if payload.get("threshold") is not None:
            lines.append(f"Threshold: {payload['threshold']}")
        lines.extend([f"Status: {status}", f"Details: {payload.get('details') or 'unavailable'}",
                      f"Run: {payload.get('account_run_id', 'unavailable')}"])
        return severity, "\n".join(lines)[:3900]
    severity = str(payload.get("severity", "INFO")).upper()
    category: str | None = None
    title = ""
    if event_type == "paper_position_opened":
        category, title = "trading", "Simulated Paper position opened"
    elif event_type in {"paper_position_closed", "trade_closed"}:
        category, title = "trading", "Simulated Paper position closed"
    elif event_type in {"risk_rejection", "risk_warning", "trading_halted", "risk_limit_exceeded", "account_floor_breach"}:
        category, title, severity = "account", "Paper risk limit or halt", "CRITICAL" if event_type in {"trading_halted", "account_floor_breach"} else "WARNING"
    elif event_type == "risk_decision":
        if payload.get("approved") is False or payload.get("accepted") is False or "reject" in str(payload.get("decision", "")).lower():
            category, title, severity = "account", "Paper risk gate rejected a trade", "WARNING"
    elif event_type in {"feed_disconnected", "ibkr_disconnected"}:
        category, title, severity = "system", "IBKR delayed feed disconnected", "CRITICAL"
    elif event_type in {"feed_connected", "ibkr_reconnected"}:
        category, title = "system", "IBKR delayed feed connected"
    elif event_type == "system_stopped":
        stopped_state = str(payload.get('state', 'UNKNOWN')).upper()
        category = 'system'
        if stopped_state in {'ERROR', 'FAILED'}:
            title, severity = 'Paper engine stopped with an error', 'CRITICAL'
        elif stopped_state == 'STOPPED':
            title, severity = 'Paper engine stopped cleanly', 'INFO'
        else:
            title, severity = 'Paper engine stopped; outcome unverified', 'WARNING'
    elif event_type == "paper_process_terminated":
        category, title, severity = "system", "Paper Engine process disappeared unexpectedly", "CRITICAL"
    elif event_type == "paper_process_recovered":
        category, title = "system", "Paper Engine process detected again"
    elif event_type == "paper_status_heartbeat_stale":
        category, title, severity = "system", "Paper Engine status heartbeat is stale", "CRITICAL"
    elif event_type == "paper_status_heartbeat_recovered":
        category, title, severity = "system", "Paper Engine status heartbeat recovered", "INFO"
    elif event_type == "paper_notification_service_stale":
        category, title, severity = "system", "Paper notification service heartbeat is stale", "CRITICAL"
    elif event_type == "paper_notification_service_recovered":
        category, title, severity = "system", "Paper notification service heartbeat recovered", "INFO"
    elif event_type in {"system_error", "checkpoint_recovery_failed", "recovery_failed"}:
        category, title, severity = "system", "Paper engine or recovery failed", "CRITICAL"
    elif event_type in {"system_critical", "error", "hmm_refit_failed", "checkpoint_failed", "backfill_failed"}:
        category, title, severity = "system", event_type.replace("_", " ").title(), "CRITICAL"
    elif event_type in {"feed_gap", "unresolved_expected_minute_gap", "data_gap", "missing_bar", "out_of_order_bar"}:
        category, title, severity = "system", "Unresolved market-data gap", "CRITICAL"
    elif event_type in {"backlog_warning", "feed_stalled", "feed_stale", "recovery_started", "backfill_started"}:
        category, title, severity = "system", event_type.replace("_", " ").title(), "WARNING"
    elif event_type in {"system_recovered", "feed_recovered", "backfill_completed", "checkpoint_restored"}:
        category, title = "system", event_type.replace("_", " ").title()
    elif event_type in {"system_warning", "contract_roll", "parity_warning"}:
        category, title, severity = "system", event_type.replace("_", " ").title(), "WARNING"
    elif event_type == "risk_warning":
        category, title, severity = "account", "Paper risk warning", "WARNING"
    elif event_type in {"drawdown_threshold_exceeded", "account_floor_warning", "account_floor_breach", "daily_summary", "significant_equity_change"}:
        category, title = "account", event_type.replace("_", " ").title()
    else:
        return None

    recovered = str(payload.get("market_data_provenance", "")).upper() == "RECOVERED_PAPER"
    if recovered:
        if category == "trading":
            if not preferences.recovered_trade_summary:
                return None
            title = "RECOVERED PAPER · " + title
        elif event_type not in {"system_error", "system_critical", "checkpoint_recovery_failed",
                               "recovery_failed", "hmm_refit_failed", "checkpoint_failed", "backfill_failed"}:
            # Catch-up market-time events are not presented as new live alerts.
            return None
    enabled = {"trading": preferences.trading, "account": preferences.account, "system": preferences.system}[category]
    if not enabled or _LEVEL.get(severity, 10) < _LEVEL[preferences.minimum_severity]:
        return None
    strategy = payload.get("strategy_name") or payload.get("strategy")
    side = payload.get("side") or payload.get("direction")
    quantity = payload.get("quantity")
    entry = payload.get("entry_price") or payload.get("entry_fill_price")
    exit_price = payload.get("exit_price") or payload.get("exit_fill_price")
    pnl = payload.get("net_pnl", payload.get("realized_pnl"))
    details = []
    if strategy: details.append(f"Strategy: {strategy}")
    if side: details.append(f"Side: {side}")
    if quantity is not None: details.append(f"Qty: {quantity}")
    contract = payload.get("contract_symbol") or payload.get("symbol")
    if contract: details.append(f"Contract: {contract}")
    if entry is not None: details.append(f"Entry: {entry}")
    if exit_price is not None: details.append(f"Exit: {exit_price}")
    if pnl is not None:
        try:
            details.append(f"Realized P&L: {float(pnl):+.2f} USD")
        except (TypeError, ValueError):
            details.append("Realized P&L: unavailable")
    if payload.get("market_data_provenance"):
        details.append(f"Provenance: {payload['market_data_provenance']}")
    if payload.get("stop_price") is not None: details.append(f"Stop: {payload['stop_price']}")
    if payload.get("target_price") is not None: details.append(f"Target: {payload['target_price']}")
    if payload.get("reason"): details.append(f"Reason: {payload['reason']}")
    stamp = str(event.get("timestamp", "unavailable"))
    body = "\n".join([f"MNQ PAPER · {severity}", title, *details, f"Event time: {stamp}"])
    return severity, body[:3900]


class TelegramTransport:
    """Minimal Telegram Bot API sender; credentials are never included in errors."""
    def __init__(self, token: str, chat_id: str, *, timeout_seconds: float = 8.0) -> None:
        if not token or not chat_id:
            raise ValueError("Telegram token and chat ID must be configured in environment")
        self._token, self._chat_id = token, chat_id
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _classify_http_error(status: int, body: bytes, headers: Any) -> tuple[str, bool, float | None]:
        """Return a safe category, retryability, and Telegram's Retry-After.

        Response text is intentionally never persisted: Telegram descriptions can
        contain chat details and are not needed for operational classification.
        """
        retry_after = None
        try:
            raw = headers.get("Retry-After") if headers is not None else None
            if raw is not None:
                retry_after = max(0.0, float(raw))
        except (TypeError, ValueError):
            retry_after = None
        description = ""
        parameters: Mapping[str, Any] = {}
        try:
            payload = json.loads(body.decode("utf-8"))
            description = str(payload.get("description", "")).lower() if isinstance(payload, dict) else ""
            parameters = payload.get("parameters", {}) if isinstance(payload, dict) else {}
        except (UnicodeError, json.JSONDecodeError):
            pass
        if status == 401:
            return "authentication", False, None
        if status == 429:
            if retry_after is None:
                try:
                    retry_after = max(0.0, float(parameters.get("retry_after")))
                except (TypeError, ValueError, AttributeError):
                    pass
            return "rate_limited", True, retry_after
        if status == 400:
            if any(term in description for term in ("chat not found", "bot was blocked", "user not found", "chat_id")):
                return "chat_id_or_access", False, None
            if any(term in description for term in ("parse entities", "can't parse", "unsupported start tag", "message is too long")):
                return "message_format", False, None
            return "request_rejected", False, None
        if status in {403, 404}:
            return "chat_id_or_access", False, None
        if status >= 500:
            return "telegram_server", True, retry_after
        return "http_error", status >= 500, retry_after

    def __call__(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        body = json.dumps({"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True}).encode()
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read(64_000))
        except Exception as exc:
            from urllib.error import HTTPError, URLError
            if isinstance(exc, HTTPError):
                try:
                    response_body = exc.read(64_000)
                except Exception:
                    response_body = b""
                category, retryable, retry_after = self._classify_http_error(
                    int(exc.code), response_body, exc.headers
                )
                raise TelegramDeliveryError(category, retryable=retryable,
                                            http_status=int(exc.code),
                                            retry_after_seconds=retry_after) from None
            if isinstance(exc, URLError):
                reason = getattr(exc, "reason", None)
                category = "network_timeout" if isinstance(reason, TimeoutError) else "network"
            elif isinstance(exc, TimeoutError):
                category = "network_timeout"
            elif isinstance(exc, (json.JSONDecodeError, UnicodeError)):
                category = "invalid_response"
            else:
                category = "transport_error"
            raise TelegramDeliveryError(category, retryable=True) from None
        if not isinstance(result, Mapping):
            raise TelegramDeliveryError("invalid_response", retryable=True)
        if not result.get("ok"):
            # HTTP 200 with ok=false is unusual but still a permanent API reject.
            code = result.get("error_code")
            category, retryable, retry_after = self._classify_http_error(
                int(code) if isinstance(code, int) else 400,
                json.dumps(result).encode("utf-8"), None,
            )
            raise TelegramDeliveryError(category, retryable=retryable,
                                        http_status=code if isinstance(code, int) else None,
                                        retry_after_seconds=retry_after)


class TelegramDeliveryError(RuntimeError):
    """Sanitized delivery failure metadata, safe for the persistent journal."""
    def __init__(self, category: str, *, retryable: bool,
                 http_status: int | None = None,
                 retry_after_seconds: float | None = None) -> None:
        self.category = str(category)
        self.retryable = bool(retryable)
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Telegram delivery {self.category}")


class PersistentRateLimiter:
    """Cross-process send interval shared by notifier and watchdog."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(self, now: float, interval_seconds: float) -> bool:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+b") as lock:
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0"); lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    value = json.loads(self.path.read_text(encoding="utf-8"))
                    last = float(value.get("last_reserved_at", 0.0))
                except FileNotFoundError:
                    last = 0.0
                except (json.JSONDecodeError, TypeError, ValueError):
                    raise RuntimeError("shared Telegram rate-limit state is corrupt") from None
                if now - last < interval_seconds:
                    return False
                fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump({"schema_version": 1, "last_reserved_at": now}, f)
                        f.flush(); os.fsync(f.fileno())
                    atomic_replace_with_retry(name, self.path)
                finally:
                    if os.path.exists(name): os.unlink(name)
                return True
            finally:
                lock.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class PersistentEventNotifier:
    """Poll an append-only engine event file and deliver eligible events.

    On first run, begins at EOF so recovered historical events are never
    replayed as if they were fresh. Cursor and pending-send state are atomic.
    """
    def __init__(self, *, event_path: str | Path, state_path: str | Path,
                 journal_path: str | Path, send: Callable[[str], None],
                 preferences: NotificationPreferences = NotificationPreferences(),
                 clock: Callable[[], float] = time.time,
                 rate_limiter: PersistentRateLimiter | None = None,
                 exclude_event_types: frozenset[str] = frozenset(),
                 replay_existing: bool = False) -> None:
        self.event_path, self.state_path, self.journal_path = Path(event_path), Path(state_path), Path(journal_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.send, self.preferences, self.clock = send, preferences, clock
        self.rate_limiter = rate_limiter
        self.exclude_event_types = frozenset(str(x).lower() for x in exclude_event_types)
        self._delivered_ids: set[str] = set()
        if self.journal_path.exists():
            with self.journal_path.open(encoding="utf-8") as journal:
                for line_number, line in enumerate(journal, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"notification delivery journal is corrupt at line {line_number}") from None
                    if not isinstance(record, dict):
                        raise RuntimeError(f"notification delivery journal has invalid record at line {line_number}")
                    if record.get("state") == "delivered" and record.get("event_id"):
                        self._delivered_ids.add(str(record["event_id"]))
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("schema_version") != 1 or self.state.get("event_path") != str(self.event_path.resolve()):
                raise ValueError("notification state is incompatible with this event source")
        else:
            offset = (self.event_path.stat().st_size if self.event_path.exists() and not replay_existing else 0)
            discard_partial = False
            if offset:
                with self.event_path.open("rb") as source:
                    source.seek(offset - 1)
                    discard_partial = source.read(1) != b"\n"
            self.state = {"schema_version": 1, "event_path": str(self.event_path.resolve()),
                          "offset": offset, "pending": None, "last_sent_at": 0.0,
                          "attempts": 0, "next_attempt_at": 0.0,
                          "status_signature": None, "status_bad": False,
                          "status_pending": None, "status_attempts": 0,
                          "status_next_attempt_at": 0.0,
                          "discard_partial_until_newline": discard_partial}
            self._save()

    def _save(self) -> None:
        fd, name = tempfile.mkstemp(prefix=self.state_path.name + ".", suffix=".tmp", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, sort_keys=True, separators=(",", ":"))
                handle.flush(); os.fsync(handle.fileno())
            atomic_replace_with_retry(name, self.state_path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def _journal(self, record: Mapping[str, Any]) -> None:
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())

    def _next_line(self) -> tuple[dict[str, Any], int] | None:
        if not self.event_path.exists(): return None
        size = self.event_path.stat().st_size
        offset = int(self.state["offset"])
        if size < offset:
            raise RuntimeError("Paper event log shrank below the durable notification cursor")
        with self.event_path.open("rb") as handle:
            handle.seek(offset)
            if self.state.get("discard_partial_until_newline"):
                remainder = handle.readline()
                if not remainder.endswith(b"\n"):
                    return None
                self.state["offset"] = handle.tell()
                self.state["discard_partial_until_newline"] = False
                self._save()
                offset = int(self.state["offset"])
            handle.seek(offset); line = handle.readline()
            if not line or not line.endswith(b"\n"): return None
            end = handle.tell()
        try: event = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid complete Paper event at byte {offset}") from exc
        if not isinstance(event, dict) or not event.get("event_id"):
            raise RuntimeError(f"Paper event lacks durable event_id at byte {offset}")
        return event, end

    def scan_once(self, *, max_skips: int = 500) -> str:
        now = float(self.clock())
        if self.state.get("delivery_blocked"):
            return "blocked"
        pending = self.state.get("pending")
        if pending is None:
            skipped_any = False
            for skipped in range(max_skips + 1):
                next_item = self._next_line()
                if next_item is None:
                    if skipped_any:
                        self._save()
                    return "skipped" if skipped else "idle"
                event, end = next_item
                if str(event["event_id"]) in self._delivered_ids:
                    self.state["offset"] = end
                    self._journal({"event_id": event["event_id"], "state": "duplicate_suppressed",
                                   "recorded_at_utc": datetime.now(UTC).isoformat()})
                    skipped_any = True
                    continue
                if (str(event.get("event_type", "")).lower() == "market_data"
                        and isinstance(event.get("payload"), Mapping)
                        and str(event["payload"].get("market_data_provenance", "")).upper() != "RECOVERED_PAPER"
                        and event.get("timestamp")):
                    self.state["last_live_market_bar_utc"] = str(event["timestamp"])
                if str(event.get("event_type", "")).lower() in self.exclude_event_types:
                    self.state["offset"] = end
                    skipped_any = True
                    continue
                formatted = format_event(event, self.preferences)
                if formatted is None:
                    self.state["offset"] = end
                    skipped_any = True
                    continue
                severity, message = formatted
                pending = {"event_id": event["event_id"], "end_offset": end,
                           "severity": severity, "message": message}
                self.state.update(pending=pending, attempts=0, next_attempt_at=0.0)
                self._journal({"event_id": pending["event_id"], "state": "queued", "severity": severity,
                               "recorded_at_utc": datetime.now(UTC).isoformat()})
                self._save()
                break
            if pending is None:
                if skipped_any:
                    self._save()
                return "scan_limit"
        if now < float(self.state.get("next_attempt_at", 0.0)):
            return "backoff"
        if now - float(self.state.get("last_sent_at", 0.0)) < self.preferences.minimum_interval_seconds:
            return "rate_limited"
        if self.rate_limiter and not self.rate_limiter.reserve(now, self.preferences.minimum_interval_seconds):
            return "rate_limited"
        try:
            self.send(str(pending["message"]))
        except Exception as exc:
            attempts = int(self.state.get("attempts", 0)) + 1
            retryable = bool(getattr(exc, "retryable", True))
            delay = min(900.0, 5.0 * (2 ** min(attempts - 1, 8)))
            server_delay = getattr(exc, "retry_after_seconds", None)
            if server_delay is not None:
                delay = max(delay, float(server_delay))
            category = str(getattr(exc, "category", "unclassified"))
            self.state.update(attempts=attempts, next_attempt_at=now + delay,
                              delivery_blocked=not retryable,
                              delivery_failure_category=category)
            record = {"event_id": pending["event_id"], "state": "retry" if retryable else "blocked",
                      "attempt": attempts,
                      "retry_after_seconds": delay if retryable else None,
                      "error_type": type(exc).__name__, "failure_category": category,
                      "http_status": getattr(exc, "http_status", None),
                      "recorded_at_utc": datetime.now(UTC).isoformat()}
            self._journal(record)
            self._save()
            return "retry" if retryable else "blocked"
        self._journal({"event_id": pending["event_id"], "state": "delivered",
                       "recorded_at_utc": datetime.now(UTC).isoformat()})
        self._delivered_ids.add(str(pending["event_id"]))
        self.state.update(offset=int(pending["end_offset"]), pending=None,
                          attempts=0, next_attempt_at=0.0, last_sent_at=now,
                          delivery_blocked=False, delivery_failure_category=None)
        self._save()
        return "delivered"

    def scan_status(self, status_path: str | Path, *, backlog_threshold: int = 60,
                    feed_stall_seconds: int = 300, monitor_incidents: bool = True) -> str:
        """Send deduplicated incidents from the run's persisted status snapshot."""
        path = Path(status_path)
        if not path.is_file():
            return "unavailable"
        try:
            status = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return "unavailable"
        system = status.get("system") if isinstance(status.get("system"), dict) else {}
        feed = system.get("feed_health") if isinstance(system.get("feed_health"), dict) else {}
        engine_state = str(system.get("state", "UNAVAILABLE")).upper()
        feed_mode = str(feed.get("mode", "UNAVAILABLE")).upper()
        connected = feed.get("connected")
        backlog = feed.get("backlog_bars")
        error = str(feed.get("last_error") or "")
        incident = None
        if monitor_incidents and engine_state in {"ERROR", "FAILED"}:
            incident = ("CRITICAL", "Paper engine failed", "engine:" + engine_state)
        elif monitor_incidents and error:
            incident = ("CRITICAL", "IBKR acquisition/recovery error", "feed-error:" + error[:240])
        elif monitor_incidents and (connected is False or feed_mode == "DISCONNECTED"):
            incident = ("CRITICAL", "IBKR delayed feed disconnected", "feed-disconnected")
        elif monitor_incidents and (feed.get("stale") is True or system.get("feed_stale") is True):
            incident = ("WARNING", "IBKR delayed feed is marked stale", "feed-stale")
        elif monitor_incidents and backlog is not None:
            try:
                backlog_count = int(backlog)
            except (TypeError, ValueError):
                backlog_count = -1
            if backlog_count > backlog_threshold:
                incident = ("WARNING", f"Delayed Paper backlog is {backlog_count} bars", "backlog-over-threshold")
        elif monitor_incidents and engine_state == "UNAVAILABLE":
            incident = ("WARNING", "Paper engine monitoring status is unavailable", "engine:UNAVAILABLE")
        elif monitor_incidents and feed.get("market_open") is True and feed.get("last_processed_bar") and feed.get("latest_available_bar"):
            try:
                latest = datetime.fromisoformat(str(feed["latest_available_bar"]).replace("Z", "+00:00"))
                processed = datetime.fromisoformat(str(feed["last_processed_bar"]).replace("Z", "+00:00"))
                lag = (latest - processed).total_seconds()
                if lag > feed_stall_seconds:
                    incident = ("WARNING", f"Paper cursor trails provider by {lag / 60:.1f} minutes", "feed-lag-over-threshold")
            except (TypeError, ValueError):
                pass
        portfolio = status.get("portfolio") if isinstance(status.get("portfolio"), dict) else {}
        try:
            equity = float(portfolio["equity"])
            baseline = self.state.get("equity_notification_reference")
            if baseline is None:
                self.state["equity_notification_reference"] = equity
                self._save()
            elif monitor_incidents and incident is None and abs(equity - float(baseline)) >= self.preferences.significant_equity_change_usd:
                incident = ("WARNING", f"Paper equity changed by {equity-float(baseline):+.2f} USD",
                            f"equity-change:{equity:.2f}")
        except (KeyError, TypeError, ValueError):
            pass

        pending = self.state.get("status_pending")
        now = float(self.clock())
        if pending:
            if self.state.get("status_delivery_blocked"):
                return "blocked"
            if now < float(self.state.get("status_next_attempt_at", 0.0)):
                return "backoff"
            if now - float(self.state.get("last_sent_at", 0.0)) < self.preferences.minimum_interval_seconds:
                return "rate_limited"
            if self.rate_limiter and not self.rate_limiter.reserve(now, self.preferences.minimum_interval_seconds):
                return "rate_limited"
            try:
                self.send(str(pending["message"]))
            except Exception as exc:
                attempts = int(self.state.get("status_attempts", 0)) + 1
                delay = min(900.0, 5.0 * (2 ** min(attempts - 1, 8)))
                retryable = bool(getattr(exc, "retryable", True))
                server_delay = getattr(exc, "retry_after_seconds", None)
                if server_delay is not None:
                    delay = max(delay, float(server_delay))
                category = str(getattr(exc, "category", "unclassified"))
                self.state.update(status_attempts=attempts, status_next_attempt_at=now + delay,
                                  status_delivery_blocked=not retryable,
                                  status_delivery_failure_category=category)
                self._journal({"event_id": pending["event_id"], "state": "retry" if retryable else "blocked",
                               "attempt": attempts,
                               "retry_after_seconds": delay if retryable else None,
                               "error_type": type(exc).__name__, "failure_category": category,
                               "http_status": getattr(exc, "http_status", None),
                               "recorded_at_utc": datetime.now(UTC).isoformat()})
                self._save()
                return "retry" if retryable else "blocked"
            self._journal({"event_id": pending["event_id"], "state": "delivered",
                           "recorded_at_utc": datetime.now(UTC).isoformat()})
            self.state.update(status_signature=pending["signature"], status_bad=bool(pending["bad"]),
                              status_pending=None, status_attempts=0,
                              status_next_attempt_at=0.0, last_sent_at=now,
                              status_delivery_blocked=False, status_delivery_failure_category=None)
            if pending.get("equity_after") is not None:
                self.state["equity_notification_reference"] = float(pending["equity_after"])
            if str(pending.get("signature", "")).startswith("daily-summary:"):
                self.state["daily_summary_date"] = str(pending["signature"]).split(":", 1)[1]
            reminder_key = pending.get("reminder_key")
            if reminder_key:
                sent = set(self.state.get("sent_reminders", []))
                sent.add(str(reminder_key))
                self.state["sent_reminders"] = sorted(sent)
            self._save()
            return "delivered"

        if incident is None and self.preferences.account:
            # Summary is tied to an actual non-recovered market_data event from
            # today's New York session, never to a historical catch-up date.
            live_stamp = self.state.get("last_live_market_bar_utc")
            portfolio = status.get("portfolio") if isinstance(status.get("portfolio"), dict) else {}
            if live_stamp and portfolio.get("daily_pnl") is not None:
                try:
                    from zoneinfo import ZoneInfo
                    now_ny = datetime.fromtimestamp(now, UTC).astimezone(ZoneInfo("America/New_York"))
                    bar_ny = datetime.fromisoformat(str(live_stamp).replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
                    date_key = now_ny.date().isoformat()
                    if (bar_ny.date() == now_ny.date() and now_ny.hour * 60 + now_ny.minute >= 16 * 60 + 15
                            and self.state.get("daily_summary_date") != date_key):
                        event_id = hashlib.sha256((str(path.resolve()) + "|daily-summary|" + date_key).encode()).hexdigest()
                        self.state["status_pending"] = {
                            "event_id": event_id, "signature": "daily-summary:" + date_key, "bad": False,
                            "message": (f"MNQ PAPER · INFO\nDaily simulated Paper summary · {date_key} New York\n"
                                        f"Realized daily P&L: {float(portfolio['daily_pnl']):+.2f} USD\n"
                                        f"Balance: {portfolio.get('balance', 'unavailable')} · Equity: {portfolio.get('equity', 'unavailable')}\n"
                                        f"Last continuous bar: {live_stamp}"),
                        }
                        self.state["status_attempts"] = 0
                        self.state["status_next_attempt_at"] = 0.0
                        self._journal({"event_id": event_id, "state": "queued", "severity": "INFO",
                                       "recorded_at_utc": datetime.now(UTC).isoformat()})
                        self._save()
                        return self.scan_status(path, backlog_threshold=backlog_threshold,
                                                feed_stall_seconds=feed_stall_seconds)
                except (TypeError, ValueError):
                    pass

        prior_bad = bool(self.state.get("status_bad", False))
        if incident is None:
            if not prior_bad:
                return "healthy"
            severity, title, signature, is_bad = "INFO", "Paper monitoring condition cleared", "healthy", False
        else:
            severity, title, signature = incident
            is_bad = True
        if signature == self.state.get("status_signature"):
            return "unchanged"
        if _LEVEL.get(severity, 10) < _LEVEL[self.preferences.minimum_severity]:
            self.state.update(status_signature=signature, status_bad=is_bad)
            self._save()
            return "filtered"
        event_id = hashlib.sha256((str(path.resolve()) + "|" + signature).encode()).hexdigest()
        message = (f"MNQ PAPER · {severity}\n{title}\nEngine: {engine_state} · Feed: {feed_mode} · "
                   f"Backlog: {backlog if backlog is not None else 'unavailable'}\n{error[:300]}")
        self.state.update(status_pending={"event_id": event_id, "signature": signature,
                                         "bad": is_bad, "message": message[:3900],
                                         "equity_after": (equity if incident and signature.startswith("equity-change:") else None)},
                          status_attempts=0, status_next_attempt_at=0.0)
        self._journal({"event_id": event_id, "state": "queued", "severity": severity,
                       "recorded_at_utc": datetime.now(UTC).isoformat()})
        self._save()
        return self.scan_status(path, backlog_threshold=backlog_threshold,
                                feed_stall_seconds=feed_stall_seconds)

    def test_message(self) -> str:
        self.send("MNQ PAPER · TEST\nThis is a notification connectivity test. No order was placed.")
        return "delivered"

    def scan_engine_process(self, process_ids: list[int] | None, status_path: str | Path,
                            *, absent_grace_seconds: float = 20.0) -> str:
        """Watch the Paper process independently of its own health reporting.

        ``None`` means process inspection is unavailable; an empty list means
        the probe succeeded and found no matching engine process.
        """
        if process_ids is None:
            return "probe_unavailable"
        path = Path(status_path)
        try:
            status = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            status = {}
        system = status.get("system") if isinstance(status.get("system"), dict) else {}
        engine_state = str(system.get("state", "UNAVAILABLE")).upper()
        ids = sorted({int(pid) for pid in process_ids if int(pid) > 0})
        now = float(self.clock())
        if ids:
            previously_missing = bool(self.state.get("watchdog_alerted"))
            self.state.update(watchdog_seen=True, watchdog_last_pids=ids,
                              watchdog_absent_since=None, watchdog_alerted=False)
            if previously_missing and self.state.get("status_pending") is None:
                self._queue_watchdog("Paper Engine process detected again", "engine-recovered", False,
                                     threshold_seconds=absent_grace_seconds)
                self._save()
                return "recovered_queued"
            self._save()
            return "running"
        if engine_state in {"STOPPED", "COMPLETED"}:
            self.state.update(watchdog_seen=False, watchdog_absent_since=None, watchdog_alerted=False)
            self._save()
            return "stopped_cleanly"
        if engine_state in {"ERROR", "FAILED"}:
            return "failed_status_reported"
        if not self.state.get("watchdog_seen") and engine_state not in {"RUNNING", "RECOVERING"}:
            return "not_seen"
        absent_since = self.state.get("watchdog_absent_since")
        if absent_since is None:
            self.state["watchdog_absent_since"] = now
            self._save()
            return "absence_grace"
        if now - float(absent_since) < absent_grace_seconds:
            return "absence_grace"
        if self.state.get("watchdog_alerted"):
            return "already_alerted"
        self._queue_watchdog("Paper Engine process disappeared unexpectedly", "engine-terminated", True,
                             threshold_seconds=absent_grace_seconds)
        self.state["watchdog_alerted"] = True
        self._save()
        return "termination_queued"

    def _queue_watchdog(self, title: str, signature: str, bad: bool, source_url: str | None = None,
                        threshold_seconds: float | None = None) -> None:
        event_id = hashlib.sha256((str(self.event_path.resolve()) + "|watchdog|" + signature).encode()).hexdigest()
        if signature.startswith("reminder:"):
            message = (f"MNQ PAPER · {'CRITICAL' if bad else 'INFO'}\n{title}\n"
                       + (f"Verified source: {source_url}\n" if source_url else "")
                       + "No trading setting was changed.")
        else:
            recovered = not bad
            message = "\n".join([
                f"{'⚠️' if recovered else '🚨'} {'Paper Engine recovered' if recovered else 'Unexpected Paper Engine termination'}",
                "Environment: PAPER", "Strategy: SYSTEM", "Symbol: MNQ",
                f"Detected: {datetime.now(UTC).isoformat()}",
                f"Expected: {'RUNNING' if not recovered else 'restart detected'}",
                f"Observed: {'process detected again' if recovered else 'process absent beyond grace period'}",
                f"Threshold: {threshold_seconds:g}s" if threshold_seconds is not None else "Threshold: unavailable",
                f"Status: {'RECOVERED' if recovered else 'ACTIVE'}",
                "Details: Detected by the separate read-only Paper process watchdog. No broker order was placed.",
            ])
        self.state["status_pending"] = {
            "event_id": event_id, "signature": "watchdog:" + signature, "bad": bad,
            "message": message,
        }
        self.state.update(status_attempts=0, status_next_attempt_at=0.0)
        self._journal({"event_id": event_id, "state": "queued", "severity": "CRITICAL" if bad else "INFO",
                       "recorded_at_utc": datetime.now(UTC).isoformat()})

    def scan_reminders(self, schedule_path: str | Path, *, now: datetime | None = None,
                       lead_days: int = 7) -> list[str]:
        """Queue reminders only from explicitly reviewed, source-backed events."""
        path = Path(schedule_path)
        if not path.is_file():
            return []
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if document.get("schema_version") != 1 or not isinstance(document.get("events"), list):
            return []
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("reminder clock must be timezone-aware")
        queued: list[str] = []
        for item in document["events"]:
            if not isinstance(item, dict) or item.get("reviewed") is not True or not item.get("source_url"):
                continue
            kind = str(item.get("kind", "")).lower()
            if kind not in {"cme_calendar", "contract_roll"}:
                continue
            try:
                effective = datetime.fromisoformat(str(item["effective_at_utc"]).replace("Z", "+00:00"))
                if effective.tzinfo is None or effective.utcoffset() != UTC.utcoffset(effective):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            source_url = str(item.get("source_url", ""))
            source_parts = urlsplit(source_url)
            if source_parts.scheme != "https" or not source_parts.netloc or source_parts.username or source_parts.password or source_parts.query or source_parts.fragment:
                continue
            delta = effective.astimezone(UTC) - current.astimezone(UTC)
            event_key = str(item.get("id") or f"{kind}:{effective.isoformat()}")
            if delta.total_seconds() < 0 or delta.total_seconds() > lead_days * 86400:
                continue
            sent = set(self.state.get("sent_reminders", []))
            if event_key in sent or self.state.get("status_pending") is not None:
                continue
            label = str(item.get("label") or ("CME calendar event" if kind == "cme_calendar" else "MNQ contract roll"))[:250]
            days = max(0, delta.days)
            title = "Upcoming CME session schedule" if kind == "cme_calendar" else "Upcoming MNQ contract-roll deadline"
            signature = "reminder:" + hashlib.sha256((str(path.resolve()) + event_key).encode()).hexdigest()
            self._queue_watchdog(f"{title}: {label} · {days} day(s)", signature, False,
                                 source_url=source_url)
            self.state["status_pending"]["reminder_key"] = event_key
            self._save()
            queued.append(event_key)
        return queued

    def run_forever(self, *, stop: Callable[[], bool], poll_seconds: float = 2.0,
                    process_probe: Callable[[], list[int] | None] | None = None,
                    reminder_schedule: str | Path | None = None) -> None:
        stop_path = self.state_path.with_name("stop.request")
        next_process_probe = 0.0
        try:
            while not stop() and not stop_path.exists():
                self.scan_once()
                status_path = self.event_path.parent / "status.json"
                self.scan_status(status_path)
                now = self.clock()
                if process_probe and now >= next_process_probe:
                    self.scan_engine_process(process_probe(), status_path)
                    next_process_probe = now + 15.0
                if reminder_schedule:
                    self.scan_reminders(reminder_schedule)
                if self.state.get("status_pending") is not None:
                    self.scan_status(status_path)
                time.sleep(poll_seconds)
        finally:
            stop_path.unlink(missing_ok=True)


def recent_failures(journal_path: str | Path, limit: int = 30) -> list[dict[str, Any]]:
    path = Path(journal_path)
    if not path.exists(): return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("state") in {"retry", "blocked"}: rows.append(row)
    return rows[-limit:]


def requeue_blocked_delivery(state_path: str | Path, *, channel: str = "event") -> bool:
    """Explicitly retry a permanently rejected pending notification after remediation.

    The event ID and cursor are preserved; this never creates a second alert.
    """
    if channel not in {"event", "status"}:
        raise ValueError("channel must be event or status")
    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    pending_key = "pending" if channel == "event" else "status_pending"
    blocked_key = "delivery_blocked" if channel == "event" else "status_delivery_blocked"
    if not state.get(pending_key) or not state.get(blocked_key):
        return False
    state[blocked_key] = False
    state["next_attempt_at" if channel == "event" else "status_next_attempt_at"] = 0.0
    state["delivery_failure_category" if channel == "event" else "status_delivery_failure_category"] = None
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True, separators=(",", ":"))
            handle.flush(); os.fsync(handle.fileno())
        atomic_replace_with_retry(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return True


class PaperProcessWatchdog:
    """Independent read-only process watcher that emits into its own event journal."""

    def __init__(self, *, run_dir: str | Path, state_path: str | Path,
                 event_path: str | Path, process_probe: Callable[[], list[int] | None],
                 clock: Callable[[], float] = time.time, grace_seconds: float = 20.0,
                 heartbeat_stale_seconds: float = 30.0,
                 notifier_heartbeat_path: str | Path | None = None) -> None:
        self.run_dir, self.state_path, self.event_path = Path(run_dir), Path(state_path), Path(event_path)
        self.process_probe, self.clock, self.grace_seconds = process_probe, clock, grace_seconds
        self.heartbeat_stale_seconds = float(heartbeat_stale_seconds)
        if self.heartbeat_stale_seconds <= 0:
            raise ValueError("heartbeat stale threshold must be positive")
        self.notifier_heartbeat_path = Path(notifier_heartbeat_path) if notifier_heartbeat_path else None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "schema_version": 1, "active": False, "pids": [], "absent_since": None,
            "alerted": False, "sequence": 0,
        }
        if self.state.get("schema_version") != 1:
            raise ValueError("watchdog state schema is incompatible")

    def _save(self) -> None:
        fd, name = tempfile.mkstemp(prefix=self.state_path.name + ".", suffix=".tmp", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.state, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
            atomic_replace_with_retry(name, self.state_path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def _emit(self, event_type: str, transition: str, payload: Mapping[str, Any]) -> None:
        event_id = hashlib.sha256((str(self.run_dir.resolve()) + "|watchdog|" + transition).encode()).hexdigest()
        if self.event_path.exists():
            with self.event_path.open(encoding="utf-8") as existing:
                for line in existing:
                    try:
                        if json.loads(line).get("event_id") == event_id:
                            self._save()
                            return
                    except json.JSONDecodeError:
                        raise RuntimeError("watchdog event journal is corrupt") from None
        self.state["sequence"] = int(self.state.get("sequence", 0)) + 1
        event = {"event_id": event_id, "sequence": self.state["sequence"], "event_type": event_type,
                 "timestamp": datetime.now(UTC).isoformat(), "payload": dict(payload)}
        with self.event_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
        self._save()

    def check_once(self) -> str:
        now = float(self.clock())
        notification = (self._check_heartbeat("notification_service", self.notifier_heartbeat_path, now)
                        if self.notifier_heartbeat_path is not None else "not_configured")
        pids = self.process_probe()
        if pids is None:
            self.state["last_probe_status"] = "unavailable"
            self.state["last_probe_at_utc"] = datetime.now(UTC).isoformat()
            self._save()
            # Process enumeration may be restricted by Windows permissions,
            # but the independent watchdog can still assess the notifier's
            # externally written heartbeat and emit through its own journal.
            return "notification_service_stale" if notification == "stale" else "probe_unavailable"
        self.state["last_probe_status"] = "ok"
        self.state["last_probe_at_utc"] = datetime.now(UTC).isoformat()
        self._save()
        pids = sorted({int(pid) for pid in pids if int(pid) > 0})
        try:
            status = json.loads((self.run_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
        system = status.get("system") if isinstance(status.get("system"), dict) else {}
        engine = str(system.get("state", "UNAVAILABLE")).upper()
        if pids:
            if self.state.get("alerted"):
                old = ",".join(str(x) for x in self.state.get("pids", [])) or "unknown"
                new = ",".join(str(x) for x in pids)
                self.state.update(active=True, pids=pids, absent_since=None, alerted=False)
                self._emit("paper_process_recovered", f"recovered:{old}:{new}", {"pids": pids, "severity": "INFO"})
                return "recovered"
            self.state.update(active=True, pids=pids, absent_since=None)
            self._save()
            health = self._check_heartbeat("engine_status", self.run_dir / "status.json", now)
            if health == "stale" or notification == "stale":
                return "heartbeat_stale"
            if health == "recovered" or notification == "recovered":
                return "heartbeat_recovered"
            return "running"
        if engine in {"STOPPED", "COMPLETED"}:
            self.state.update(active=False, pids=[], absent_since=None, alerted=False)
            self._save()
            return "stopped_cleanly"
        if engine in {"ERROR", "FAILED"}:
            # The event/status notifier reports the actual failure; do not add a
            # second watchdog alert for the same engine error.
            return "heartbeat_stale" if notification == "stale" else "failed_status_reported"
        if self.state.get("absent_since") is None:
            self.state["active"] = self.state.get("active", False) or engine in {"RUNNING", "RECOVERING"}
            self.state["absent_since"] = now
            self._save()
            return "absence_grace"
        if not self.state.get("active"):
            return "not_seen"
        if now - float(self.state["absent_since"]) < self.grace_seconds:
            return "absence_grace"
        if self.state.get("alerted"):
            return "already_alerted"
        pids_before = ",".join(str(x) for x in self.state.get("pids", [])) or "not observed in this session"
        self.state["alerted"] = True
        self._emit("paper_process_terminated", f"terminated:{pids_before}:{int(self.state['absent_since'])}",
                   {"severity": "CRITICAL", "last_known_pids": self.state.get("pids", []),
                    "status_state": engine, "run": self.run_dir.name})
        return "termination_detected"

    def _check_heartbeat(self, key: str, path: Path, now: float) -> str:
        try:
            age = now - path.stat().st_mtime if path.is_file() else None
        except OSError:
            age = None
        stale = age is None or age >= self.heartbeat_stale_seconds
        fields = {"engine_status": ("paper_status_heartbeat_stale", "Paper engine status heartbeat is stale",
                                    "paper_status_heartbeat_recovered", "Paper engine status heartbeat recovered"),
                  "notification_service": ("paper_notification_service_stale", "Paper notification service heartbeat is stale",
                                            "paper_notification_service_recovered", "Paper notification service heartbeat recovered")}[key]
        since_key = key + "_stale_since"
        active_key = key + "_stale_alerted"
        if stale:
            if self.state.get(since_key) is None:
                self.state[since_key] = now
                self._save()
                return "pending"
            if now - float(self.state[since_key]) < self.heartbeat_stale_seconds:
                return "pending"
            if not self.state.get(active_key):
                self.state[active_key] = True
                self._emit(fields[0], f"{key}:stale:{int(self.state[since_key])}", {
                    "severity": "CRITICAL", "age_seconds": age,
                    "threshold_seconds": self.heartbeat_stale_seconds,
                    "details": f"Independent watchdog observed no fresh {key.replace('_', ' ')} heartbeat."})
            return "stale"
        if self.state.get(active_key):
            self.state[active_key] = False
            self.state[since_key] = None
            self._emit(fields[2], f"{key}:recovered:{int(now)}", {
                "severity": "INFO", "age_seconds": age,
                "details": f"Independent watchdog observed a fresh {key.replace('_', ' ')} heartbeat."})
            return "recovered"
        self.state[since_key] = None
        self._save()
        return "fresh"
