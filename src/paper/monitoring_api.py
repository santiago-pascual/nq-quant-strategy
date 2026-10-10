"""Versioned, authenticated, read-only HTTP API for Paper analytics."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit
from datetime import date, datetime, time, timedelta, timezone

from src.paper.analytics import PaperAnalyticsReader


API_SCHEMA_VERSION = "1.0"
ROUTES = {
    "/v1/health", "/v1/account", "/v1/strategies", "/v1/positions",
    "/v1/trades", "/v1/hmm", "/v1/refits", "/v1/feed",
    "/v1/checkpoint", "/v1/parity", "/v1/events", "/v1/candidates",
    "/v1/daily-reports", "/v1/risk", "/v1/dashboard", "/v1/analytics", "/v1/bars",
    "/v1/alerts",
    "/v1/operational-health",
}


def _read_fill_rows(db: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    """Read a bounded fill sample and expose only explicitly-unitized slippage."""
    available = {str(row[1]) for row in db.execute("PRAGMA table_info(fills)")}
    fields = ("fill_id", "order_id", "timestamp_utc", "strategy", "contract", "quantity", "price", "side",
              "commission", "exchange_fee", "regulatory_fee", "artificial_slippage", "observed_spread")
    select = [field if field in available else f"NULL AS {field}" for field in fields]
    if "payload_json" in available:
        select.append("payload_json")
    rows = [dict(row) for row in db.execute(
        f"SELECT {','.join(select)} FROM fills ORDER BY rowid DESC LIMIT ?", (max(1, min(int(limit), 100)),)
    )]
    for fill in rows:
        try:
            payload_raw = fill.pop("payload_json", None)
            payload = json.loads(payload_raw) if payload_raw else {}
            value = payload.get("artificial_slippage_ticks") if isinstance(payload, dict) else None
            fill["artificial_slippage_ticks"] = float(value) if value is not None else None
        except (TypeError, ValueError, json.JSONDecodeError):
            fill.pop("payload_json", None)
            fill["artificial_slippage_ticks"] = None
    return rows


def _run_kind(scope: dict[str, Any] | None, status: dict[str, Any] | None) -> str:
    """Classify only when a persisted run manifest or provider status supports it."""
    if scope:
        return "HISTORICAL_REPLAY"
    if not status or status.get("mode") != "PAPER":
        return "UNAVAILABLE"
    system = status.get("system") or {}
    source = ((system.get("feed_health") or {}).get("source")
              or status.get("source"))
    if source == "deterministic_replay":
        return "HISTORICAL_REPLAY"
    if system.get("state") and source:
        return "REALTIME_PAPER"
    return "UNAVAILABLE"


def create_monitoring_server(
    output_dir: str | Path, *, token: str, host: str = "127.0.0.1", port: int = 8765,
) -> ThreadingHTTPServer:
    """Build a GET-only server. Authentication is mandatory on all interfaces."""
    if not token or len(token) < 24:
        raise ValueError("PAPER_MONITORING_TOKEN must contain at least 24 characters")
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("monitoring API binds to loopback only; use an authenticated TLS proxy or SSH/VPN tunnel for remote access")
    directory = Path(output_dir)
    database = directory / "paper_analytics.sqlite3"
    status_file = directory / "status.json"

    class Handler(BaseHTTPRequestHandler):
        server_version = "MNQPaperMonitor/1.0"

        def _send(self, status: int, value: Any) -> None:
            body = json.dumps({"schema_version": API_SCHEMA_VERSION, "data": value},
                              separators=(",", ":"), default=str).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Browser refreshes, mobile network handoffs, and canceled
                # Streamlit requests can close their socket while a read-only
                # response is being assembled. The response is disposable; do
                # not turn a client disconnect into an API/server traceback.
                return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            supplied = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not supplied.startswith(prefix) or not hmac.compare_digest(supplied[len(prefix):], token):
                self._send(401, {"error": "unauthorized"})
                return
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            if route.startswith("/v1/trades/"):
                trade_id = route.removeprefix("/v1/trades/")
                data = self._with_reader(lambda reader: reader.trade_detail(trade_id))
                self._send(200 if data is not None else 404,
                           data if data is not None else {"error": "not_found"})
                return
            if route.startswith("/v1/trade-bars/"):
                trade_id = route.removeprefix("/v1/trade-bars/")
                data = self._with_reader(lambda reader: reader.trade_bars(trade_id))
                self._send(200 if data is not None else 503, data if data is not None else {"error": "analytics_database_unavailable"})
                return
            if route not in ROUTES:
                self._send(404, {"error": "not_found"})
                return
            if route == "/v1/dashboard":
                try:
                    self._send(200, self._dashboard_snapshot())
                except Exception as exc:
                    self._send(503, {"error": "dashboard_read_failed", "type": type(exc).__name__})
                return
            if route == "/v1/alerts":
                self._send(200, self._alerts_snapshot())
                return
            if route == "/v1/operational-health":
                try:
                    from src.paper.operational_health import inspect
                    from src.paper.notification_cli import _reviewed_calendar_for_run
                    self._send(200, inspect(directory, _reviewed_calendar_for_run(directory)))
                except Exception as exc:
                    self._send(503, {"error": "operational_health_unavailable", "type": type(exc).__name__})
                return
            if not database.is_file():
                self._send(503, {"error": "analytics_database_unavailable"})
                return
            try:
                self._send(200, self._data(route, query))
            except Exception as exc:
                self._send(503, {"error": "analytics_read_failed", "type": type(exc).__name__})

        def _with_reader(self, callback):
            if not database.is_file():
                return None
            reader = PaperAnalyticsReader(str(database))
            try:
                return callback(reader)
            finally:
                reader.close()

        @staticmethod
        def _filters(query: dict[str, list[str]]) -> dict[str, Any]:
            def one(name: str) -> str | None:
                values = query.get(name)
                return values[0] if values and values[0] else None
            strategy = one("strategy")
            if strategy and strategy not in {"MRL1", "MRS2", "S2R", "ORB"}:
                raise ValueError("unknown strategy filter")
            start = end = None
            if one("start"):
                start = datetime.combine(date.fromisoformat(one("start")), time.min, timezone.utc).isoformat()
            if one("end"):
                end = datetime.combine(date.fromisoformat(one("end")) + timedelta(days=1), time.min, timezone.utc).isoformat()
            if start and end and start >= end:
                raise ValueError("start date must be on or before end date")
            return {"strategy": strategy, "start": start, "end": end}

        def _dashboard_snapshot(self) -> dict[str, Any]:
            status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.is_file() else None
            scope = json.loads((directory / "oos_run_scope.json").read_text(encoding="utf-8")) \
                if (directory / "oos_run_scope.json").is_file() else None
            replay = bool(scope)
            system_state = (status or {}).get("system", {}).get("state")
            last_bar = (status or {}).get("system", {}).get("last_bar")
            replay_state = "UNAVAILABLE"
            if system_state == "ERROR":
                replay_state = "FAILED"
            elif system_state == "STOPPED":
                end = scope.get("replay_end_utc_exclusive") if scope else None
                if end and last_bar:
                    try:
                        from datetime import datetime, timedelta
                        last = datetime.fromisoformat(str(last_bar).replace("Z", "+00:00"))
                        exclusive = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
                        replay_state = "COMPLETED" if last + timedelta(minutes=1) >= exclusive else "STOPPED"
                    except (TypeError, ValueError):
                        replay_state = "STOPPED"
                else:
                    replay_state = "STOPPED"
            elif system_state in {"RUNNING", "PAUSED_REFIT", "DEGRADED"}:
                replay_state = "RUNNING" if system_state != "DEGRADED" else "DEGRADED"

            base: dict[str, Any] = {
                "run": {
                    "kind": _run_kind(scope, status),
                    "status": replay_state if replay else (system_state or "UNAVAILABLE"),
                    "scope": scope,
                    "processed_bars": (status or {}).get("system", {}).get("bars_processed"),
                    "start_utc": scope.get("start_utc_inclusive") if scope else None,
                    "end_utc_exclusive": scope.get("replay_end_utc_exclusive") if scope else None,
                    "last_processed_bar": last_bar,
                    "progress_pct": None,
                    "progress_note": "Unavailable: total eligible bars are not persisted.",
                },
                "status": status,
                "available": database.is_file(),
                "account": None,
                "portfolio": None,
                "analytics_sample": None,
                "strategies": None,
                "positions": [],
                "orders": [],
                "fills": [],
                "trades": [],
                "equity_curve": [],
                "risk": None,
                "system": {
                    "latest_event": None,
                    "feed": None,
                    "errors": (status or {}).get("system", {}).get("errors", []),
                    "warnings": (status or {}).get("system", {}).get("warnings", []),
                    "hmm": (status or {}).get("hmm"),
                },
                "alerts": self._alerts_snapshot(),
                "unavailable_fields": ["initial_balance", "account_floor", "failure_conditions", "progress_pct"],
            }
            if not database.is_file():
                return base

            reader = PaperAnalyticsReader(str(database))
            try:
                db = reader._db
                db.execute("PRAGMA busy_timeout=250")
                account = reader.status()
                # Initial equity is shown only when the persisted engine configuration contains it.
                initial_balance = None
                config = db.execute(
                    "SELECT source_json FROM configuration_versions WHERE config_kind='paper_engine' ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                if config:
                    try:
                        source = json.loads(config[0])
                        initial_balance = source.get("engine", {}).get("initial_equity")
                    except (TypeError, json.JSONDecodeError):
                        initial_balance = None
                if account.get("status") == "UNAVAILABLE":
                    account = None
                if account is not None:
                    account["initial_balance"] = initial_balance

                orders = [dict(row) for row in db.execute(
                    "SELECT order_id,timestamp_utc,strategy,status,quantity,reference_price FROM orders ORDER BY rowid DESC LIMIT 100"
                )]
                fills = _read_fill_rows(db)
                curve = [dict(row) for row in db.execute(
                    "SELECT timestamp_utc,balance,equity,realized_pnl,unrealized_pnl,drawdown FROM account_snapshots ORDER BY timestamp_utc DESC LIMIT 500"
                )]
                curve.reverse()
                analytics = reader.dashboard_analytics(limit=5000)
                base.update({
                    "available": True,
                    "account": account,
                    "portfolio": analytics["metrics"],
                    "analytics_sample": analytics["sample"],
                    "strategies": analytics["strategies"],
                    "positions": ((status or {}).get("portfolio", {}).get("open_positions")
                                  or reader.positions()),
                    "orders": orders,
                    "fills": fills,
                    "trades": reader.recent_trades(250),
                    "equity_curve": curve,
                    "risk": reader.risk_status(),
                })
                events = reader.event_timeline(1)
                base["system"]["latest_event"] = events[0] if events else None
                base["system"]["feed"] = reader.feed_status()
                return base
            finally:
                reader.close()

        def _data(self, route: str, query: dict[str, list[str]] | None = None) -> Any:
            query = query or {}
            if route == "/v1/health":
                status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.is_file() else {"system": {"state": "UNAVAILABLE"}}
                feed = self._with_reader(lambda r: r.feed_status())
                checkpoint = self._with_reader(lambda r: r.latest_checkpoint())
                return {"mode": "PAPER", "status": status, "feed": feed, "checkpoint": checkpoint}
            if route == "/v1/alerts":
                return self._alerts_snapshot()
            if route == "/v1/account":
                return self._with_reader(lambda r: {"account": r.status(), "portfolio": r.portfolio_statistics()})
            if route == "/v1/strategies":
                return self._with_reader(lambda r: r.strategy_statistics())
            if route == "/v1/positions":
                return self._with_reader(lambda r: r.positions())
            if route == "/v1/trades":
                filters = self._filters(query)
                raw_limit = (query.get("limit") or ["500"])[0]
                limit = max(1, min(int(raw_limit), 1000))
                return self._with_reader(lambda r: r.filtered_trades(limit=limit, **filters))
            if route == "/v1/analytics":
                filters = self._filters(query)
                raw_limit = (query.get("limit") or ["5000"])[0]
                limit = max(1, min(int(raw_limit), 5000))
                return self._with_reader(lambda r: r.dashboard_analytics(limit=limit, **filters))
            if route == "/v1/bars":
                raw_limit = (query.get("limit") or ["1000"])[0]
                limit = max(1, min(int(raw_limit), 1500))
                return self._with_reader(lambda r: r.recent_market_bars(limit))
            if route == "/v1/hmm":
                return self._with_reader(lambda r: r.hmm_state())
            if route == "/v1/refits":
                return self._with_reader(lambda r: r.model_refits())
            if route == "/v1/feed":
                return self._with_reader(lambda r: r.feed_status())
            if route == "/v1/checkpoint":
                return self._with_reader(lambda r: r.latest_checkpoint())
            if route == "/v1/parity":
                return self._with_reader(lambda r: r.shadow_parity())
            if route == "/v1/events":
                return self._with_reader(lambda r: r.event_timeline(100))
            if route == "/v1/candidates":
                return self._with_reader(lambda r: r.candidate_diagnostics())
            if route == "/v1/daily-reports":
                return self._with_reader(lambda r: r.daily_reports(30))
            if route == "/v1/risk":
                return self._with_reader(lambda r: r.risk_status())
            return {"error": "not_found"}

        def _alerts_snapshot(self) -> dict[str, Any]:
            # Alert state is written by a separate read-only sidecar; this GET
            # exposes only its bounded snapshot and never acknowledges/mutates it.
            alert_path = directory.parent / ".notifications" / directory.name / "alert_state.json"
            if not alert_path.is_file():
                return {"available": False, "active": [], "history": [],
                        "capabilities": {"state": "NOT_STARTED"}}
            try:
                saved = json.loads(alert_path.read_text(encoding="utf-8"))
                public = lambda row: {key: value for key, value in row.items() if not str(key).startswith("_")}
                runtime = alert_path.parent
                now = datetime.now(timezone.utc)

                def read_json(name: str) -> dict[str, Any] | None:
                    try:
                        value = json.loads((runtime / name).read_text(encoding="utf-8"))
                        return value if isinstance(value, dict) else None
                    except (OSError, json.JSONDecodeError, TypeError):
                        return None

                def age_seconds(value: Any) -> float | None:
                    try:
                        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                        if stamp.tzinfo is None:
                            stamp = stamp.replace(tzinfo=timezone.utc)
                        return max(0.0, (now - stamp.astimezone(timezone.utc)).total_seconds())
                    except (TypeError, ValueError):
                        return None

                notifier = read_json("notifier_heartbeat.json")
                notifier_age = age_seconds((notifier or {}).get("timestamp_utc"))
                watchdog = read_json("engine_watchdog_state.json")
                watchdog_age = age_seconds((watchdog or {}).get("last_probe_at_utc"))
                services = {
                    "notification_service": {
                        "state": "UNAVAILABLE" if notifier_age is None else ("HEALTHY" if notifier_age < 30 else "STALE"),
                        "pid": (notifier or {}).get("pid"),
                        "heartbeat_utc": (notifier or {}).get("timestamp_utc"),
                        "heartbeat_age_seconds": notifier_age,
                        "stale_after_seconds": 30,
                    },
                    "process_watchdog": {
                        "state": "UNAVAILABLE" if watchdog_age is None else ("HEALTHY" if watchdog_age < 45 else "STALE"),
                        "last_probe_utc": (watchdog or {}).get("last_probe_at_utc"),
                        "last_probe_age_seconds": watchdog_age,
                        "process_probe": (watchdog or {}).get("last_probe_status", "UNAVAILABLE"),
                        "probe_interval_seconds": 15,
                    },
                }
                return {"available": True, "active": [public(row) for row in saved.get("active", {}).values()],
                        "history": [public(row) for row in saved.get("history", [])[-100:]],
                        "capabilities": saved.get("capabilities", {}),
                        "services": services,
                        "last_evaluated_utc": saved.get("last_evaluated_utc"),
                        "market_state": saved.get("market_state", {"state": "MARKET_UNKNOWN", "reason": "not_evaluated"}),
                        "feed_assessment": saved.get("feed_assessment", {"state": "UNAVAILABLE"}),
                        "thresholds": saved.get("thresholds", {})}
            except (OSError, json.JSONDecodeError, TypeError):
                return {"available": False, "active": [], "history": [],
                        "capabilities": {"state": "CORRUPT_OR_UNREADABLE"}}

        def do_POST(self) -> None:  # noqa: N802
            self._send(405, {"error": "method_not_allowed"})

        def do_PUT(self) -> None:  # noqa: N802
            self._send(405, {"error": "method_not_allowed"})

        def do_DELETE(self) -> None:  # noqa: N802
            self._send(405, {"error": "method_not_allowed"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)
