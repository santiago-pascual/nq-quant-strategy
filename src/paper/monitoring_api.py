"""Versioned, authenticated, read-only HTTP API for Paper analytics."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from src.paper.analytics import PaperAnalyticsReader


API_SCHEMA_VERSION = "1.0"
ROUTES = {
    "/v1/health", "/v1/account", "/v1/strategies", "/v1/positions",
    "/v1/trades", "/v1/hmm", "/v1/refits", "/v1/feed",
    "/v1/checkpoint", "/v1/parity", "/v1/events", "/v1/candidates",
    "/v1/daily-reports", "/v1/risk",
}


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
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            supplied = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not supplied.startswith(prefix) or not hmac.compare_digest(supplied[len(prefix):], token):
                self._send(401, {"error": "unauthorized"})
                return
            route = unquote(urlsplit(self.path).path)
            if route.startswith("/v1/trades/"):
                trade_id = route.removeprefix("/v1/trades/")
                data = self._with_reader(lambda reader: reader.trade_detail(trade_id))
                self._send(200 if data is not None else 404,
                           data if data is not None else {"error": "not_found"})
                return
            if route not in ROUTES:
                self._send(404, {"error": "not_found"})
                return
            if not database.is_file():
                self._send(503, {"error": "analytics_database_unavailable"})
                return
            try:
                self._send(200, self._data(route))
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

        def _data(self, route: str) -> Any:
            if route == "/v1/health":
                status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {"system": {"state": "UNAVAILABLE"}}
                feed = self._with_reader(lambda r: r.feed_status())
                checkpoint = self._with_reader(lambda r: r.latest_checkpoint())
                return {"mode": "PAPER", "status": status, "feed": feed, "checkpoint": checkpoint}
            if route == "/v1/account":
                return self._with_reader(lambda r: {"account": r.status(), "portfolio": r.portfolio_statistics()})
            if route == "/v1/strategies":
                return self._with_reader(lambda r: r.strategy_statistics())
            if route == "/v1/positions":
                return self._with_reader(lambda r: r.positions())
            if route == "/v1/trades":
                return self._with_reader(lambda r: r.recent_trades(100))
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

        def do_POST(self) -> None:  # noqa: N802
            self._send(405, {"error": "method_not_allowed"})

        def do_PUT(self) -> None:  # noqa: N802
            self._send(405, {"error": "method_not_allowed"})

        def do_DELETE(self) -> None:  # noqa: N802
            self._send(405, {"error": "method_not_allowed"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)
