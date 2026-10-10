import json
import sqlite3
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.paper.analytics import PaperAnalyticsReader
from src.paper.monitoring_api import _read_fill_rows, _run_kind, create_monitoring_server


def _database(path):
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE trades(
        trade_id TEXT PRIMARY KEY, strategy TEXT, status TEXT, exit_timestamp_utc TEXT,
        entry_timestamp_utc TEXT, net_pnl REAL, gross_pnl REAL, total_costs REAL,
        commission REAL, exchange_fees REAL, regulatory_fees REAL, realized_r REAL,
        duration_seconds REAL, quantity INTEGER, mae_points REAL, mfe_points REAL,
        exit_reason TEXT, direction TEXT, entry_fill_price REAL, exit_fill_price REAL
    )""")
    db.executemany("INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("a", "MRL1", "CLOSED", "2024-01-02T15:00:00+00:00", "2024-01-02T14:30:00+00:00", 9, 10, 1, .5, .4, .1, .9, 1800, 1, 2, 5, "target", "long", 100, 101),
        ("b", "MRS2", "CLOSED", "2024-01-02T16:00:00+00:00", "2024-01-02T15:30:00+00:00", -5, -4, 1, .5, .4, .1, -.5, 1800, 1, 5, 2, "stop", "short", 100, 101),
        ("c", "ORB", "CLOSED", "2024-01-03T16:00:00+00:00", "2024-01-03T15:30:00+00:00", 2, 2.5, .5, .25, .2, .05, .4, 600, 1, 1, 3, "target", "long", 100, 101),
    ])
    db.commit()
    db.close()


def _assert_metrics(path):
    _database(path)
    reader = PaperAnalyticsReader(str(path))
    try:
        result = reader.dashboard_analytics(limit=2)
    finally:
        reader.close()
    assert result["sample"] == {
        "total_closed_trades": 3, "included_closed_trades": 2, "limit": 2,
        "truncated": True,
        "basis": "Most recent closed trades by exit time within the selected filters.",
    }
    metrics = result["metrics"]
    assert metrics["trades"] == 2
    assert metrics["gross_pnl"] == pytest.approx(-1.5)
    assert metrics["net_pnl"] == pytest.approx(-3)
    assert metrics["total_costs"] == pytest.approx(1.5)
    assert metrics["commissions"] == pytest.approx(.75)
    assert metrics["profit_factor"] == pytest.approx(2 / 5)
    assert metrics["win_rate"] == pytest.approx(.5)
    assert metrics["expectancy_r"] == pytest.approx(-.05)
    assert [row["date_et"] for row in result["daily"]] == ["2024-01-02", "2024-01-03"]
    assert [row["net_pnl"] for row in result["daily"]] == [-5, 2]
    assert result["cumulative_net_pnl"][-1]["cumulative_net_pnl"] == pytest.approx(-3)
    assert result["strategies"]["S2R"]["trades"] == 0
    assert [row["trade_id"] for row in result["trade_outcomes"]] == ["b", "c"]
    assert all(row["hmm_state"] is None for row in result["trade_outcomes"])


def test_run_type_requires_persisted_replay_or_provider_evidence():
    assert _run_kind(None, None) == "UNAVAILABLE"
    assert _run_kind({"start_utc_inclusive": "2026-08-27T00:00:00Z"}, None) == "HISTORICAL_REPLAY"
    assert _run_kind(None, {"mode": "PAPER", "system": {"state": "RUNNING", "feed_health": {"source": "deterministic_replay"}}}) == "HISTORICAL_REPLAY"
    assert _run_kind(None, {"mode": "PAPER", "system": {"state": "RUNNING", "feed_health": {"source": "ibkr_delayed"}}}) == "REALTIME_PAPER"


def test_fill_projection_uses_explicit_slippage_ticks_and_handles_missing_payload():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE fills(fill_id TEXT,order_id TEXT,timestamp_utc TEXT,strategy TEXT,contract TEXT,quantity INTEGER,price REAL,side TEXT,commission REAL,exchange_fee REAL,regulatory_fee REAL,artificial_slippage REAL,observed_spread REAL,payload_json TEXT)")
    db.executemany("INSERT INTO fills VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("f1", "o1", "2026-10-08T15:00:00Z", "ORB", "MNQZ6", 1, 100, "BUY", .25, .35, .01, 0.0, None, '{"artificial_slippage_ticks":0}'),
        ("f2", "o2", "2026-10-08T15:01:00Z", "ORB", "MNQZ6", 1, 101, "SELL", .25, .35, .01, 0.0, None, "not-json"),
    ])
    rows = _read_fill_rows(db)
    db.close()
    assert rows[0]["fill_id"] == "f2"  # newest-first API order
    assert rows[0]["artificial_slippage_ticks"] is None
    assert rows[1]["artificial_slippage_ticks"] == 0
    assert "payload_json" not in rows[0]


def test_dashboard_financial_metrics_are_bounded_and_use_persisted_values():
    path = Path.cwd() / ".dashboard_analytics_fixture.sqlite3"
    assert not path.exists(), "refusing to overwrite an existing file"
    try:
        _assert_metrics(path)
    finally:
        path.unlink(missing_ok=True)


def test_monitoring_api_analytics_filters_and_trade_page_are_read_only():
    output = Path.cwd()
    path = output / "paper_analytics.sqlite3"
    assert not path.exists(), "refusing to overwrite an existing analytics database"
    _database(path)
    server = create_monitoring_server(output, token="dashboard-test-token-at-least-24-chars", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        headers = {"Authorization": "Bearer dashboard-test-token-at-least-24-chars"}
        with urlopen(Request(base + "/v1/analytics?strategy=MRL1&start=2024-01-02&end=2024-01-02", headers=headers), timeout=3) as response:
            payload = json.loads(response.read())["data"]
        assert payload["sample"]["total_closed_trades"] == 1
        assert payload["metrics"]["net_pnl"] == pytest.approx(9)
        assert [row["trade_id"] for row in payload["trade_outcomes"]] == ["a"]
        with urlopen(Request(base + "/v1/trades?strategy=MRL1&limit=10", headers=headers), timeout=3) as response:
            trades = json.loads(response.read())["data"]
        assert [row["trade_id"] for row in trades] == ["a"]
        try:
            urlopen(Request(base + "/v1/analytics", method="POST", headers=headers), timeout=3)
        except HTTPError as exc:
            assert exc.code == 405
        else:
            raise AssertionError("POST must remain rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        path.unlink(missing_ok=True)
