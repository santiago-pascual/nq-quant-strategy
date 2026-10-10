import json
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

from src.paper.monitoring_alerts import (AlertConfig, AlertStateStore, PaperAlertMonitor,
                                         evaluate_cme_market_state, evaluate_status)
from src.paper.notifications import NotificationPreferences, format_event


def _root() -> Path:
    path = Path.cwd() / "results" / "paper" / f"alert_monitor_test_{uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _status(**updates):
    result = {"mode": "PAPER", "run_id": "fixture-run",
              "system": {"state": "RUNNING", "feed_health": {"connected": True, "provider_connected": True,
                  "mode": "CAUGHT_UP", "backlog_bars": 0, "market_open": False}},
              "portfolio": {"daily_pnl": 0.0, "drawdown": 0.0, "open_positions": []}}
    for section, values in updates.items():
        result.setdefault(section, {}).update(values)
    return result


def _append(path: Path, event_type: str, payload: dict, event_id="event-1"):
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"event_id": event_id, "event_type": event_type,
                            "timestamp": "2026-10-09T15:00:00+00:00", "payload": payload}) + "\n")


def _db(path: Path, *, loss_limit=500.0, open_trades=()):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE configuration_versions(rowid INTEGER PRIMARY KEY, config_kind TEXT, source_json TEXT)")
        db.execute("INSERT INTO configuration_versions(config_kind,source_json) VALUES('paper_engine',?)",
                   (json.dumps({"risk_limits": {"max_daily_loss": loss_limit}}),))
        db.execute("CREATE TABLE trades(strategy TEXT,contract TEXT,direction TEXT,quantity INTEGER,status TEXT)")
        db.executemany("INSERT INTO trades VALUES(?,?,?,?, 'OPEN')", list(open_trades))
        db.execute("CREATE TABLE orders(order_id TEXT,quantity INTEGER,status TEXT,strategy TEXT,payload_json TEXT,reference_price REAL)")
        db.execute("CREATE TABLE fills(order_id TEXT,quantity INTEGER)")


def test_status_detectors_use_existing_limits_and_do_not_infer_feed_stall_when_market_closed():
    config = AlertConfig(backlog_warning_bars=10, drawdown_warning_usd=100, drawdown_critical_usd=200)
    status = _status(system={"feed_health": {"connected": True, "provider_connected": True,
        "stale": True, "market_open": False, "backlog_bars": 11}},
        portfolio={"daily_pnl": -500.0, "drawdown": -220.0})
    status["system"]["market_state"] = {"state": "MARKET_CLOSED"}
    alerts, caps = evaluate_status(status, config=config, risk_limits={"max_daily_loss": 500})
    names = {a["alert_type"] for a in alerts}
    assert "Daily loss limit reached" in names
    assert "Drawdown beyond expected threshold" in names
    assert "Feed backlog excessive" in names
    assert "Data feed disconnected" not in names
    assert caps["daily_loss"] == "AVAILABLE"
    assert all(a["environment"] == "PAPER" for a in alerts)


def _reviewed_calendar():
    from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
    project = Path(__file__).resolve().parents[2]
    return CMETradingCalendar(CMECalendarSnapshot.from_json(
        project / "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json"))


def test_cme_state_uses_reviewed_calendar_timezone_weekend_maintenance_and_coverage():
    from datetime import datetime, timezone
    from datetime import date
    from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
    calendar = _reviewed_calendar()
    assert evaluate_cme_market_state(calendar, datetime(2026, 10, 10, 13, 51, tzinfo=timezone.utc))["state"] == "MARKET_CLOSED"
    assert evaluate_cme_market_state(calendar, datetime(2026, 10, 12, 21, 30, tzinfo=timezone.utc))["state"] == "MARKET_BREAK"
    assert evaluate_cme_market_state(calendar, datetime(2026, 10, 12, 22, 1, tzinfo=timezone.utc))["state"] == "MARKET_OPEN"
    # 2026-11-02 14:30Z is 09:30 ET after DST has ended; outside reviewed coverage is UNKNOWN.
    result = evaluate_cme_market_state(calendar, datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc))
    assert result["state"] == "MARKET_UNKNOWN"
    assert evaluate_cme_market_state(None, datetime(2026, 10, 10, tzinfo=timezone.utc))["state"] == "MARKET_UNKNOWN"
    dst_calendar = CMETradingCalendar(CMECalendarSnapshot(version="dst-test", source="https://www.cmegroup.com/trading-hours.html",
        coverage_start=date(2026, 10, 30), coverage_end=date(2026, 11, 3)))
    assert evaluate_cme_market_state(dst_calendar, datetime(2026, 11, 1, 22, 30, tzinfo=timezone.utc))["state"] == "MARKET_BREAK"
    assert evaluate_cme_market_state(dst_calendar, datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc))["state"] == "MARKET_OPEN"


def test_provider_delay_and_engine_processing_stalls_are_separate_and_only_when_market_open():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 12, 22, 30, tzinfo=timezone.utc)
    # MNQ starts Sunday 18:00 ET (=22:00 UTC); this timestamp is inside the reviewed Monday session.
    status = _status(system={"feed_health": {"connected": True, "provider_connected": True,
        "latest_available_bar": (now - timedelta(seconds=1400)).isoformat(),
        "last_processed_bar": (now - timedelta(seconds=1900)).isoformat(), "backlog_bars": 12}})
    status["system"]["market_state"] = {"state": "MARKET_OPEN"}
    monitor = AlertConfig(provider_stall_warning_seconds=300, provider_stall_critical_seconds=900,
                          processing_stall_warning_seconds=300, processing_stall_critical_seconds=900)
    alerts, _ = evaluate_status(status, config=monitor, risk_limits=None, now=now)
    names = {row["alert_type"] for row in alerts}
    assert "Provider data stalled" in names
    assert "Paper processing stalled" in names
    closed = _status(system={"feed_health": {"connected": True, "provider_connected": True,
        "latest_available_bar": "2026-10-09T20:59:00+00:00", "last_processed_bar": "2026-10-09T20:59:00+00:00",
        "backlog_bars": 0}})
    closed["system"]["market_state"] = {"state": "MARKET_CLOSED"}
    closed_alerts, _ = evaluate_status(closed, config=monitor, risk_limits=None, now=now)
    assert not any(row["alert_type"] in {"Provider data stalled", "Paper processing stalled"} for row in closed_alerts)


def test_stall_alert_requires_persisted_debounce_and_market_state_is_saved():
    from datetime import datetime, timedelta, timezone
    root = _root()
    try:
        (root / "events.jsonl").write_text("", encoding="utf-8")
        status = _status(system={"feed_health": {"connected": True, "provider_connected": True,
            "latest_available_bar": "2026-10-12T21:00:00+00:00", "last_processed_bar": "2026-10-12T21:00:00+00:00",
            "backlog_bars": 0}})
        (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
        now = datetime(2026, 10, 12, 22, 30, tzinfo=timezone.utc)
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json", events_path=root / "alert_events.jsonl",
            calendar=_reviewed_calendar(), config=AlertConfig(provider_stall_warning_seconds=60,
            provider_stall_critical_seconds=300, stall_debounce_seconds=60))
        first = monitor.poll_once(now=now)
        assert not any(a["alert_type"] == "Provider data stalled" for a in first["emitted"])
        second = monitor.poll_once(now=now + timedelta(seconds=61))
        assert any(a["alert_type"] == "Provider data stalled" for a in second["emitted"])
        assert second["state"]["market_state"]["state"] == "MARKET_OPEN"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_all_event_alert_categories_and_simulated_slippage_and_latency_are_labeled():
    root = _root()
    try:
        (root / "events.jsonl").write_text("", encoding="utf-8")
        (root / "status.json").write_text(json.dumps(_status()), encoding="utf-8")
        _db(root / "paper_analytics.sqlite3")
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json",
                                    events_path=root / "alert_events.jsonl",
                                    config=AlertConfig(slippage_warning_ticks=1, execution_latency_warning_ms=10))
        _append(root / "events.jsonl", "paper_order_rejected", {"strategy_name": "ORB", "reason": "simulator rejection"}, "reject")
        _append(root / "events.jsonl", "position_size_mismatch", {"expected_quantity": 2, "actual_quantity": 1}, "size")
        _append(root / "events.jsonl", "strategy_invariant_violation", {"strategy_name": "MRL1", "message": "invalid signal input"}, "signal")
        _append(root / "events.jsonl", "paper_order_filled", {"strategy_name": "ORB", "price": 100.5,
            "reference_price": 100.0, "market_data_provenance": "CONTINUOUS_PAPER"}, "slip")
        _append(root / "events.jsonl", "order_submitted", {"strategy_name": "ORB", "execution_processing_latency_ms": 25,
            "market_data_provenance": "CONTINUOUS_PAPER"}, "latency")
        monitor.poll_once()
        rows = [json.loads(line)["payload"] for line in (root / "alert_events.jsonl").read_text().splitlines()]
        names = {row["alert_type"] for row in rows}
        assert "Simulated order rejected" in names
        assert "Position size mismatch" in names
        assert "Strategy generated unexpected signal" in names
        assert "Slippage > threshold" in names
        assert "Execution latency abnormal" in names
        slippage = next(row for row in rows if row["alert_type"] == "Slippage > threshold")
        assert "SIMULATED" in slippage["details"]
        assert slippage["environment"] == "PAPER"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_position_mismatch_daily_loss_feed_and_unexpected_termination_are_read_only_alerts():
    root = _root()
    try:
        (root / "events.jsonl").write_text("", encoding="utf-8")
        status = _status(system={"state": "FAILED", "feed_health": {"connected": False,
            "provider_connected": False, "mode": "DISCONNECTED", "backlog_bars": 100}},
            portfolio={"daily_pnl": -510, "open_positions": [{"strategy": "ORB", "contract": "MNQZ6",
                "direction": "long", "quantity": 2}]})
        (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
        _db(root / "paper_analytics.sqlite3", open_trades=[("ORB", "MNQZ6", "long", 1)])
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json",
                                    events_path=root / "alert_events.jsonl")
        result = monitor.poll_once()
        active = list(result["state"]["active"].values())
        names = {a["alert_type"] for a in active}
        assert {"Paper Engine error state", "Data feed disconnected", "Daily loss limit reached",
                "Position size mismatch"}.issubset(names)
        assert result["state"]["capabilities"]["live_broker_reconciliation"] == "DISABLED"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unexpected_position_and_filled_order_size_mismatch_use_durable_records():
    root = _root()
    try:
        (root / "events.jsonl").write_text("", encoding="utf-8")
        (root / "status.json").write_text(json.dumps(_status(portfolio={"open_positions": [
            {"strategy": "ORB", "contract": "MNQZ6", "direction": "short", "quantity": 1}]})), encoding="utf-8")
        _db(root / "paper_analytics.sqlite3")
        with sqlite3.connect(root / "paper_analytics.sqlite3") as db:
            db.execute("INSERT INTO orders VALUES('SIM-1',2,'paper_order_filled','ORB','{}',100)")
            db.execute("INSERT INTO fills VALUES('SIM-1',1)")
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json", events_path=root / "alert_events.jsonl")
        active = monitor.poll_once()["state"]["active"].values()
        names = {a["alert_type"] for a in active}
        assert "Unexpected position" in names
        assert "Position size mismatch" in names
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_first_monitor_start_skips_old_event_history_but_processes_new_events():
    root = _root()
    try:
        events = root / "events.jsonl"
        _append(events, "paper_order_rejected", {"reason": "old"}, "old-reject")
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json", events_path=root / "alert_events.jsonl")
        assert monitor.poll_once()["state"]["active"] == {}
        _append(events, "paper_order_rejected", {"reason": "new"}, "new-reject")
        result = monitor.poll_once()
        assert any(a["source_event_id"] == "new-reject" for a in result["emitted"])
        assert len(result["state"]["active"]) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_feed_status_and_event_incident_do_not_duplicate_the_same_telegram_alert():
    root = _root()
    try:
        events = root / "events.jsonl"
        events.write_text("", encoding="utf-8")
        status = _status(system={"feed_health": {"connected": False, "provider_connected": False,
            "mode": "DISCONNECTED", "last_error": "TWS connection lost"}})
        (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json", events_path=root / "alert_events.jsonl")
        _append(events, "feed_disconnected", {"run_id": "fixture-run", "message": "feed disconnected"}, "feed-down")
        result = monitor.poll_once()
        transition_rows = [json.loads(line)["payload"] for line in (root / "alert_events.jsonl").read_text().splitlines()]
        matching = [row for row in transition_rows if row["alert_type"] == "Data feed disconnected"]
        assert len(matching) == 1
        assert result["state"]["event_ids"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reconnect_outage_remains_one_alert_through_catchup_then_resolves_once():
    root = _root()
    try:
        events = root / "events.jsonl"
        events.write_text("", encoding="utf-8")
        state_path = root / "alert_state.json"
        status_path = root / "status.json"
        status = _status(system={"feed_health": {
            "connected": False, "provider_connected": False, "mode": "RECONNECTING",
            "connection_state": "RECONNECTING", "incident_id": "incident-1",
            "retry_count": 3, "next_retry_at_utc": "2026-10-10T12:00:05+00:00",
            "last_error": "TWS connectivity lost (1100)", "backlog_bars": 0,
        }})
        status["system"]["market_state"] = {"state": "MARKET_CLOSED"}
        status_path.write_text(json.dumps(status), encoding="utf-8")
        monitor = PaperAlertMonitor(run_dir=root, state_path=state_path, events_path=root / "alerts.jsonl")
        first = monitor.poll_once()
        assert [item["alert_type"] for item in first["state"]["active"].values()] == ["Data feed disconnected"]

        # Handshake is back, but catch-up is still underway: retain the same
        # incident and do not send a recovery notification yet.
        status["system"]["feed_health"].update({"connected": True, "provider_connected": True,
            "mode": "RECOVERING", "connection_state": "RECOVERING", "retry_count": 3,
            "last_successful_handshake_utc": "2026-10-10T12:01:00+00:00",
            "last_error": "awaiting chronological recovery", "backlog_bars": 12})
        status_path.write_text(json.dumps(status), encoding="utf-8")
        during = monitor.poll_once()
        assert [item["alert_type"] for item in during["state"]["active"].values()] == ["Data feed disconnected"]

        status["system"]["feed_health"].update({"mode": "CAUGHT_UP", "connection_state": "CONNECTED",
            "incident_id": None, "next_retry_at_utc": None, "last_error": None,
            "backlog_bars": 0, "catchup_active": False})
        status_path.write_text(json.dumps(status), encoding="utf-8")
        recovered = monitor.poll_once()
        assert recovered["state"]["active"] == {}
        assert any(item["status"] == "RECOVERED" for item in recovered["emitted"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_alert_recovery_acknowledgment_restart_and_dedup_are_durable():
    root = _root()
    try:
        state_path, event_path = root / "state.json", root / "alerts.jsonl"
        store = AlertStateStore(state_path, event_path)
        warning = {"alert_id": "feed", "event_id": "feed-active", "alert_type": "Data feed disconnected",
                   "severity": "CRITICAL", "environment": "PAPER", "strategy": "SYSTEM", "symbol": "MNQ",
                   "timestamp_utc": "2026-10-09T12:00:00+00:00", "account_run_id": "r",
                   "expected": True, "observed": False, "threshold": None, "status": "ACTIVE", "details": "down"}
        assert len(store.observe_conditions([warning])) == 1
        assert store.observe_conditions([warning]) == []
        restarted = AlertStateStore(state_path, event_path)
        recovered = restarted.observe_conditions([])
        assert recovered[0]["status"] == "RECOVERED"
        assert len(restarted.state["history"]) == 2
        repeated = restarted.observe_conditions([warning])[0]
        assert repeated["event_id"] != restarted.state["history"][0]["event_id"]
        assert restarted.record_event_alert({**warning, "alert_id": "reject", "event_id": "order-reject"})
        assert not restarted.record_event_alert({**warning, "alert_id": "reject", "event_id": "order-reject"})
        assert restarted.acknowledge("reject")
        assert restarted.state["active"]["reject"]["status"] == "ACKNOWLEDGED"
        assert restarted.acknowledge("reject", resolved=True)
        assert "reject" not in restarted.state["active"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_condition_measurement_updates_do_not_emit_repeated_telegram_transitions():
    root = _root()
    try:
        store = AlertStateStore(root / "state.json", root / "alerts.jsonl")
        base = {"alert_id": "provider-stall", "event_id": "first", "alert_type": "Provider data stalled",
                "severity": "WARNING", "environment": "PAPER", "strategy": "SYSTEM", "symbol": "MNQ",
                "timestamp_utc": "2026-10-12T22:00:00Z", "account_run_id": "r", "expected": "frontier",
                "observed": {"lag_seconds": 301}, "threshold": 300, "status": "ACTIVE", "details": "lagging"}
        assert len(store.observe_conditions([base])) == 1
        assert store.observe_conditions([{**base, "event_id": "second", "observed": {"lag_seconds": 303},
                                         "details": "lagging by 303 seconds"}]) == []
        assert store.state["active"]["provider-stall"]["observed"]["lag_seconds"] == 303
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_legacy_error_status_is_not_mislabeled_as_unexpected_process_termination():
    root = _root()
    try:
        (root / "events.jsonl").write_text("", encoding="utf-8")
        status = _status(system={"state": "ERROR", "feed_health": {"connected": False,
            "provider_connected": False, "mode": "DISCONNECTED"}})
        (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
        store = AlertStateStore(root / "alert_state.json", root / "alert_events.jsonl")
        old = {"alert_type": "Unexpected Paper Engine termination", "event_id": "legacy", "alert_id": "legacy-id",
               "severity": "CRITICAL", "environment": "PAPER", "strategy": "SYSTEM", "symbol": "MNQ",
               "timestamp_utc": "2026-10-10T00:00:00Z", "account_run_id": "r", "expected": "RUNNING",
               "observed": "ERROR", "threshold": None, "status": "ACTIVE", "details": "legacy status mapping",
               "source_event_id": None}
        store.record_event_alert(old)
        monitor = PaperAlertMonitor(run_dir=root, state_path=root / "alert_state.json",
                                    events_path=root / "alert_events.jsonl")
        active = monitor.poll_once()["state"]["active"]
        assert "legacy-id" not in active
        assert any(a["alert_type"] == "Paper Engine error state" for a in active.values())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_telegram_alert_schema_is_complete_and_missing_capabilities_are_not_healthy():
    from src.paper.monitoring_alerts import evaluate_status
    status = {"mode": "PAPER", "system": {"state": "UNAVAILABLE"}, "portfolio": {}}
    alerts, caps = evaluate_status(status, config=AlertConfig(), risk_limits=None)
    assert caps["daily_loss"] == "UNAVAILABLE"
    assert caps["feed"] == "UNAVAILABLE"
    assert alerts == []  # Unknown data is reported as unavailable, not as a healthy alert state.
    alert = {"event_id": "id", "event_type": "monitoring_alert", "timestamp": "2026-10-09T12:00:00Z",
             "payload": {"alert_type": "Unexpected position", "severity": "CRITICAL", "environment": "PAPER",
                         "strategy": "SYSTEM", "symbol": "MNQ", "timestamp_utc": "2026-10-09T12:00:00Z",
                         "expected": "flat", "observed": "long 1", "threshold": None, "status": "ACTIVE",
                         "details": "ledger mismatch", "account_run_id": "r"}}
    formatted = format_event(alert, NotificationPreferences())
    assert formatted and all(part in formatted[1] for part in ("Environment: PAPER", "Expected: flat", "Observed: long 1", "Status: ACTIVE"))


def test_alert_api_is_authenticated_get_only_without_opening_database():
    from threading import Thread
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from src.paper.monitoring_api import create_monitoring_server
    root = _root()
    server = None
    try:
        runtime = root.parent / ".notifications" / root.name
        runtime.mkdir(parents=True)
        state = {"schema_version": 1, "active": {"x": {"status": "ACTIVE", "alert_type": "Feed down"}},
                 "history": [], "capabilities": {"daily_loss": "UNAVAILABLE"},
                 "market_state": {"state": "MARKET_CLOSED", "reason": "weekend"},
                 "feed_assessment": {"connection": "DISCONNECTED"},
                 "thresholds": {"daily_loss_limit_usd": {"value": 500}}}
        (runtime / "alert_state.json").write_text(json.dumps(state), encoding="utf-8")
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        (runtime / "notifier_heartbeat.json").write_text(json.dumps({"pid": 1, "timestamp_utc": now}), encoding="utf-8")
        (runtime / "engine_watchdog_state.json").write_text(json.dumps({"last_probe_status": "unavailable",
            "last_probe_at_utc": now}), encoding="utf-8")
        server = create_monitoring_server(root, token="test-monitoring-token-at-least-24", port=0)
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f"http://127.0.0.1:{server.server_port}/v1/alerts"
        def request(token=None, method="GET"):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            try:
                with urlopen(Request(url, headers=headers, method=method), timeout=3) as response:
                    return response.status, json.loads(response.read())
            except HTTPError as exc:
                return exc.code, json.loads(exc.read())
        assert request()[0] == 401
        code, body = request("test-monitoring-token-at-least-24")
        assert code == 200 and body["data"]["active"][0]["alert_type"] == "Feed down"
        assert body["data"]["capabilities"]["daily_loss"] == "UNAVAILABLE"
        assert body["data"]["market_state"]["state"] == "MARKET_CLOSED"
        assert body["data"]["services"]["notification_service"]["state"] == "HEALTHY"
        assert body["data"]["services"]["process_watchdog"]["process_probe"] == "unavailable"
        assert body["data"]["feed_assessment"]["connection"] == "DISCONNECTED"
        assert body["data"]["thresholds"]["daily_loss_limit_usd"]["value"] == 500
        assert request("test-monitoring-token-at-least-24", "POST")[0] == 405
        assert json.loads((runtime / "alert_state.json").read_text(encoding="utf-8")) == state
    finally:
        if server:
            server.shutdown(); server.server_close()
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(root.parent / ".notifications" / root.name, ignore_errors=True)
