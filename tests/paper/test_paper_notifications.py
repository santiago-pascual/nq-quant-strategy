import json
from pathlib import Path
import shutil
from uuid import uuid4

from src.paper.notifications import NotificationPreferences, PersistentEventNotifier, format_event
from src.paper.notification_cli import _reviewed_calendar_for_run


def _root():
    return Path.cwd() / "results" / "paper" / f"notification_test_{uuid4().hex}"


def _event(kind="paper_position_opened", *, provenance="CONTINUOUS_PAPER"):
    return {"event_id": uuid4().hex, "event_type": kind, "timestamp": "2026-10-09T20:00:00+00:00",
            "payload": {"severity": "INFO", "strategy_name": "ORB", "side": "long",
                        "quantity": 1, "entry_price": 21000, "market_data_provenance": provenance}}


def _append(path: Path, event):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event) + "\n")


def test_new_events_deliver_once_and_recovered_trade_is_suppressed():
    root = _root()
    try:
        root.mkdir(parents=True)
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        sent = []
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0))
        _append(events, _event())
        assert notifier.scan_once() == "delivered"
        assert notifier.scan_once() == "idle"
        assert len(sent) == 1 and "Simulated Paper position opened" in sent[0]

        _append(events, _event(provenance="RECOVERED_PAPER"))
        assert notifier.scan_once() == "skipped"
        assert len(sent) == 1
        restarted = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0))
        assert restarted.scan_once() == "idle"
        assert len(sent) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_calendar_loader_skips_unverified_template_and_matches_reviewed_run_identity():
    from datetime import date
    from src.paper.cme_calendar import CMECalendarSnapshot

    root = _root(); run = root / "run"; config = root / "config"
    run.mkdir(parents=True); config.mkdir()
    try:
        snapshot = CMECalendarSnapshot(version="test-v1", source="https://www.cmegroup.com/trading-hours.html",
            coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31))
        (run / "status.json").write_text(json.dumps({"system": {"cme_calendar": {"identity": snapshot.identity}}}), encoding="utf-8")
        # This template matches the same glob but does not use the runtime schema.
        (config / "cme_mnq_calendar_2026-10-07_REVIEW_TEMPLATE.json").write_text(
            json.dumps({"review_status": "TEMPLATE", "coverage_dates": []}), encoding="utf-8")
        valid_path = config / "cme_mnq_calendar_2026-10-08_2026-10-31.json"
        valid_path.write_text(json.dumps(snapshot.to_mapping()), encoding="utf-8")
        (config / "cme_mnq_calendar_2026-10-08_2026-10-31.review.json").write_text(json.dumps({
            "review_status": "REVIEWED_PRODUCT_FILTERED_CME_GLOBEX", "product": "MNQ",
            "snapshot_identity": snapshot.identity}), encoding="utf-8")
        loaded = _reviewed_calendar_for_run(run, config_dir=config)
        assert loaded is not None
        assert loaded.snapshot.identity == snapshot.identity
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_duplicate_event_id_at_a_later_offset_is_suppressed_from_delivery_journal():
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        sent = []
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0))
        event = _event()
        _append(events, event)
        assert notifier.scan_once() == "delivered"
        _append(events, event)
        assert notifier.scan_once() == "skipped"
        assert len(sent) == 1
        assert any(row["state"] == "duplicate_suppressed" for row in
                   map(json.loads, (root / "delivery.jsonl").read_text().splitlines()))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_retry_backoff_persists_pending_event_and_test_message_is_labeled():
    root = _root(); now = [1000.0]; calls = []
    try:
        root.mkdir(parents=True)
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        def send(message):
            calls.append(message)
            if len(calls) == 1: raise OSError("network unavailable")
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=send,
            preferences=NotificationPreferences(minimum_interval_seconds=0), clock=lambda: now[0])
        _append(events, _event("system_error"))
        assert notifier.scan_once() == "retry"
        assert notifier.scan_once() == "backoff"
        now[0] += 5
        assert notifier.scan_once() == "delivered"
        assert json.loads((root / "state.json").read_text()) ["pending"] is None
        notifier.test_message()
        assert "TEST" in calls[-1] and "No order was placed" in calls[-1]
        journal = [json.loads(line) for line in (root / "delivery.jsonl").read_text().splitlines()]
        assert [row["state"] for row in journal] == ["queued", "retry", "delivered"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_notification_mapping_keeps_recovery_provenance_distinct():
    prefs = NotificationPreferences(recovered_trade_summary=True)
    result = format_event(_event(provenance="RECOVERED_PAPER"), prefs)
    assert result and "RECOVERED PAPER" in result[1]


def test_status_incident_is_deduplicated_and_recovery_notifies_once():
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        status = root / "status.json"
        status.write_text(json.dumps({"system": {"state": "RUNNING", "feed_health": {
            "connected": True, "mode": "RECOVERING", "backlog_bars": 80,
            "last_processed_bar": "2026-10-09T10:00:00+00:00",
            "latest_available_bar": "2026-10-09T12:00:00+00:00"}}}), encoding="utf-8")
        sent = []
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0), clock=lambda: 2000)
        assert notifier.scan_status(status) == "delivered"
        assert notifier.scan_status(status) == "unchanged"
        assert len(sent) == 1 and "backlog is 80" in sent[0]

        status.write_text(json.dumps({"system": {"state": "RUNNING", "feed_health": {
            "connected": True, "mode": "CAUGHT_UP", "backlog_bars": 0}}}), encoding="utf-8")
        assert notifier.scan_status(status) == "delivered"
        assert "condition cleared" in sent[-1].lower()
        assert notifier.scan_status(status) == "healthy"
        assert len(sent) == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_daily_summary_requires_continuous_bar_from_current_new_york_date():
    from datetime import datetime, timezone
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        status = root / "status.json"
        status.write_text(json.dumps({"system": {"state": "RUNNING", "feed_health": {
            "connected": True, "mode": "CAUGHT_UP", "backlog_bars": 0}},
            "portfolio": {"daily_pnl": 12.5, "balance": 50012.0, "equity": 50020.0}}), encoding="utf-8")
        sent = []
        epoch = datetime(2026, 10, 9, 21, 30, tzinfo=timezone.utc).timestamp()
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0), clock=lambda: epoch)
        _append(events, {"event_id": "live-bar", "event_type": "market_data",
                         "timestamp": "2026-10-09T20:00:00+00:00",
                         "payload": {"market_data_provenance": "CONTINUOUS_PAPER"}})
        assert notifier.scan_once() == "skipped"
        assert notifier.scan_status(status) == "delivered"
        assert "Daily simulated Paper summary" in sent[0]
        assert "12.50 USD" in sent[0]
        assert notifier.scan_status(status) == "healthy"
        assert len(sent) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_significant_equity_change_alert_uses_persisted_account_snapshots():
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        status = root / "status.json"
        def write(equity):
            status.write_text(json.dumps({"system": {"state": "RUNNING", "feed_health": {
                "connected": True, "mode": "CAUGHT_UP", "backlog_bars": 0}},
                "portfolio": {"equity": equity}}), encoding="utf-8")
        write(50000)
        sent = []
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0, significant_equity_change_usd=500),
            clock=lambda: 1000)
        assert notifier.scan_status(status) == "healthy"
        write(49499)
        assert notifier.scan_status(status) == "delivered"
        assert "changed by -501.00 USD" in sent[0]
        write(49499)
        assert notifier.scan_status(status) == "delivered"
        assert "condition cleared" in sent[1].lower()
        assert notifier.scan_status(status) == "healthy"
        assert len(sent) == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_trade_alert_uses_real_contract_cost_and_recovery_provenance_fields():
    close = _event("paper_position_closed")
    close["payload"].update({"contract_symbol": "MNQZ6", "net_pnl": -12.34,
                             "market_data_provenance": "CONTINUOUS_PAPER"})
    message = format_event(close, NotificationPreferences())
    assert message and "Contract: MNQZ6" in message[1]
    assert "Qty: 1" in message[1] and "Realized P&L: -12.34 USD" in message[1]
    assert "Provenance: CONTINUOUS_PAPER" in message[1]


def test_risk_halt_is_alerted_and_recovered_trade_events_are_not_catchup_spam():
    prefs = NotificationPreferences()
    halted = _event("trading_halted")
    assert format_event(halted, prefs) and "CRITICAL" in format_event(halted, prefs)[0]
    recovered = _event("paper_position_closed", provenance="RECOVERED_PAPER")
    assert format_event(recovered, prefs) is None
    # Operational recovery failures remain visible even when they occur during catch-up.
    failed = _event("recovery_failed", provenance="RECOVERED_PAPER")
    assert format_event(failed, prefs) and "CRITICAL" in format_event(failed, prefs)[0]


def test_first_start_discards_a_partial_preexisting_jsonl_tail():
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"
        original = json.dumps(_event())
        partial = original[:-3]
        events.write_text(partial, encoding="utf-8")
        sent = []
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0))
        with events.open("a", encoding="utf-8") as f:
            f.write(original[-3:] + "\n")
            f.write(json.dumps(_event()) + "\n")
        assert notifier.scan_once() == "delivered"
        assert len(sent) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_independent_watchdog_detects_unexpected_exit_and_recovery_once():
    from src.paper.notifications import PaperProcessWatchdog
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        status = root / "status.json"
        status.write_text(json.dumps({"system": {"state": "RUNNING", "feed_health": {"connected": True,
            "mode": "CAUGHT_UP", "backlog_bars": 0}}}), encoding="utf-8")
        sent = []; now = [100.0]; pids = [[4242]]
        notifier = PersistentEventNotifier(event_path=root / "watchdog_events.jsonl",
            state_path=root / "watchdog_notify.json", journal_path=root / "watchdog_delivery.jsonl",
            send=sent.append, preferences=NotificationPreferences(minimum_interval_seconds=0), clock=lambda: now[0])
        watchdog = PaperProcessWatchdog(run_dir=root, state_path=root / "watchdog_state.json",
            event_path=root / "watchdog_events.jsonl", process_probe=lambda: pids[0], clock=lambda: now[0],
            grace_seconds=20)
        assert watchdog.check_once() == "running"
        pids[0] = []
        assert watchdog.check_once() == "absence_grace"
        now[0] += 21
        assert watchdog.check_once() == "termination_detected"
        assert notifier.scan_once() == "delivered"
        assert "disappeared unexpectedly" in sent[-1]
        assert watchdog.check_once() == "already_alerted"
        pids[0] = [4300]
        assert watchdog.check_once() == "recovered"
        assert notifier.scan_once() == "delivered"
        assert "detected again" in sent[-1]
        assert len(sent) == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_watchdog_distinguishes_clean_stop_and_reported_engine_error():
    from src.paper.notifications import PaperProcessWatchdog
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        status = root / "status.json"
        state = root / "watchdog.json"
        status.write_text(json.dumps({"system": {"state": "STOPPED"}}), encoding="utf-8")
        watchdog = PaperProcessWatchdog(run_dir=root, state_path=state, event_path=events,
            process_probe=lambda: [], clock=lambda: 100)
        assert watchdog.check_once() == "stopped_cleanly"
        status.write_text(json.dumps({"system": {"state": "ERROR"}}), encoding="utf-8")
        assert watchdog.check_once() == "failed_status_reported"
        assert not events.read_text(encoding="utf-8").strip()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_notification_does_not_call_an_error_clean_shutdown():
    event = {'event_type': 'system_stopped', 'timestamp': '2026-10-10T13:45:50Z',
             'payload': {'state': 'ERROR'}}
    severity, message = format_event(event, NotificationPreferences())
    assert severity == 'CRITICAL'
    assert 'stopped with an error' in message
    assert 'cleanly' not in message
    event['payload']['state'] = 'STOPPED'
    assert format_event(event, NotificationPreferences())[0] == 'INFO'
    event['payload'].pop('state')
    assert 'outcome unverified' in format_event(event, NotificationPreferences())[1]


def test_watchdog_detects_stale_engine_and_notifier_heartbeats_then_recovery():
    import os
    from src.paper.notifications import PaperProcessWatchdog
    root = _root(); root.mkdir(parents=True)
    try:
        (root / "status.json").write_text(json.dumps({"system": {"state": "RUNNING"}}), encoding="utf-8")
        notifier_heartbeat = root / "notifier_heartbeat.json"
        notifier_heartbeat.write_text("{}", encoding="utf-8")
        now = [2000.0]
        for path in (root / "status.json", notifier_heartbeat):
            os.utime(path, (now[0], now[0]))
        watchdog = PaperProcessWatchdog(run_dir=root, state_path=root / "watchdog.json",
            event_path=root / "watchdog_events.jsonl", process_probe=lambda: [4321], clock=lambda: now[0],
            heartbeat_stale_seconds=30, notifier_heartbeat_path=notifier_heartbeat)
        assert watchdog.check_once() == "running"
        now[0] += 31
        assert watchdog.check_once() == "running"
        now[0] += 31
        assert watchdog.check_once() == "heartbeat_stale"
        events = [json.loads(line)["event_type"] for line in (root / "watchdog_events.jsonl").read_text().splitlines()]
        assert "paper_status_heartbeat_stale" in events
        assert "paper_notification_service_stale" in events
        now[0] += 1
        for path in (root / "status.json", notifier_heartbeat):
            os.utime(path, (now[0], now[0]))
        assert watchdog.check_once() == "heartbeat_recovered"
        events = [json.loads(line)["event_type"] for line in (root / "watchdog_events.jsonl").read_text().splitlines()]
        assert "paper_status_heartbeat_recovered" in events
        assert "paper_notification_service_recovered" in events
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_watchdog_checks_notifier_heartbeat_even_when_process_probe_is_unavailable():
    import os
    from src.paper.notifications import PaperProcessWatchdog
    root = _root(); root.mkdir(parents=True)
    try:
        heartbeat = root / "notifier_heartbeat.json"
        heartbeat.write_text("{}", encoding="utf-8")
        now = [3000.0]
        os.utime(heartbeat, (now[0], now[0]))
        watchdog = PaperProcessWatchdog(run_dir=root, state_path=root / "watchdog.json",
            event_path=root / "watchdog_events.jsonl", process_probe=lambda: None, clock=lambda: now[0],
            heartbeat_stale_seconds=30, notifier_heartbeat_path=heartbeat)
        assert watchdog.check_once() == "probe_unavailable"
        now[0] += 31
        assert watchdog.check_once() == "probe_unavailable"
        now[0] += 31
        assert watchdog.check_once() == "notification_service_stale"
        rows = [json.loads(line) for line in (root / "watchdog_events.jsonl").read_text().splitlines()]
        assert [row["event_type"] for row in rows] == ["paper_notification_service_stale"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sidecar_atomic_snapshot_retries_transient_windows_sharing_violation(monkeypatch):
    import os
    from src.paper.atomic_io import atomic_replace_with_retry
    root = _root(); root.mkdir(parents=True)
    try:
        source, destination = root / "state.tmp", root / "state.json"
        source.write_text('{"fresh":true}', encoding="utf-8")
        original_replace = os.replace
        calls = [0]

        def replace_once_locked(src, dst):
            calls[0] += 1
            if calls[0] == 1:
                raise PermissionError("synthetic transient Windows sharing lock")
            return original_replace(src, dst)

        monkeypatch.setattr("src.paper.atomic_io.os.replace", replace_once_locked)
        atomic_replace_with_retry(source, destination, attempts=3, initial_delay_seconds=0)
        assert calls[0] == 2
        assert destination.read_text(encoding="utf-8") == '{"fresh":true}'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reviewed_calendar_and_roll_reminders_are_once_and_source_backed():
    from datetime import datetime, timezone
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"; events.write_text("", encoding="utf-8")
        status = root / "status.json"
        status.write_text(json.dumps({"system": {"state": "RUNNING", "feed_health": {"mode": "CAUGHT_UP"}}}), encoding="utf-8")
        schedule = root / "schedule.json"
        schedule.write_text(json.dumps({"schema_version": 1, "events": [
            {"id": "unreviewed", "kind": "contract_roll", "label": "ignored", "effective_at_utc": "2026-10-12T00:00:00Z", "reviewed": False, "source_url": "https://example.invalid"},
            {"id": "holiday", "kind": "cme_calendar", "label": "Reviewed early close", "effective_at_utc": "2026-10-14T00:00:00Z", "reviewed": True, "source_url": "https://www.cmegroup.com/trading-hours.html"},
            {"id": "roll", "kind": "contract_roll", "label": "Reviewed contract transition", "effective_at_utc": "2026-10-15T00:00:00Z", "reviewed": True, "source_url": "https://example.com/review"}
        ]}), encoding="utf-8")
        sent = []
        notifier = PersistentEventNotifier(event_path=events, state_path=root / "state.json",
            journal_path=root / "delivery.jsonl", send=sent.append,
            preferences=NotificationPreferences(minimum_interval_seconds=0), clock=lambda: 1791849600)
        now = datetime(2026, 10, 10, tzinfo=timezone.utc)
        assert notifier.scan_reminders(schedule, now=now, lead_days=7) == ["holiday"]
        assert notifier.scan_status(status) == "delivered"
        assert "Verified source: https://www.cmegroup.com/trading-hours.html" in sent[-1]
        assert notifier.scan_reminders(schedule, now=now, lead_days=7) == ["roll"]
        assert notifier.scan_status(status) == "delivered"
        assert notifier.scan_reminders(schedule, now=now, lead_days=7) == []
        assert len(sent) == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_windows_dpapi_credentials_are_encrypted_and_round_trip():
    import os
    import pytest
    if os.name != "nt":
        pytest.skip("Windows DPAPI test")
    from src.paper.notification_credentials import load_credentials, save_credentials
    root = _root(); root.mkdir(parents=True)
    path = root / "credentials.json"
    token, chat_id = "123456:synthetic-test-secret", "-100123456"
    try:
        save_credentials(token, chat_id, path)
        stored = path.read_text(encoding="utf-8")
        assert token not in stored and chat_id not in stored
        assert load_credentials(path).token == token
        assert load_credentials(path).chat_id == chat_id
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_telegram_transport_redacts_token_from_transport_errors(monkeypatch):
    import src.paper.notifications as module
    token = "synthetic-private-token"
    def fail(*args, **kwargs):
        raise OSError(f"failed URL includes {token}")
    monkeypatch.setattr(module, "urlopen", fail)
    from src.paper.notifications import TelegramTransport
    try:
        TelegramTransport(token, "123")("MNQ PAPER · TEST")
    except RuntimeError as exc:
        assert token not in str(exc)
    else:
        raise AssertionError("expected redacted transport failure")


def test_telegram_http_failures_are_sanitized_and_classified(monkeypatch):
    import io
    from urllib.error import HTTPError
    import src.paper.notifications as module
    from src.paper.notifications import TelegramDeliveryError, TelegramTransport
    cases = [
        (401, {"ok": False, "description": "Unauthorized"}, "authentication", False, None),
        (400, {"ok": False, "description": "Bad Request: chat not found"}, "chat_id_or_access", False, None),
        (400, {"ok": False, "description": "Bad Request: can't parse entities"}, "message_format", False, None),
        (429, {"ok": False, "parameters": {"retry_after": 17}}, "rate_limited", True, 17.0),
        (503, {"ok": False}, "telegram_server", True, None),
    ]
    for status, payload, expected_category, expected_retryable, expected_after in cases:
        def fail(request, *, _status=status, _payload=payload, **kwargs):
            raise HTTPError(request.full_url, _status, "private details", {},
                            io.BytesIO(json.dumps(_payload).encode()))
        monkeypatch.setattr(module, "urlopen", fail)
        try:
            TelegramTransport("private-token", "private-chat")("MNQ PAPER · TEST")
        except TelegramDeliveryError as exc:
            assert (exc.category, exc.retryable, exc.retry_after_seconds) == (
                expected_category, expected_retryable, expected_after)
            assert "private-token" not in str(exc) and "private-chat" not in str(exc)
        else:
            raise AssertionError("expected TelegramDeliveryError")


def test_permanent_telegram_rejection_blocks_same_pending_event_until_manual_retry():
    from src.paper.notifications import (NotificationPreferences, PersistentEventNotifier,
                                         TelegramDeliveryError, requeue_blocked_delivery)
    root = _root(); root.mkdir(parents=True)
    try:
        events = root / "events.jsonl"
        events.write_text("", encoding="utf-8")
        attempts = []
        def reject(_):
            attempts.append(1)
            raise TelegramDeliveryError("authentication", retryable=False, http_status=401)
        state_path, journal_path = root / "state.json", root / "delivery.jsonl"
        notifier = PersistentEventNotifier(event_path=events, state_path=state_path,
            journal_path=journal_path, send=reject,
            preferences=NotificationPreferences(minimum_interval_seconds=0), clock=lambda: 100)
        _append(events, _event())
        assert notifier.scan_once() == "blocked"
        assert notifier.scan_once() == "blocked"
        assert len(attempts) == 1
        state = json.loads(state_path.read_text(encoding="utf-8"))
        event_id = state["pending"]["event_id"]
        assert state["offset"] == 0 and state["delivery_blocked"] is True
        assert requeue_blocked_delivery(state_path, channel="event") is True
        assert requeue_blocked_delivery(state_path, channel="event") is False
        rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
        assert rows[-1]["state"] == "blocked"
        assert rows[-1]["failure_category"] == "authentication"
        assert rows[-1]["http_status"] == 401
        assert rows[-1]["event_id"] == event_id
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_windows_pid_identity_probe_handles_real_isolated_process():
    import os
    import subprocess
    import sys
    import time
    import pytest
    if os.name != "nt":
        pytest.skip("Windows direct PID identity test")
    from src.paper.notification_cli import _process_probe
    from src.paper.process_identity import PaperProcessIdentityLease
    run = _root() / "isolated-run"
    run.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        from src.paper.process_identity import make_identity_record
        record = make_identity_record(run.name, child.pid)
        (run / "paper_process_identity.json").write_text(json.dumps(record), encoding="utf-8")
        assert _process_probe(run) == [child.pid]
        reused = dict(record, created_filetime="wrong-start-time")
        (run / "paper_process_identity.json").write_text(json.dumps(reused), encoding="utf-8")
        assert _process_probe(run) == []
        (run / "paper_process_identity.json").write_text(json.dumps(record), encoding="utf-8")
        child.terminate(); child.wait(timeout=5)
        assert _process_probe(run) == []
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)
        shutil.rmtree(run.parent, ignore_errors=True)


def test_process_identity_lease_is_removed_on_normal_exit():
    import os
    import pytest
    if os.name != "nt":
        pytest.skip("Windows process identity lease test")
    from src.paper.process_identity import PaperProcessIdentityLease, verify_identity_record
    root = _root(); root.mkdir(parents=True)
    try:
        path = root / "paper_process_identity.json"
        with PaperProcessIdentityLease(path, "isolated-run") as record:
            assert verify_identity_record(record, expected_run_id="isolated-run") == "running"
            assert path.exists()
        assert not path.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shared_rate_limit_is_enforced_across_sidecar_processes():
    from src.paper.notifications import PersistentRateLimiter
    root = _root(); root.mkdir(parents=True)
    try:
        first = PersistentRateLimiter(root / "shared-rate.json")
        second = PersistentRateLimiter(root / "shared-rate.json")
        assert first.reserve(100.0, 3.0)
        assert not second.reserve(102.0, 3.0)
        assert second.reserve(103.0, 3.0)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_telegram_chat_lookup_returns_ids_only_and_redacts_token(monkeypatch):
    import io
    import src.paper.notification_credentials as credentials
    response = io.BytesIO(json.dumps({"ok": True, "result": [{"update_id": 1,
        "message": {"chat": {"id": 987654, "type": "private", "username": "paper_user"},
                    "text": "private message must not be surfaced"}}]}).encode())
    monkeypatch.setattr(credentials, "urlopen", lambda *args, **kwargs: response)
    assert credentials.available_chat_ids("synthetic-token") == [("987654", "paper_user")]
    monkeypatch.setattr(credentials, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("synthetic-token leaked?")))
    try:
        credentials.available_chat_ids("synthetic-token")
    except RuntimeError as exc:
        assert "synthetic-token" not in str(exc)
    else:
        raise AssertionError("expected Telegram lookup failure")
