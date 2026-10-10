from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import pytest

from src.paper import windows_supervisor as supervisor
from src.paper.realtime_checkpoint import AtomicCheckpointStore, runtime_identity
from src.paper.ibkr_paper_recovery import AppendOnlyPaperDeliveryLedger
from src.paper.causal_bootstrap import CausalBootstrapCheckpointStore
from src.paper.notifications import PersistentEventNotifier


@pytest.fixture
def scratch():
    root = Path(__file__).resolve().parents[2] / ".test_scratch"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="windows-supervisor-", dir=root) as directory:
        yield Path(directory)


def _prepared_run(tmp_path, *, matching_delivery=True, engine_state="ERROR"):
    run = tmp_path / "delayed_test_run"
    run.mkdir()
    (run / "paper_writer.lock").write_bytes(b"0\n")
    (run / "delayed_paper_run.json").write_text(json.dumps({
        "mode": "DELAYED_IBKR_PAPER", "paper_only": True, "orders_enabled": False,
        "contract": {"con_id": 815824267, "local_symbol": "MNQZ6", "expiry": "20261218"},
        "calendar_identity": "reviewed-calendar-id",
    }), encoding="utf-8")
    timestamp = datetime(2026, 10, 9, 20, 59, tzinfo=timezone.utc)
    stamp = timestamp.isoformat()
    (run / "status.json").write_text(json.dumps({
        "run_id": "delayed-ibkr-paper", "system": {
            "state": engine_state, "last_bar": stamp,
            "feed_health": {"last_processed_bar": stamp, "latest_available_bar": stamp,
                            "backlog_bars": 0, "mode": "CAUGHT_UP"}},
        "portfolio": {"equity": 50000.0, "gross_realized_pnl": 0.0, "open_positions": []},
    }), encoding="utf-8")
    (run / "events.jsonl").write_text(json.dumps({"event_id": "evt-1", "event_type": "system_started"}) + "\n", encoding="utf-8")
    payload = {
        "system": {"runtime_identity": runtime_identity(), "calendar_identity": "reviewed-calendar-id"},
        "runtime": {"run_id": "delayed-ibkr-paper", "last_bar": {"timestamp": stamp}},
        "context": {"hmm": {"mr": {}, "s2r": {}}},
        "engine": {"account_equity": 50000.0, "realized_pnl": 0.0,
                   "strategies": {name: {} for name in ("MRL1", "MRS2", "S2R", "ORB")},
                   "pending_execution": {},
                   "execution": {"processed_intents": [{"id": "a"}, {"id": "b"}]},
                   "broker": {"orders": [], "fills": [], "positions": {}}},
    }
    checkpoint = run / "paper_checkpoint.json"
    AtomicCheckpointStore(checkpoint).save(payload)
    digest = supervisor._sha256(checkpoint)
    ledger = AppendOnlyPaperDeliveryLedger(
        run / "paper_delivery.jsonl", contract_id=815824267,
        local_symbol="MNQZ6", expiry="20261218", calendar_identity="reviewed-calendar-id",
    )
    ledger.stage({
        "exchange_bar_timestamp_utc": stamp,
        "first_observed_at_utc": stamp,
        "finalized_at_utc": stamp,
        "provider": "fixture", "contract_id": 815824267,
        "local_symbol": "MNQZ6", "expiry": "20261218",
        "data_quality_status": "validated_stable_completed_calendar_covered",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    }, recovered=False)
    ledger.acknowledge_through(stamp, digest if matching_delivery else "0" * 64)
    return run


def test_unsubstantiated_checkpoint_digest_mismatch_blocks_recovery_without_mutation(scratch, monkeypatch):
    run = _prepared_run(scratch, matching_delivery=False)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(supervisor, "probe_tws_contract", lambda *a, **k: {"verified": True})
    report = supervisor.inspect_run(run, check_tws_handshake=True, allow_automatic_resume=True)
    after = {p.name: p.read_bytes() for p in run.iterdir()}
    assert report["safe_to_resume"] is False
    assert any("no equivalent valid same-cursor backup" in item for item in report["blockers"])
    assert report["paper_process_started"] is False
    assert after == before


def test_same_bar_checkpoint_resave_is_reconciled_by_cursor_and_equivalent_state(scratch):
    run = _prepared_run(scratch)
    checkpoint_path = run / "paper_checkpoint.json"
    payload = AtomicCheckpointStore(checkpoint_path).load()
    acknowledged_digest = supervisor._sha256(checkpoint_path)
    payload["system"]["created_at_utc"] = "2026-10-09T21:05:00+00:00"
    # processed_intents is a set in memory; the checkpoint codec serializes it
    # as a list, so a different order is an equivalent persisted state.
    payload["engine"]["execution"]["processed_intents"] = [{"id": "b"}, {"id": "a"}]
    AtomicCheckpointStore(checkpoint_path).save(payload)
    rewritten_digest = supervisor._sha256(checkpoint_path)
    assert rewritten_digest != acknowledged_digest
    assert supervisor._sha256(checkpoint_path.with_suffix(".json.bak")) == acknowledged_digest
    primary = supervisor._load_checkpoint_candidate(checkpoint_path)[0]
    backup = supervisor._load_checkpoint_candidate(checkpoint_path.with_suffix(".json.bak"))[0]
    result = supervisor._delivery_check(
        run / "paper_delivery.jsonl", payload["runtime"]["last_bar"]["timestamp"],
        rewritten_digest,
        {"con_id": 815824267, "local_symbol": "MNQZ6", "expiry": "20261218"},
        "reviewed-calendar-id", checkpoint_path=checkpoint_path,
        checkpoint_payload=primary, backup_payload=backup,
    )
    assert result["valid"] is True, result
    assert result["same_cursor_checkpoint_rewrite"] is True
    assert result["acknowledged_checkpoint_sha256"] == [acknowledged_digest]


def test_same_bar_checkpoint_state_change_is_not_accepted_as_resave(scratch):
    run = _prepared_run(scratch)
    checkpoint_path = run / "paper_checkpoint.json"
    payload = AtomicCheckpointStore(checkpoint_path).load()
    payload["engine"]["account_equity"] += 1.0
    AtomicCheckpointStore(checkpoint_path).save(payload)
    result = supervisor._delivery_check(
        run / "paper_delivery.jsonl", payload["runtime"]["last_bar"]["timestamp"],
        supervisor._sha256(checkpoint_path),
        {"con_id": 815824267, "local_symbol": "MNQZ6", "expiry": "20261218"},
        "reviewed-calendar-id", checkpoint_path=checkpoint_path,
        checkpoint_payload=supervisor._load_checkpoint_candidate(checkpoint_path)[0],
        backup_payload=supervisor._load_checkpoint_candidate(checkpoint_path.with_suffix(".json.bak"))[0],
    )
    assert result["valid"] is False
    assert "equivalent valid same-cursor backup" in result["reason"]


def test_checkpoint_corruption_is_rejected_by_candidate_validator(scratch):
    run = _prepared_run(scratch)
    path = run / "paper_checkpoint.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["engine"]["account_equity"] += 1.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        supervisor._load_checkpoint_candidate(path)


def test_checkpoint_runtime_mismatch_remains_a_blocker(scratch, monkeypatch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    current = runtime_identity()
    current["source_sha256"] = dict(current["source_sha256"])
    current["source_sha256"]["src/strategies/orb/strategy.py"] = "f" * 64
    monkeypatch.setattr(supervisor, "runtime_identity", lambda: current)
    report = supervisor.inspect_run(run, check_tws_handshake=False)
    assert report["gates"]["checkpoint"]["runtime_identity_matches"] is False
    assert "src/strategies/orb/strategy.py" in report["gates"]["checkpoint"]["runtime_identity_differences"]["source_sha256"]
    assert any("runtime fingerprint differs" in item for item in report["blockers"])


def test_monitoring_source_hash_difference_is_reported_not_misclassified_as_environment(scratch, monkeypatch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    current = runtime_identity()
    current["source_sha256"] = dict(current["source_sha256"])
    current["source_sha256"]["src/paper/analytics.py"] = "a" * 64
    monkeypatch.setattr(supervisor, "runtime_identity", lambda: current)
    report = supervisor.inspect_run(run, check_tws_handshake=False)
    differences = report["gates"]["checkpoint"]["runtime_identity_differences"]
    assert differences == {"source_sha256": ["src/paper/analytics.py"]}
    assert report["gates"]["checkpoint"]["runtime_identity_matches"] is False


def test_delivery_cursor_mismatch_blocks_recovery(scratch):
    run = _prepared_run(scratch)
    path = run / "paper_checkpoint.json"
    payload = AtomicCheckpointStore(path).load()
    stamp = "2026-10-09T21:00:00+00:00"
    result = supervisor._delivery_check(
        run / "paper_delivery.jsonl", stamp, supervisor._sha256(path),
        {"con_id": 815824267, "local_symbol": "MNQZ6", "expiry": "20261218"},
        "reviewed-calendar-id", checkpoint_path=path,
        checkpoint_payload=payload, backup_payload=payload,
    )
    assert result["valid"] is False
    assert "cursor and checkpoint market cursor differ" in result["reason"]


def test_duplicate_delivery_acknowledgment_is_rejected(scratch):
    run = _prepared_run(scratch)
    journal = run / "paper_delivery.jsonl"
    first = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(first) + "\n")
    payload = AtomicCheckpointStore(run / "paper_checkpoint.json").load()
    result = supervisor._delivery_check(
        journal, payload["runtime"]["last_bar"]["timestamp"],
        supervisor._sha256(run / "paper_checkpoint.json"),
        {"con_id": 815824267, "local_symbol": "MNQZ6", "expiry": "20261218"},
        "reviewed-calendar-id", checkpoint_payload=payload,
        backup_payload=None,
    )
    assert result["valid"] is False
    assert "duplicate bar acknowledgments" in result["reason"]


def test_duplicate_writer_identity_blocks_recovery(scratch, monkeypatch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    (run / "paper_process_identity.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(supervisor, "verify_identity_record", lambda *a, **k: "running")
    report = supervisor.inspect_run(run, check_tws_handshake=False)
    assert report["gates"]["process_identity"] == "running"
    assert any("process stop is not proven" in item for item in report["blockers"])


def test_bootstrap_artifact_is_verified_without_comparing_lifecycle_fingerprints(scratch):
    run = scratch / "delayed_test_run"
    run.mkdir()
    activation = "2026-10-08T13:03:00Z"
    source_last = "2026-10-08T13:02:00Z"
    bootstrap_runtime = runtime_identity()
    identity = {
        "configuration": {
            "activation_timestamp_utc": activation,
            "runtime_identity": bootstrap_runtime,
            "feature_frame_sha256": "a" * 64,
        },
        "source": {
            "first_timestamp_utc": "2019-05-05T22:03:00Z",
            "last_timestamp_utc": source_last,
            "rows": 100,
            "source_frame_sha256": "b" * 64,
        },
    }
    artifact = run.parent / "bootstrap_fixture" / "checkpoint.json"
    CausalBootstrapCheckpointStore(artifact).save({
        "identity": identity, "rows_processed": 100,
        "last_processed_timestamp_utc": source_last, "context": {},
    })
    (artifact.parent / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    checkpoint_runtime = json.loads(json.dumps(bootstrap_runtime))
    checkpoint_runtime["source_sha256"]["src/paper/analytics.py"] = "c" * 64
    checkpoint_runtime["source_sha256"]["src/paper/delayed_paper_cli.py"] = "d" * 64
    manifest = {
        "activation_timestamp_utc": activation,
        "bootstrap_identity": identity,
        "bootstrap_sha256": supervisor._sha256(artifact),
    }
    result = supervisor._bootstrap_identity_check(manifest, checkpoint_runtime, run)
    assert result["valid"] is True, result
    assert result["artifact_verified"] is True
    assert result["runtime_identity_matches"] is False
    assert result["critical_runtime_matches"] is True

    checkpoint_runtime["source_sha256"]["src/strategies/orb/strategy.py"] = "e" * 64
    unsafe = supervisor._bootstrap_identity_check(manifest, checkpoint_runtime, run)
    assert unsafe["valid"] is False
    assert unsafe["critical_runtime_matches"] is False


def test_even_all_green_preflight_keeps_unapproved_restart_disabled(scratch, monkeypatch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    monkeypatch.setattr(supervisor, "probe_tws_contract", lambda *a, **k: {"verified": True})
    monkeypatch.setattr(supervisor, "_reviewed_calendar", lambda *a, **k: {"valid": True, "version": "fixture"})
    monkeypatch.setattr(supervisor, "_bootstrap_identity_check", lambda *a, **k: {"valid": True})
    report = supervisor.inspect_run(run, check_tws_handshake=True, allow_automatic_resume=False)
    assert report["gates"]["checkpoint"]["valid_internal_checksum"] is True
    assert report["gates"]["delivery_reconciliation"]["valid"] is True
    assert report["gates"]["tws_readonly_handshake"]["verified"] is True
    assert report["safe_to_resume"] is False
    assert "automatic Paper resume is disabled pending explicit authorization" in report["blockers"]


def test_missing_readonly_handshake_is_a_blocker(scratch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    report = supervisor.inspect_run(run, check_tws_handshake=False)
    assert report["gates"]["tws_readonly_handshake"]["state"] == "NOT_CHECKED"
    assert any("handshake was not checked" in item for item in report["blockers"])


def test_active_writer_lock_blocks_resume(scratch, monkeypatch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    monkeypatch.setattr(supervisor, "_writer_lock_available", lambda path: False)
    monkeypatch.setattr(supervisor, "probe_tws_contract", lambda *a, **k: {"verified": True})
    report = supervisor.inspect_run(run, check_tws_handshake=True)
    assert report["gates"]["writer_lock"]["available"] is False
    assert any("writer lock is active" in item for item in report["blockers"])


def test_contract_mismatch_blocks_resume(scratch, monkeypatch):
    run = _prepared_run(scratch, engine_state="RUNNING")
    manifest = json.loads((run / "delayed_paper_run.json").read_text(encoding="utf-8"))
    manifest["contract"]["con_id"] = 123
    (run / "delayed_paper_run.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(supervisor, "probe_tws_contract", lambda *a, **k: {"verified": True})
    report = supervisor.inspect_run(run, check_tws_handshake=True, allow_automatic_resume=True)
    assert report["gates"]["contract_identity"]["valid"] is False
    assert any("approved MNQZ6 mapping" in item for item in report["blockers"])


def test_persisted_engine_error_always_requires_manual_review(scratch, monkeypatch):
    run = _prepared_run(scratch)
    monkeypatch.setattr(supervisor, "probe_tws_contract", lambda *a, **k: {"verified": True})
    report = supervisor.inspect_run(run, check_tws_handshake=True, allow_automatic_resume=True)
    assert any("ERROR/FAILED state requires recovery review" in item for item in report["blockers"])
    assert report["safe_to_resume"] is False


def test_tws_alert_transitions_are_persistent_and_deduplicated(scratch):
    run = scratch / "delayed_test_run"
    run.mkdir()
    first = supervisor.persist_tws_condition(run, active=True, details="API handshake failed", observed="unavailable")
    second = supervisor.persist_tws_condition(run, active=True, details="API handshake failed", observed="unavailable")
    recovered = supervisor.persist_tws_condition(run, active=False, details="API handshake restored", observed="verified")
    assert first["queued_transitions"] == 1
    assert second["queued_transitions"] == 0
    assert recovered["queued_transitions"] == 1
    event_path = run.parent / ".notifications" / run.name / "supervisor_events.jsonl"
    rows = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert [row["payload"]["status"] for row in rows] == ["ACTIVE", "RECOVERED"]
    notifier = PersistentEventNotifier(event_path=event_path,
        state_path=event_path.with_name("supervisor_notifier_state.json"),
        journal_path=event_path.with_name("supervisor_delivery.jsonl"), send=lambda _: None,
        replay_existing=True)
    assert notifier.state["offset"] == 0
