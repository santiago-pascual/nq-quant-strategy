"""Read-only Windows recovery gates for the delayed MNQ Paper service.

This module deliberately does not start/stop Paper or mutate its journals. It
provides a conservative preflight for an operator/supervisor. Automatic Paper
resume remains disabled until separately authorized and all gates pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import socket
import threading
from typing import Any

from src.paper.atomic_io import atomic_replace_with_retry
from src.paper.process_identity import current_process_identity, verify_identity_record
from src.paper.realtime_checkpoint import CHECKPOINT_SCHEMA_VERSION, runtime_identity

EXPECTED_CONTRACT = {"symbol": "MNQ", "local_symbol": "MNQZ6", "con_id": 815824267,
                     "expiry": "20261218", "exchange": "CME"}
ACTIVE_STATES = {"RUNNING", "RECOVERING", "DEGRADED", "PAUSED_REFIT"}
BOOTSTRAP_LINEAGE_NON_EXECUTION_SOURCES = {
    # These are relevant to reporting / bootstrap orchestration, not the
    # strategy/HMM state being carried into an already-running Paper account.
    "src/paper/analytics.py",
    "src/paper/delayed_paper_cli.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _load_checkpoint_candidate(path: Path) -> tuple[dict[str, Any], str]:
    """Validate one checkpoint file with AtomicCheckpointStore's envelope rules.

    AtomicCheckpointStore.load() intentionally falls back to ``.bak``. The
    supervisor needs to know which physical file is valid so it never pairs a
    fallback payload with the hash of a corrupt primary file.
    """
    document = _read_json(path)
    if int(document.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not an object")
    envelope = {"schema_version": document["schema_version"], "payload": payload}
    body = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=True)
    if hashlib.sha256(body.encode()).hexdigest() != document.get("sha256"):
        raise ValueError("checkpoint checksum mismatch")
    return payload, _sha256(path)


def _canonical_checkpoint_payload(payload: dict[str, Any]) -> str:
    """Canonicalize only known non-semantic serialization differences.

    Checkpoint timestamps identify save operations, not Paper state. The
    execution engine stores processed intents as a set; its JSON codec emits
    that set as a list whose order can vary between saves.
    """
    normalized = json.loads(json.dumps(payload, sort_keys=True, allow_nan=True))
    system = normalized.get("system")
    if isinstance(system, dict):
        system.pop("created_at_utc", None)
    processed = normalized.get("engine", {}).get("execution", {}).get("processed_intents")
    if isinstance(processed, list):
        processed.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _checkpoint_payloads_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _canonical_checkpoint_payload(left) == _canonical_checkpoint_payload(right)


def _writer_lock_available(path: Path) -> bool | None:
    """Probe an existing OS lock without creating or modifying its lock file."""
    if not path.exists():
        return True
    try:
        with path.open("r+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    return False
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    return None
                return True
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                return None
            return True
    except OSError:
        return None


def _delivery_check(path: Path, last_bar: str | None, checkpoint_digest: str,
                    expected_contract: dict[str, Any] | None = None,
                    expected_calendar: str | None = None, *,
                    checkpoint_path: Path | None = None,
                    checkpoint_payload: dict[str, Any] | None = None,
                    backup_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(f"delivery row {number} is not an object")
                    rows.append(item)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return {"valid": False, "reason": f"delivery journal unreadable: {type(exc).__name__}"}
    commits = [row for row in rows if row.get("kind") == "paper_commit"]
    staged = [row for row in rows if row.get("kind") == "finalized_pending"]
    if not commits:
        return {"valid": False, "reason": "delivery journal has no acknowledged Paper commits"}
    try:
        acknowledged = max(commits, key=lambda row: int(row["bar_start_epoch_utc"]))
        ack_epoch = int(acknowledged["bar_start_epoch_utc"])
        ack_stamp = datetime.fromtimestamp(ack_epoch, timezone.utc).isoformat()
    except (KeyError, TypeError, ValueError, OSError):
        return {"valid": False, "reason": "delivery journal contains malformed commit identity"}
    if not last_bar:
        return {"valid": False, "reason": "checkpoint has no committed market bar"}
    normalized = last_bar.replace("Z", "+00:00")
    try:
        checkpoint_stamp = datetime.fromisoformat(normalized).astimezone(timezone.utc).isoformat()
    except ValueError:
        return {"valid": False, "reason": "checkpoint committed timestamp is invalid"}
    if ack_stamp != checkpoint_stamp:
        return {"valid": False, "reason": "delivery cursor and checkpoint market cursor differ",
                "delivery_last_bar": ack_stamp, "checkpoint_last_bar": checkpoint_stamp}
    commit_epochs: list[int] = []
    staged_epochs: list[int] = []
    try:
        commit_epochs = [int(row["bar_start_epoch_utc"]) for row in commits]
        staged_epochs = [int(row["bar_start_epoch_utc"]) for row in staged]
    except (KeyError, TypeError, ValueError):
        return {"valid": False, "reason": "delivery journal has malformed bar timestamps"}
    if len(commit_epochs) != len(set(commit_epochs)):
        return {"valid": False, "reason": "delivery journal contains duplicate bar acknowledgments"}
    if len(staged_epochs) != len(set(staged_epochs)):
        return {"valid": False, "reason": "delivery journal contains duplicate staged bar timestamps"}
    try:
        ack_hashes = {str(row.get("checkpoint_sha256", "")).lower()
                      for row in commits if int(row.get("bar_start_epoch_utc", -1)) == ack_epoch}
    except (TypeError, ValueError):
        return {"valid": False, "reason": "delivery journal has malformed bar timestamps"}
    identity = acknowledged.get("ledger_identity") or {}
    if expected_contract and (identity.get("contract_id") != expected_contract.get("con_id")
            or identity.get("local_symbol") != expected_contract.get("local_symbol")
            or str(identity.get("expiry")) != str(expected_contract.get("expiry"))):
        return {"valid": False, "reason": "delivery contract identity differs from run manifest"}
    if expected_calendar and identity.get("calendar_identity") != expected_calendar:
        return {"valid": False, "reason": "delivery calendar identity differs from run manifest"}
    if expected_contract and expected_calendar:
        try:
            from src.paper.ibkr_paper_recovery import AppendOnlyPaperDeliveryLedger
            ledger = AppendOnlyPaperDeliveryLedger(
                path, contract_id=int(expected_contract["con_id"]),
                local_symbol=str(expected_contract["local_symbol"]),
                expiry=str(expected_contract["expiry"]), calendar_identity=expected_calendar,
            )
            committed = set(ledger.committed_timestamps)
            staged_set = set(staged_epochs)
            if committed != set(commit_epochs):
                return {"valid": False, "reason": "delivery ledger commit cursor set is inconsistent"}
            unacknowledged_through_cursor = sorted(
                epoch for epoch in staged_set if epoch <= ack_epoch and epoch not in committed
            )
            if unacknowledged_through_cursor:
                return {"valid": False,
                        "reason": "delivery ledger has staged bars at or before checkpoint cursor without acknowledgments",
                        "unacknowledged_through_cursor": len(unacknowledged_through_cursor)}
            # Use the same rollback rule as the service's authoritative ledger
            # reconciliation: any acknowledgment ahead of the checkpoint is
            # fatal, while an already-acknowledged same cursor is reconciled by
            # timestamp and staged-bar digest, not by a mutable whole-file hash.
            ledger.pending_after(checkpoint_stamp)
        except (KeyError, TypeError, ValueError) as exc:
            return {"valid": False, "reason": f"authoritative delivery ledger validation failed: {str(exc)[:180]}"}

    same_cursor_rewrite = False
    if checkpoint_digest.lower() not in ack_hashes:
        # A service can durably save its unchanged same-bar state again (for
        # example on shutdown). The delivery ledger acknowledges the bar and
        # the service's restore reconciles by cursor. Accept a changed file
        # digest only when both intact checkpoint generations describe the
        # same state, modulo the known save timestamp / set ordering above.
        if (checkpoint_payload is None or backup_payload is None
                or not _checkpoint_payloads_equivalent(checkpoint_payload, backup_payload)):
            return {"valid": False,
                    "reason": "checkpoint digest differs from acknowledgment and no equivalent valid same-cursor backup proves a resave",
                    "delivery_checkpoint_sha256": sorted(ack_hashes),
                    "checkpoint_sha256": checkpoint_digest}
        same_cursor_rewrite = True
    return {"valid": True, "last_bar": checkpoint_stamp, "checkpoint_sha256": checkpoint_digest,
            "acknowledged_rows": len(commits), "ledger_identity": identity,
            "same_cursor_checkpoint_rewrite": same_cursor_rewrite,
            "acknowledged_checkpoint_sha256": sorted(ack_hashes),
            "staged_rows": len(staged), "uncommitted_staged_rows": len(staged_epochs) - len(commit_epochs)}


def _reviewed_calendar(identity: str | None, config_dir: Path) -> dict[str, Any]:
    if not identity:
        return {"valid": False, "reason": "calendar identity is missing"}
    try:
        from src.paper.cme_calendar import CMECalendarSnapshot
        for path in sorted(config_dir.glob("cme_mnq_calendar_*.json")):
            review_path = path.with_name(path.stem + ".review.json")
            try:
                snapshot = CMECalendarSnapshot.from_json(path)
                review = _read_json(review_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
                continue
            if (snapshot.identity == identity and review.get("snapshot_identity") == identity
                    and review.get("product") == "MNQ"
                    and review.get("review_status") == "REVIEWED_PRODUCT_FILTERED_CME_GLOBEX"):
                return {"valid": True, "version": snapshot.version,
                        "coverage_start": str(snapshot.coverage_start),
                        "coverage_end": str(snapshot.coverage_end),
                        "snapshot": str(path), "review_record": str(review_path)}
    except ImportError:
        pass
    return {"valid": False, "reason": "matching reviewed product-specific CME snapshot was not found"}


def _identity_differences(left: Any, right: Any) -> dict[str, Any]:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {"identity": {"saved": left, "current": right}}
    differences: dict[str, Any] = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) == right.get(key):
            continue
        if key == "source_sha256" and isinstance(left.get(key), dict) and isinstance(right.get(key), dict):
            changed = sorted(k for k in set(left[key]) | set(right[key]) if left[key].get(k) != right[key].get(k))
            differences[key] = changed
        else:
            differences[key] = {"saved": left.get(key), "current": right.get(key)}
    return differences


def _bootstrap_identity_check(manifest: dict[str, Any], checkpoint_runtime: Any,
                              run_dir: Path | None = None) -> dict[str, Any]:
    identity = manifest.get("bootstrap_identity")
    digest = str(manifest.get("bootstrap_sha256", ""))
    if not isinstance(identity, dict) or len(digest) != 64:
        return {"valid": False, "reason": "run manifest lacks bootstrap identity or source artifact hash"}
    config = identity.get("configuration") or {}
    source = identity.get("source") or {}
    bootstrap_runtime = config.get("runtime_identity")
    activation = str(manifest.get("activation_timestamp_utc", "")).replace("Z", "+00:00")
    try:
        activation_time = datetime.fromisoformat(activation).astimezone(timezone.utc)
        bootstrap_activation = datetime.fromisoformat(
            str(config["activation_timestamp_utc"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        source_first = datetime.fromisoformat(str(source["first_timestamp_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        source_last = datetime.fromisoformat(str(source["last_timestamp_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        valid = (bootstrap_activation == activation_time
                 and isinstance(bootstrap_runtime, dict)
                 and isinstance(bootstrap_runtime.get("source_sha256"), dict)
                 and source_first < source_last < activation_time
                 and int(source.get("rows", 0)) > 0
                 and len(str(source.get("source_frame_sha256", ""))) == 64
                 and len(str(config.get("feature_frame_sha256", ""))) == 64
                 and all(ch in "0123456789abcdef" for ch in digest.lower())
                 and all(ch in "0123456789abcdef" for ch in str(source.get("source_frame_sha256", "")).lower())
                 and all(ch in "0123456789abcdef" for ch in str(config.get("feature_frame_sha256", "")).lower()))
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    artifact_path: str | None = None
    artifact_verified = False
    artifact_error: str | None = None
    if valid and run_dir is not None:
        # The manifest stores the SHA of the exact bootstrap file supplied at
        # Paper creation. Locate it through the small identity sidecar first;
        # do not deserialize a hundreds-of-megabytes model checkpoint just to
        # compare its historical runtime lineage during a stopped-run audit.
        for sidecar in sorted(run_dir.parent.glob("bootstrap_*/identity.json")):
            try:
                sidecar_value = _read_json(sidecar)
                sidecar_identity = sidecar_value.get("identity", sidecar_value)
                if sidecar_identity != identity:
                    continue
                candidate = sidecar.parent / "checkpoint.json"
                if _sha256(candidate).lower() != digest.lower():
                    continue
                artifact_path = str(candidate)
                artifact_verified = True
                break
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                artifact_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    if run_dir is not None and not artifact_verified:
        valid = False
    from src.paper.runtime_compatibility import bootstrap_lineage_matches
    critical_runtime_matches = bootstrap_lineage_matches(bootstrap_runtime, checkpoint_runtime)
    critical_differences = ({} if critical_runtime_matches else
                            _identity_differences(bootstrap_runtime, checkpoint_runtime))
    valid = valid and critical_runtime_matches
    return {"valid": valid, "bootstrap_sha256": digest,
            "activation_timestamp_utc": manifest.get("activation_timestamp_utc"),
            "source_last_timestamp_utc": source.get("last_timestamp_utc"),
            "source_rows": source.get("rows"),
            "artifact_verified": artifact_verified, "artifact_path": artifact_path,
            "artifact_error": artifact_error,
            "runtime_identity_matches": bootstrap_runtime == checkpoint_runtime,
            "runtime_differences": _identity_differences(bootstrap_runtime, checkpoint_runtime),
            "critical_runtime_matches": critical_runtime_matches,
            "critical_runtime_differences": critical_differences}


def _event_journal_check(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "reason": "Paper event journal is missing"}
    ids: set[str] = set()
    duplicates = 0
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict) or not event.get("event_id") or not event.get("event_type"):
                    return {"valid": False, "reason": f"event journal row {number} lacks identity"}
                rows += 1
                event_id = str(event["event_id"])
                if event_id in ids:
                    duplicates += 1
                ids.add(event_id)
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "reason": f"event journal unreadable ({type(exc).__name__})"}
    return {"valid": duplicates == 0, "rows": rows, "duplicate_event_ids": duplicates,
            "reason": None if duplicates == 0 else "duplicate event identities detected"}


def inspect_run(run_dir: str | Path, *, tws_host: str = "127.0.0.1", tws_port: int = 7497,
                check_tws_handshake: bool = False, allow_automatic_resume: bool = False) -> dict[str, Any]:
    """Read-only recovery assessment. No Paper process is launched or changed."""
    run = Path(run_dir).resolve()
    blockers: list[str] = []
    gates: dict[str, Any] = {"engine_autorestart_enabled": bool(allow_automatic_resume)}
    if not run.is_dir():
        return {"run_dir": str(run), "safe_to_resume": False,
                "blockers": ["run directory is unavailable"], "gates": gates}
    try:
        manifest = _read_json(run / "delayed_paper_run.json")
        gates["run_manifest"] = {"valid": manifest.get("mode") == "DELAYED_IBKR_PAPER"
                                  and manifest.get("paper_only") is True
                                  and manifest.get("orders_enabled") is False}
        contract = manifest.get("contract") or {}
        gates["contract_identity"] = {"valid": all(contract.get(k) == v for k, v in
                                                  (("con_id", EXPECTED_CONTRACT["con_id"]),
                                                   ("local_symbol", EXPECTED_CONTRACT["local_symbol"]),
                                                   ("expiry", EXPECTED_CONTRACT["expiry"]))),
                                      "observed": contract}
        if not gates["run_manifest"]["valid"]:
            blockers.append("run manifest is not a verified Paper-only delayed-IBKR run")
        if not gates["contract_identity"]["valid"]:
            blockers.append("run manifest contract does not match approved MNQZ6 mapping")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = {}
        blockers.append(f"run manifest unavailable or invalid ({type(exc).__name__})")
        gates["run_manifest"] = {"valid": False}

    try:
        status = _read_json(run / "status.json")
        state = str((status.get("system") or {}).get("state", "UNKNOWN")).upper()
        gates["persisted_engine_state"] = state
        if state not in ACTIVE_STATES | {"ERROR", "FAILED", "STOPPED"}:
            blockers.append("persisted Engine state is unknown")
        if state in {"ERROR", "FAILED"}:
            blockers.append("persisted Engine ERROR/FAILED state requires recovery review")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status = {}
        state = "UNKNOWN"
        gates["persisted_engine_state"] = state
        blockers.append(f"Paper status unavailable ({type(exc).__name__})")

    checkpoint_path = run / "paper_checkpoint.json"
    backup_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak")
    checkpoint_digest = None
    payload: dict[str, Any] = {}
    backup_payload: dict[str, Any] | None = None
    try:
        primary_payload: dict[str, Any] | None = None
        primary_digest: str | None = None
        primary_error: str | None = None
        try:
            primary_payload, primary_digest = _load_checkpoint_candidate(checkpoint_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            primary_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        try:
            backup_payload, backup_digest = _load_checkpoint_candidate(backup_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            backup_payload = None
            backup_digest = None
        # Match AtomicCheckpointStore.load()'s schema/envelope/checksum rules
        # while retaining which physical generation was selected. The ledger
        # below is validated by its authoritative AppendOnlyPaperDeliveryLedger.
        if primary_payload is not None:
            payload, checkpoint_digest = primary_payload, primary_digest
        elif backup_payload is not None:
            payload, checkpoint_digest = backup_payload, backup_digest
        else:
            raise ValueError(primary_error or "no valid primary or backup checkpoint")
        runtime = payload.get("runtime") or {}
        system = payload.get("system") or {}
        checkpoint_run = runtime.get("run_id")
        last_bar = (runtime.get("last_bar") or {}).get("timestamp")
        current_identity = runtime_identity()
        saved_identity = system.get("runtime_identity")
        from src.paper.runtime_compatibility import validate_runtime_identity
        identity_compatibility = validate_runtime_identity(saved_identity, current_identity)
        identity_matches = identity_compatibility.accepted
        gates["checkpoint"] = {"valid_internal_checksum": True,
                                "schema_version": 1, "sha256": checkpoint_digest,
                                "last_bar": last_bar, "run_id": checkpoint_run,
                                "runtime_identity_matches": identity_matches,
                                "runtime_compatibility": identity_compatibility.as_dict(),
                                "runtime_identity_differences": _identity_differences(saved_identity, current_identity),
                                "loaded_from": "primary" if primary_payload is not None else "backup",
                                "primary_valid": primary_payload is not None,
                                "backup_valid": backup_payload is not None}
        if checkpoint_run != str(status.get("run_id")):
            blockers.append("checkpoint run identity differs from Paper status")
        if not identity_matches:
            blockers.append("checkpoint runtime fingerprint differs from current runtime")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        last_bar = None
        blockers.append(f"checkpoint failed integrity/schema validation ({type(exc).__name__})")
        gates["checkpoint"] = {"valid_internal_checksum": False,
                                "reason": str(exc)[:240]}

    if checkpoint_digest:
        expected_calendar = (payload.get("system") or {}).get("calendar_identity")
        gates["delivery_reconciliation"] = _delivery_check(
            run / "paper_delivery.jsonl", last_bar, checkpoint_digest,
            manifest.get("contract") if manifest else None, expected_calendar,
            checkpoint_path=checkpoint_path, checkpoint_payload=primary_payload,
            backup_payload=backup_payload)
        if not gates["delivery_reconciliation"].get("valid"):
            blockers.append(str(gates["delivery_reconciliation"].get("reason", "delivery reconciliation failed")))
    else:
        gates["delivery_reconciliation"] = {"valid": False, "reason": "checkpoint unavailable"}

    event_check = _event_journal_check(run / "events.jsonl")
    gates["event_journal"] = event_check
    if not event_check.get("valid"):
        blockers.append(str(event_check.get("reason", "Paper event journal validation failed")))

    try:
        status_stamp = datetime.fromisoformat(str((status.get("system") or {}).get("last_bar")).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        checkpoint_stamp = datetime.fromisoformat(str(last_bar).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        feed = (status.get("system") or {}).get("feed_health") or {}
        feed_processed = datetime.fromisoformat(str(feed.get("last_processed_bar")).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        frontier = datetime.fromisoformat(str(feed.get("latest_available_bar")).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        cursors_match = status_stamp == checkpoint_stamp == feed_processed and frontier >= checkpoint_stamp
        gates["market_cursors"] = {"match": cursors_match, "checkpoint": checkpoint_stamp,
                                   "status": status_stamp, "paper_cursor": feed_processed,
                                   "provider_frontier": frontier,
                                   "backlog_bars": feed.get("backlog_bars"),
                                   "acquisition_mode": feed.get("mode")}
        if not cursors_match:
            blockers.append("checkpoint, status, and Paper cursor do not reconcile with provider frontier")
    except (TypeError, ValueError):
        gates["market_cursors"] = {"match": False, "available": False}
        blockers.append("market-data cursor/frontier telemetry is incomplete or invalid")

    lock_state = _writer_lock_available(run / "paper_writer.lock")
    gates["writer_lock"] = {"available": lock_state}
    if lock_state is not True:
        blockers.append("Paper writer lock is active or cannot be inspected")

    identity_file = run / "paper_process_identity.json"
    if identity_file.exists():
        try:
            record = _read_json(identity_file)
            process_state = verify_identity_record(record, expected_run_id=run.name)
        except (OSError, ValueError, json.JSONDecodeError):
            process_state = "unavailable"
        gates["process_identity"] = process_state
        if process_state != "absent":
            blockers.append(f"Paper PID identity is {process_state}; process stop is not proven")
    else:
        gates["process_identity"] = "no active PID lease"

    if check_tws_handshake:
        gates["tws_readonly_handshake"] = probe_tws_contract(tws_host, tws_port)
        if not gates["tws_readonly_handshake"].get("verified"):
            blockers.append("read-only TWS handshake/approved contract verification failed")
    else:
        gates["tws_readonly_handshake"] = {"verified": False, "state": "NOT_CHECKED"}
        blockers.append("read-only TWS handshake was not checked")

    calendar_identity = (payload.get("system") or {}).get("calendar_identity")
    calendar_review = _reviewed_calendar(calendar_identity, Path(__file__).resolve().parent / "config")
    gates["calendar_identity"] = {"identity": calendar_identity, **calendar_review,
                                   "matches_run_manifest": bool(manifest.get("calendar_identity") == calendar_identity)}
    if not calendar_review.get("valid") or manifest.get("calendar_identity") != calendar_identity:
        blockers.append("checkpoint/run identity does not match a currently reviewed CME MNQ calendar snapshot")
    bootstrap_check = _bootstrap_identity_check(
        manifest, (payload.get("system") or {}).get("runtime_identity"), run)
    gates["bootstrap_identity"] = bootstrap_check
    if not bootstrap_check.get("valid"):
        blockers.append("embedded bootstrap identity is invalid or differs from the runtime checkpoint")
    engine_state = payload.get("engine") or {}
    broker_state = engine_state.get("broker") or {}
    engine_positions = broker_state.get("positions")
    status_portfolio = status.get("portfolio") or {}
    try:
        equity_delta = abs(float(engine_state["account_equity"]) - float(status_portfolio["equity"]))
        realized_delta = abs(float(engine_state["realized_pnl"]) - float(status_portfolio["gross_realized_pnl"]))
        status_positions = status_portfolio["open_positions"]
        position_count_match = len(engine_positions) == len(status_positions)
        account_match = equity_delta <= 0.02 and realized_delta <= 0.02 and position_count_match
        gates["account_reconciliation"] = {"available": True, "matches_status": account_match,
                                            "equity_delta": equity_delta,
                                            "gross_realized_pnl_delta": realized_delta,
                                            "position_count_checkpoint": len(engine_positions),
                                            "position_count_status": len(status_positions)}
        if not account_match:
            blockers.append("checkpoint account/equity/position summary differs from persisted status")
    except (KeyError, TypeError, ValueError):
        gates["account_reconciliation"] = {"available": False, "matches_status": False}
        blockers.append("checkpoint and status do not expose comparable account/position values")
    strategy_keys = set((engine_state.get("strategies") or {}).keys())
    gates["strategy_state"] = {"available": bool(strategy_keys), "strategies": sorted(strategy_keys),
                                "complete_four_strategy_state": strategy_keys == {"MRL1", "MRS2", "S2R", "ORB"}}
    if not gates["strategy_state"]["complete_four_strategy_state"]:
        blockers.append("checkpoint does not contain all four strategy states")
    gates["causal_context_present"] = bool(payload.get("context"))
    if not gates["causal_context_present"]:
        blockers.append("checkpoint has no causal feature/HMM context")
    gates["pending_execution_state_present"] = "pending_execution" in engine_state
    if not gates["pending_execution_state_present"]:
        blockers.append("checkpoint has no unsettled simulated-order lifecycle state")

    # This policy is deliberately hard-off. Passing preflight is necessary but
    # never sufficient to activate unattended automatic restarts.
    if not allow_automatic_resume:
        blockers.append("automatic Paper resume is disabled pending explicit authorization")
    return {"run_dir": str(run), "mode": "PAPER_ONLY", "paper_process_started": False,
            "safe_to_resume": not blockers, "blockers": blockers, "gates": gates,
            "observed_at_utc": datetime.now(timezone.utc).isoformat()}


def probe_tws_contract(host: str, port: int, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """One bounded, read-only IB API handshake and exact contract lookup."""
    try:
        with socket.create_connection((host, int(port)), timeout=min(2.0, timeout_seconds)):
            pass
    except OSError as exc:
        return {"verified": False, "socket_reachable": False, "error": type(exc).__name__}
    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError:
        return {"verified": False, "socket_reachable": True, "error": "ibapi unavailable"}

    class Probe(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.ready = threading.Event()
            self.done = threading.Event()
            self.matches: list[dict[str, Any]] = []
            self.errors: list[int] = []

        def nextValidId(self, orderId):  # noqa: N802
            self.ready.set()

        def contractDetails(self, reqId, details):  # noqa: N802
            c = details.contract
            self.matches.append({"con_id": int(c.conId), "symbol": str(c.symbol),
                                 "local_symbol": str(c.localSymbol),
                                 "expiry": str(c.lastTradeDateOrContractMonth),
                                 "exchange": str(c.exchange)})

        def contractDetailsEnd(self, reqId):  # noqa: N802
            self.done.set()

        def error(self, reqId, code, message, *args):
            self.errors.append(int(code))
            if int(reqId) == 1:
                self.done.set()

    app = Probe()
    thread = None
    try:
        app.connect(host, int(port), 180 + secrets.randbelow(800))
        thread = threading.Thread(target=app.run, name="mnq-supervisor-readonly", daemon=True)
        thread.start()
        if not app.ready.wait(timeout_seconds):
            return {"verified": False, "socket_reachable": True, "error": "IBKR handshake timeout"}
        query = Contract()
        query.conId = EXPECTED_CONTRACT["con_id"]
        query.symbol, query.secType, query.exchange, query.currency = "MNQ", "FUT", "CME", "USD"
        query.lastTradeDateOrContractMonth = EXPECTED_CONTRACT["expiry"]
        app.reqContractDetails(1, query)
        if not app.done.wait(timeout_seconds):
            return {"verified": False, "socket_reachable": True, "error": "contract lookup timeout"}
        match = any(row["con_id"] == EXPECTED_CONTRACT["con_id"]
                    and row["symbol"] == EXPECTED_CONTRACT["symbol"]
                    and row["local_symbol"] == EXPECTED_CONTRACT["local_symbol"]
                    and row["expiry"].startswith(EXPECTED_CONTRACT["expiry"])
                    and row["exchange"] in {"CME", "GLOBEX"}
                    for row in app.matches)
        return {"verified": match, "socket_reachable": True, "contract_match": match,
                "matches": app.matches[:4], "error_codes": app.errors[:8],
                "read_only": True, "orders_submitted": False}
    except Exception as exc:
        return {"verified": False, "socket_reachable": True, "error": type(exc).__name__}
    finally:
        try:
            app.disconnect()
        except Exception:
            pass


def persist_tws_condition(run_dir: str | Path, *, active: bool, details: str,
                          observed: Any = None) -> dict[str, Any]:
    """Queue a deduplicated supervisor alert in the notifier's durable sidecar journal."""
    run = Path(run_dir).resolve()
    return persist_supervisor_condition(run, condition="tws_api", title="TWS API unavailable / manual authentication may be required",
                                        active=active, details=details, observed=observed, severity="CRITICAL")


def persist_supervisor_condition(run_dir: str | Path, *, condition: str, title: str,
                                 active: bool, details: str, observed: Any = None,
                                 severity: str = "WARNING") -> dict[str, Any]:
    """Persist a deduplicated non-trading incident in a supervisor-owned journal."""
    from src.paper.monitoring_alerts import make_alert
    run = Path(run_dir).resolve()
    runtime = run.parent / ".notifications" / run.name
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "supervisor_conditions.json"
    event_path = runtime / "supervisor_events.jsonl"
    lock_path = runtime / "supervisor_conditions.lock"
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
            state = _read_json(state_path) if state_path.exists() else {
                "schema_version": 1, "active": {}, "generations": {}}
            if state.get("schema_version") != 1 or not isinstance(state.get("active"), dict):
                raise ValueError("supervisor condition state is incompatible")
            key = str(condition)
            previous = state["active"].get(key)
            if active and previous is not None:
                return {"queued_transitions": 0, "active": True, "journal": str(event_path)}
            if not active and previous is None:
                return {"queued_transitions": 0, "active": False, "journal": str(event_path)}
            generation = int(state.setdefault("generations", {}).get(key, 0)) + (1 if active else 0)
            now = datetime.now(timezone.utc).isoformat()
            if active:
                alert = make_alert(alert_type=title, severity=severity,
                    expected="component healthy", observed=observed, threshold=None,
                    details=details, strategy="SYSTEM", run_id=run.name)
                alert.update(alert_id=f"windows-supervisor-{key}", status="ACTIVE", timestamp_utc=now)
                alert["event_id"] = hashlib.sha256(f"windows-supervisor|{key}|ACTIVE|{generation}".encode()).hexdigest()
                state["active"][key] = {"alert": alert, "generation": generation}
                state["generations"][key] = generation
            else:
                alert = dict(previous["alert"])
                alert.update(severity="INFO", status="RECOVERED", timestamp_utc=now,
                             details=details, observed=observed)
                alert["event_id"] = hashlib.sha256(f"windows-supervisor|{key}|RECOVERED|{int(previous['generation'])}".encode()).hexdigest()
                del state["active"][key]
            envelope = {"event_id": alert["event_id"], "event_type": "monitoring_alert",
                        "timestamp": alert["timestamp_utc"], "payload": alert}
            line = (json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode("utf-8")
            descriptor = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, line); os.fsync(descriptor)
            finally:
                os.close(descriptor)
            fd, temporary = tempfile.mkstemp(prefix=state_path.name + ".", suffix=".tmp", dir=runtime)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    json.dump(state, output, sort_keys=True, separators=(",", ":")); output.flush(); os.fsync(output.fileno())
                atomic_replace_with_retry(temporary, state_path)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
            return {"queued_transitions": 1, "active": active, "journal": str(event_path)}
        finally:
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only MNQ Windows Paper recovery preflight")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--tws-host", default="127.0.0.1")
    parser.add_argument("--tws-port", default=7497, type=int)
    parser.add_argument("--check-tws", action="store_true", help="Perform one bounded read-only API handshake")
    parser.add_argument("--tws-probe-only", action="store_true",
                        help="Perform one bounded read-only handshake/contract lookup only")
    parser.add_argument("--record-tws-condition", choices=("active", "recovered"))
    parser.add_argument("--record-condition", choices=("active", "recovered"))
    parser.add_argument("--condition", default="tws_api")
    parser.add_argument("--title", default="Windows Paper supervisor incident")
    parser.add_argument("--severity", choices=("WARNING", "CRITICAL"), default="WARNING")
    parser.add_argument("--allow-automatic-resume", action="store_true",
                        help="Explicit policy input; does not start Paper and still requires every recovery gate")
    parser.add_argument("--details", default="TWS API state unavailable")
    parser.add_argument("--observed", default="unavailable")
    args = parser.parse_args(argv)
    if args.record_tws_condition:
        if args.run_dir is None:
            parser.error("--record-tws-condition requires --run-dir")
        result = persist_tws_condition(args.run_dir,
            active=args.record_tws_condition == "active", details=args.details,
            observed=args.observed)
        print(json.dumps(result, indent=2))
        return 0
    if args.record_condition:
        if args.run_dir is None:
            parser.error("--record-condition requires --run-dir")
        result = persist_supervisor_condition(args.run_dir, condition=args.condition, title=args.title,
            active=args.record_condition == "active", details=args.details, observed=args.observed,
            severity=args.severity)
        print(json.dumps(result, indent=2))
        return 0
    if args.tws_probe_only:
        result = probe_tws_contract(args.tws_host, args.tws_port)
        print(json.dumps(result, indent=2))
        return 0 if result.get("verified") else 2
    if args.run_dir is None:
        parser.error("--run-dir is required unless --tws-probe-only is used")
    report = inspect_run(args.run_dir, tws_host=args.tws_host, tws_port=args.tws_port,
                         check_tws_handshake=args.check_tws,
                         allow_automatic_resume=args.allow_automatic_resume)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["safe_to_resume"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
