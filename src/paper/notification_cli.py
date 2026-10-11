"""Run or inspect the read-only Paper event notification sidecar."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager

from src.paper.atomic_io import atomic_replace_with_retry
from src.paper.notifications import (NotificationPreferences, PaperProcessWatchdog,
                                     PersistentEventNotifier, PersistentRateLimiter,
                                     TelegramTransport, recent_failures)
from src.paper.notification_credentials import configure_interactively, credentials_path, load_credentials
from src.paper.monitoring_alerts import AlertStateStore, PaperAlertMonitor


ROOT = Path(__file__).resolve().parents[2]


def _log_sidecar_failure(exc: Exception) -> None:
    """Useful stack locations without exception messages, URLs or credentials."""
    frames = [{'file': Path(frame.filename).name, 'line': frame.lineno,
               'function': frame.name} for frame in traceback.extract_tb(exc.__traceback__)]
    print(json.dumps({'component':'paper_notification_service',
        'error_type':type(exc).__name__, 'errno':getattr(exc,'errno',None),
        'winerror':getattr(exc,'winerror',None), 'frames':frames},sort_keys=True),file=sys.stderr)
DEFAULT_RUN = "results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3"


def _paths(run: Path) -> tuple[Path, Path, Path]:
    runtime = ROOT / "results" / "paper" / ".notifications" / run.name
    return run / "events.jsonl", runtime / "state.json", runtime / "delivery.jsonl"


def _rate_limiter(run: Path) -> PersistentRateLimiter:
    return PersistentRateLimiter(ROOT / "results" / "paper" / ".notifications" / run.name / "telegram_rate_limit.json")


def _alert_paths(run: Path) -> tuple[Path, Path, Path]:
    runtime = ROOT / "results" / "paper" / ".notifications" / run.name
    alert_events = runtime / "alert_events.jsonl"
    alert_state = runtime / "alert_state.json"
    delivery = runtime / "alert_delivery.jsonl"
    return alert_events, alert_state, delivery


def _write_heartbeat(path: Path, *, component: str, run: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "component": component, "pid": os.getpid(),
               "run_id": run.name, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush(); os.fsync(handle.fileno())
        atomic_replace_with_retry(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _alert_notifier(run: Path) -> PersistentEventNotifier:
    credentials = load_credentials()
    event_path, state_path, journal_path = _alert_paths(run)
    delivery_state = state_path.with_name("alert_notifier_state.json")
    return PersistentEventNotifier(event_path=event_path, state_path=delivery_state,
        journal_path=journal_path, send=TelegramTransport(credentials.token, credentials.chat_id),
        preferences=NotificationPreferences.from_environment(), rate_limiter=_rate_limiter(run))


def _supervisor_notifier(run: Path) -> PersistentEventNotifier:
    """Deliver durable Windows supervisor incidents, including events queued offline."""
    credentials = load_credentials()
    runtime = ROOT / "results" / "paper" / ".notifications" / run.name
    return PersistentEventNotifier(
        event_path=runtime / "supervisor_events.jsonl",
        state_path=runtime / "supervisor_notifier_state.json",
        journal_path=runtime / "supervisor_delivery.jsonl",
        send=TelegramTransport(credentials.token, credentials.chat_id),
        preferences=NotificationPreferences.from_environment(),
        rate_limiter=_rate_limiter(run), replay_existing=True)


def _drain_quant_reports(run: Path) -> str:
    """Retry the report spool with fresh state under its shared OS lock.

    The reporting command can enqueue/send independently; fresh construction
    prevents a long-running notifier from using a stale delivery cursor.
    """
    from src.paper.single_writer import PaperWriterLock, WriterAlreadyActive
    directory = ROOT / 'results' / 'diagnostics' / 'quant_reports' / run.name
    events = directory / 'report_events.jsonl'
    if not events.is_file():
        return 'unavailable'
    try:
        with PaperWriterLock(directory / 'report_delivery.lock'):
            credentials = load_credentials()
            notifier = PersistentEventNotifier(event_path=events,
                state_path=directory / 'report_delivery_state.json',
                journal_path=directory / 'report_delivery.jsonl', replay_existing=True,
                preferences=NotificationPreferences.from_environment(),
                send=TelegramTransport(credentials.token, credentials.chat_id), rate_limiter=_rate_limiter(run))
            return notifier.scan_once()
    except WriterAlreadyActive:
        return 'writer_busy'


def _reviewed_calendar_for_run(run: Path, *, config_dir: Path | None = None):
    """Load only a local MNQ snapshot whose identity matches this run's record."""
    try:
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        identity = (((status.get("system") or {}).get("cme_calendar") or {}).get("identity"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not identity:
        return None
    from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
    config_root = config_dir or (ROOT / "src" / "paper" / "config")
    for path in sorted(config_root.glob("cme_mnq_calendar_*.json")):
        try:
            snapshot = CMECalendarSnapshot.from_json(path)
            review_path = path.with_name(path.stem + ".review.json")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if (snapshot.identity == identity
                    and review.get("snapshot_identity") == identity
                    and review.get("product") == "MNQ"
                    and review.get("review_status") == "REVIEWED_PRODUCT_FILTERED_CME_GLOBEX"):
                return CMETradingCalendar(snapshot)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            continue
    return None


def _default_maintenance_schedule(run: Path) -> Path | None:
    calendar = _reviewed_calendar_for_run(run)
    if calendar is None:
        return None
    path = ROOT / 'src/paper/config/paper_monitoring_maintenance.json'
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        events = document['events']
        if (document.get('schema_version') != 1 or not events or
                any(item.get('snapshot_identity') != calendar.snapshot.identity for item in events)):
            return None
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return None
    return path


def _notifier(run: Path) -> PersistentEventNotifier:
    credentials = load_credentials()
    event_path, state_path, journal_path = _paths(run)
    return PersistentEventNotifier(event_path=event_path, state_path=state_path,
                                  journal_path=journal_path, send=TelegramTransport(credentials.token, credentials.chat_id),
                                  preferences=NotificationPreferences.from_environment(),
                                  rate_limiter=_rate_limiter(run),
                                  exclude_event_types=PaperAlertMonitor.STRUCTURED_EVENT_TYPES)


def _watchdog_notifier(run: Path, engine_pid: int | None = None) -> tuple[PersistentEventNotifier, PaperProcessWatchdog, Path]:
    credentials = load_credentials()
    runtime = ROOT / "results" / "paper" / ".notifications" / run.name
    events = runtime / "watchdog_events.jsonl"
    state = runtime / "watchdog_notifier_state.json"
    journal = runtime / "watchdog_delivery.jsonl"
    notifier = PersistentEventNotifier(event_path=events, state_path=state, journal_path=journal,
        send=TelegramTransport(credentials.token, credentials.chat_id),
        preferences=NotificationPreferences.from_environment(), rate_limiter=_rate_limiter(run))
    watchdog_state = runtime / "engine_watchdog_state.json"
    watchdog = PaperProcessWatchdog(run_dir=run, state_path=watchdog_state, event_path=events,
                                    process_probe=lambda: _process_probe(run, engine_pid),
                                    notifier_heartbeat_path=runtime / "notifier_heartbeat.json")
    return notifier, watchdog, watchdog_state


@contextmanager
def _process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
    except OSError:
        handle.close()
        raise RuntimeError("notification sidecar already holds its single-instance lock") from None
    try:
        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _process_probe(run: Path, engine_pid: int | None = None):
    """Probe the per-run PID identity lease before falling back to CIM.

    The lease is checked against OS PID, process creation time and executable;
    this remains reliable when Windows denies global Win32_Process queries.
    """
    if os.name != "nt":
        return None
    identity_path = run / "paper_process_identity.json"
    if identity_path.is_file():
        from src.paper.process_identity import verify_identity_record
        try:
            record = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        result = verify_identity_record(record, expected_run_id=run.name)
        if result == "running":
            return [int(record["pid"])]
        if result in {"absent", "mismatch"}:
            return []
        return None
    if engine_pid is not None:
        if engine_pid <= 0:
            raise ValueError("engine PID must be positive")
        from src.paper.process_identity import current_process_identity
        identity = current_process_identity(engine_pid)
        if identity is False:
            return []
        if identity is None:
            return None
        if Path(str(identity.get("executable", ""))).name.lower() in {"python.exe", "pythonw.exe"}:
            return [int(engine_pid)]
        return []
    else:
        target = run.name.replace("'", "''").lower()
        script = ("$all=@(Get-CimInstance Win32_Process -ErrorAction Stop); "
                  "if($all.Count -eq 0){ Write-Output 'MNQ_PROBE_UNAVAILABLE'; exit 3 }; "
                  f"$needle='{target}'; $rows=$all | Where-Object "
                  "{ $_.CommandLine -and $_.CommandLine.ToLowerInvariant().Contains($needle) "
                  "-and $_.CommandLine -match 'run_realtime_paper|delayed_paper_cli|ibkr_paper_runner|run_autonomous' "
                  "-and $_.CommandLine -notmatch 'notification_cli' }; "
                  "$rows | ForEach-Object { $_.ProcessId }")
    try:
        result = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=10, check=False)
        if result.returncode != 0:
            return None
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only MNQ Paper notifications")
    parser.add_argument("command", choices=("configure", "run", "watchdog", "test", "status", "monitor-once", "acknowledge", "retry-pending"))
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--reminder-schedule", help="Optional reviewed JSON with CME exception/roll reminder dates")
    parser.add_argument("--engine-pid", type=int, help="Explicit Paper Python PID when automatic Windows process inspection is unavailable")
    parser.add_argument("--alert-id", help="Alert ID for acknowledge command")
    parser.add_argument("--resolve", action="store_true", help="Resolve rather than acknowledge the alert")
    parser.add_argument("--channel", choices=("event", "status"), default="event",
                        help="Pending notification channel to unblock after fixing its configuration")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    if not run.is_absolute(): run = ROOT / run
    run = run.resolve()
    if args.command == "configure":
        try:
            saved = configure_interactively()
        except Exception as exc:
            print(f"Credential setup failed ({type(exc).__name__}); values were not logged", file=sys.stderr)
            return 1
        print(f"Telegram credentials encrypted for this Windows user at: {saved}")
        return 0
    if args.command == "status":
        event_path, state_path, journal_path = _paths(run)
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        runtime = state_path.parent
        alert_events, alert_state_path, alert_delivery = _alert_paths(run)
        alert_state = json.loads(alert_state_path.read_text(encoding="utf-8")) if alert_state_path.exists() else None
        watchdog_state_path = runtime / "engine_watchdog_state.json"
        watchdog_state = json.loads(watchdog_state_path.read_text(encoding="utf-8")) if watchdog_state_path.exists() else None
        print(json.dumps({"run": run.name, "state": state, "watchdog": watchdog_state,
                          "alerts": alert_state, "alert_event_log": str(alert_events),
                          "alert_delivery_log": str(alert_delivery),
                          "recent_failures": recent_failures(journal_path),
                          "recent_alert_failures": recent_failures(alert_delivery),
                          "credential_store_configured": credentials_path().is_file()}, indent=2))
        return 0
    if args.command == "retry-pending":
        from src.paper.notifications import requeue_blocked_delivery
        runtime = ROOT / "results" / "paper" / ".notifications" / run.name
        state_path = runtime / ("alert_notifier_state.json" if args.channel == "event" else "state.json")
        try:
            changed = requeue_blocked_delivery(state_path, channel=args.channel)
        except (OSError, ValueError, json.JSONDecodeError):
            print("Could not update pending notification state; no event or Paper history was changed.", file=sys.stderr)
            return 1
        print("Pending notification unblocked without changing its event ID." if changed
              else "No permanently blocked pending notification found.")
        return 0 if changed else 2
    if args.command == "test":
        try:
            notifier = _notifier(run)
            notifier.test_message()
        except Exception as exc:
            category = getattr(exc, "category", type(exc).__name__)
            status = getattr(exc, "http_status", None)
            suffix = f"; HTTP {status}" if status is not None else ""
            print(f"Telegram TEST failed: {category}{suffix}; credentials and response details redacted", file=sys.stderr)
            return 1
        print("Clearly labeled Telegram test message delivered; no order was placed.")
        return 0
    if args.command == "monitor-once":
        try:
            result = PaperAlertMonitor(run_dir=run, state_path=_alert_paths(run)[1], events_path=_alert_paths(run)[0],
                                       calendar=_reviewed_calendar_for_run(run)).poll_once()
            print(json.dumps({"run": run.name, "emitted_alert_transitions": len(result["emitted"]),
                              "active_alerts": len(result["state"].get("active", {})),
                              "capabilities": result["state"].get("capabilities")}, indent=2))
            return 0
        except Exception as exc:
            print(f"Alert monitor failed ({type(exc).__name__}); no Paper state was changed", file=sys.stderr)
            return 1
    if args.command == "acknowledge":
        if not args.alert_id:
            parser.error("acknowledge requires --alert-id")
        alert_events, alert_state_path, _ = _alert_paths(run)
        try:
            store = AlertStateStore(alert_state_path, alert_events)
            changed = store.acknowledge(args.alert_id, resolved=args.resolve)
            if not changed:
                print("No matching active alert. Paper state was not changed.", file=sys.stderr)
                return 2
            print("Alert resolved." if args.resolve else "Alert acknowledged.")
            return 0
        except Exception as exc:
            print(f"Alert state update failed ({type(exc).__name__})", file=sys.stderr)
            return 1
    runtime = ROOT / "results" / "paper" / ".notifications" / run.name
    try:
        if args.command == "run":
            _, state_path, _ = _paths(run)
            with _process_lock(state_path.with_name("notifier.lock")):
                print(f"Read-only Paper event notifier started for {run.name}; Paper Engine is not controlled.")
                notifier = _notifier(run)
                alert_notifier = _alert_notifier(run)
                supervisor_notifier = _supervisor_notifier(run)
                monitor = PaperAlertMonitor(run_dir=run, state_path=_alert_paths(run)[1], events_path=_alert_paths(run)[0],
                                            calendar=_reviewed_calendar_for_run(run))
                runtime = ROOT / "results" / "paper" / ".notifications" / run.name
                heartbeat = runtime / "notifier_heartbeat.json"
                stop_path = state_path.with_name("stop.request")
                next_report_delivery = 0.0
                reminder_schedule = args.reminder_schedule or _default_maintenance_schedule(run)
                try:
                    while not stop_path.exists():
                        notifier.scan_once()
                        notifier.scan_status(run / "status.json", monitor_incidents=False)
                        try:
                            monitor.poll_once()
                        except Exception as exc:
                            # Sidecar errors are logged locally and never touch Paper execution.
                            print(f"Alert evaluation unavailable ({type(exc).__name__}); persisted state retained", file=sys.stderr)
                        alert_notifier.scan_once()
                        supervisor_notifier.scan_once()
                        if time.monotonic() >= next_report_delivery:
                            try:
                                _drain_quant_reports(run)
                            except Exception as exc:
                                print(f"Report delivery unavailable ({type(exc).__name__}); spool retained", file=sys.stderr)
                            next_report_delivery = time.monotonic() + 30.0
                        _write_heartbeat(heartbeat, component="paper_notification_service", run=run)
                        if reminder_schedule:
                            notifier.scan_reminders(reminder_schedule)
                        time.sleep(2.0)
                finally:
                    stop_path.unlink(missing_ok=True)
        else:
            notifier, watchdog, watchdog_state_path = _watchdog_notifier(run, args.engine_pid)
            with _process_lock(watchdog_state_path.with_name("watchdog.lock")):
                print(f"Independent read-only Paper process watchdog started for {run.name}.")
                stop_path = watchdog_state_path.with_name("watchdog.stop.request")
                try:
                    next_probe = 0.0
                    while not stop_path.exists():
                        now = time.monotonic()
                        if now >= next_probe:
                            watchdog.check_once()
                            next_probe = now + 15.0
                        notifier.scan_once()
                        time.sleep(2.0)
                finally:
                    stop_path.unlink(missing_ok=True)
        print("Notification sidecar stopped.")
        return 0
    except KeyboardInterrupt:
        print("Notification sidecar stopped by operator.")
        return 0
    except RuntimeError as exc:
        if "credential" in str(exc).lower():
            print(f"Notification credentials unavailable ({type(exc).__name__}); run the configure command.", file=sys.stderr)
        else:
            _log_sidecar_failure(exc)
        return 1
    except Exception as exc:
        _log_sidecar_failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
