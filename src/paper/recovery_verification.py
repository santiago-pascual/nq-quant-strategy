"""Post-launch evidence for the disabled supervisor; no trading actions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paper.operational_health import read_snapshot, _stamp
from src.paper.process_identity import current_process_identity, write_identity_record


def evaluate_launch(*, expected_identity: dict, observed_identity: dict | bool | None,
                    status: dict | None, status_modified_at: float, launched_at: float,
                    previous_cursor: str, now: float, market_state: str,
                    writer_owned: bool | None) -> dict[str, Any]:
    """A fresh process, state and writer proof precede any recovery claim.

    Connectivity alone is never sufficient. Closed-session waiting is separate
    from observed durable market progression. No child is killed on failure.
    """
    if observed_identity is False:
        return {'state': 'PROCESS_EXITED', 'verified': False}
    if not isinstance(observed_identity, dict):
        return {'state': 'IDENTITY_UNAVAILABLE', 'verified': False}
    if observed_identity != expected_identity:
        return {'state': 'IDENTITY_MISMATCH', 'verified': False}
    if not status or status_modified_at < launched_at or status_modified_at > now + 5 or now - status_modified_at > 120:
        return {'state': 'WAITING_FOR_FRESH_STATUS', 'verified': False}
    system = status.get('system') or {}
    if system.get('state') in {'ERROR', 'FAILED', 'STOPPED'}:
        return {'state': 'ENGINE_FAILED', 'verified': False}
    if system.get('state') not in {'RUNNING', 'RECOVERING', 'DEGRADED', 'PAUSED_REFIT'}:
        return {'state': 'ENGINE_STATE_UNAVAILABLE', 'verified': False}
    if writer_owned is not True:
        return {'state': 'WRITER_OWNERSHIP_UNVERIFIED', 'verified': False}
    old, committed = _stamp(previous_cursor), _stamp(system.get('last_bar'))
    if old is None or committed is None or committed < old:
        return {'state': 'CURSOR_INVALID', 'verified': False}
    feed = system.get('feed_health') or {}
    if feed.get('connection_state') != 'CONNECTED' or feed.get('provider_connected') is not True:
        return {'state': 'WAITING_FOR_HANDSHAKE_AND_DATA', 'verified': False}
    frontier = _stamp(feed.get('latest_available_bar'))
    if frontier is None or committed > frontier:
        return {'state': 'FRONTIER_INVALID', 'verified': False}
    if feed.get('backlog_bars') is None:
        return {'state': 'BACKLOG_UNAVAILABLE', 'verified': False}
    if feed['backlog_bars'] or feed.get('mode') in {'RECOVERING', 'BLOCKED', 'DISCONNECTED'}:
        return {'state': 'RECOVERING', 'verified': False}
    progressed = committed > old
    if not progressed and market_state not in {'MARKET_CLOSED', 'MARKET_BREAK'}:
        return {'state': 'WAITING_FOR_DURABLE_PROGRESS', 'verified': False}
    return {'state': 'RECOVERED_PROGRESS_VERIFIED' if progressed else 'VERIFIED_CLOSED_SESSION_WAIT',
            'verified': True, 'market_progress_verified': progressed,
            'committed_cursor': committed.isoformat(), 'provider_frontier': frontier.isoformat()}


def inspect_launched_process(run: Path, *, expected_identity: dict, launched_at: float,
                             previous_cursor: str, market_state: str,
                             now: float | None = None, register: bool = False) -> dict:
    """Check only the child identity captured by the trusted launcher.

    Optional PID lease registration is metadata-only and occurs after evidence
    passes. This function is not invoked for the production Engine in tests.
    """
    from src.paper.windows_supervisor import _writer_lock_available
    status_path = run / 'status.json'
    observed = current_process_identity(int(expected_identity['pid']))
    try:
        modified = status_path.stat().st_mtime
    except OSError:
        modified = 0
    writer_available = _writer_lock_available(run / 'paper_writer.lock')
    writer_owned = None
    if writer_available is False:
        try:
            # Windows locks the first byte; reading it would itself fail. The
            # existing lock writer stores "pid=<worker>\n". Read only its tail.
            with (run / 'paper_writer.lock').open('rb') as handle:
                handle.seek(1)
                owner = handle.read(128).decode('ascii').strip()
            writer_owned = owner == f"id={expected_identity['pid']}"
        except (OSError, UnicodeError):
            pass
    result = evaluate_launch(expected_identity=expected_identity, observed_identity=observed,
        status=read_snapshot(status_path), status_modified_at=modified, launched_at=launched_at,
        previous_cursor=previous_cursor, now=now or datetime.now(timezone.utc).timestamp(),
        market_state=market_state, writer_owned=writer_owned)
    if register and result['verified']:
        # Recheck creation identity immediately before registration (PID reuse).
        if current_process_identity(int(expected_identity['pid'])) != expected_identity:
            return {'state': 'IDENTITY_CHANGED', 'verified': False}
        write_identity_record(run / 'paper_process_identity.json', run.name, int(expected_identity['pid']))
        result['watchdog_registered'] = True
    return result
