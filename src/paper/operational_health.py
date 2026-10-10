"""Bounded, read-only delayed Paper health evidence; never advances a cursor."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from src.paper.atomic_io import atomic_replace_with_retry
from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.monitoring_alerts import evaluate_cme_market_state

UTC = timezone.utc


def _stamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(UTC) if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
        return result if isinstance(result, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def journal_tail(path: Path, *, maximum_bytes: int = 2_000_000) -> dict[str, Any]:
    """Read complete trailing records only; a partial append is not corruption."""
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            start = max(0, size - maximum_bytes)
            stream.seek(start)
            raw = stream.read(maximum_bytes)
    except OSError:
        return {'available': False, 'records': [], 'complete_journal': False}
    if start:
        raw = raw.partition(b'\n')[2]
    partial = bool(raw and not raw.endswith(b'\n'))
    lines = raw.split(b'\n')[:-1]
    records, invalid = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('not an object')
            records.append(record)
        except (ValueError, UnicodeError):
            invalid += 1
    return {'available': True, 'records': records, 'complete_journal': start == 0,
            'partial_append': partial, 'invalid_complete_records': invalid,
            'bytes_read': len(raw)}


def evaluate(status: dict[str, Any] | None, market: dict[str, Any], *,
             now: datetime, previous: dict[str, Any] | None = None,
             status_age_seconds: float | None = None) -> dict[str, Any]:
    """Separate session closure, connectivity, recovery and measured progression."""
    system = (status or {}).get('system') or {}
    feed = system.get('feed_health') or {}
    last, frontier = _stamp(system.get('last_bar')), _stamp(feed.get('latest_available_bar'))
    old = (previous or {}).get('evidence') or {}
    old_last, old_frontier = _stamp(old.get('last_committed_bar')), _stamp(old.get('provider_frontier'))
    problems = []
    if old_last and last and last < old_last:
        problems.append('committed_cursor_regressed')
    if last and frontier and last > frontier:
        problems.append('committed_cursor_ahead_of_provider')
    if old_frontier and frontier and frontier < old_frontier:
        problems.append('provider_frontier_regressed')
    cursor, old_cursor = feed.get('backfill_next_start_epoch_utc'), old.get('acquisition_cursor')
    if isinstance(cursor, (int, float)) and isinstance(old_cursor, (int, float)) and cursor < old_cursor:
        problems.append('acquisition_cursor_regressed')
    state = 'UNKNOWN'
    if not status:
        state = 'UNAVAILABLE'
    elif system.get('state') in {'ERROR', 'FAILED'} or problems:
        state = 'ERROR'
    elif system.get('state') == 'STOPPED':
        state = 'STOPPED'
    elif market.get('state') in {'MARKET_CLOSED', 'MARKET_BREAK', 'MARKET_UNKNOWN'}:
        state = market['state']
    elif feed.get('connection_state') in {'DISCONNECTED', 'RECONNECTING'}:
        state = feed['connection_state']
    elif feed.get('mode') == 'RECOVERING' or feed.get('backlog_bars', 0):
        state = 'RECOVERING'
    elif status_age_seconds is not None and status_age_seconds > 120:
        state = 'STALLED'
    elif feed.get('provider_connected') is True and last:
        state = 'CONNECTED' if old_last and last > old_last else 'WAITING_FOR_DELAYED_DATA'
    progressing = bool(old_last and last and last > old_last)
    return {'schema_version': 1, 'environment': 'PAPER', 'live_execution_enabled': False,
            'observed_at_utc': now.astimezone(UTC).isoformat(), 'state': state,
            'market': market, 'integrity_findings': problems,
            'real_session_validation': 'OBSERVED_COMMIT_PROGRESSION' if progressing else 'PENDING',
            'evidence': {'engine_state': system.get('state'), 'connection_state': feed.get('connection_state'),
                         'last_committed_bar': last.isoformat() if last else None,
                         'provider_frontier': frontier.isoformat() if frontier else None,
                         'provider_progressed': bool(old_frontier and frontier and frontier > old_frontier),
                         'commits_progressed': progressing, 'committed_bars': system.get('bars_processed'),
                         'backlog_bars': feed.get('backlog_bars'), 'status_age_seconds': status_age_seconds,
                         'acquisition_cursor': feed.get('backfill_next_start_epoch_utc'),
                         'last_successful_request_utc': feed.get('last_successful_request_utc')},
            'account': (status or {}).get('portfolio'), 'hmm': (status or {}).get('hmm'),
            'strategy_activity': (status or {}).get('strategies'),
            'errors': system.get('errors', []), 'warnings': system.get('warnings', [])}


def inspect(run: Path, calendar: CMETradingCalendar | None, *, previous: dict | None = None,
            now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    status = read_snapshot(run / 'status.json')
    try:
        age = max(0.0, now.timestamp() - (run / 'status.json').stat().st_mtime)
    except OSError:
        age = None
    report = evaluate(status, evaluate_cme_market_state(calendar, now), now=now,
                      previous=previous, status_age_seconds=age)
    report['run_id'] = run.name
    identity = read_snapshot(run / 'paper_process_identity.json')
    if identity:
        from src.paper.process_identity import verify_identity_record
        report['process_identity'] = verify_identity_record(identity, expected_run_id=run.name)
    else:
        report['process_identity'] = 'UNAVAILABLE'
    sidecar = run.parent / '.notifications' / run.name
    report['notifications'] = {}
    for channel, journal in [('events', 'delivery.jsonl'), ('alerts', 'alert_delivery.jsonl')]:
        delivery = journal_tail(sidecar / journal, maximum_bytes=128_000)
        records = delivery.pop('records')
        delivered = [row for row in records if row.get('state') == 'delivered']
        latest = records[-1] if records else {}
        report['notifications'][channel] = {
            'available': delivery['available'],
            'last_observed_success_utc': delivered[-1].get('recorded_at_utc') if delivered else None,
            'latest_state': latest.get('state'), 'failure_category': latest.get('failure_category'),
            'attempt': latest.get('attempt'),
            'delivery_confirmed': bool(latest.get('state') == 'delivered'),
            'scope': 'bounded_delivery_journal_tail'}
    sample = journal_tail(run / 'events.jsonl')
    rows = sample.pop('records')
    ids = [row['event_id'] for row in rows if row.get('event_id')]
    bars = [_stamp((row.get('payload') or {}).get('timestamp')) for row in rows
            if row.get('event_type') == 'market_data']
    bars = [bar for bar in bars if bar]
    sample.update(sample_records=len(rows), duplicate_event_ids=len(ids) - len(set(ids)),
                  event_counts=dict(Counter(row.get('event_type') for row in rows)),
                  market_bar_order_valid=all(a < b for a, b in zip(bars, bars[1:])),
                  scope='bounded_tail_only')
    report['event_evidence'] = sample
    from src.paper.continuity_validation import validate_bar_events
    report['processing_validation'] = validate_bar_events(rows, calendar)
    if report['processing_validation']['issues']:
        report['integrity_findings'].extend(report['processing_validation']['issues'])
        report['state'] = 'ERROR'
    elif report['processing_validation']['state'] == 'UNRESOLVED' and report['state'] != 'ERROR':
        report['state'] = 'REQUIRES_REVIEW'
    if sample.get('invalid_complete_records') or sample.get('duplicate_event_ids') or not sample['market_bar_order_valid']:
        report['integrity_findings'].append('event_tail_integrity_failure')
        report['state'] = 'ERROR'
    report['unavailable_checks'] = ['full_checkpoint_integrity_not_checked_by_lightweight_report',
                                  'full_journal_uniqueness_not_proven_by_tail',
                                  'finalization_and_causal_state_equivalence_require_recovery_tests']
    if calendar:
        manifest = read_snapshot(run / 'delayed_paper_run.json') or {}
        contract = manifest.get('contract') or {}
        try:
            expiry = datetime.strptime(str(contract.get('expiry')), '%Y%m%d').date()
            expiry_days = (expiry - now.date()).days
        except ValueError:
            expiry_days = None
        report['maintenance'] = {'calendar_coverage_end': calendar.snapshot.coverage_end.isoformat(),
                                'calendar_days_remaining': (calendar.snapshot.coverage_end - now.date()).days,
                                'contract': contract, 'contract_expiry_days_remaining': expiry_days,
                                'replacement_contract_approved': False,
                                'checkpoint_backup_exists': (run / 'paper_checkpoint.json.bak').is_file(),
                                'checkpoint_backup_integrity': 'NOT_CHECKED_BY_LIGHTWEIGHT_REPORT'}
    return report


def save_report(destination: Path, report: dict[str, Any], run: Path) -> None:
    """Reports may never be written anywhere inside the observed production run."""
    destination, run = destination.resolve(), run.resolve()
    if destination == run or run in destination.parents:
        raise ValueError('health reports must be outside the Paper run')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.health-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        atomic_replace_with_retry(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--calendar', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    calendar = None
    if args.calendar:
        calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(args.calendar))
        manifest = read_snapshot(args.run_dir / 'delayed_paper_run.json') or {}
        if manifest.get('calendar_identity') != calendar.snapshot.identity:
            parser.error('calendar identity differs from the observed run')
    report = inspect(args.run_dir, calendar, previous=read_snapshot(args.previous) if args.previous else None)
    if args.output:
        save_report(args.output, report, args.run_dir)
    print(json.dumps(report, indent=2))
    return 2 if report['state'] in {'ERROR', 'UNAVAILABLE', 'MARKET_UNKNOWN', 'REQUIRES_REVIEW'} else 0


if __name__ == '__main__':
    raise SystemExit(main())
