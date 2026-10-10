"""Persist daily/weekly read-only reports outside the Paper execution run."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from src.paper.analytics import PaperAnalyticsReader
from src.paper.operational_health import read_snapshot, save_report
from paper_dashboard.research_analytics import load_validated_research
from paper_dashboard.forward_diagnostics import forward_comparison


from paper_dashboard.closed_outcome_metrics import closed_trade_drawdown


def period_bounds(day: date, period: str) -> tuple[str, str]:
    if period not in {'daily', 'weekly'}:
        raise ValueError('period must be daily or weekly')
    first = day if period == 'daily' else day - timedelta(days=day.weekday())
    following = first + timedelta(days=1 if period == 'daily' else 7)
    zone = ZoneInfo('America/New_York')
    return tuple(datetime.combine(item, time(), zone).astimezone(timezone.utc).isoformat()
                 for item in (first, following))


def build_report(run: Path, *, day: date, period: str, root: Path) -> dict:
    start, end = period_bounds(day, period)
    status = read_snapshot(run / 'status.json')
    reader = PaperAnalyticsReader(str(run / 'paper_analytics.sqlite3'))
    try:
        analytics = reader.dashboard_analytics(start=start, end=end, limit=5000)
        lifetime = reader.dashboard_analytics(limit=5000)
    finally:
        reader.close()
    research = load_validated_research(root)
    # Do not reuse the reader's sample curve drawdown: older implementations
    # started at the first outcome and could omit an initial losing trade.
    analytics['metrics'].pop('max_drawdown_usd', None)
    analytics['metrics'].pop('current_drawdown_usd', None)
    analytics['metrics']['closed_trade_sample_drawdown'] = closed_trade_drawdown(analytics['trade_outcomes'])
    comparison = forward_comparison(research['trades'], lifetime['trade_outcomes'])
    if lifetime['sample']['truncated']:
        comparison['limitations'].append('Forward outcomes truncated to the latest 5000 trades.')
    report = {'schema_version': 1, 'environment': 'PAPER', 'run_id': run.name,
              'period': period, 'period_start_utc': start, 'period_end_utc_exclusive': end,
              'timezone': 'America/New_York', 'execution': 'INTERNAL_SIMULATED_FILLS',
              'metrics': analytics['metrics'], 'strategies': analytics['strategies'],
              'hmm_raw_state_metrics': analytics['hmm_raw_state_metrics'],
              'sample': analytics['sample'], 'assumptions': analytics['assumptions'],
              'forward_diagnostics': comparison, 'research_provenance': research['provenance'],
              'current_account_snapshot': (status or {}).get('portfolio'),
              'last_committed_bar': ((status or {}).get('system') or {}).get('last_bar'),
              'unavailable': ['verified_feed_uptime_for_period', 'broker_execution_quality',
                              'period_end_account_equity_when_snapshot_not_at_boundary',
                              'complete_market_data_coverage_for_report_period'],
              'period_has_ended': datetime.now(timezone.utc).isoformat() >= end,
              'complete_period_validation': 'UNVERIFIED'}
    feed = ((status or {}).get('system') or {}).get('feed_health') or {}
    report['reconciliation'] = {
        'pairwise_signal_parity': 'UNAVAILABLE_DIFFERENT_MARKET_PERIODS',
        'research_period': '2020-06-23 through 2026-06-19',
        'research_units': 'stored Research r_multiple', 'paper_units': 'stored net realized_r and USD',
        'source_difference': {'paper_provider': feed.get('source'),
                              'contract': (read_snapshot(run / 'delayed_paper_run.json') or {}).get('contract')},
        'execution': {'paper': 'internal simulated fills; broker execution unavailable',
                      'cost_profile': ((status or {}).get('system') or {}).get('cost_profile_id')},
        'operational_failures': ((status or {}).get('system') or {}).get('errors', []),
        'performance_difference_attribution': 'UNAVAILABLE_WITHOUT_MATCHED_INPUTS',
        'research_gap': 'No observations or performance inferred between June and October 2026'}
    report['report_id'] = hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    parser.add_argument('--period', choices=['daily', 'weekly'], default='daily')
    parser.add_argument('--output-dir', type=Path, default=Path('results/diagnostics/quant_reports'))
    parser.add_argument('--send', action='store_true', help='Send through a separate durable Telegram report journal')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    report = build_report(args.run_dir.resolve(), day=args.date, period=args.period, root=root)
    destination = args.output_dir / args.run_dir.name / f"{args.period}_{args.date}_{report['report_id'][:12]}.json"
    save_report(destination, report, args.run_dir)
    delivery = 'NOT_REQUESTED'
    if args.send:
        from src.paper.notification_credentials import load_credentials
        from src.paper.notifications import NotificationPreferences, PersistentEventNotifier, TelegramTransport
        from src.paper.notification_cli import _rate_limiter
        import os
        directory = destination.parent
        spool = directory / 'report_events.jsonl'
        metric = report['metrics']
        message = (f"MNQ PAPER · {args.period.upper()} REPORT\n"
                   f"Period: {report['period_start_utc']} to {report['period_end_utc_exclusive']} exclusive\n"
                   f"Closed trades: {metric['trades']} · Net PnL: ${metric['net_pnl']:.2f}\n"
                   f"Data coverage: UNVERIFIED · Internal simulated fills only")
        notification_id = hashlib.sha256(
            f"quant-report|{args.run_dir.resolve()}|{args.period}|{report['period_start_utc']}|{report['period_end_utc_exclusive']}".encode()
        ).hexdigest()
        envelope = {'event_id': notification_id, 'event_type': 'monitoring_alert',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'payload': {'event_id': notification_id, 'alert_id': notification_id,
                                'report_id': report['report_id'],
                                'alert_type': message, 'severity': 'INFO', 'environment': 'PAPER',
                                'run_id': args.run_dir.name, 'strategy': 'SYSTEM', 'symbol': 'MNQ',
                                'status': 'RESOLVED', 'details': 'Persisted quantitative report; complete period coverage unverified.'}}
        from src.paper.single_writer import PaperWriterLock, WriterAlreadyActive
        # Enqueue before acquiring the delivery lock: a busy sender must not
        # prevent durable queueing. Producers serialize appends independently.
        with PaperWriterLock(directory / 'report_enqueue.lock'):
            fd = os.open(spool, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                os.write(fd, (json.dumps(envelope, separators=(',', ':')) + '\n').encode()); os.fsync(fd)
            finally:
                os.close(fd)
        try:
            with PaperWriterLock(directory / 'report_delivery.lock'):
                credentials = load_credentials()
                notifier = PersistentEventNotifier(event_path=spool, state_path=directory / 'report_delivery_state.json',
                    journal_path=directory / 'report_delivery.jsonl', replay_existing=True,
                    preferences=NotificationPreferences.from_environment(),
                    send=TelegramTransport(credentials.token, credentials.chat_id),
                    rate_limiter=_rate_limiter(args.run_dir.resolve()))
                delivery = notifier.scan_once()
        except WriterAlreadyActive:
            delivery = 'QUEUED_DELIVERY_BUSY'
    print(json.dumps({'report_path': str(destination), 'report_id': report['report_id'],
                      'period_has_ended': report['period_has_ended'], 'metrics': report['metrics'],
                      'telegram_delivery': delivery}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
