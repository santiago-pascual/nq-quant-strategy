"""Isolated causal June--October simulation; never extends frozen Research.

Prepare validates real local data. Run requires an activation-specific seed and
reviewed calendar; an October seed cannot initialize June retroactively.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp('2026-06-20T00:00:00Z')
END = pd.Timestamp('2026-10-08T00:00:00Z')
DEFAULT_OUTPUT = ROOT / 'results/paper/historical_extension_20260620_20261008'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def isolated_output(path: Path) -> Path:
    path = path.resolve()
    relative = path.relative_to(ROOT / 'results/paper')
    if len(relative.parts) != 1 or not relative.name.startswith('historical_extension_'):
        raise ValueError('extension must use a dedicated results/paper/historical_extension_* directory')
    return path


def validate_frame(frame: pd.DataFrame) -> dict:
    stamps = pd.to_datetime(frame['timestamp'], utc=True, errors='raise')
    if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise ValueError('invalid, duplicate or nonchronological timestamps')
    if set(frame['symbol'].astype(str)) != {'MNQ.v.0'}:
        raise ValueError('canonical continuous symbol must be MNQ.v.0')
    ids = pd.to_numeric(frame['instrument_id'], errors='raise')
    if ids.isna().any() or (ids <= 0).any():
        raise ValueError('invalid instrument identity')
    numbers = frame[['open','high','low','close','volume']].apply(pd.to_numeric, errors='raise')
    import numpy as np
    if not np.isfinite(numbers.to_numpy()).all() or (numbers['volume'] < 0).any():
        raise ValueError('invalid OHLCV')
    if (numbers[['open','high','low','close']] <= 0).any().any() or (
        numbers['high'] < numbers[['open','low','close']].max(axis=1)).any() or (
        numbers['low'] > numbers[['open','high','close']].min(axis=1)).any():
        raise ValueError('invalid price relationships')
    selected = stamps[(stamps >= START) & (stamps < END)]
    if selected.empty or stamps.iloc[0] > START or stamps.iloc[-1] < END - pd.Timedelta(minutes=1):
        raise ValueError('history or extension endpoint is unavailable')
    return {'observed_rows': len(selected), 'first_observed_utc': selected.iloc[0].isoformat(),
            'last_observed_utc': selected.iloc[-1].isoformat(),
            'source_first_utc': stamps.iloc[0].isoformat(), 'source_last_utc': stamps.iloc[-1].isoformat(),
            'minute_grid_completeness_claimed': False}


def prerequisites(args, raw) -> tuple[dict, object | None]:
    from src.paper.causal_bootstrap_cli import _source_file_check
    certificate = json.loads(args.certificate.read_text(encoding='utf-8'))
    match, errors = _source_file_check(certificate)
    if not match:
        raise ValueError('; '.join(errors))
    report = {'schema_version': 1, 'classification': 'HISTORICAL_CAUSAL_SIMULATION',
              'paper_only': True, 'broker_orders_enabled': False,
              'start_utc_inclusive': START.isoformat(), 'end_utc_exclusive': END.isoformat(),
              'source_certificate_sha256': sha256(args.certificate),
              'data': validate_frame(raw), 'blockers': []}
    calendar = None
    try:
        from scripts.validate_cme_snapshot import validate
        from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
        if not args.calendar or not args.calendar_review:
            raise ValueError('reviewed June 20--October 7 calendar is required')
        validate(args.calendar, args.calendar_review)
        calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(args.calendar))
        for day in pd.date_range(START, END - pd.Timedelta(days=1)):
            calendar.snapshot.session_for_rth_date(day.date())
        report['calendar_identity'] = calendar.snapshot.identity
    except (OSError, ValueError, RuntimeError) as exc:
        report['blockers'].append(f'calendar: {exc}')
    try:
        from src.paper.causal_bootstrap import CausalBootstrapCheckpointStore, activation_payload_errors
        from src.paper.runtime_compatibility import validate_runtime_identity
        from src.paper.realtime_checkpoint import runtime_identity
        if not args.seed or not args.seed_identity:
            raise ValueError('June 20 activation-ready causal seed and identity are required; October seed is future data')
        identity = json.loads(args.seed_identity.read_text(encoding='utf-8'))
        payload = CausalBootstrapCheckpointStore(args.seed).load(expected_identity=identity)
        problems = activation_payload_errors(payload, activation_timestamp=START, expected_identity=identity)
        if problems:
            raise ValueError('; '.join(problems))
        # Store.load already checks its identity. Also require the current runtime
        # through the bootstrap's own runtime field rather than inventing a migration.
        saved_runtime = (identity.get('configuration') or {}).get('runtime_identity')
        if not saved_runtime or not validate_runtime_identity(saved_runtime, runtime_identity()).accepted:
            raise ValueError('bootstrap runtime incompatibility')
        from src.paper.causal_bootstrap import _hash_frame
        history = raw.loc[raw.timestamp < START].reset_index(drop=True)
        if _hash_frame(history) != identity.get('source', {}).get('source_frame_sha256'):
            raise ValueError('seed historical source fingerprint differs from the current past-only prefix')
        report['seed_sha256'] = sha256(args.seed)
        report['seed_identity'] = identity
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        report['blockers'].append(f'bootstrap: {exc}')
    from src.paper.costs import PaperCostPolicy
    costs = PaperCostPolicy.from_json(args.costs)
    if costs.artificial_slippage_ticks != 0:
        raise ValueError('extension must preserve current zero artificial-slippage policy')
    report['cost_profile'] = asdict(costs)
    report['ready'] = not report['blockers']
    return report, calendar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','resume'])
    parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--calendar', type=Path)
    parser.add_argument('--calendar-review', type=Path)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--seed-identity', type=Path)
    parser.add_argument('--costs', type=Path, default=ROOT/'src/paper/config/topstepx_mnq_fees_2026-07.json')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path, help='Prepare-only diagnostic JSON outside Paper runs')
    args = parser.parse_args(argv)
    output = isolated_output(args.output_dir)
    from src.paper.autonomous_runner import load_canonical_raw_mnq
    import contextlib
    import sys
    with contextlib.redirect_stdout(sys.stderr):
        raw = load_canonical_raw_mnq(include_contract_metadata=True)
    report, calendar = prerequisites(args, raw)
    if args.command == 'prepare':
        if args.report:
            from src.paper.operational_health import save_report
            save_report(args.report, report, ROOT/'results/paper')
        print(json.dumps(report, indent=2, default=str))
        return 0 if report['ready'] else 2
    if not report['ready']:
        print(json.dumps(report, indent=2, default=str)); return 2
    from src.paper.single_writer import PaperWriterLock
    # A separate orchestration lock prevents competing manifest writes before
    # RealtimePaperService acquires its authoritative execution writer lock.
    output.mkdir(parents=True, exist_ok=True)
    with PaperWriterLock(output/'extension_orchestration.lock'):
        manifest = output/'historical_extension.json'
        resume = args.command == 'resume'
        if resume:
            if json.loads(manifest.read_text()) != report:
                raise ValueError('extension input/configuration identity changed')
        else:
            if any(path.name != 'extension_orchestration.lock' for path in output.iterdir()):
                raise ValueError('extension output is not empty')
            import os
            from src.paper.atomic_io import atomic_replace_with_retry
            temporary = manifest.with_suffix('.tmp')
            with temporary.open('x', encoding='utf-8') as handle:
                handle.write(json.dumps(report, sort_keys=True, indent=2)+'\n')
                handle.flush(); os.fsync(handle.fileno())
            atomic_replace_with_retry(temporary, manifest)
        from src.paper.causal_bootstrap import CausalBootstrapCheckpointStore, load_context_seed_for_new_account
        from src.paper.run_autonomous import build_real_paper_engine
        from src.paper.delayed_paper_cli import _expected_risk_configuration
        from src.paper.costs import PaperCostPolicy
        from src.paper.realtime_market_data import ReplayMarketDataSource, mark_replay_session_final_bars
        from src.paper.realtime_service import RealtimePaperService, RealtimePaperConfig
        from src.paper.run_realtime_paper import _run_with_graceful_signals
        costs = PaperCostPolicy.from_json(args.costs)
        store = CausalBootstrapCheckpointStore(args.seed)
        payload = store.load(expected_identity=report['seed_identity'])
        if payload['account_seed']['risk_configuration'] != _expected_risk_configuration():
            raise ValueError('seed risk differs from frozen Paper policy')
        engine, adapter = build_real_paper_engine(output,
            initial_equity=payload['account_seed']['initial_equity'],
            commission_per_contract=costs.commission_per_contract_side,
            exchange_fee_per_contract=costs.exchange_fee_per_contract_side,
            regulatory_fee_per_contract=costs.regulatory_fee_per_contract_side,
            recover_trailing_event=resume)
        if not resume:
            load_context_seed_for_new_account(store=store, context=adapter.context,
                expected_identity=report['seed_identity'], activation_timestamp=START)
        selected = raw.loc[(raw.timestamp >= START) & (raw.timestamp < END)].copy()
        if resume:
            from src.paper.realtime_checkpoint import AtomicCheckpointStore
            saved = AtomicCheckpointStore(output/'paper_checkpoint.json').load()
            selected = selected.loc[selected.timestamp > pd.Timestamp(saved['runtime']['last_bar']['timestamp'])]
        selected = mark_replay_session_final_bars(selected)
        service = RealtimePaperService(source=ReplayMarketDataSource(selected.to_dict('records')),
            engine=engine, context_adapter=adapter, calendar=calendar, cost_policy=costs,
            config=RealtimePaperConfig(mode='PAPER', output_dir=output, checkpoint_every_bars=1))
        from src.paper.retry_checkpoint import configure_checkpoint_retry
        configure_checkpoint_retry(service)
        _run_with_graceful_signals(service, restore=resume)
        if service._state == 'STOPPED':
            publish_completed_extension(output, report, len(raw.loc[(raw.timestamp >= START) & (raw.timestamp < END)]))
    return 0 if service._state == 'STOPPED' else 2


def publish_completed_extension(output: Path, report: dict, expected_rows: int) -> None:
    """Publish only an exhausted, reconciled replay; early stops stay unavailable."""
    from src.paper.snapshot_reader import read_snapshot_json
    from src.paper.analytics import PaperAnalyticsReader
    status = read_snapshot_json(output/'status.json')
    system = status['system']
    # Status is the persisted authority; no partial replay is promoted to complete.
    if system.get('state') != 'STOPPED' or system.get('bars_processed') != expected_rows:
        raise ValueError('extension did not complete every selected observation')
    from src.paper.realtime_checkpoint import AtomicCheckpointStore
    checkpoint = AtomicCheckpointStore(output/'paper_checkpoint.json').load()
    last = pd.Timestamp(checkpoint['runtime']['last_bar']['timestamp'])
    if last != pd.Timestamp(report['data']['last_observed_utc']):
        raise ValueError('extension final checkpoint does not reach its observed endpoint')
    reader = PaperAnalyticsReader(str(output/'paper_analytics.sqlite3'))
    try:
        analytics = reader.dashboard_analytics(limit=100000)
    finally:
        reader.close()
    if analytics['sample']['truncated']:
        raise ValueError('extension ledger export is incomplete')
    trades = analytics['trade_outcomes']
    from paper_dashboard.research_analytics import reconstruct_paper_outcomes
    normalized = reconstruct_paper_outcomes(trades)
    if not normalized['available'] or normalized.get('invalid_rows'):
        raise ValueError('extension ledger contains invalid or duplicate outcomes')
    artifacts = {name: sha256(output/name) for name in ('status.json','events.jsonl','historical_extension.json')}
    summary = {'schema_version': 1, 'classification': 'HISTORICAL_CAUSAL_SIMULATION',
               'completed': True, 'start_utc_inclusive': START.isoformat(),
               'end_utc_exclusive': END.isoformat(), 'observed_rows': expected_rows,
               'cost_profile': report['cost_profile'], 'artifacts_sha256': artifacts,
               'trades': normalized['trades'], 'research_parity_claimed': False}
    import os
    from src.paper.atomic_io import atomic_replace_with_retry
    target = output/'extension_analysis.json'
    temporary = target.with_suffix('.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    atomic_replace_with_retry(temporary, target)


if __name__ == '__main__':
    raise SystemExit(main())
