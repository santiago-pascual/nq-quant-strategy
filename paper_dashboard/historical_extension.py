"""Read-only extension boundaries and completed simulation provenance."""
from pathlib import Path

from src.paper.historical_extension import START, END, DEFAULT_OUTPUT, sha256
from src.paper.snapshot_reader import read_snapshot_json
from paper_dashboard.research_analytics import reconstruct_paper_outcomes


def load_extension(root: Path) -> dict:
    directory = root/'results/paper'/DEFAULT_OUTPUT.name
    path = directory/'extension_analysis.json'
    if not path.is_file():
        return {'available': False, 'reason': 'Historical causal extension has not completed validation',
                'start_utc_inclusive': START.isoformat(), 'end_utc_exclusive': END.isoformat(), 'trades': []}
    try:
        value = read_snapshot_json(path, maximum_bytes=20_000_000)
        if value.get('schema_version') != 1 or value.get('classification') != 'HISTORICAL_CAUSAL_SIMULATION' or value.get('completed') is not True:
            raise ValueError('extension is not a completed causal simulation')
        if value.get('start_utc_inclusive') != START.isoformat() or value.get('end_utc_exclusive') != END.isoformat():
            raise ValueError('extension bounds differ from the documented transition gap')
        evidence = value.get('artifacts_sha256', {})
        if set(evidence) != {'status.json','events.jsonl','historical_extension.json'}:
            raise ValueError('extension provenance is incomplete')
        for name, digest in evidence.items():
            if sha256(directory/name) != digest:
                raise ValueError('extension source changed after validation')
        normalized = reconstruct_paper_outcomes(value['trades'])
        if not normalized['available'] or normalized.get('invalid_rows'):
            raise ValueError('extension outcomes are invalid')
        for row in normalized['trades']:
            import pandas as pd
            entry = pd.Timestamp(row['entry_timestamp_utc'])
            exit_time = pd.Timestamp(row['exit_timestamp_utc'])
            if pd.isna(entry) or pd.isna(exit_time) or entry.tzinfo is None or exit_time.tzinfo is None:
                raise ValueError('extension trade timestamps must be finite UTC-aware observations')
            if entry < START or exit_time >= END or exit_time < entry:
                raise ValueError('extension trade violates segment boundaries')
        return {**value, 'available': True, 'trades': normalized['trades']}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'available': False, 'reason': str(exc), 'trades': []}


def source_timeline(root: Path, activation: str | None) -> list[dict]:
    """Dates denote data/validation scopes, never a fabricated return curve."""
    actual = root/'results/diagnostics/historical_extension_preparation_v1.json'
    data = {}
    try:
        report = read_snapshot_json(actual)
        data = report.get('data', {})
        observed = data.get('observed_rows')
    except (OSError, ValueError, TypeError):
        observed = None
    return [
        {'segment': 'Historical raw inputs', 'start': data.get('source_first_utc'), 'end': data.get('source_last_utc'),
         'units': 'Observed MNQ.v.0 OHLCV; not performance',
         'validation': 'Source integrity plus explicit Paper-only residual quality acceptance' if data else 'Source validation report unavailable'},
        {'segment': 'Frozen Research common OOS', 'start': '2020-06-23T00:00:00Z', 'end': '2026-06-19T23:59:59Z',
         'units': 'Research R', 'validation': 'Frozen independent reproduction; 3,255 trades'},
        {'segment': 'Historical causal simulation extension', 'start': START.isoformat(), 'end': END.isoformat(),
         'units': 'Separate simulated USD and net R',
         'validation': f'{observed:,} observed bars; results pending seed/calendar validation' if isinstance(observed,int) else 'Results unavailable; source validation report pending'},
        {'segment': 'Forward delayed Paper', 'start': activation, 'end': None,
         'units': 'Selected-run simulated account USD / persisted net R', 'validation': 'Persisted account/events only; no inherited Research PnL'}]
