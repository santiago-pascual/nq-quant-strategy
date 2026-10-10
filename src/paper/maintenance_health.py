"""Bounded read-only maintenance evidence; never repairs production artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
import time

from src.paper.operational_health import read_snapshot, save_report


def checkpoint_health(path: Path, *, include_runtime: bool = False) -> dict:
    from src.paper.realtime_checkpoint import CHECKPOINT_SCHEMA_VERSION
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
        envelope = {'schema_version': document['schema_version'], 'payload': document['payload']}
        body = json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=True)
        valid = (sha256(body.encode()).hexdigest() == document.get('sha256')
                 and document['schema_version'] == CHECKPOINT_SCHEMA_VERSION)
        result = {'state': 'VALID' if valid else 'INVALID', 'bytes': len(raw),
                  'internal_checksum_valid': valid, 'sha256': sha256(raw).hexdigest()}
        if include_runtime and valid:
            result['_runtime_identity'] = (document['payload'].get('system') or {}).get('runtime_identity')
        return result
    except FileNotFoundError:
        return {'state': 'MISSING', 'internal_checksum_valid': False}
    except (OSError, ValueError, KeyError, TypeError):
        return {'state': 'INVALID_OR_UNREADABLE', 'internal_checksum_valid': False}


def database_health(path: Path, *, maximum_seconds: float = 2) -> dict:
    if not path.is_file():
        return {'state': 'MISSING'}
    if maximum_seconds <= 0:
        raise ValueError('database check must have a positive bounded duration')
    began = time.monotonic()
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)
        connection.execute('PRAGMA query_only=ON')
        connection.set_progress_handler(lambda: int(time.monotonic() - began > maximum_seconds), 1000)
        result = [row[0] for row in connection.execute('PRAGMA quick_check(10)').fetchall()]
        return {'state': 'VALID' if result == ['ok'] else 'INVALID', 'results': result,
                'duration_seconds': time.monotonic() - began, 'scope': 'SQLite quick_check; logical trade reconciliation separate'}
    except sqlite3.Error as exc:
        return {'state': 'UNAVAILABLE', 'error_type': type(exc).__name__,
                'duration_seconds': time.monotonic() - began}
    finally:
        if connection: connection.close()


def inspect_maintenance(run: Path) -> dict:
    from src.paper.realtime_checkpoint import runtime_identity
    from src.paper.runtime_compatibility import validate_runtime_identity
    run = run.resolve()
    primary = run / 'paper_checkpoint.json'
    backup = run / 'paper_checkpoint.json.bak'
    checkpoints = {'primary': checkpoint_health(primary, include_runtime=True), 'backup': checkpoint_health(backup)}
    runtime = {'accepted': False, 'state': 'UNAVAILABLE'}
    if checkpoints['primary']['internal_checksum_valid']:
        try:
            runtime = validate_runtime_identity(checkpoints['primary'].pop('_runtime_identity', None), runtime_identity()).as_dict()
        except (OSError, ValueError, KeyError):
            pass
    disk = shutil.disk_usage(run)
    checkpoint_bytes = sum(item.get('bytes', 0) for item in checkpoints.values())
    logs = list((run.parent / '.automation' / 'logs').glob('*.log'))
    status = read_snapshot(run / 'status.json') or {}
    manifest = read_snapshot(run / 'delayed_paper_run.json') or {}
    return {'schema_version': 1, 'environment': 'PAPER', 'run_id': run.name,
            'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoints': checkpoints, 'runtime_compatibility': runtime,
            'database': database_health(run / 'paper_analytics.sqlite3'),
            'disk': {'free_bytes': disk.free, 'total_bytes': disk.total,
                     'checkpoint_pair_bytes': checkpoint_bytes,
                     'can_store_two_more_checkpoint_pairs': disk.free > max(1, checkpoint_bytes * 2),
                     'forecast_available': False},
            'calendar': (status.get('system') or {}).get('cme_calendar'),
            'contract': manifest.get('contract'), 'automatic_calendar_approval': False,
            'automatic_contract_roll': False,
            'automation_logs': {'files': len(logs), 'bytes': sum(p.stat().st_size for p in logs),
                                'rotation_required': len(logs) > 30},
            'actions_taken': [], 'resume_authorization': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect_maintenance(args.run_dir)
    save_report(args.output, report, args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if all(row['internal_checksum_valid'] for row in report['checkpoints'].values()) and report['database']['state'] == 'VALID' and report['runtime_compatibility'].get('accepted') else 2


if __name__ == '__main__':
    raise SystemExit(main())
