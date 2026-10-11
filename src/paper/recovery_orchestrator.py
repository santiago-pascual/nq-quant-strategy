"""Disabled-by-default restart controller using authoritative recovery gates.

This is a supervision primitive, not a production restart task. Installation
and activation still require separate authorization. A reserved launch counts
against the crash budget even if the supervisor itself dies during launch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from uuid import uuid4
from typing import Callable, Any

from src.paper.atomic_io import atomic_replace_with_retry
from src.paper.single_writer import PaperWriterLock


class RecoveryController:
    def __init__(self, directory: Path, *, run_id: str, cooldown_seconds: float = 300,
                 maximum_attempts: int = 3, window_seconds: float = 3600):
        if not run_id or maximum_attempts < 1 or cooldown_seconds <= 0 or window_seconds <= 0:
            raise ValueError('invalid recovery policy')
        self.directory = Path(directory)
        self.run_id = run_id
        self.cooldown = cooldown_seconds
        self.maximum = maximum_attempts
        self.window = window_seconds

    def _read(self) -> dict[str, Any]:
        path = self.directory / 'recovery_state.json'
        if not path.exists():
            return {'schema_version': 1, 'run_id': self.run_id, 'attempts': []}
        state = json.loads(path.read_text(encoding='utf-8'))
        saved_digest = state.pop('sha256', None)
        if saved_digest != hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest():
            raise ValueError('recovery controller checksum mismatch')
        if (state.get('schema_version') != 1 or state.get('run_id') != self.run_id
                or not isinstance(state.get('attempts'), list)):
            raise ValueError('incompatible recovery controller state')
        if any(not isinstance(item, dict) or not isinstance(item.get('timestamp'), (int, float))
               or isinstance(item.get('timestamp'), bool) or not math.isfinite(item['timestamp'])
               for item in state['attempts']):
            raise ValueError('invalid recovery attempt history')
        return state

    def _save(self, state: dict) -> None:
        fd, temp = tempfile.mkstemp(prefix='.recovery-', dir=self.directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                document = dict(state)
                document['sha256'] = hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
                json.dump(document, stream, sort_keys=True, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            atomic_replace_with_retry(temp, self.directory / 'recovery_state.json')
        finally:
            Path(temp).unlink(missing_ok=True)

    def verify_started(self, run: Path, *, expected_identity: dict, launched_at: float,
                       previous_cursor: str, market_state: str, register: bool = False) -> dict:
        """Explicit post-launch step, not a production retry loop.

        Caller must supply the actual Python worker identity, not a Windows
        venv launcher PID. Only a matching durable launch reservation is accepted.
        """
        from src.paper.recovery_verification import inspect_launched_process
        with PaperWriterLock(self.directory / 'recovery_controller.lock'):
            state = self._read()
            if run.name != self.run_id or not state['attempts']:
                raise ValueError('no matching launch reservation')
            reservation = state['attempts'][-1]
            from src.paper.recovery_worker import prove_reserved_worker
            if reservation['timestamp'] != launched_at or not prove_reserved_worker(
                    reservation.get('launcher_identity'), expected_identity):
                raise ValueError('process identity differs from reserved launch')
            result = inspect_launched_process(run, expected_identity=expected_identity,
                launched_at=launched_at, previous_cursor=previous_cursor,
                market_state=market_state, register=register)
            reservation['post_launch'] = result
            reservation['worker_identity'] = expected_identity
            if result.get('verified') is True:
                reservation['state'] = 'VERIFIED'
            self._save(state)
            return result

    def attempt(self, *, enabled: bool, authorized: bool, now: float,
                validate: Callable[[], dict], launch: Callable[[], int]) -> dict:
        """Serializes launch reservations; validates again immediately before launch.

        `validate` must invoke windows_supervisor.inspect_run with a fresh TWS
        handshake. The Engine retains its own OS writer lock as the final race
        guard. No checker result is repaired or overridden here.
        """
        if not enabled or not authorized:
            return {'state': 'DISABLED', 'process_started': False}
        if isinstance(now, bool) or not math.isfinite(now) or now < 0:
            raise ValueError('invalid recovery clock')
        self.directory.mkdir(parents=True, exist_ok=True)
        with PaperWriterLock(self.directory / 'recovery_controller.lock'):
            state = self._read()
            recent = [item for item in state['attempts'] if now - item['timestamp'] <= self.window]
            if recent and now - recent[-1]['timestamp'] < self.cooldown:
                return {'state': 'COOLDOWN', 'process_started': False,
                        'retry_after_seconds': self.cooldown - (now - recent[-1]['timestamp'])}
            if len(recent) >= self.maximum:
                return {'state': 'ESCALATED', 'process_started': False,
                        'reason': 'restart budget exhausted; operator intervention required'}
            if state['attempts'] and state['attempts'][-1].get('state') in {'LAUNCH_RESERVED', 'LAUNCHED_UNVERIFIED'}:
                # A supervisor crash may leave a child that has not acquired its
                # writer lock yet. Cooldown expiry alone never proves absence.
                return {'state': 'RECONCILIATION_REQUIRED', 'process_started': False,
                        'reason': 'prior launch lacks verified worker identity and post-launch evidence'}
            report = validate()
            if report.get('safe_to_resume') is not True or report.get('blockers'):
                return {'state': 'BLOCKED', 'process_started': False, 'blockers': report.get('blockers', ['missing gates'])}
            gates = report.get('gates') or {}
            # A bare safe_to_resume Boolean is insufficient evidence.
            required = (gates.get('writer_lock', {}).get('available') is True,
                        gates.get('process_identity') in {'absent', 'no active PID lease'},
                        gates.get('checkpoint', {}).get('valid_internal_checksum') is True,
                        gates.get('checkpoint', {}).get('runtime_identity_matches') is True,
                        gates.get('delivery_reconciliation', {}).get('valid') is True,
                        gates.get('market_cursors', {}).get('match') is True,
                        gates.get('event_journal', {}).get('valid') is True,
                        gates.get('account_reconciliation', {}).get('matches_status') is True,
                        gates.get('bootstrap_identity', {}).get('valid') is True,
                        gates.get('calendar_identity', {}).get('valid') is True,
                        gates.get('contract_identity', {}).get('valid') is True,
                        gates.get('tws_readonly_handshake', {}).get('verified') is True,
                        gates.get('strategy_state', {}).get('complete_four_strategy_state') is True,
                        gates.get('causal_context_present') is True,
                        gates.get('pending_execution_state_present') is True)
            if not all(required):
                return {'state': 'BLOCKED', 'process_started': False, 'reason': 'authoritative gate evidence incomplete'}
            reservation = {'timestamp': now, 'state': 'LAUNCH_RESERVED',
                           'evidence_sha256': hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()}
            state['attempts'].append(reservation)
            self._save(state)  # durable before any child can start
            try:
                pid = launch()
                if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                    raise ValueError('launcher did not return a valid PID')
                reservation.update(state='LAUNCHED_UNVERIFIED', pid=pid)
                from src.paper.process_identity import current_process_identity
                identity = current_process_identity(pid)
                reservation['launcher_identity'] = identity if isinstance(identity, dict) else None
            except Exception as exc:
                reservation.update(state='LAUNCH_FAILED', error_type=type(exc).__name__)
                self._save(state)
                return {'state': 'LAUNCH_FAILED', 'process_started': False, 'error_type': type(exc).__name__}
            self._save(state)
            # Process creation is never reported as successful market recovery.
            return {'state': 'LAUNCHED_UNVERIFIED', 'process_started': True, 'pid': pid}


def supervised_resume_once(run: Path, directory: Path, *, recovery_validation: Path,
                           enabled: bool = False, authorized: bool = False) -> dict:
    """Concrete Windows launcher, deliberately not installed or enabled.

    The authoritative checker is called here, rather than supplied by an
    operator-configurable command. Persisted ERROR still requires manual review.
    The child's own readiness, restore and writer-lock checks remain mandatory.
    """
    run, directory = run.resolve(), directory.resolve()
    if directory == run or run in directory.parents:
        raise ValueError('controller state must be outside the Paper run')
    root = Path(__file__).resolve().parents[2]
    evidence = {}
    def validate():
        from src.paper.windows_supervisor import inspect_run
        report = inspect_run(run, tws_host='127.0.0.1', tws_port=7497,
                             check_tws_handshake=True, allow_automatic_resume=True)
        evidence.update(report)
        return report
    def launch():
        from src.paper.operational_health import read_snapshot
        manifest = read_snapshot(run / 'delayed_paper_run.json') or {}
        status = read_snapshot(run / 'status.json') or {}
        calendar = evidence['gates']['calendar_identity']
        bootstrap = Path(evidence['gates']['bootstrap_identity']['artifact_path'])
        costs = root / 'src/paper/config/topstepx_mnq_fees_2026-07.json'
        from src.paper.costs import PaperCostPolicy
        cost_policy = PaperCostPolicy.from_json(costs)
        if cost_policy.profile_id != (status.get('system') or {}).get('cost_profile_id'):
            raise ValueError('launch fee profile differs from existing run')
        command = [str(root / '.venv/Scripts/python.exe'), '-m', 'src.paper.delayed_paper_cli', 'start',
                   '--activation-utc', manifest['activation_timestamp_utc'],
                   '--calendar', calendar['snapshot'], '--calendar-review', calendar['review_record'],
                   '--bootstrap', str(bootstrap), '--bootstrap-identity', str(bootstrap.with_name('identity.json')),
                   '--recovery-validation', str(recovery_validation.resolve()), '--cost-config', str(costs),
                   '--output-dir', str(run), '--tws-host', '127.0.0.1', '--tws-port', '7497',
                   '--resume', '--confirm-delayed-paper']
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LOKY_MAX_CPU_COUNT='4')
        label = uuid4().hex
        with (directory / f'resume-{label}.stdout.log').open('xb') as output, \
             (directory / f'resume-{label}.stderr.log').open('xb') as errors:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=output, stderr=errors,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return child.pid
    return RecoveryController(directory, run_id=run.name).attempt(
        enabled=enabled, authorized=authorized, now=time.time(), validate=validate, launch=launch)
