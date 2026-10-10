import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time

import pytest

from src.paper.recovery_orchestrator import RecoveryController, supervised_resume_once
from src.paper.single_writer import PaperWriterLock, WriterAlreadyActive


@pytest.fixture
def scratch():
    root = Path(__file__).resolve().parents[2] / '.test_scratch'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        yield Path(temp)


def gates():
    return {'safe_to_resume': True, 'blockers': [], 'gates': {
        'writer_lock': {'available': True}, 'process_identity': 'absent',
        'checkpoint': {'valid_internal_checksum': True, 'runtime_identity_matches': True},
        'delivery_reconciliation': {'valid': True}, 'event_journal': {'valid': True},
        'market_cursors': {'match': True},
        'account_reconciliation': {'matches_status': True}, 'bootstrap_identity': {'valid': True},
        'calendar_identity': {'valid': True}, 'contract_identity': {'valid': True},
        'tws_readonly_handshake': {'verified': True},
        'strategy_state': {'complete_four_strategy_state': True},
        'causal_context_present': True, 'pending_execution_state_present': True}}


def test_disabled_controller_never_probes_or_launches(scratch):
    controller = RecoveryController(scratch / 'not_created', run_id='test')
    def forbidden():
        pytest.fail('disabled controller performed work')
    assert controller.attempt(enabled=False, authorized=True, now=0, validate=forbidden, launch=forbidden)['state'] == 'DISABLED'
    assert not (scratch / 'not_created').exists()


@pytest.mark.parametrize('gate,field', [('checkpoint', 'valid_internal_checksum'),
    ('checkpoint', 'runtime_identity_matches'), ('writer_lock', 'available'),
    ('delivery_reconciliation', 'valid'), ('tws_readonly_handshake', 'verified'),
    ('market_cursors', 'match'), ('account_reconciliation', 'matches_status')])
def test_each_integrity_failure_blocks_launch(scratch, gate, field):
    report = gates(); report['gates'][gate][field] = False
    def forbidden():
        pytest.fail('failed integrity gate launched a process')
    controller = RecoveryController(scratch, run_id='test')
    assert controller.attempt(enabled=True, authorized=True, now=0, validate=lambda: report,
                              launch=forbidden)['state'] == 'BLOCKED'


def test_launch_reservation_survives_supervisor_power_loss_and_cooldown(scratch):
    controller = RecoveryController(scratch, run_id='test')
    def power_loss():
        state = json.loads((scratch / 'recovery_state.json').read_text())
        assert state['attempts'][-1]['state'] == 'LAUNCH_RESERVED'
        raise SystemExit('simulated supervisor termination')
    with pytest.raises(SystemExit):
        controller.attempt(enabled=True, authorized=True, now=100, validate=gates, launch=power_loss)
    restarted = RecoveryController(scratch, run_id='test')
    assert restarted.attempt(enabled=True, authorized=True, now=110, validate=gates,
                             launch=lambda: 1)['state'] == 'COOLDOWN'


def test_crash_loop_escalates_across_restarts(scratch):
    def failure():
        raise OSError('launch failed')
    for timestamp in (0, 301, 602):
        result = RecoveryController(scratch, run_id='test').attempt(enabled=True, authorized=True,
                    now=timestamp, validate=gates, launch=failure)
        assert result['state'] == 'LAUNCH_FAILED'
    result = RecoveryController(scratch, run_id='test').attempt(enabled=True, authorized=True,
                now=903, validate=gates, launch=lambda: 123)
    assert result['state'] == 'ESCALATED'


def test_duplicate_supervisor_lock_blocks_launch(scratch):
    with PaperWriterLock(scratch / 'recovery_controller.lock'):
        with pytest.raises(WriterAlreadyActive):
            RecoveryController(scratch, run_id='test').attempt(enabled=True, authorized=True,
                        now=0, validate=gates, launch=lambda: 123)


def test_launch_is_unverified_and_corrupt_state_blocks(scratch):
    result = RecoveryController(scratch, run_id='test').attempt(enabled=True, authorized=True,
                now=0, validate=gates, launch=lambda: 123)
    assert result == {'state': 'LAUNCHED_UNVERIFIED', 'process_started': True, 'pid': 123}
    (scratch / 'recovery_state.json').write_text('{broken')
    with pytest.raises(ValueError):
        RecoveryController(scratch, run_id='test').attempt(enabled=True, authorized=True,
                    now=1000, validate=gates, launch=lambda: 123)


def test_hard_supervisor_process_exit_preserves_launch_reservation(scratch):
    report_path = scratch / 'gates.json'
    report_path.write_text(json.dumps(gates()))
    script = '''
import json,os,sys
from pathlib import Path
from src.paper.recovery_orchestrator import RecoveryController
directory=Path(sys.argv[1])
report=json.loads((directory/'gates.json').read_text())
RecoveryController(directory,run_id='test').attempt(enabled=True,authorized=True,now=100,
    validate=lambda:report,launch=lambda:os._exit(71))
'''
    # Terminate the actual stdlib-only worker, not a Windows venv launcher whose
    # inherited pipe can outlive subprocess.run's launcher timeout. Worker
    # ancestry through the real venv is covered separately below.
    child = subprocess.run([sys._base_executable, '-c', script, str(scratch)], timeout=20,
                           cwd=Path(__file__).resolve().parents[2], capture_output=True)
    assert child.returncode == 71
    state = json.loads((scratch / 'recovery_state.json').read_text())
    assert state['attempts'][-1]['state'] == 'LAUNCH_RESERVED'
    result = RecoveryController(scratch, run_id='test').attempt(enabled=True, authorized=True,
        now=110, validate=gates, launch=lambda: 123)
    assert result['state'] == 'COOLDOWN'


def test_concrete_resume_entrypoint_is_disabled_and_cannot_write_into_run(scratch):
    run = scratch / 'run'; run.mkdir()
    result = supervised_resume_once(run, scratch / 'controller', recovery_validation=scratch / 'evidence.json')
    assert result == {'state': 'DISABLED', 'process_started': False}
    assert list(run.iterdir()) == []
    with pytest.raises(ValueError, match='outside'):
        supervised_resume_once(run, run / 'controller', recovery_validation=scratch / 'evidence.json')


def test_valid_json_controller_corruption_is_rejected(scratch):
    controller = RecoveryController(scratch, run_id='test')
    controller.attempt(enabled=True, authorized=True, now=100, validate=gates, launch=lambda: 123)
    path = scratch / 'recovery_state.json'
    document = json.loads(path.read_text())
    document['attempts'] = []
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='checksum'):
        controller.attempt(enabled=True, authorized=True, now=500, validate=gates, launch=lambda: 123)


def test_post_launch_cannot_verify_an_unreserved_pid(scratch):
    controller = RecoveryController(scratch, run_id='test')
    controller.attempt(enabled=True, authorized=True, now=100, validate=gates, launch=lambda: 123)
    with pytest.raises(ValueError, match='reserved'):
        controller.verify_started(scratch / 'test', expected_identity={'pid': 456},
            launched_at=100, previous_cursor='2026-10-09T20:59:00Z', market_state='MARKET_CLOSED')


def test_controller_binds_real_venv_worker_and_registers_only_after_proof(scratch):
    run = scratch / 'test'; run.mkdir()
    directory = scratch / 'controller'
    controller = RecoveryController(directory, run_id=run.name)
    launched_at = time.time()
    script = '''
import json,sys,time
from pathlib import Path
from src.paper.single_writer import PaperWriterLock
from src.paper.process_identity import current_process_identity
run=Path(sys.argv[1])
with PaperWriterLock(run/'paper_writer.lock'):
 (run/'worker.json').write_text(json.dumps(current_process_identity()))
 (run/'status.json').write_text(json.dumps({'system':{'state':'RUNNING',
  'last_bar':'2026-10-09T20:59:00Z','feed_health':{'connection_state':'CONNECTED',
  'provider_connected':True,'latest_available_bar':'2026-10-09T20:59:00Z','backlog_bars':0}}}))
 while not (run/'stop').exists():time.sleep(.05)
'''
    child = None
    def launch():
        nonlocal child
        child = subprocess.Popen([sys.executable, '-c', script, str(run)],
                                 cwd=Path(__file__).resolve().parents[2])
        return child.pid
    try:
        assert controller.attempt(enabled=True, authorized=True, now=launched_at,
            validate=gates, launch=launch)['state'] == 'LAUNCHED_UNVERIFIED'
        deadline = time.monotonic() + 15
        while not (run/'status.json').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        worker = json.loads((run/'worker.json').read_text())
        result = controller.verify_started(run, expected_identity=worker,
            launched_at=launched_at, previous_cursor='2026-10-09T20:59:00Z',
            market_state='MARKET_CLOSED', register=True)
        assert result['state'] == 'VERIFIED_CLOSED_SESSION_WAIT'
        assert result['watchdog_registered'] and not result['market_progress_verified']
        assert controller._read()['attempts'][-1]['worker_identity'] == worker
        assert json.loads((run/'paper_process_identity.json').read_text())['pid'] == worker['pid']
    finally:
        (run/'stop').touch()
        if child is not None: child.wait(timeout=15)
