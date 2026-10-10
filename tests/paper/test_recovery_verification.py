import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from src.paper.recovery_verification import evaluate_launch, inspect_launched_process
from src.paper.process_identity import current_process_identity


def inputs():
    return dict(expected_identity={'pid': 12, 'created_filetime': 'x', 'executable': 'python'},
        observed_identity={'pid': 12, 'created_filetime': 'x', 'executable': 'python'},
        status={'system': {'state': 'RUNNING', 'last_bar': '2026-10-09T20:59:00Z',
                 'feed_health': {'connection_state': 'CONNECTED', 'provider_connected': True,
                     'latest_available_bar': '2026-10-09T20:59:00Z', 'backlog_bars': 0}}},
        status_modified_at=101, launched_at=100, previous_cursor='2026-10-09T20:58:00Z',
        now=102, market_state='MARKET_OPEN', writer_owned=True)


@pytest.mark.parametrize('change,expected', [
    ({'observed_identity': False}, 'PROCESS_EXITED'),
    ({'observed_identity': None}, 'IDENTITY_UNAVAILABLE'),
    ({'observed_identity': {'pid': 12}}, 'IDENTITY_MISMATCH'),
    ({'status_modified_at': 99}, 'WAITING_FOR_FRESH_STATUS'),
    ({'now': 300}, 'WAITING_FOR_FRESH_STATUS'),
    ({'status_modified_at': 110}, 'WAITING_FOR_FRESH_STATUS'),
    ({'writer_owned': False}, 'WRITER_OWNERSHIP_UNVERIFIED'),
    ({'previous_cursor': '2026-10-09T21:00:00Z'}, 'CURSOR_INVALID'),
])
def test_failure_gates(change, expected):
    args = inputs(); args.update(change)
    result = evaluate_launch(**args)
    assert result['state'] == expected and not result['verified']


def test_open_session_requires_commits_and_disconnection_is_not_recovery():
    args = inputs(); args['previous_cursor'] = '2026-10-09T20:59:00Z'
    assert evaluate_launch(**args)['state'] == 'WAITING_FOR_DURABLE_PROGRESS'
    args['market_state'] = 'MARKET_CLOSED'
    assert evaluate_launch(**args)['state'] == 'VERIFIED_CLOSED_SESSION_WAIT'
    args['status']['system']['feed_health']['connection_state'] = 'RECONNECTING'
    assert not evaluate_launch(**args)['verified']


def test_prolonged_outage_catchup_and_new_commit_proof():
    args = inputs(); args['previous_cursor'] = '2026-10-09T19:59:00Z'
    args['status']['system']['feed_health']['backlog_bars'] = 60
    assert evaluate_launch(**args)['state'] == 'RECOVERING'
    args['status']['system']['feed_health']['backlog_bars'] = 0
    assert evaluate_launch(**args)['state'] == 'RECOVERED_PROGRESS_VERIFIED'


def test_actual_child_writer_lease_registration_and_exit():
    root = Path(__file__).resolve().parents[2]; scratch = root / '.test_scratch'; scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        run = Path(directory); launched = time.time()
        status = inputs()['status']
        script = '''
import sys,json,time
from pathlib import Path
from src.paper.single_writer import PaperWriterLock
run=Path(sys.argv[1])
with PaperWriterLock(run/'paper_writer.lock'):
 (run/'status.json').write_text(sys.argv[2])
 while not (run/'stop').exists():time.sleep(.05)
'''
        # Use the actual interpreter: the Windows venv launcher is a different
        # process from the Python worker owning the lock.
        child = subprocess.Popen([sys._base_executable, '-c', script, str(run), json.dumps(status)], cwd=root)
        try:
            deadline = time.time() + 10
            while not (run/'status.json').exists() and time.time() < deadline: time.sleep(.05)
            expected = current_process_identity(child.pid)
            result = inspect_launched_process(run, expected_identity=expected, launched_at=launched,
                previous_cursor='2026-10-09T20:58:00Z', market_state='MARKET_OPEN', register=True)
            assert result['verified'] and result['watchdog_registered'], result
            lease = json.loads((run/'paper_process_identity.json').read_text())
            assert lease['pid'] == child.pid and lease['run_id'] == run.name
        finally:
            (run/'stop').touch(); child.wait(timeout=10)
        assert inspect_launched_process(run, expected_identity=expected, launched_at=launched,
            previous_cursor='2026-10-09T20:58:00Z', market_state='MARKET_OPEN')['state'] == 'PROCESS_EXITED'
