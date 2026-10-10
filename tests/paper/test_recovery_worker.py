import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from src.paper import recovery_worker
from src.paper.process_identity import current_process_identity


def test_missing_or_reused_launcher_is_rejected(monkeypatch):
    launcher = {'pid': 12, 'created_filetime': '1', 'executable': 'python.exe'}
    worker = {'pid': 13, 'created_filetime': '2', 'executable': 'python.exe'}
    assert not recovery_worker.prove_reserved_worker(None, worker)
    monkeypatch.setattr(recovery_worker, 'current_process_identity', lambda pid: False)
    assert not recovery_worker.prove_reserved_worker(launcher, worker)


@pytest.mark.parametrize('parent,created,accepted', [(12, '2', True), (99, '2', False), (12, '0', False)])
def test_lineage_and_creation_time(parent, created, accepted, monkeypatch):
    launcher = {'pid': 12, 'created_filetime': '1', 'executable': 'python.exe'}
    worker = {'pid': 13, 'created_filetime': created, 'executable': 'python.exe'}
    monkeypatch.setattr(recovery_worker, 'current_process_identity', lambda pid: {12: launcher, 13: worker}.get(pid))
    monkeypatch.setattr(recovery_worker, 'parent_pid', lambda pid: parent)
    assert recovery_worker.prove_reserved_worker(launcher, worker) is accepted


def test_actual_venv_launcher_worker_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    worker_file = tmp_path / 'worker.json'
    stop = tmp_path / 'stop'
    script = '''
import json,sys,time
from pathlib import Path
from src.paper.process_identity import current_process_identity
Path(sys.argv[1]).write_text(json.dumps(current_process_identity()))
while not Path(sys.argv[2]).exists():time.sleep(.05)
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(worker_file), str(stop)], cwd=root)
    try:
        deadline = time.monotonic() + 15
        while not worker_file.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        worker = json.loads(worker_file.read_text())
        launcher = current_process_identity(child.pid)
        assert recovery_worker.prove_reserved_worker(launcher, worker)
        assert not recovery_worker.prove_reserved_worker(current_process_identity(), worker)
    finally:
        stop.touch(); child.wait(timeout=15)
    assert not recovery_worker.prove_reserved_worker(launcher, worker)
