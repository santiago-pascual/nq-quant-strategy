import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from src.paper.snapshot_reader import open_snapshot,read_snapshot_json
from src.paper.atomic_io import atomic_replace_with_retry


def test_open_version_survives_replacement(tmp_path):
    destination=tmp_path/'status.json'; destination.write_text('{"bar":1}')
    replacement=tmp_path/'new.json'; replacement.write_text('{"bar":2}')
    with open_snapshot(destination) as old:
        atomic_replace_with_retry(replacement,destination)
        assert json.loads(old.read())=={'bar':1}
    assert read_snapshot_json(destination)=={'bar':2}


def test_concurrent_readers_never_see_partial_json(tmp_path):
    destination=tmp_path/'status.json'; destination.write_text('{"bar":0}')
    def writer():
        for number in range(1,101):
            replacement=tmp_path/'new.json'
            replacement.write_text(json.dumps({'bar':number,'payload':'x'*1000}))
            atomic_replace_with_retry(replacement,destination)
    def reader():
        for _ in range(300):
            value=read_snapshot_json(destination)
            assert 0<=value['bar']<=100
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(writer)]+[pool.submit(reader) for _ in range(3)]
        for task in tasks:task.result(timeout=30)
    assert read_snapshot_json(destination)['bar']==100


def test_invalid_or_missing_snapshots_do_not_modify_files(tmp_path):
    path=tmp_path/'status.json'; path.write_text('{partial')
    with pytest.raises(ValueError):read_snapshot_json(path)
    assert path.read_text()=='{partial'
    path.write_text('"too large"')
    with pytest.raises(ValueError,match='limit'):read_snapshot_json(path,maximum_bytes=2)
    with pytest.raises(OSError):read_snapshot_json(tmp_path/'missing')
