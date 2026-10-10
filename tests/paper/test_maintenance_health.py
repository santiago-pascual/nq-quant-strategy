import json
from pathlib import Path
import sqlite3
import tempfile

from src.paper.maintenance_health import checkpoint_health, database_health
from src.paper.realtime_checkpoint import AtomicCheckpointStore


def test_primary_and_backup_are_independently_validated_without_fallback():
    root = Path(__file__).resolve().parents[2] / '.test_scratch'; root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        path = Path(temp) / 'checkpoint.json'; store = AtomicCheckpointStore(path)
        store.save({'bar': 1}); store.save({'bar': 2})
        assert checkpoint_health(path)['state'] == 'VALID'
        assert checkpoint_health(store.backup_path)['state'] == 'VALID'
        path.write_text('{broken')
        before = store.backup_path.read_bytes()
        assert checkpoint_health(path)['state'] == 'INVALID_OR_UNREADABLE'
        assert checkpoint_health(store.backup_path)['state'] == 'VALID'
        assert store.backup_path.read_bytes() == before
        assert checkpoint_health(Path(temp) / 'absent')['state'] == 'MISSING'


def test_database_check_is_readonly_and_rejects_corruption():
    root = Path(__file__).resolve().parents[2] / '.test_scratch'; root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        path = Path(temp) / 'db.sqlite'; db = sqlite3.connect(path)
        db.execute('create table trades(id integer)'); db.execute('insert into trades values (1)'); db.commit(); db.close()
        before = path.read_bytes()
        assert database_health(path)['state'] == 'VALID'
        assert path.read_bytes() == before
        path.write_bytes(b'not a database')
        assert database_health(path)['state'] == 'UNAVAILABLE'
