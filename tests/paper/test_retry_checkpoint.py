from pathlib import Path
import tempfile
import pytest
from src.paper.realtime_checkpoint import AtomicCheckpointStore
from src.paper.retry_checkpoint import RetrySafeCheckpointStore


def test_retry_store_is_byte_equivalent_and_preserves_fallback(monkeypatch):
    import src.paper.atomic_io as io
    root=Path(__file__).resolve().parents[2]/'.test_scratch';root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        a=AtomicCheckpointStore(Path(temp)/'a.json'); b=RetrySafeCheckpointStore(Path(temp)/'b.json')
        a.save({'state':1}); b.save({'state':1})
        original=io.os.replace; attempts=[]
        def transient(source,target):
            attempts.append(str(target))
            if len(attempts) in (1,3): raise PermissionError('isolated transient sharing failure')
            return original(source,target)
        monkeypatch.setattr(io.os,'replace',transient)
        b.save({'state':2})
        monkeypatch.setattr(io.os,'replace',original)
        a.save({'state':2})
        assert a.path.read_bytes()==b.path.read_bytes()
        assert a.backup_path.read_bytes()==b.backup_path.read_bytes()
        b.path.write_text('{corrupt')
        assert b.load()=={'state':1}


def test_permanent_replace_failure_still_fails_closed(monkeypatch):
    import src.paper.atomic_io as io
    root=Path(__file__).resolve().parents[2]/'.test_scratch';root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        store=RetrySafeCheckpointStore(Path(temp)/'a.json');store.save({'state':1})
        before=store.path.read_bytes()
        def fail(*args):raise PermissionError('isolated permanent denial')
        monkeypatch.setattr(io.os,'replace',fail);monkeypatch.setattr(io.time,'sleep',lambda seconds:None)
        with pytest.raises(PermissionError):store.save({'state':2})
        assert store.path.read_bytes()==before and store.load()=={'state':1}
