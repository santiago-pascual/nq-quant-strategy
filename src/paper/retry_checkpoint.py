"""Future opt-in checkpoint IO hardening; not installed in the active Engine.

Same canonical envelope/checksum as AtomicCheckpointStore, with bounded sharing
violation retries. Execution state and fallback validation remain unchanged.
"""
import json
import os
from hashlib import sha256
from pathlib import Path
from src.paper.atomic_io import atomic_replace_with_retry
from src.paper.realtime_checkpoint import AtomicCheckpointStore, CHECKPOINT_SCHEMA_VERSION


class RetrySafeCheckpointStore(AtomicCheckpointStore):
    def save(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {'schema_version':CHECKPOINT_SCHEMA_VERSION,'payload':payload}
        body = json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=True)
        document = json.dumps({**envelope,'sha256':sha256(body.encode()).hexdigest()},
                              sort_keys=True,separators=(',', ':'),allow_nan=True)
        temporary = self.path.with_suffix(self.path.suffix+'.tmp')
        with temporary.open('w',encoding='utf-8',newline='\n') as handle:
            handle.write(document); handle.flush(); os.fsync(handle.fileno())
        if self.path.exists():
            atomic_replace_with_retry(self.path,self.backup_path)
        atomic_replace_with_retry(temporary,self.path)
        try:
            fd = os.open(self.path.parent,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except OSError:
            pass  # Same Windows directory-fsync limitation as authoritative store.
