"""Crash-released process ownership lock for one Paper output directory."""

from __future__ import annotations

import os
from pathlib import Path


class WriterAlreadyActive(RuntimeError):
    """Raised when another process owns the Paper output directory."""


class PaperWriterLock:
    """Advisory OS lock; the operating system releases it after process death."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> "PaperWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise WriterAlreadyActive(f"Paper output directory is already owned: {self.path.parent}") from exc
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise WriterAlreadyActive(f"Paper output directory is already owned: {self.path.parent}") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n".encode("ascii"))
            handle.flush()
            self._handle = handle
            return self
        except BaseException:
            handle.close()
            raise

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "PaperWriterLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
