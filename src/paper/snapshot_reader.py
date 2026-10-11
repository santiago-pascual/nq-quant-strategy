"""Short-lived read-only snapshot handles that permit Windows atomic rename."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
import time
from pathlib import Path


@contextmanager
def open_snapshot(path: str | Path):
    """The opened inode/version stays stable while another writer replaces it.

    No fallback writes, ACL changes or repair. Windows FILE_SHARE_DELETE avoids
    readers themselves preventing atomic replacement; other blocking handles
    and antivirus can still cause genuine writer errors.
    """
    if os.name != 'nt':
        with Path(path).open('rb') as stream:
            yield stream
        return
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000,
        1 | 2 | 4, None, 3, 0x80, None)  # READ, share READ/WRITE/DELETE, OPEN_EXISTING
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        yield stream  # fd owns and closes the native handle, including on errors


def read_snapshot_json(path: str | Path, *, maximum_bytes: int = 4_000_000):
    """Parse one stable version. Corrupt/incomplete/oversized data raises."""
    if maximum_bytes < 1:
        raise ValueError('snapshot size limit must be positive')
    for attempt in range(3):
        try:
            with open_snapshot(path) as stream:
                raw = stream.read(maximum_bytes + 1)
            break
        except (FileNotFoundError, PermissionError):
            # Windows ReplaceFile can have a brief namespace transition. Retry
            # only opening; missing/denied after this bounded wait stays an error.
            if attempt == 2:
                raise
            time.sleep(.01 * (attempt+1))
    if len(raw) > maximum_bytes:
        raise ValueError('snapshot exceeds read-only size limit')
    return json.loads(raw.decode('utf-8-sig'))
