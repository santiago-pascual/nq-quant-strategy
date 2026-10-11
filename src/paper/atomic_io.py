"""Small Windows-safe helpers for atomic sidecar state replacement."""
from __future__ import annotations

import os
from pathlib import Path
import time


def _replace_once(source: str | Path, destination: str | Path) -> None:
    """Replace an existing Windows snapshot with the native replacement API.

    MoveFileEx (used by os.replace) can deny replacing an open destination even
    when readers permit delete sharing. ReplaceFileW preserves target ACLs and
    stable old reader handles. No deletion/in-place-write fallback is allowed.
    """
    try:
        os.replace(source, destination)
        return
    except PermissionError:
        if os.name != 'nt' or not Path(destination).exists():
            raise
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.ReplaceFileW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    kernel.ReplaceFileW.restype = wintypes.BOOL
    if not kernel.ReplaceFileW(str(Path(destination).resolve()),
                              str(Path(source).resolve()), None, 0, None, None):
        raise ctypes.WinError(ctypes.get_last_error())


def atomic_replace_with_retry(source: str | Path, destination: str | Path, *, attempts: int = 10,
                             initial_delay_seconds: float = 0.05) -> None:
    """Retry transient Windows sharing violations during atomic replacement.

    Read-only dashboard requests can briefly hold an atomic JSON snapshot open
    while a sidecar attempts to replace it. Preserve atomicity and fail after a
    bounded wait if the destination remains locked.
    """
    if attempts < 1 or initial_delay_seconds < 0:
        raise ValueError("atomic replace retry settings are invalid")
    for attempt in range(attempts):
        try:
            _replace_once(source, destination)
            return
        except OSError as exc:
            if not isinstance(exc, PermissionError) and getattr(exc, 'winerror', None) not in {32,33,1175}:
                raise
            if attempt + 1 >= attempts:
                raise
            time.sleep(initial_delay_seconds * (attempt + 1))
