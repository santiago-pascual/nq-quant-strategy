"""Prove a Windows venv worker belongs to a reserved launch; no process control."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path

from src.paper.process_identity import current_process_identity


def parent_pid(pid: int) -> int | None:
    """Kernel process lineage, without command-line enumeration or credentials."""
    if os.name != 'nt':
        try:
            text = Path(f'/proc/{pid}/stat').read_text()
            return int(text[text.rfind(')') + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            return None
    class Entry(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD),
            ('pid', wintypes.DWORD), ('heap', ctypes.c_size_t),
            ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
            ('parent', wintypes.DWORD), ('priority', wintypes.LONG),
            ('flags', wintypes.DWORD), ('executable', wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ('Process32FirstW', 'Process32NextW'):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        function.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return None
    try:
        entry = Entry(); entry.size = ctypes.sizeof(entry)
        found = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            if entry.pid == pid:
                return int(entry.parent)
            found = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        return None
    finally:
        kernel.CloseHandle(snapshot)


def prove_reserved_worker(launcher: dict | None, worker: dict) -> bool:
    """Accept only the still-identical launcher or its direct Python worker.

    Ambiguous lineage, reused PIDs and exited launchers fail closed. The caller
    separately proves writer ownership, fresh status, cursors and feed recovery.
    """
    if not isinstance(launcher, dict) or not isinstance(worker, dict):
        return False
    if current_process_identity(launcher.get('pid', 0)) != launcher:
        return False
    if current_process_identity(worker.get('pid', 0)) != worker:
        return False
    if worker == launcher:
        return True
    if Path(worker.get('executable', '')).name.lower() not in {'python', 'python.exe', 'pythonw', 'pythonw.exe'}:
        return False
    try:
        base = 16 if os.name == 'nt' else 10
        if int(worker['created_filetime'], base) < int(launcher['created_filetime'], base):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    related = parent_pid(worker['pid']) == launcher['pid']
    # Recheck both identities after enumerating; never trust PID ancestry alone.
    return related and current_process_identity(launcher['pid']) == launcher and current_process_identity(worker['pid']) == worker
