"""PID-reuse-safe identity lease for the delayed Paper process watchdog.

The lease is metadata only. It does not control or signal the Paper process.
Windows verification uses the kernel process handle, creation FILETIME and
executable path, so it does not depend on system-wide CIM enumeration.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from src.paper.atomic_io import atomic_replace_with_retry


def _windows_process_identity(pid: int) -> dict[str, Any] | None | bool:
    """Return identity, False if known absent, or None if inspection is denied."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                         ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                         ctypes.POINTER(wintypes.FILETIME)]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                     wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    process = kernel32.OpenProcess(0x1000 | 0x00100000, False, int(pid))  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    if not process:
        error = ctypes.get_last_error()
        if error in {87, 1168, 6}:  # invalid parameter / not found / invalid handle
            return False
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
            return None
        if exit_code.value != 259:  # STILL_ACTIVE
            return False
        creation = wintypes.FILETIME(); exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME(); user_time = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(process, ctypes.byref(creation), ctypes.byref(exit_time),
                                        ctypes.byref(kernel_time), ctypes.byref(user_time)):
            return None
        length = wintypes.DWORD(32768)
        image = ctypes.create_unicode_buffer(length.value)
        if not kernel32.QueryFullProcessImageNameW(process, 0, image, ctypes.byref(length)):
            return None
        filetime = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return {"pid": int(pid), "created_filetime": f"{filetime:016x}",
                "executable": os.path.normcase(os.path.abspath(image.value))}
    finally:
        kernel32.CloseHandle(process)


def current_process_identity(pid: int | None = None) -> dict[str, Any] | None | bool:
    target = int(pid if pid is not None else os.getpid())
    if target <= 0:
        raise ValueError("PID must be positive")
    if os.name == "nt":
        return _windows_process_identity(target)
    stat_path = Path(f"/proc/{target}/stat")
    try:
        stat = stat_path.read_text(encoding="ascii")
        close = stat.rfind(")")
        fields = stat[close + 2:].split()
        start_ticks = fields[19]  # proc stat field 22 (starttime), after pid/comm.
        executable = os.path.realpath(f"/proc/{target}/exe")
    except FileNotFoundError:
        return False
    except (OSError, IndexError, ValueError):
        return None
    return {"pid": target, "created_filetime": str(start_ticks),
            "executable": os.path.normcase(executable)}


def make_identity_record(run_id: str, pid: int | None = None) -> dict[str, Any]:
    identity = current_process_identity(pid)
    if not isinstance(identity, dict):
        raise RuntimeError("cannot verify current Paper process identity")
    if Path(str(identity.get("executable", ""))).name.lower() not in {"python", "python.exe", "pythonw", "pythonw.exe"}:
        raise RuntimeError("registered process is not Python")
    return {"schema_version": 1, "run_id": str(run_id), **identity}


def write_identity_record(path: str | Path, run_id: str, pid: int) -> dict[str, Any]:
    target = Path(path)
    record = make_identity_record(run_id, pid)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.flush(); os.fsync(handle.fileno())
        atomic_replace_with_retry(name, target)
    finally:
        if os.path.exists(name): os.unlink(name)
    return record


def verify_identity_record(record: Any, *, expected_run_id: str) -> str:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        return "invalid"
    if record.get("run_id") != expected_run_id:
        return "mismatch"
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return "invalid"
    current = current_process_identity(pid)
    if current is False:
        return "absent"
    if current is None:
        return "unavailable"
    if (current.get("created_filetime") != record.get("created_filetime")
            or current.get("executable") != record.get("executable")):
        return "mismatch"
    return "running"


class PaperProcessIdentityLease(AbstractContextManager[dict[str, Any]]):
    """Publish this process identity while delayed Paper is running."""
    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.record = make_identity_record(run_id)

    def __enter__(self) -> dict[str, Any]:
        write_identity_record(self.path, str(self.record["run_id"]), int(self.record["pid"]))
        return self.record

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved == self.record:
                self.path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
        return None
