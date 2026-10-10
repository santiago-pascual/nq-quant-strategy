"""Small Windows-safe helpers for atomic sidecar state replacement."""
from __future__ import annotations

import os
from pathlib import Path
import time


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
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(initial_delay_seconds * (attempt + 1))
