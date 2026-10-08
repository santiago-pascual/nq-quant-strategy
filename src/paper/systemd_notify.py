"""Optional systemd readiness/watchdog notifications (no-op outside systemd)."""

from __future__ import annotations

import os
import socket


def notify_systemd(message: str) -> bool:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        client.connect(address)
        client.sendall(message.encode("utf-8"))
        return True
    except OSError:
        return False
    finally:
        client.close()
