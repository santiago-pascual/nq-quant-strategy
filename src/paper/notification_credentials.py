"""Current-user DPAPI storage for local Telegram notification credentials.

The encrypted file lives under LOCALAPPDATA and is bound to the Windows user
profile that created it. Plaintext values are held only in process memory.
"""
from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import getpass
import json
import os
from pathlib import Path
import tempfile
from typing import NamedTuple
from urllib.request import Request, urlopen


class TelegramCredentials(NamedTuple):
    token: str
    chat_id: str


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def credentials_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        raise RuntimeError("LOCALAPPDATA is unavailable; Windows user profile required")
    return Path(root) / "MNQPaperDashboard" / "telegram.credentials.json"


def _crypt(data: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Telegram credential storage requires Windows DPAPI")
    source_buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = _Blob()
    crypt = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    crypt.argtypes = ([ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.POINTER(_Blob),
                       wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_Blob)]
                      if protect else [ctypes.POINTER(_Blob), ctypes.POINTER(wintypes.LPWSTR),
                       ctypes.POINTER(_Blob), wintypes.LPVOID, wintypes.LPVOID,
                       wintypes.DWORD, ctypes.POINTER(_Blob)])
    crypt.restype = wintypes.BOOL
    if protect:
        ok = crypt(ctypes.byref(source), "MNQ Paper Telegram credentials", None, None, None,
                   0x1, ctypes.byref(result))
    else:
        description = wintypes.LPWSTR()
        ok = crypt(ctypes.byref(source), ctypes.byref(description), None, None, None,
                   0x1, ctypes.byref(result))
        if description:
            ctypes.windll.kernel32.LocalFree(description)
    if not ok:
        raise RuntimeError("Windows DPAPI credential operation failed")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(result.pbData)


def save_credentials(token: str, chat_id: str, path: str | Path | None = None) -> Path:
    token, chat_id = token.strip(), chat_id.strip()
    if not token or not chat_id:
        raise ValueError("Telegram bot token and chat ID are required")
    destination = Path(path) if path else credentials_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"token": token, "chat_id": chat_id}, separators=(",", ":")).encode("utf-8")
    envelope = {"schema_version": 1, "protection": "Windows DPAPI CurrentUser",
                "blob": base64.b64encode(_crypt(raw, protect=True)).decode("ascii")}
    fd, name = tempfile.mkstemp(prefix="telegram.credentials.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(envelope, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(name, destination)
    finally:
        if os.path.exists(name): os.unlink(name)
    return destination


def load_credentials(path: str | Path | None = None) -> TelegramCredentials:
    source = Path(path) if path else credentials_path()
    try:
        envelope = json.loads(source.read_text(encoding="utf-8"))
        if envelope.get("schema_version") != 1 or envelope.get("protection") != "Windows DPAPI CurrentUser":
            raise ValueError("unsupported credential-store schema")
        value = json.loads(_crypt(base64.b64decode(envelope["blob"], validate=True), protect=False))
        token, chat_id = str(value["token"]), str(value["chat_id"])
        if not token or not chat_id:
            raise ValueError("empty credential")
        return TelegramCredentials(token, chat_id)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Telegram credential store unavailable or invalid ({type(exc).__name__})") from None


def available_chat_ids(token: str) -> list[tuple[str, str]]:
    """Read Telegram updates and expose only chat identifiers/labels, not messages."""
    request = Request(f"https://api.telegram.org/bot{token}/getUpdates", method="GET")
    try:
        with urlopen(request, timeout=8.0) as response:
            result = json.loads(response.read(256_000))
    except Exception as exc:
        raise RuntimeError(f"Telegram chat lookup failed ({type(exc).__name__}); token redacted") from None
    if not isinstance(result, dict) or not result.get("ok") or not isinstance(result.get("result"), list):
        raise RuntimeError("Telegram chat lookup was rejected; response details redacted")
    chats: dict[str, str] = {}
    for update in result["result"]:
        if not isinstance(update, dict):
            continue
        for key in ("message", "edited_message", "channel_post", "edited_channel_post", "my_chat_member"):
            value = update.get(key)
            chat = value.get("chat") if isinstance(value, dict) else None
            if not isinstance(chat, dict) or chat.get("id") is None:
                continue
            chat_id = str(chat["id"])
            label = str(chat.get("title") or chat.get("username") or chat.get("first_name") or chat.get("type") or "chat")
            chats[chat_id] = label[:80]
    return sorted(chats.items())


def configure_interactively(path: str | Path | None = None) -> Path:
    token = getpass.getpass("Telegram bot token (input hidden): ")
    try:
        chats = available_chat_ids(token)
    except RuntimeError:
        chats = []
        print("Telegram chat lookup unavailable; enter the chat ID manually if known.")
    if chats:
        print("Recent Telegram chats (IDs only; no message contents):")
        for chat_id, label in chats:
            print(f"  {chat_id} · {label}")
    else:
        print("No Telegram chats found. Send /start to your bot, then rerun configure; or enter a known chat ID.")
    chat_id = getpass.getpass("Telegram chat ID (input hidden): ")
    saved = save_credentials(token, chat_id, path)
    token = ""  # drop references as soon as practical
    return saved
