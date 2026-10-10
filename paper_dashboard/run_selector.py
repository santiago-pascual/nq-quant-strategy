"""Read-only discovery and classification of persisted Paper run directories."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunOption:
    path: Path
    kind: str
    status: str
    last_modified: float

    @property
    def label(self) -> str:
        return f"{self.kind.replace('_', ' ')} · {self.status.replace('_', ' ')} · {self.path.name}"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def classify_run(path: Path) -> RunOption | None:
    """Classify a directory using only its own persisted manifest/status."""
    path = path.resolve()
    status_data = _read_json(path / "status.json")
    scope = _read_json(path / "oos_run_scope.json")
    summary = _read_json(path / "run_summary.json")
    delayed_manifest = _read_json(path / "delayed_paper_run.json")
    if not any((status_data, scope, summary, delayed_manifest)):
        return None

    system = (status_data or {}).get("system") or {}
    feed = system.get("feed_health") or {}
    source = str(feed.get("source") or (status_data or {}).get("source") or "")
    state = str(system.get("state") or (status_data or {}).get("status") or "UNAVAILABLE").upper()
    if scope is not None or source == "deterministic_replay" or summary is not None:
        kind = "HISTORICAL_REPLAY"
    elif delayed_manifest is not None or source.startswith("ibkr_delayed"):
        kind = "DELAYED_PAPER"
    elif status_data is not None:
        kind = "PAPER_SESSION"
    else:
        kind = "UNAVAILABLE"
    try:
        modified = (path / "status.json").stat().st_mtime
    except OSError:
        modified = path.stat().st_mtime if path.exists() else 0.0
    return RunOption(path=path, kind=kind, status=state, last_modified=modified)


def discover_runs(root: Path, *, preferred: Path | None = None) -> list[RunOption]:
    """Discover isolated runs without reading their analytics databases."""
    options: list[RunOption] = []
    try:
        children = list(root.iterdir())
    except OSError:
        children = []
    if preferred is not None:
        preferred = preferred.resolve()
        if preferred.is_dir() and all(p.path != preferred for p in options):
            option = classify_run(preferred)
            if option:
                options.append(option)
    for child in children:
        try:
            if not child.is_dir():
                continue
            option = classify_run(child)
        except OSError:
            continue
        if option and all(existing.path != option.path for existing in options):
            options.append(option)
    return sorted(options, key=lambda option: (option.last_modified, option.path.name), reverse=True)


def default_run(options: list[RunOption], preferred: Path | None = None) -> RunOption | None:
    """Prefer an active delayed Paper session over a stale replay default."""
    if preferred is not None:
        target = preferred.resolve()
        exact = next((item for item in options if item.path == target), None)
        if exact is not None and exact.kind == "DELAYED_PAPER":
            return exact
    active_delayed = [item for item in options if item.kind == "DELAYED_PAPER" and item.status in {"RUNNING", "RECOVERING"}]
    if active_delayed:
        return max(active_delayed, key=lambda item: (item.last_modified, item.path.name))
    if preferred is not None:
        target = preferred.resolve()
        exact = next((item for item in options if item.path == target), None)
        if exact is not None:
            return exact
    return options[0] if options else None
