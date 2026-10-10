"""Append-only versions and as-of visibility for IBKR bar observations.

This ledger is a diagnostic boundary only. It is deliberately not imported by
the Paper execution engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping


def _utc_iso(value: str | datetime) -> str:
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat()


def bar_value_hash(bar: Mapping[str, Any]) -> str:
    payload = {field: float(bar[field]) for field in ("open", "high", "low", "close", "volume")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AppendOnlyBarObservationLedger:
    """Durably append every bar observation, including unchanged poll repeats."""

    def __init__(self, path: str | Path, *, run_id: str, source: str = "IBKR_TWS_HISTORICAL",
                 provenance: str = "TWS_DEMO_UNVERIFIED", requested_market_data_type: int = 3) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.source = source
        self.provenance = provenance
        self.requested_market_data_type = requested_market_data_type
        self._lock = threading.Lock()
        self._latest: dict[tuple[int, int], tuple[str, int]] = {}
        self._sequence = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                prior = json.loads(line)
                key = (int(prior["contract_id"]), int(prior["bar_start_epoch_utc"]))
                version = int(prior["value_version"])
                self._latest[key] = (prior["bar_value_sha256"], version)
                self._sequence = max(self._sequence, int(prior["observation_sequence"]))

    def append(self, bar: Mapping[str, Any], *, contract: Mapping[str, Any],
               observed_at: str | datetime, request_id: int, poll_number: int,
               eligible_for_finalization: bool = True) -> dict[str, Any]:
        observed_utc = _utc_iso(observed_at)
        con_id = int(contract["con_id"])
        start = int(bar["timestamp"])
        digest = bar_value_hash(bar)
        key = (con_id, start)
        with self._lock:
            previous = self._latest.get(key)
            is_revision = previous is not None and previous[0] != digest
            version = (previous[1] + 1) if is_revision else (previous[1] if previous else 1)
            self._sequence += 1
            record = {
                "observation_sequence": self._sequence,
                "run_id": self.run_id,
                "source": self.source,
                "provenance": self.provenance,
                "requested_market_data_type": self.requested_market_data_type,
                "request_id": int(request_id),
                "poll_number": int(poll_number),
                "contract_id": con_id,
                "local_symbol": str(contract.get("local_symbol", "")),
                "expiry": str(contract.get("expiry", "")),
                "bar_start_epoch_utc": start,
                "bar_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                "exchange_bar_end_utc": datetime.fromtimestamp(start + 60, timezone.utc).isoformat(),
                "observed_at_utc": observed_utc,
                "bar_complete_when_observed": datetime.fromisoformat(observed_utc).timestamp() >= start + 60,
                "eligible_for_finalization": bool(eligible_for_finalization),
                "ohlcv": {field: float(bar[field]) for field in ("open", "high", "low", "close", "volume")},
                "bar_value_sha256": digest,
                "value_version": version,
                "is_revision": is_revision,
                "previous_value_sha256": previous[0] if is_revision else None,
            }
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._latest[key] = (digest, version)
            return record


def visible_bar_version(records: Iterable[Mapping[str, Any]], *, contract_id: int,
                        bar_start_epoch_utc: int, as_of: str | datetime) -> Mapping[str, Any] | None:
    """Return latest observed value known at `as_of`; never leak later revisions."""
    cutoff = datetime.fromisoformat(_utc_iso(as_of))
    eligible = []
    for record in records:
        if int(record["contract_id"]) != int(contract_id) or int(record["bar_start_epoch_utc"]) != int(bar_start_epoch_utc):
            continue
        seen = datetime.fromisoformat(_utc_iso(record["observed_at_utc"]))
        if seen <= cutoff and bool(record.get("bar_complete_when_observed", False)):
            eligible.append((seen, int(record["observation_sequence"]), record))
    return max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None


def load_observation_ledger(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
