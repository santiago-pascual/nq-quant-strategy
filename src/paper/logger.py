from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping
from uuid import uuid4
from hashlib import sha256


class PaperEventType(str, Enum):
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_RECOVERED = "system_recovered"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    SYSTEM_CRITICAL = "system_critical"
    FEED_CONNECTED = "feed_connected"
    FEED_DISCONNECTED = "feed_disconnected"
    FEED_STALE = "feed_stale"
    FEED_RECOVERED = "feed_recovered"
    CONTRACT_ROLL = "contract_roll"
    MISSING_BAR = "missing_bar"
    DUPLICATE_BAR = "duplicate_bar"
    OUT_OF_ORDER_BAR = "out_of_order_bar"
    BACKFILL_STARTED = "backfill_started"
    BACKFILL_COMPLETED = "backfill_completed"
    BACKFILL_FAILED = "backfill_failed"
    HMM_STATE = "hmm_state"
    HMM_REFIT_STARTED = "hmm_refit_started"
    HMM_REFIT_COMPLETED = "hmm_refit_completed"
    HMM_REFIT_FAILED = "hmm_refit_failed"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_REJECTED = "candidate_rejected"
    PAPER_ORDER_CREATED = "paper_order_created"
    PAPER_ORDER_FILLED = "paper_order_filled"
    PAPER_POSITION_OPENED = "paper_position_opened"
    PAPER_POSITION_CLOSED = "paper_position_closed"
    RISK_REJECTION = "risk_rejection"
    RISK_WARNING = "risk_warning"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    CHECKPOINT_FAILED = "checkpoint_failed"
    PARITY_WARNING = "parity_warning"
    MARKET_DATA = "market_data"
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_DECISION = "strategy_decision"
    CANDIDATE_EVALUATION = "candidate_evaluation"

    RISK_REQUEST = "risk_request"
    RISK_DECISION = "risk_decision"

    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    FILL = "fill"

    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"

    ERROR = "error"


@dataclass(frozen=True)
class PaperEvent:
    """
    Immutable event stored by the paper-trading event log.

    Every event has:
        event_id
        sequence
        event_type
        timestamp
        payload
    """

    event_id: str
    sequence: int
    event_type: PaperEventType
    timestamp: datetime
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")

        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")

        if not isinstance(self.event_type, PaperEventType):
            raise TypeError("event_type must be a PaperEventType")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "PaperEvent":

        required = {
            "event_id",
            "sequence",
            "event_type",
            "timestamp",
            "payload",
        }

        missing = required - set(data)

        if missing:
            raise ValueError(f"Missing event fields: {sorted(missing)}")

        timestamp = datetime.fromisoformat(str(data["timestamp"]))

        if timestamp.tzinfo is None:
            raise ValueError("Stored event timestamp must be timezone-aware")

        return cls(
            event_id=str(data["event_id"]),
            sequence=int(data["sequence"]),
            event_type=PaperEventType(str(data["event_type"])),
            timestamp=timestamp.astimezone(timezone.utc),
            payload=dict(data["payload"]),
        )


class PaperEventLogger:
    """
    Append-only JSONL event logger for paper trading.

    Properties:
    - deterministic ordering
    - monotonic sequence numbers
    - UTC timestamps
    - one JSON object per line
    - thread-safe writes
    - replayable events
    """

    def __init__(
        self,
        path: str | Path,
        *,
        recover_trailing_partial: bool = False,
    ) -> None:

        self.path = Path(path)
        self.recover_trailing_partial = bool(recover_trailing_partial)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = Lock()
        self._listeners: list[Callable[[PaperEvent], None]] = []
        self._event_context: dict[str, Any] = {}
        self._idempotency_scope: str | None = None
        self._known_idempotent_events: dict[str, tuple[str, int, datetime]] = {}
        self._next_sequence = self._discover_next_sequence()

    def set_event_context(self, **fields: Any) -> None:
        """Set stable envelope fields applied to subsequent events.

        The realtime service uses this to attach its run and symbol identity
        to events emitted internally by the Paper engine as well as its own
        lifecycle events.
        """
        with self._lock:
            self._event_context.update(fields)

    def set_idempotency_scope(self, scope: str | None) -> None:
        """Set the current completed-bar transaction key for durable deduplication."""
        with self._lock:
            self._idempotency_scope = scope

    def subscribe(self, listener: Callable[[PaperEvent], None]) -> None:
        """Register an in-process observer for monitoring/status updates."""
        with self._lock:
            self._listeners.append(listener)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover_next_sequence(self) -> int:
        if not self.path.exists():
            return 0

        if self.recover_trailing_partial:
            with self.path.open("r+b") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                if end:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        scan_end = end
                        cut = 0
                        while scan_end > 0:
                            start = max(0, scan_end - 8192)
                            handle.seek(start)
                            block = handle.read(scan_end - start)
                            newline = block.rfind(b"\n")
                            if newline >= 0:
                                cut = start + newline + 1
                                break
                            scan_end = start
                        handle.seek(cut)
                        tail = handle.read(end - cut)
                        try:
                            json.loads(tail.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            handle.truncate(cut)
                            handle.flush()
                            os.fsync(handle.fileno())
                        else:
                            handle.seek(0, os.SEEK_END)
                            handle.write(b"\n")
                            handle.flush()
                            os.fsync(handle.fileno())

        last_sequence = -1

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sequence = int(data["sequence"])
                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValueError(
                        f"Invalid event log at line {line_number}"
                    ) from exc

                if sequence <= last_sequence:
                    raise ValueError("Event log sequence is not strictly increasing")
                key = data.get("payload", {}).get("_idempotency_key")
                if key:
                    self._known_idempotent_events[str(key)] = (
                        str(data["event_id"]), sequence,
                        datetime.fromisoformat(str(data["timestamp"])).astimezone(timezone.utc),
                    )

                last_sequence = sequence

        return last_sequence + 1

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: PaperEventType,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime | None = None,
    ) -> PaperEvent:

        if not isinstance(event_type, PaperEventType):
            raise TypeError("event_type must be a PaperEventType")

        event_timestamp = self._utc_now() if timestamp is None else timestamp

        if event_timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        event_timestamp = event_timestamp.astimezone(timezone.utc)

        with self._lock:
            event_payload = {**self._event_context, **dict(payload)}
            idempotency_key = None
            if self._idempotency_scope is not None:
                identity = json.dumps(
                    [self._idempotency_scope, event_type.value, event_payload],
                    ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
                )
                idempotency_key = sha256(identity.encode()).hexdigest()
                prior = self._known_idempotent_events.get(idempotency_key)
                if prior is not None:
                    listeners = tuple(self._listeners)
                    event_payload["_idempotency_key"] = idempotency_key
                    event = PaperEvent(
                        event_id=prior[0], sequence=prior[1],
                        event_type=event_type, timestamp=prior[2],
                        payload=event_payload,
                    )
                    for_listener_only = True
                else:
                    event_payload["_idempotency_key"] = idempotency_key
                    for_listener_only = False
            else:
                for_listener_only = False
            if for_listener_only:
                pass
            else:
                event = PaperEvent(
                    event_id=str(uuid4()),
                    sequence=self._next_sequence,
                    event_type=event_type,
                    timestamp=event_timestamp,
                    payload=event_payload,
                )

                serialized = json.dumps(
                    event.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )

                with self.path.open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(serialized)
                    handle.write("\n")
                    handle.flush()
                    if event.event_type in {
                        PaperEventType.POSITION_OPENED,
                        PaperEventType.POSITION_CLOSED,
                        PaperEventType.ORDER_SUBMITTED,
                        PaperEventType.FILL,
                        PaperEventType.SYSTEM_STOPPED,
                        PaperEventType.CHECKPOINT_SAVED,
                    }:
                        os.fsync(handle.fileno())

                self._next_sequence += 1
                if idempotency_key is not None:
                    self._known_idempotent_events[idempotency_key] = (
                        event.event_id, event.sequence, event.timestamp
                    )
                listeners = tuple(self._listeners)

        for listener in listeners:
            listener(event)

        return event

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def log_market_data(
        self,
        market_data: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.MARKET_DATA,
            market_data,
            timestamp=timestamp,
        )

    def log_strategy_signal(
        self,
        *,
        strategy_name: str,
        signal: str,
        timestamp: datetime,
        reason: str | None = None,
    ) -> PaperEvent:

        payload: dict[str, Any] = {
            "strategy_name": strategy_name,
            "signal": signal,
        }

        if reason is not None:
            payload["reason"] = reason

        return self.append(
            PaperEventType.STRATEGY_SIGNAL,
            payload,
            timestamp=timestamp,
        )

    def log_strategy_decision(
        self,
        *,
        strategy_name: str,
        action: str,
        signal: str,
        timestamp: datetime,
        reason: str | None = None,
    ) -> PaperEvent:

        payload: dict[str, Any] = {
            "strategy_name": strategy_name,
            "action": action,
            "signal": signal,
        }

        if reason is not None:
            payload["reason"] = reason

        return self.append(
            PaperEventType.STRATEGY_DECISION,
            payload,
            timestamp=timestamp,
        )

    def log_risk_request(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.RISK_REQUEST,
            payload,
            timestamp=timestamp,
        )

    def log_risk_decision(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.RISK_DECISION,
            payload,
            timestamp=timestamp,
        )

    def log_order_created(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.ORDER_CREATED,
            payload,
            timestamp=timestamp,
        )

    def log_order_submitted(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.ORDER_SUBMITTED,
            payload,
            timestamp=timestamp,
        )

    def log_fill(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.FILL,
            payload,
            timestamp=timestamp,
        )

    def log_position_opened(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.POSITION_OPENED,
            payload,
            timestamp=timestamp,
        )

    def log_position_updated(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.POSITION_UPDATED,
            payload,
            timestamp=timestamp,
        )

    def log_position_closed(
        self,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime,
    ) -> PaperEvent:

        return self.append(
            PaperEventType.POSITION_CLOSED,
            payload,
            timestamp=timestamp,
        )

    def log_error(
        self,
        error: Exception | str,
        *,
        timestamp: datetime,
        context: Mapping[str, Any] | None = None,
    ) -> PaperEvent:

        payload: dict[str, Any] = {
            "error_type": (
                type(error).__name__ if isinstance(error, Exception) else "Error"
            ),
            "message": str(error),
        }

        if context is not None:
            payload["context"] = dict(context)

        return self.append(
            PaperEventType.ERROR,
            payload,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Reading / replay
    # ------------------------------------------------------------------

    def read_all(self) -> list[PaperEvent]:
        if not self.path.exists():
            return []

        events: list[PaperEvent] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                    event = PaperEvent.from_dict(data)
                except Exception as exc:
                    raise ValueError(f"Invalid event at line {line_number}") from exc

                events.append(event)

        return events

    def count(self) -> int:
        return len(self.read_all())

    def last_event(self) -> PaperEvent | None:
        events = self.read_all()

        if not events:
            return None

        return events[-1]

    def clear(self) -> None:
        """
        Explicit test/development helper.

        Production code should not use this method because the paper
        trading log is intended to be append-only.
        """

        with self._lock:
            self.path.unlink(missing_ok=True)
            self._next_sequence = 0
