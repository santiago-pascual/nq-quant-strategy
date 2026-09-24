from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4


class PaperEventType(str, Enum):
    MARKET_DATA = "market_data"
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_DECISION = "strategy_decision"

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
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = Lock()
        self._next_sequence = self._discover_next_sequence()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover_next_sequence(self) -> int:
        if not self.path.exists():
            return 0

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

        event = PaperEvent(
            event_id=str(uuid4()),
            sequence=self._next_sequence,
            event_type=event_type,
            timestamp=event_timestamp,
            payload=dict(payload),
        )

        serialized = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._lock:
            # Re-check sequence in case this logger was re-used after
            # another process appended to the same file.
            self._next_sequence = self._discover_next_sequence()

            event = PaperEvent(
                event_id=event.event_id,
                sequence=self._next_sequence,
                event_type=event.event_type,
                timestamp=event.timestamp,
                payload=event.payload,
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

            self._next_sequence += 1

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
