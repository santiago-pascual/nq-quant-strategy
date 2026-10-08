"""CME equity-index session schedule interface for MNQ Paper.

The frozen strategy RTH remains 09:30--16:00 America/New_York on a regular
session. CME holiday changes are supplied as a versioned, source-attributed
snapshot; this module deliberately does not infer special-session times.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


NEW_YORK = ZoneInfo("America/New_York")
CME_HOLIDAY_SOURCE = "https://www.cmegroup.com/trading-hours.html"
CME_MNQ_HOURS_SOURCE = (
    "https://www.cmegroup.com/trading/equity-index/files/cme-micro-e-mini-futures-fact-card.pdf"
)


class CalendarUnavailable(RuntimeError):
    """Raised when no authoritative schedule covers the requested date."""


@dataclass(frozen=True)
class CMETradingSession:
    trading_date: date
    session_type: str
    rth_start: datetime | None
    rth_end: datetime | None
    globex_open: datetime | None
    globex_close: datetime | None
    calendar_source: str
    calendar_version: str

    @property
    def final_rth_bar(self) -> datetime | None:
        """Timestamp of the last eligible one-minute bar (bar-start convention)."""
        if self.rth_end is None:
            return None
        return self.rth_end - timedelta(minutes=1)


@dataclass(frozen=True)
class CMECalendarSnapshot:
    """A reviewed CME schedule snapshot with explicit date coverage.

    ``exceptions`` maps ISO trading dates to ``closed``, ``early_close`` or
    ``special`` records. For an early close, ``rth_end`` is required. Special
    Globex opens/closes must be supplied when they differ from the regular
    template. Every snapshot must cite a CME source and version.
    """

    version: str
    source: str
    coverage_start: date
    coverage_end: date
    exceptions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    globex_open: time = time(18, 0)
    globex_close: time = time(17, 0)
    maintenance_start: time = time(17, 0)
    maintenance_end: time = time(18, 0)

    def __post_init__(self) -> None:
        if not self.version.strip() or not self.source.startswith("https://"):
            raise ValueError("CME calendar snapshot requires a version and HTTPS source")
        if self.coverage_end < self.coverage_start:
            raise ValueError("calendar coverage end precedes start")
        for key, row in self.exceptions.items():
            date.fromisoformat(key)
            kind = row.get("session_type")
            if kind not in {"closed", "early_close", "special"}:
                raise ValueError(f"invalid CME session_type for {key}: {kind!r}")
            if kind == "early_close" and not row.get("rth_end"):
                raise ValueError(f"CME early-close record {key} must include rth_end")
            if kind == "early_close" and not row.get("globex_close"):
                raise ValueError(f"CME early-close record {key} must include globex_close")
            if kind == "special" and not all(
                row.get(name) for name in
                ("rth_start", "rth_end", "globex_open", "globex_close")
            ):
                raise ValueError(f"CME special-session record {key} must give all session boundaries")

    @classmethod
    def from_json(cls, path: str | Path) -> "CMECalendarSnapshot":
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(body)

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "CMECalendarSnapshot":
        body = dict(body)
        for key in ("coverage_start", "coverage_end"):
            body[key] = date.fromisoformat(body[key])
        for key in ("rth_start", "rth_end", "globex_open", "globex_close",
                    "maintenance_start", "maintenance_end"):
            if key in body:
                body[key] = time.fromisoformat(body[key])
        return cls(**body)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version, "source": self.source,
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "exceptions": {key: dict(value) for key, value in self.exceptions.items()},
            "rth_start": self.rth_start.isoformat(), "rth_end": self.rth_end.isoformat(),
            "globex_open": self.globex_open.isoformat(), "globex_close": self.globex_close.isoformat(),
            "maintenance_start": self.maintenance_start.isoformat(),
            "maintenance_end": self.maintenance_end.isoformat(),
        }

    @property
    def identity(self) -> str:
        encoded = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode()).hexdigest()

    def session_for_rth_date(self, session_date: date) -> CMETradingSession:
        if not self.coverage_start <= session_date <= self.coverage_end:
            raise CalendarUnavailable(
                f"CME schedule {self.version} does not cover {session_date}; "
                "refusing to assume a regular or holiday session"
            )
        exception = self.exceptions.get(session_date.isoformat())
        if session_date.weekday() >= 5:
            return self._closed(session_date, "weekend")
        if exception:
            kind = str(exception["session_type"])
            if kind == "closed":
                return self._closed(session_date, "holiday_closed")
            rth_start = _local(session_date, exception.get("rth_start", self.rth_start))
            rth_end = _local(session_date, exception["rth_end"] if kind == "early_close" else exception.get("rth_end", self.rth_end))
            globex_open_value = exception.get("globex_open", self.globex_open)
            globex_open_day = session_date - timedelta(days=1)
            globex_open = _local(globex_open_day, globex_open_value)
            globex_close = _local(session_date, exception.get("globex_close", self.globex_close))
            return CMETradingSession(
                session_date, kind, rth_start, rth_end, globex_open, globex_close,
                self.source, self.version,
            )
        return CMETradingSession(
            session_date, "regular", _local(session_date, self.rth_start),
            _local(session_date, self.rth_end),
            _local(session_date - timedelta(days=1), self.globex_open),
            _local(session_date, self.globex_close), self.source, self.version,
        )

    def _closed(self, session_date: date, kind: str) -> CMETradingSession:
        return CMETradingSession(
            session_date, kind, None, None, None, None, self.source, self.version,
        )


class CMETradingCalendar:
    """Resolve CME sessions without changing the strategy's RTH definition."""

    def __init__(self, snapshot: CMECalendarSnapshot) -> None:
        self.snapshot = snapshot

    @property
    def version(self) -> str:
        return self.snapshot.version

    def session_for_timestamp(self, timestamp: datetime | pd.Timestamp) -> CMETradingSession:
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            raise ValueError("calendar timestamp must be timezone-aware")
        return self.snapshot.session_for_rth_date(stamp.tz_convert(NEW_YORK).date())

    def is_rth(self, timestamp: datetime | pd.Timestamp) -> bool:
        session = self.session_for_timestamp(timestamp)
        stamp = pd.Timestamp(timestamp).tz_convert(NEW_YORK)
        return bool(session.rth_start and session.rth_end and session.rth_start <= stamp < session.rth_end)

    def is_final_rth_bar(self, timestamp: datetime | pd.Timestamp) -> bool:
        session = self.session_for_timestamp(timestamp)
        stamp = pd.Timestamp(timestamp).tz_convert(NEW_YORK)
        return session.final_rth_bar == stamp

    def trading_date_for_timestamp(self, timestamp: datetime | pd.Timestamp) -> date | None:
        """CME Globex trade date, or ``None`` during a scheduled closure."""
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            raise ValueError("calendar timestamp must be timezone-aware")
        local = stamp.tz_convert(NEW_YORK)
        day = local.date()
        minute = local.hour * 60 + local.minute
        weekday = local.weekday()
        if weekday == 5 or (weekday == 6 and minute < 18 * 60):
            return None
        if weekday == 4 and minute >= 17 * 60:
            return None
        if 17 * 60 <= minute < 18 * 60:
            return None
        if minute >= 18 * 60:
            day += timedelta(days=1)
            while day.weekday() >= 5:
                day += timedelta(days=1)
        session = self.snapshot.session_for_rth_date(day)
        if not session.globex_open or not session.globex_close:
            return None
        if not session.globex_open <= local.to_pydatetime() < session.globex_close:
            return None
        exception = self.snapshot.exceptions.get(day.isoformat(), {})
        for start_value, end_value in exception.get("closed_intervals", []):
            start_minute = time.fromisoformat(start_value).hour * 60 + time.fromisoformat(start_value).minute
            end_minute = time.fromisoformat(end_value).hour * 60 + time.fromisoformat(end_value).minute
            if start_minute <= minute < end_minute:
                return None
        return day

    def expected_globex_minute(self, timestamp: datetime | pd.Timestamp) -> bool:
        """Whether a timestamp is inside its snapshotted Globex trading date."""
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            raise ValueError("calendar timestamp must be timezone-aware")
        local = stamp.tz_convert(NEW_YORK)
        return self.trading_date_for_timestamp(local) is not None

    def expected_missing_minutes(
        self, after_timestamp: datetime | pd.Timestamp,
        before_timestamp: datetime | pd.Timestamp,
    ) -> list[pd.Timestamp]:
        after = pd.Timestamp(after_timestamp).tz_convert("UTC")
        before = pd.Timestamp(before_timestamp).tz_convert("UTC")
        if before <= after:
            raise ValueError("before_timestamp must be after after_timestamp")
        missing: list[pd.Timestamp] = []
        cursor = after.floor("min") + pd.Timedelta(minutes=1)
        while cursor < before:
            if self.expected_globex_minute(cursor):
                missing.append(cursor)
            cursor += pd.Timedelta(minutes=1)
        return missing


def _local(day: date, value: time | str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=NEW_YORK) if value.tzinfo is None else value.astimezone(NEW_YORK)
    if isinstance(value, str) and "T" in value:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError("special-session datetime boundaries must have a timezone")
        return stamp.tz_convert(NEW_YORK).to_pydatetime()
    clock = time.fromisoformat(value) if isinstance(value, str) else value
    return datetime.combine(day, clock, tzinfo=NEW_YORK)
