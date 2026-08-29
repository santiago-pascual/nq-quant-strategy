from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class Clock(ABC):
    """
    Abstract time source.

    All system components that require the current time should depend on
    this interface rather than directly calling datetime.now().
    """

    @abstractmethod
    def now(self) -> datetime:
        """Return the current timezone-aware timestamp."""


class SystemClock(Clock):
    """Real wall-clock implementation used by paper/live environments."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock(Clock):
    """
    Deterministic clock for tests.

    The returned timestamp remains constant until explicitly changed.
    """

    def __init__(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime.")

        self._timestamp = timestamp

    def now(self) -> datetime:
        return self._timestamp

    def set(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime.")

        self._timestamp = timestamp
