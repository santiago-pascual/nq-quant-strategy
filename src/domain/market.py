from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketData:
    """
    Immutable market observation.

    Represents the information available to the strategy at a single
    decision point.

    No future information should be included in this object.
    """

    timestamp: datetime

    symbol: str

    open: float
    high: float
    low: float
    close: float

    volume: float | None = None

    timeframe: str | None = None

    bid: float | None = None
    ask: float | None = None

    @property
    def midpoint(self) -> float | None:
        """Return bid/ask midpoint when both sides are available."""

        if self.bid is None or self.ask is None:
            return None

        return (self.bid + self.ask) / 2.0
