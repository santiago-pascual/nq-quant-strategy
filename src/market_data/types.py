from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketDataBar:
    """
    Immutable normalized market-data bar.

    All timestamps must be timezone-aware.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")

        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        if self.open <= 0:
            raise ValueError("open must be > 0")

        if self.high <= 0:
            raise ValueError("high must be > 0")

        if self.low <= 0:
            raise ValueError("low must be > 0")

        if self.close <= 0:
            raise ValueError("close must be > 0")

        if self.volume < 0:
            raise ValueError("volume must be >= 0")

        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open and close")

        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open and close")
