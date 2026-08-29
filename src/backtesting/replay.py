from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

from src.domain import MarketData


@dataclass(frozen=True)
class MarketEvent:
    """
    A single chronological market observation emitted by the replay engine.
    """

    timestamp: datetime
    data: MarketData


class MarketReplay:
    """
    Deterministic historical market-data replay.

    Converts a finite sequence of MarketData observations into chronological
    MarketEvents.

    The replay engine does not know anything about:
        - strategies
        - risk
        - execution
        - portfolio state
    """

    def __init__(self, market_data: Iterable[MarketData]) -> None:
        self._market_data = tuple(market_data)

        self._validate()

    def _validate(self) -> None:
        """Validate replay input before execution."""

        if not self._market_data:
            raise ValueError("MarketReplay requires at least one observation.")

        for observation in self._market_data:
            if observation.timestamp.tzinfo is None:
                raise ValueError("All MarketData timestamps must be timezone-aware.")

        timestamps = [observation.timestamp for observation in self._market_data]

        if timestamps != sorted(timestamps):
            raise ValueError("MarketReplay requires MarketData sorted chronologically.")

    def __iter__(self) -> Iterator[MarketEvent]:
        """Yield market events in deterministic chronological order."""

        for observation in self._market_data:
            yield MarketEvent(
                timestamp=observation.timestamp,
                data=observation,
            )

    def __len__(self) -> int:
        """Return the number of replay observations."""

        return len(self._market_data)

    @property
    def start_time(self) -> datetime:
        """Return the first replay timestamp."""

        return self._market_data[0].timestamp

    @property
    def end_time(self) -> datetime:
        """Return the final replay timestamp."""

        return self._market_data[-1].timestamp
