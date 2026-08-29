from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PositionSide(Enum):
    """Current directional side of a position."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Position:
    """
    Immutable representation of an open position.

    Position contains portfolio state only.
    Strategy-specific behavior such as MAE, recovery, or exit rules
    belongs to the strategy layer.
    """

    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_timestamp: object

    @property
    def is_long(self) -> bool:
        """Return True when the position is long."""

        return self.side is PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Return True when the position is short."""

        return self.side is PositionSide.SHORT
