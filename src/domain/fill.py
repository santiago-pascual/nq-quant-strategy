from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.domain.order import OrderSide


@dataclass(frozen=True)
class Fill:
    """
    Immutable execution result.

    A Fill represents an executed quantity at an actual execution price.
    It does not contain broker-specific behavior.
    """

    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime

    order_id: str | None = None

    commission: float = 0.0
    slippage: float = 0.0
