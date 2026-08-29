from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    """Directional side of an order."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    """Lifecycle status of an order."""

    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Order:
    """
    Immutable order instruction.

    Order describes what should be executed. It does not perform execution
    and contains no broker-specific behavior.
    """

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET

    limit_price: float | None = None
    stop_price: float | None = None

    client_order_id: str | None = None
