from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.strategies.base import StrategyAction, StrategySignal


class OrderType(Enum):
    MARKET = "market"


class OrderStatus(Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(Enum):
    FLAT = "flat"
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class ExecutionIntent:
    strategy_name: str
    strategy_version: str
    signal: StrategySignal
    action: StrategyAction
    timestamp: datetime
    reason: str | None = None

    def __post_init__(self):
        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if not self.strategy_version:
            raise ValueError("strategy_version must not be empty")

        if self.timestamp.tzinfo is None:
            raise ValueError("Execution intent timestamp must be timezone-aware")

        if self.action is StrategyAction.ENTER and self.signal is StrategySignal.FLAT:
            raise ValueError("ENTER intent cannot have FLAT signal")

        if (
            self.action is StrategyAction.EXIT
            and self.signal is not StrategySignal.FLAT
        ):
            raise ValueError("EXIT intent must have FLAT signal")


@dataclass(frozen=True)
class Order:
    order_id: str
    strategy_name: str
    side: StrategySignal
    order_type: OrderType
    quantity: int
    created_at: datetime
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    average_fill_price: float | None = None

    def __post_init__(self):
        if not self.order_id:
            raise ValueError("order_id must not be empty")

        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if self.side not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Order side must be LONG or SHORT")

        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative")

        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed order quantity")

        if self.average_fill_price is not None:
            if self.average_fill_price <= 0:
                raise ValueError("average_fill_price must be positive")

        if self.created_at.tzinfo is None:
            raise ValueError("Order timestamp must be timezone-aware")

        if self.status is OrderStatus.FILLED:
            if self.filled_quantity != self.quantity:
                raise ValueError("FILLED order must have all quantity filled")

            if self.average_fill_price is None:
                raise ValueError("FILLED order requires average_fill_price")

        if self.status is OrderStatus.PARTIALLY_FILLED:
            if not (0 < self.filled_quantity < self.quantity):
                raise ValueError(
                    "PARTIALLY_FILLED order must have partial quantity filled"
                )

            if self.average_fill_price is None:
                raise ValueError("PARTIALLY_FILLED order requires average_fill_price")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    strategy_name: str
    side: StrategySignal
    quantity: int
    price: float
    timestamp: datetime

    def __post_init__(self):
        if not self.fill_id:
            raise ValueError("fill_id must not be empty")

        if not self.order_id:
            raise ValueError("order_id must not be empty")

        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if self.side not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Fill side must be LONG or SHORT")

        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive")

        if self.price <= 0:
            raise ValueError("Fill price must be positive")

        if self.timestamp.tzinfo is None:
            raise ValueError("Fill timestamp must be timezone-aware")


@dataclass
class Position:
    strategy_name: str
    side: StrategySignal
    quantity: int
    entry_price: float
    entry_timestamp: datetime
    status: PositionStatus = PositionStatus.OPEN

    def __post_init__(self):
        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if self.side not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Position side must be LONG or SHORT")

        if self.quantity <= 0:
            raise ValueError("Position quantity must be positive")

        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")

        if self.entry_timestamp.tzinfo is None:
            raise ValueError("Position timestamp must be timezone-aware")
