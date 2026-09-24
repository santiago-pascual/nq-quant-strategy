from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.strategies.base import StrategySignal


class BrokerOrderStatus(Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class BrokerOrderRequest:
    strategy_name: str
    strategy_version: str
    signal: StrategySignal
    quantity: int
    order_type: str = "MARKET"

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if not self.strategy_version:
            raise ValueError("strategy_version must not be empty")

        if self.signal not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Broker order signal must be LONG or SHORT")

        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")

        if not self.order_type:
            raise ValueError("order_type must not be empty")


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    request: BrokerOrderRequest
    status: BrokerOrderStatus


@dataclass(frozen=True)
class BrokerFill:
    broker_order_id: str
    fill_id: str
    quantity: int
    price: float
    signal: StrategySignal

    def __post_init__(self) -> None:
        if not self.broker_order_id:
            raise ValueError("broker_order_id must not be empty")

        if not self.fill_id:
            raise ValueError("fill_id must not be empty")

        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")

        if self.price <= 0:
            raise ValueError("price must be > 0")

        if self.signal not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Broker fill signal must be LONG or SHORT")


@dataclass(frozen=True)
class BrokerPosition:
    strategy_name: str
    symbol: str
    signal: StrategySignal
    quantity: int
    average_price: float

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")

        if not self.symbol:
            raise ValueError("symbol must not be empty")

        if self.signal not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Position signal must be LONG or SHORT")

        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")

        if self.average_price <= 0:
            raise ValueError("average_price must be > 0")
