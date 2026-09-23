from __future__ import annotations

from dataclasses import dataclass

from src.strategies.base import StrategySignal

from .types import Fill, Position, PositionStatus


@dataclass(frozen=True)
class PositionClose:
    strategy_name: str
    side: StrategySignal
    quantity: int
    entry_price: float
    exit_price: float
    entry_timestamp: object
    exit_timestamp: object
    status: PositionStatus = PositionStatus.CLOSED

    @property
    def points(self) -> float:
        if self.side is StrategySignal.LONG:
            return self.exit_price - self.entry_price

        return self.entry_price - self.exit_price


def open_position(
    *,
    strategy_name: str,
    side: StrategySignal,
    quantity: int,
    entry_price: float,
    entry_timestamp,
) -> Position:
    return Position(
        strategy_name=strategy_name,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        entry_timestamp=entry_timestamp,
        status=PositionStatus.OPEN,
    )


def add_position_fill(
    position: Position,
    fill: Fill,
) -> Position:
    if position.status is not PositionStatus.OPEN:
        raise RuntimeError("Cannot add fill to closed position")

    if fill.strategy_name != position.strategy_name:
        raise ValueError("Fill strategy does not match position strategy")

    if fill.side is not position.side:
        raise ValueError("Fill side does not match position side")

    previous_quantity = position.quantity
    previous_average = position.entry_price

    new_quantity = previous_quantity + fill.quantity

    new_average = (
        (previous_average * previous_quantity) + (fill.price * fill.quantity)
    ) / new_quantity

    return Position(
        strategy_name=position.strategy_name,
        side=position.side,
        quantity=new_quantity,
        entry_price=new_average,
        entry_timestamp=position.entry_timestamp,
        status=PositionStatus.OPEN,
    )


def close_position(
    position: Position,
    *,
    exit_price: float,
    exit_timestamp,
) -> PositionClose:
    if position.status is not PositionStatus.OPEN:
        raise RuntimeError("Only OPEN positions can be closed")

    if exit_price <= 0:
        raise ValueError("exit_price must be positive")

    return PositionClose(
        strategy_name=position.strategy_name,
        side=position.side,
        quantity=position.quantity,
        entry_price=position.entry_price,
        exit_price=exit_price,
        entry_timestamp=position.entry_timestamp,
        exit_timestamp=exit_timestamp,
    )
