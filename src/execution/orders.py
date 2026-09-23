from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from src.strategies.base import StrategySignal

from .types import (
    ExecutionIntent,
    Order,
    OrderStatus,
    OrderType,
)


def create_market_order(
    intent: ExecutionIntent,
    quantity: int,
) -> Order:
    if quantity <= 0:
        raise ValueError("Order quantity must be positive")

    if intent.signal is StrategySignal.FLAT:
        raise ValueError("Order side cannot be FLAT")

    return Order(
        order_id=str(uuid4()),
        strategy_name=intent.strategy_name,
        side=intent.signal,
        order_type=OrderType.MARKET,
        quantity=quantity,
        created_at=intent.timestamp,
        status=OrderStatus.CREATED,
    )


def create_market_order_direct(
    *,
    order_id: str | None = None,
    strategy_name: str,
    side: StrategySignal,
    quantity: int,
    created_at,
) -> Order:
    """
    Create a market order directly.

    Used by the EXIT path because EXIT intents intentionally
    have signal=FLAT. The engine determines the actual opposite
    execution side from the open position.
    """

    if quantity <= 0:
        raise ValueError("Order quantity must be positive")

    if side is StrategySignal.FLAT:
        raise ValueError("Order side cannot be FLAT")

    return Order(
        order_id=order_id or str(uuid4()),
        strategy_name=strategy_name,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        created_at=created_at,
        status=OrderStatus.CREATED,
    )


def submit_order(order: Order) -> Order:
    if order.status is not OrderStatus.CREATED:
        raise RuntimeError("Only CREATED orders can be submitted")

    return replace(
        order,
        status=OrderStatus.SUBMITTED,
    )


def apply_fill(
    order: Order,
    *,
    fill_quantity: int,
    average_fill_price: float,
) -> Order:
    if order.status not in (
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
    ):
        raise RuntimeError(f"Order {order.order_id} is not fillable")

    if fill_quantity <= 0:
        raise ValueError("Fill quantity must be positive")

    if fill_quantity > order.remaining_quantity:
        raise ValueError("Fill quantity exceeds remaining order quantity")

    if average_fill_price <= 0:
        raise ValueError("average_fill_price must be positive")

    new_quantity = order.filled_quantity + fill_quantity

    if new_quantity == order.quantity:
        status = OrderStatus.FILLED
    else:
        status = OrderStatus.PARTIALLY_FILLED

    return replace(
        order,
        status=status,
        filled_quantity=new_quantity,
        average_fill_price=average_fill_price,
    )


def cancel_order(order: Order) -> Order:
    if order.status not in (
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
    ):
        raise RuntimeError(f"Order {order.order_id} cannot be cancelled")

    return replace(
        order,
        status=OrderStatus.CANCELLED,
    )


def reject_order(order: Order) -> Order:
    if order.status not in (
        OrderStatus.CREATED,
        OrderStatus.SUBMITTED,
    ):
        raise RuntimeError(f"Order {order.order_id} cannot be rejected")

    return replace(
        order,
        status=OrderStatus.REJECTED,
    )
