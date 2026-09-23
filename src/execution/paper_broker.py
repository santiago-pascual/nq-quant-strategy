from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .orders import apply_fill, cancel_order, reject_order
from .types import Fill, Order, OrderStatus


class PaperBroker:
    """
    Deterministic in-memory broker used for execution-engine tests.

    The broker accepts both CREATED orders and orders that have already
    transitioned to SUBMITTED in the execution engine.
    """

    def __init__(self) -> None:
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []

    def submit_order(self, order: Order) -> Order:
        if order.order_id in self._orders:
            raise RuntimeError(f"Order already exists: {order.order_id}")

        if order.status is OrderStatus.CREATED:
            order = replace(
                order,
                status=OrderStatus.SUBMITTED,
            )
        elif order.status is not OrderStatus.SUBMITTED:
            raise RuntimeError(
                f"Order {order.order_id} cannot be submitted "
                f"from status {order.status.value}"
            )

        self._orders[order.order_id] = order
        return order

    def get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise KeyError(f"Unknown order: {order_id}") from exc

    def get_order_status(self, order_id: str) -> OrderStatus:
        return self.get_order(order_id).status

    def get_open_orders(self) -> list[Order]:
        return [
            order
            for order in self._orders.values()
            if order.status
            in {
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
            }
        ]

    def cancel_order(self, order_id: str) -> Order:
        order = self.get_order(order_id)
        updated = cancel_order(order)
        self._orders[order_id] = updated
        return updated

    def reject_order(self, order_id: str) -> Order:
        order = self.get_order(order_id)
        updated = reject_order(order)
        self._orders[order_id] = updated
        return updated

    def inject_fill(self, fill: Fill) -> Order:
        order = self.get_order(fill.order_id)

        if order.status not in {
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise RuntimeError(f"Order {order.order_id} is not fillable")

        if fill.quantity <= 0:
            raise ValueError("Fill quantity must be positive")

        if fill.quantity > order.remaining_quantity:
            raise ValueError(
                "Fill quantity exceeds order quantity; exceeds remaining order quantity"
            )

        if fill.price <= 0:
            raise ValueError("Fill price must be positive")

        previous_quantity = order.filled_quantity
        previous_average = order.average_fill_price or 0.0
        new_quantity = previous_quantity + fill.quantity

        if previous_quantity == 0:
            new_average = fill.price
        else:
            new_average = (
                previous_average * previous_quantity + fill.price * fill.quantity
            ) / new_quantity

        updated = apply_fill(
            order,
            fill_quantity=fill.quantity,
            average_fill_price=new_average,
        )

        self._orders[order.order_id] = updated
        self._fills.append(fill)

        return updated

    def get_fills(self) -> list[Fill]:
        return list(self._fills)
