from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from .orders import (
    apply_fill,
    cancel_order,
    create_market_order,
    create_market_order_direct,
    reject_order,
    submit_order,
)
from .positions import PositionClose, add_position_fill, close_position, open_position
from .types import ExecutionIntent, Fill, Order, OrderStatus, Position, PositionStatus
from src.strategies.base import StrategyAction, StrategyDecision, StrategySignal


class ExecutionEngine:
    """Deterministic execution engine for strategy orders and positions."""

    def __init__(self, broker=None) -> None:
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._closed_positions: Dict[str, PositionClose] = {}
        self._processed_intents: set[ExecutionIntent] = set()
        self._order_intents: Dict[str, ExecutionIntent] = {}
        self._processed_fill_ids: set[str] = set()
        self._broker = broker

    @property
    def broker(self):
        return self._broker

    def attach_broker(self, broker) -> None:
        self._broker = broker

    def build_intent(
        self,
        strategy_name: str,
        strategy_version: str,
        decision: StrategyDecision,
        timestamp: datetime,
    ) -> ExecutionIntent:
        if (
            decision.action is StrategyAction.ENTER
            and decision.signal is StrategySignal.FLAT
        ):
            raise ValueError("ENTER decision cannot have FLAT signal")
        if (
            decision.action is StrategyAction.EXIT
            and decision.signal is not StrategySignal.FLAT
        ):
            raise ValueError("EXIT decision must have FLAT signal")
        return ExecutionIntent(
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            signal=decision.signal,
            action=decision.action,
            reason=decision.reason,
            timestamp=timestamp,
        )

    def _register_intent(self, intent: ExecutionIntent) -> None:
        if intent in self._processed_intents:
            raise RuntimeError("Duplicate execution intent")
        self._processed_intents.add(intent)

    def _release_intent(self, intent: ExecutionIntent) -> None:
        self._processed_intents.discard(intent)

    def submit_entry(self, intent: ExecutionIntent, quantity: int) -> Order:
        if intent.action is not StrategyAction.ENTER:
            raise ValueError("submit_entry requires an ENTER intent")
        if intent.signal is StrategySignal.FLAT:
            raise ValueError("submit_entry requires LONG or SHORT signal")
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.has_open_position(intent.strategy_name):
            raise RuntimeError(
                f"Strategy {intent.strategy_name} already has an open position"
            )
        self._register_intent(intent)
        try:
            order = submit_order(create_market_order(intent=intent, quantity=quantity))
            self._orders[order.order_id] = order
            self._order_intents[order.order_id] = intent
            if self._broker is not None:
                order = self._broker.submit_order(order)
                self._orders[order.order_id] = order
            return order
        except Exception:
            self._release_intent(intent)
            raise

    def submit_exit(self, intent: ExecutionIntent) -> Order:
        if intent.action is not StrategyAction.EXIT:
            raise ValueError("submit_exit requires an EXIT intent")
        if intent.signal is not StrategySignal.FLAT:
            raise ValueError("EXIT intent must use FLAT signal")
        position = self.get_position(intent.strategy_name)
        if position is None or position.status is not PositionStatus.OPEN:
            raise RuntimeError(f"Strategy {intent.strategy_name} has no open position")
        self._register_intent(intent)
        try:
            if position.side is StrategySignal.LONG:
                exit_side = StrategySignal.SHORT
            elif position.side is StrategySignal.SHORT:
                exit_side = StrategySignal.LONG
            else:
                raise ValueError("Cannot create exit order for FLAT position")
            order = create_market_order_direct(
                strategy_name=intent.strategy_name,
                side=exit_side,
                quantity=position.quantity,
                created_at=intent.timestamp,
            )
            order = submit_order(order)
            self._orders[order.order_id] = order
            self._order_intents[order.order_id] = intent
            if self._broker is not None:
                order = self._broker.submit_order(order)
                self._orders[order.order_id] = order
            return order
        except Exception:
            self._release_intent(intent)
            raise

    def process_entry_fill(
        self,
        order_id: str,
        fill_price: float,
        timestamp: datetime,
        fill_quantity: Optional[int] = None,
    ) -> Fill:
        order = self.get_order(order_id)
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
            raise RuntimeError(f"Order {order.order_id} is not fillable")
        quantity = order.remaining_quantity if fill_quantity is None else fill_quantity
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if quantity > order.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining order quantity")
        fill = Fill(
            fill_id=str(uuid4()),
            order_id=order.order_id,
            strategy_name=order.strategy_name,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            timestamp=timestamp,
        )
        return self._process_entry_fill_object(fill)

    def process_exit_fill(
        self,
        order_id: str,
        fill_price: float,
        timestamp: datetime,
        fill_quantity: Optional[int] = None,
    ):
        order = self.get_order(order_id)
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
            raise RuntimeError(f"Order {order.order_id} is not fillable")
        position = self.get_position(order.strategy_name)
        if position is None or position.status is not PositionStatus.OPEN:
            raise RuntimeError(f"Strategy {order.strategy_name} has no open position")
        quantity = position.quantity if fill_quantity is None else fill_quantity
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if quantity > position.quantity:
            raise ValueError("Fill quantity exceeds position quantity")
        if quantity > order.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining order quantity")
        fill = Fill(
            fill_id=str(uuid4()),
            order_id=order.order_id,
            strategy_name=order.strategy_name,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            timestamp=timestamp,
        )
        return self._process_exit_fill_object(fill)

    def process_broker_fill(self, fill: Fill):
        if fill.fill_id in self._processed_fill_ids:
            raise RuntimeError("Duplicate fill")
        order = self.get_order(fill.order_id)
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
            raise RuntimeError(f"Order {order.order_id} is not fillable")
        position = self.get_position(fill.strategy_name)
        if position is None or position.side is order.side:
            result = self._process_entry_fill_object(fill)
        else:
            result = self._process_exit_fill_object(fill)
        self._processed_fill_ids.add(fill.fill_id)
        return result

    @staticmethod
    def _fill_average(order: Order, fill: Fill) -> float:
        old_qty = order.filled_quantity
        old_avg = order.average_fill_price
        if old_qty == 0 or old_avg is None:
            return float(fill.price)
        return float(
            (old_qty * old_avg + fill.quantity * fill.price) / (old_qty + fill.quantity)
        )

    def _process_entry_fill_object(self, fill: Fill) -> Fill:
        order = self.get_order(fill.order_id)
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
            raise RuntimeError(f"Order {order.order_id} is not fillable")
        if fill.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if fill.quantity > order.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining order quantity")
        new_average = self._fill_average(order, fill)
        self._orders[order.order_id] = apply_fill(
            order, fill_quantity=fill.quantity, average_fill_price=new_average
        )
        position = self.get_position(fill.strategy_name)
        if position is None:
            position = open_position(
                strategy_name=fill.strategy_name,
                side=fill.side,
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_timestamp=fill.timestamp,
            )
        else:
            if position.side is not fill.side:
                raise RuntimeError(
                    f"Strategy {fill.strategy_name} cannot accumulate fills with opposite side"
                )
            position = add_position_fill(position, fill)
        self._positions[fill.strategy_name] = position
        return fill

    def _process_exit_fill_object(self, fill: Fill):
        order = self.get_order(fill.order_id)
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
            raise RuntimeError(f"Order {order.order_id} is not fillable")
        if fill.quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if fill.quantity > order.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining order quantity")
        position = self.get_position(fill.strategy_name)
        if position is None or position.status is not PositionStatus.OPEN:
            raise RuntimeError(f"Strategy {fill.strategy_name} has no open position")
        if fill.quantity > position.quantity:
            raise ValueError("Fill quantity exceeds position quantity")
        if fill.side is position.side:
            raise ValueError("Exit fill must be opposite to the open position")
        new_average = self._fill_average(order, fill)
        self._orders[order.order_id] = apply_fill(
            order, fill_quantity=fill.quantity, average_fill_price=new_average
        )
        if fill.quantity < position.quantity:
            self._positions[fill.strategy_name] = replace(
                position, quantity=position.quantity - fill.quantity
            )
            return fill, None
        closed_position = close_position(
            position=position, exit_price=fill.price, exit_timestamp=fill.timestamp
        )
        self._closed_positions[fill.strategy_name] = closed_position
        del self._positions[fill.strategy_name]
        return fill, closed_position

    def get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise KeyError(f"Unknown order: {order_id}") from exc

    def get_order_status(self, order_id: str) -> OrderStatus:
        return self.get_order(order_id).status

    def cancel_order(self, order_id: str) -> Order:
        order = self.get_order(order_id)
        if order.status not in {
            OrderStatus.CREATED,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise RuntimeError(f"Order {order.order_id} cannot be cancelled")
        updated = cancel_order(order)
        self._orders[order.order_id] = updated
        if self._broker is not None:
            updated = self._broker.cancel_order(order.order_id)
            self._orders[order.order_id] = updated
        return updated

    def reject_order(self, order_id: str) -> Order:
        order = self.get_order(order_id)
        if order.status not in {
            OrderStatus.CREATED,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise RuntimeError(f"Order {order.order_id} cannot be rejected")
        updated = reject_order(order)
        self._orders[order.order_id] = updated
        intent = self._order_intents.get(order.order_id)
        if intent is not None:
            self._release_intent(intent)
        if self._broker is not None:
            updated = self._broker.reject_order(order.order_id)
            self._orders[order.order_id] = updated
        return updated

    def get_position(self, strategy_name: str) -> Optional[Position]:
        return self._positions.get(strategy_name)

    def has_open_position(self, strategy_name: str) -> bool:
        position = self.get_position(strategy_name)
        return (
            position is not None
            and position.status is PositionStatus.OPEN
            and position.quantity > 0
        )

    def get_closed_position(self, strategy_name: str) -> Optional[PositionClose]:
        return self._closed_positions.get(strategy_name)

    def get_orders(self) -> list[Order]:
        return list(self._orders.values())

    def get_open_orders(self) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if o.status in {OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        ]

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_closed_positions(self) -> list[PositionClose]:
        return list(self._closed_positions.values())
