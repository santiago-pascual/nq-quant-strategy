from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.broker import (
    BrokerAdapter,
    BrokerFill,
    BrokerOrder,
    BrokerOrderRequest,
)
from src.execution import ExecutionEngine
from src.execution.types import ExecutionIntent, Fill
from src.strategies.base import StrategyAction, StrategySignal


@dataclass(frozen=True)
class BrokerSubmission:
    """
    Mapping between an internal execution order and a broker order.
    """

    broker_order: BrokerOrder
    execution_intent: ExecutionIntent
    execution_order_id: str

    @property
    def broker_order_id(self) -> str:
        return self.broker_order.broker_order_id

    @property
    def request(self):
        return self.broker_order.request


class BrokerExecutionCoordinator:
    """
    Coordinates the internal execution engine with a provider-neutral
    broker adapter.
    """

    def __init__(
        self,
        *,
        execution_engine: ExecutionEngine,
        broker: BrokerAdapter,
    ) -> None:

        if execution_engine is None:
            raise ValueError("execution_engine cannot be None")

        if broker is None:
            raise ValueError("broker cannot be None")

        self.execution_engine = execution_engine
        self.broker = broker

        self._broker_orders: dict[
            str,
            BrokerSubmission,
        ] = {}

    @property
    def connected(self) -> bool:
        return self.broker.is_connected()

    def connect(self) -> None:
        self.broker.connect()

    def disconnect(self) -> None:
        self.broker.disconnect()

    # ------------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------------

    def submit_entry(
        self,
        intent: ExecutionIntent,
        *,
        quantity: int,
    ) -> BrokerOrder:

        if intent.action is not StrategyAction.ENTER:
            raise ValueError("Broker entry submission requires ENTER action")

        if intent.signal not in (
            StrategySignal.LONG,
            StrategySignal.SHORT,
        ):
            raise ValueError("Broker entry submission requires LONG or SHORT signal")

        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        execution_order = self.execution_engine.submit_entry(
            intent,
            quantity=quantity,
        )

        broker_request = BrokerOrderRequest(
            strategy_name=intent.strategy_name,
            strategy_version=intent.strategy_version,
            signal=intent.signal,
            quantity=quantity,
        )

        try:
            broker_order = self.broker.submit_order(
                broker_request,
            )
        except Exception:
            raise

        submission = BrokerSubmission(
            broker_order=broker_order,
            execution_intent=intent,
            execution_order_id=execution_order.order_id,
        )

        self._broker_orders[broker_order.broker_order_id] = submission

        # Public API returns BrokerOrder.
        return broker_order

    # ------------------------------------------------------------------
    # EXIT
    # ------------------------------------------------------------------

    def submit_exit(
        self,
        intent: ExecutionIntent,
    ) -> BrokerOrder:

        if intent.action is not StrategyAction.EXIT:
            raise ValueError("Broker exit submission requires EXIT action")

        if intent.signal is not StrategySignal.FLAT:
            raise ValueError("Broker exit submission requires FLAT signal")

        position = self.execution_engine.get_position(
            intent.strategy_name,
        )

        if position is None:
            raise RuntimeError(
                f"no open execution position for strategy {intent.strategy_name}"
            )

        execution_order = self.execution_engine.submit_exit(
            intent,
        )

        if position.side is StrategySignal.LONG:
            broker_signal = StrategySignal.SHORT
        elif position.side is StrategySignal.SHORT:
            broker_signal = StrategySignal.LONG
        else:
            raise RuntimeError("Cannot create broker exit for FLAT position")

        broker_request = BrokerOrderRequest(
            strategy_name=intent.strategy_name,
            strategy_version=intent.strategy_version,
            signal=broker_signal,
            quantity=position.quantity,
        )

        broker_order = self.broker.submit_order(
            broker_request,
        )

        submission = BrokerSubmission(
            broker_order=broker_order,
            execution_intent=intent,
            execution_order_id=execution_order.order_id,
        )

        self._broker_orders[broker_order.broker_order_id] = submission

        return broker_order

    # ------------------------------------------------------------------
    # LOOKUP
    # ------------------------------------------------------------------

    def broker_order(
        self,
        broker_order_id: str,
    ) -> BrokerSubmission | None:

        return self._broker_orders.get(
            broker_order_id,
        )

    def submission(
        self,
        broker_order_id: str,
    ) -> BrokerSubmission | None:

        return self._broker_orders.get(
            broker_order_id,
        )

    # ------------------------------------------------------------------
    # CANCEL
    # ------------------------------------------------------------------

    def cancel(
        self,
        broker_order_id: str,
    ) -> BrokerOrder:

        submission = self._broker_orders.get(
            broker_order_id,
        )

        if submission is None:
            raise KeyError("Unknown coordinated broker order")

        return self.broker.cancel_order(
            broker_order_id,
        )

    # ------------------------------------------------------------------
    # BROKER FILL
    # ------------------------------------------------------------------

    def process_broker_fill(
        self,
        broker_fill: BrokerFill,
        *,
        timestamp: Any,
    ) -> Fill:

        submission = self._broker_orders.get(
            broker_fill.broker_order_id,
        )

        if submission is None:
            raise KeyError("Broker fill does not belong to a coordinated order")

        execution_intent = submission.execution_intent

        execution_fill = Fill(
            fill_id=broker_fill.fill_id,
            order_id=submission.execution_order_id,
            strategy_name=execution_intent.strategy_name,
            side=broker_fill.signal,
            quantity=broker_fill.quantity,
            price=broker_fill.price,
            timestamp=timestamp,
        )

        if execution_intent.action is StrategyAction.ENTER:
            self.execution_engine.process_entry_fill(
                order_id=submission.execution_order_id,
                fill_price=broker_fill.price,
                timestamp=timestamp,
                fill_quantity=broker_fill.quantity,
            )

        elif execution_intent.action is StrategyAction.EXIT:
            self.execution_engine.process_exit_fill(
                order_id=submission.execution_order_id,
                fill_price=broker_fill.price,
                timestamp=timestamp,
                fill_quantity=broker_fill.quantity,
            )

        else:
            raise RuntimeError("Unsupported execution action for broker fill")

        return execution_fill
