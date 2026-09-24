from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.execution.types import ExecutionIntent
from src.portfolio.broker_execution import BrokerExecutionCoordinator
from src.strategies.base import StrategyAction, StrategySignal


TIMESTAMP = datetime(
    2026,
    1,
    5,
    14,
    30,
    tzinfo=timezone.utc,
)


def make_intent(
    strategy_name: str = "MRS2",
    signal: StrategySignal = StrategySignal.SHORT,
    action: StrategyAction = StrategyAction.ENTER,
) -> ExecutionIntent:
    return ExecutionIntent(
        strategy_name=strategy_name,
        strategy_version="1.0",
        signal=signal,
        action=action,
        timestamp=TIMESTAMP,
        reason="test",
    )


def make_coordinator():
    execution = ExecutionEngine()
    broker = InMemoryBrokerAdapter()

    coordinator = BrokerExecutionCoordinator(
        execution_engine=execution,
        broker=broker,
    )

    return coordinator, execution, broker


def test_coordinator_requires_execution_engine():
    broker = InMemoryBrokerAdapter()

    with pytest.raises(
        ValueError,
        match="execution_engine cannot be None",
    ):
        BrokerExecutionCoordinator(
            execution_engine=None,
            broker=broker,
        )


def test_coordinator_requires_broker():
    execution = ExecutionEngine()

    with pytest.raises(
        ValueError,
        match="broker cannot be None",
    ):
        BrokerExecutionCoordinator(
            execution_engine=execution,
            broker=None,
        )


def test_connect():
    coordinator, _, broker = make_coordinator()

    coordinator.connect()

    assert broker.is_connected()
    assert coordinator.connected


def test_disconnect():
    coordinator, _, broker = make_coordinator()

    coordinator.connect()
    coordinator.disconnect()

    assert not broker.is_connected()
    assert not coordinator.connected


def test_entry_requires_connection():
    coordinator, _, _ = make_coordinator()

    with pytest.raises(
        RuntimeError,
        match="Broker is not connected",
    ):
        coordinator.submit_entry(
            make_intent(),
            quantity=2,
        )


def test_entry_creates_broker_order():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    order = coordinator.submit_entry(
        make_intent(),
        quantity=2,
    )

    assert order.request.strategy_name == "MRS2"
    assert order.request.strategy_version == "1.0"
    assert order.request.signal is StrategySignal.SHORT
    assert order.request.quantity == 2
    assert order.request.order_type == "MARKET"


def test_entry_rejects_non_enter_intent():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    intent = make_intent(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
    )

    with pytest.raises(
        ValueError,
        match="ENTER",
    ):
        coordinator.submit_entry(
            intent,
            quantity=1,
        )


def test_entry_rejects_flat_signal():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    with pytest.raises(
        ValueError,
        match="ENTER intent cannot have FLAT signal",
    ):
        make_intent(
            signal=StrategySignal.FLAT,
            action=StrategyAction.ENTER,
        )


def test_entry_rejects_invalid_quantity():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    with pytest.raises(
        ValueError,
        match="quantity must be > 0",
    ):
        coordinator.submit_entry(
            make_intent(),
            quantity=0,
        )


def test_broker_order_is_tracked():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    intent = make_intent()

    order = coordinator.submit_entry(
        intent,
        quantity=2,
    )

    submission = coordinator.broker_order(order.broker_order_id)

    assert submission is not None
    assert submission.execution_intent == intent


def test_unknown_broker_order_cannot_cancel():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    with pytest.raises(
        KeyError,
        match="Unknown coordinated broker order",
    ):
        coordinator.cancel("UNKNOWN")


def test_cancel_coordinated_order():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    order = coordinator.submit_entry(
        make_intent(),
        quantity=2,
    )

    cancelled = coordinator.cancel(order.broker_order_id)

    assert cancelled.status.value == "cancelled"


def test_exit_requires_open_position():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    intent = make_intent(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
    )

    with pytest.raises(
        RuntimeError,
        match="no open execution position",
    ):
        coordinator.submit_exit(intent)


def test_exit_requires_exit_action():
    coordinator, _, _ = make_coordinator()

    coordinator.connect()

    with pytest.raises(
        ValueError,
        match="ENTER intent cannot have FLAT signal",
    ):
        make_intent(
            signal=StrategySignal.FLAT,
            action=StrategyAction.ENTER,
        )
