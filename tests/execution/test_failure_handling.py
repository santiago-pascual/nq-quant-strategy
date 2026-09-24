from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.execution.engine import ExecutionEngine
from src.execution.types import (
    ExecutionIntent,
    Fill,
    OrderStatus,
    StrategyAction,
    StrategySignal,
)


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def make_entry_intent(
    strategy_name: str,
    *,
    side: StrategySignal,
    timestamp: str = "2026-09-23T14:30:00+00:00",
) -> ExecutionIntent:
    return ExecutionIntent(
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        signal=side,
        action=StrategyAction.ENTER,
        timestamp=ts(timestamp),
        reason="failure handling test",
    )


def make_exit_intent(
    strategy_name: str,
    *,
    timestamp: str = "2026-09-23T14:31:00+00:00",
) -> ExecutionIntent:
    return ExecutionIntent(
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        timestamp=ts(timestamp),
        reason="failure handling test",
    )


def make_fill(
    *,
    order_id: str,
    strategy_name: str,
    side: StrategySignal,
    quantity: int,
    price: float,
    timestamp: str = "2026-09-23T14:30:01+00:00",
) -> Fill:
    return Fill(
        fill_id=str(uuid4()),
        order_id=order_id,
        strategy_name=strategy_name,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=ts(timestamp),
    )


def test_rejected_entry_does_not_create_position():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "MRL1",
        side=StrategySignal.LONG,
    )

    order = engine.submit_entry(intent, quantity=2)
    order = engine.reject_order(order.order_id)

    assert order.status is OrderStatus.REJECTED
    assert engine.get_position("MRL1") is None


def test_cancelled_entry_does_not_create_position():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "MRS2",
        side=StrategySignal.SHORT,
    )

    order = engine.submit_entry(intent, quantity=1)
    order = engine.cancel_order(order.order_id)

    assert order.status is OrderStatus.CANCELLED
    assert engine.get_position("MRS2") is None


def test_rejected_order_can_be_retried_with_same_strategy():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "S2R",
        side=StrategySignal.SHORT,
    )

    first_order = engine.submit_entry(intent, quantity=1)
    first_order = engine.reject_order(first_order.order_id)

    second_order = engine.submit_entry(intent, quantity=1)

    assert first_order.status is OrderStatus.REJECTED
    assert second_order.order_id != first_order.order_id
    assert second_order.status is OrderStatus.SUBMITTED


def test_successful_duplicate_execution_intent_is_rejected():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "MRS2",
        side=StrategySignal.SHORT,
    )

    order = engine.submit_entry(intent, quantity=1)

    fill = make_fill(
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25000.0,
    )

    engine.process_broker_fill(fill)

    # Once the intent has successfully produced an open position,
    # the engine's public contract rejects the second entry because
    # the strategy already has an open position.
    with pytest.raises(
        RuntimeError,
        match="already has an open position",
    ):
        engine.submit_entry(
            intent,
            quantity=1,
        )


def test_duplicate_fill_is_rejected():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "MRL1",
        side=StrategySignal.LONG,
    )

    order = engine.submit_entry(intent, quantity=1)

    fill = make_fill(
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
    )

    engine.process_broker_fill(fill)

    with pytest.raises(
        RuntimeError,
        match="Duplicate fill",
    ):
        engine.process_broker_fill(fill)


def test_overfill_is_rejected():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "S2R",
        side=StrategySignal.SHORT,
    )

    order = engine.submit_entry(intent, quantity=2)

    fill = make_fill(
        order_id=order.order_id,
        strategy_name="S2R",
        side=StrategySignal.SHORT,
        quantity=3,
        price=25000.0,
    )

    with pytest.raises(
        ValueError,
        match="remaining order quantity",
    ):
        engine.process_broker_fill(fill)


def test_filled_order_cannot_receive_another_fill():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "MRS2",
        side=StrategySignal.SHORT,
    )

    order = engine.submit_entry(intent, quantity=1)

    fill = make_fill(
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25000.0,
    )

    engine.process_broker_fill(fill)

    second_fill = make_fill(
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=1,
        price=24990.0,
        timestamp="2026-09-23T14:30:02+00:00",
    )

    with pytest.raises(RuntimeError):
        engine.process_broker_fill(second_fill)


def test_cancelled_order_cannot_receive_fill():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "MRL1",
        side=StrategySignal.LONG,
    )

    order = engine.submit_entry(intent, quantity=1)
    order = engine.cancel_order(order.order_id)

    fill = make_fill(
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
    )

    with pytest.raises(RuntimeError):
        engine.process_broker_fill(fill)

    assert engine.get_position("MRL1") is None


def test_rejected_order_cannot_receive_fill():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "S2R",
        side=StrategySignal.SHORT,
    )

    order = engine.submit_entry(intent, quantity=1)
    order = engine.reject_order(order.order_id)

    fill = make_fill(
        order_id=order.order_id,
        strategy_name="S2R",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25000.0,
    )

    with pytest.raises(RuntimeError):
        engine.process_broker_fill(fill)

    assert engine.get_position("S2R") is None


def test_failed_exit_keeps_position_open():
    engine = ExecutionEngine()

    entry_intent = make_entry_intent(
        "MRS2",
        side=StrategySignal.SHORT,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    entry_fill = make_fill(
        order_id=entry_order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25000.0,
    )

    engine.process_broker_fill(entry_fill)

    exit_intent = make_exit_intent("MRS2")

    # submit_exit derives the exit quantity from the open position.
    exit_order = engine.submit_exit(exit_intent)

    exit_order = engine.reject_order(exit_order.order_id)

    assert exit_order.status is OrderStatus.REJECTED
    assert engine.get_position("MRS2") is not None


def test_failed_exit_can_be_retried():
    engine = ExecutionEngine()

    entry_intent = make_entry_intent(
        "MRL1",
        side=StrategySignal.LONG,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    entry_fill = make_fill(
        order_id=entry_order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
    )

    engine.process_broker_fill(entry_fill)

    exit_intent = make_exit_intent("MRL1")

    first_exit = engine.submit_exit(exit_intent)
    first_exit = engine.reject_order(first_exit.order_id)

    second_exit = engine.submit_exit(exit_intent)

    assert first_exit.status is OrderStatus.REJECTED
    assert second_exit.order_id != first_exit.order_id
    assert second_exit.status is OrderStatus.SUBMITTED
    assert engine.get_position("MRL1") is not None


def test_orb_uses_the_same_failure_contract_as_other_strategies():
    engine = ExecutionEngine()

    intent = make_entry_intent(
        "ORB",
        side=StrategySignal.LONG,
    )

    order = engine.submit_entry(intent, quantity=1)

    assert order.status is OrderStatus.SUBMITTED

    order = engine.reject_order(order.order_id)

    assert order.status is OrderStatus.REJECTED
    assert engine.get_position("ORB") is None


def test_orb_successful_entry_then_failed_exit_preserves_position():
    engine = ExecutionEngine()

    entry_intent = make_entry_intent(
        "ORB",
        side=StrategySignal.SHORT,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    entry_fill = make_fill(
        order_id=entry_order.order_id,
        strategy_name="ORB",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25000.0,
    )

    engine.process_broker_fill(entry_fill)

    assert engine.get_position("ORB") is not None

    exit_intent = make_exit_intent("ORB")

    exit_order = engine.submit_exit(exit_intent)
    exit_order = engine.reject_order(exit_order.order_id)

    assert exit_order.status is OrderStatus.REJECTED
    assert engine.get_position("ORB") is not None
