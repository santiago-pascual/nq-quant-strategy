from datetime import datetime, timezone

import pytest

from src.execution.types import Fill, OrderStatus
from src.execution.engine import ExecutionEngine
from src.strategies.base import (
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


def make_entry_intent(
    engine: ExecutionEngine,
    *,
    strategy_name: str = "MRL1",
):
    timestamp = datetime.now(timezone.utc)

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
        reason="test",
    )

    return engine.build_intent(
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )


def test_duplicate_intent_is_rejected():
    engine = ExecutionEngine()

    intent = make_entry_intent(engine)

    engine.submit_entry(intent, quantity=1)

    with pytest.raises(RuntimeError, match="Duplicate execution intent"):
        engine.submit_entry(intent, quantity=1)


def test_second_entry_while_position_open_is_rejected():
    engine = ExecutionEngine()

    intent = make_entry_intent(engine)

    order = engine.submit_entry(intent, quantity=1)

    engine.process_entry_fill(
        order_id=order.order_id,
        fill_price=25000.0,
        timestamp=intent.timestamp,
    )

    second_intent = make_entry_intent(
        engine,
        strategy_name="MRL1",
    )

    with pytest.raises(RuntimeError, match="already has an open position"):
        engine.submit_entry(second_intent, quantity=1)


def test_full_entry_exit_lifecycle():
    engine = ExecutionEngine()

    entry_time = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)
    exit_time = datetime(2026, 9, 23, 14, 5, tzinfo=timezone.utc)

    entry_decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
        reason="entry",
    )

    entry_intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=entry_decision,
        timestamp=entry_time,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    engine.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=25000.0,
        timestamp=entry_time,
    )

    assert engine.has_open_position("MRL1")

    exit_decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="take_profit",
    )

    exit_intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=exit_decision,
        timestamp=exit_time,
    )

    exit_order = engine.submit_exit(exit_intent)

    exit_fill, closed_position = engine.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=25025.0,
        timestamp=exit_time,
    )

    assert exit_fill.price == 25025.0
    assert closed_position.entry_price == 25000.0
    assert closed_position.exit_price == 25025.0
    assert closed_position.points == 25.0
    assert not engine.has_open_position("MRL1")


def test_exit_without_position_is_rejected():
    engine = ExecutionEngine()

    timestamp = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)

    decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    with pytest.raises(RuntimeError, match="has no open position"):
        engine.submit_exit(intent)


def test_duplicate_exit_intent_is_rejected():
    engine = ExecutionEngine()

    entry_time = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)

    entry_decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    entry_intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=entry_decision,
        timestamp=entry_time,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    engine.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=25000.0,
        timestamp=entry_time,
    )

    exit_decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
    )

    exit_intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=exit_decision,
        timestamp=entry_time,
    )

    engine.submit_exit(exit_intent)

    with pytest.raises(RuntimeError, match="Duplicate execution intent"):
        engine.submit_exit(exit_intent)


def test_short_entry_exit_lifecycle():
    engine = ExecutionEngine()

    entry_time = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )
    exit_time = datetime(
        2026,
        9,
        23,
        14,
        5,
        tzinfo=timezone.utc,
    )

    entry_decision = StrategyDecision(
        signal=StrategySignal.SHORT,
        action=StrategyAction.ENTER,
        reason="short_entry",
    )

    entry_intent = engine.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=entry_decision,
        timestamp=entry_time,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    assert entry_order.side is StrategySignal.SHORT

    engine.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=25000.0,
        timestamp=entry_time,
    )

    position = engine.get_position("MRS2")

    assert position is not None
    assert position.side is StrategySignal.SHORT
    assert position.entry_price == 25000.0

    exit_decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="take_profit",
    )

    exit_intent = engine.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=exit_decision,
        timestamp=exit_time,
    )

    exit_order = engine.submit_exit(exit_intent)

    assert exit_order.side is StrategySignal.LONG

    exit_fill, closed_position = engine.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=24975.0,
        timestamp=exit_time,
    )

    assert exit_fill.price == 24975.0
    assert closed_position.entry_price == 25000.0
    assert closed_position.exit_price == 24975.0

    # SHORT: entry - exit
    assert closed_position.points == 25.0

    assert not engine.has_open_position("MRS2")

    stored_closed = engine.get_closed_position("MRS2")

    assert stored_closed is not None
    assert stored_closed.points == 25.0


def test_order_submission_changes_state_to_submitted():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    assert order.status is OrderStatus.SUBMITTED
    assert engine.get_order(order.order_id).status is OrderStatus.SUBMITTED


def test_submitted_order_can_be_filled():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    assert order.status is OrderStatus.SUBMITTED

    engine.process_entry_fill(
        order_id=order.order_id,
        fill_price=25000.0,
        timestamp=timestamp,
    )

    assert engine.get_order(order.order_id).status is OrderStatus.FILLED

    assert engine.has_open_position("MRL1")


def test_rejected_order_does_not_create_position():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    rejected = engine.reject_order(order.order_id)

    assert rejected.status is OrderStatus.REJECTED
    assert not engine.has_open_position("MRL1")


def test_cancelled_order_does_not_create_position():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    cancelled = engine.cancel_order(order.order_id)

    assert cancelled.status is OrderStatus.CANCELLED
    assert not engine.has_open_position("MRL1")


def test_rejected_order_can_be_retried():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    first_order = engine.submit_entry(
        intent,
        quantity=1,
    )

    engine.reject_order(first_order.order_id)

    second_order = engine.submit_entry(
        intent,
        quantity=1,
    )

    assert second_order.order_id != first_order.order_id
    assert second_order.status is OrderStatus.SUBMITTED


def test_filled_order_cannot_be_filled_again():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    engine.process_entry_fill(
        order_id=order.order_id,
        fill_price=25000.0,
        timestamp=timestamp,
    )

    with pytest.raises(RuntimeError, match="not fillable"):
        engine.process_entry_fill(
            order_id=order.order_id,
            fill_price=25001.0,
            timestamp=timestamp,
        )


def test_filled_order_cannot_be_cancelled():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    engine.process_entry_fill(
        order_id=order.order_id,
        fill_price=25000.0,
        timestamp=timestamp,
    )

    with pytest.raises(RuntimeError, match="cannot be cancelled"):
        engine.cancel_order(order.order_id)


def test_filled_order_cannot_be_rejected():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    engine.process_entry_fill(
        order_id=order.order_id,
        fill_price=25000.0,
        timestamp=timestamp,
    )

    with pytest.raises(RuntimeError, match="cannot be rejected"):
        engine.reject_order(order.order_id)


def test_cancelled_order_cannot_be_filled():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    engine.cancel_order(order.order_id)

    with pytest.raises(RuntimeError, match="not fillable"):
        engine.process_entry_fill(
            order_id=order.order_id,
            fill_price=25000.0,
            timestamp=timestamp,
        )


def test_rejected_order_cannot_be_filled():
    engine = ExecutionEngine()

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    engine.reject_order(order.order_id)

    with pytest.raises(RuntimeError, match="not fillable"):
        engine.process_entry_fill(
            order_id=order.order_id,
            fill_price=25000.0,
            timestamp=timestamp,
        )


def test_engine_submits_order_to_paper_broker():
    from src.execution.paper_broker import PaperBroker

    broker = PaperBroker()
    engine = ExecutionEngine(broker=broker)

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    assert order.status is OrderStatus.SUBMITTED

    assert broker.get_order_status(order.order_id) is OrderStatus.SUBMITTED

    assert len(broker.get_open_orders()) == 1


def test_engine_consumes_paper_broker_fill():
    from src.execution.paper_broker import PaperBroker

    broker = PaperBroker()
    engine = ExecutionEngine(broker=broker)

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=1,
    )

    fill = Fill(
        fill_id="paper-fill-1",
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
        timestamp=timestamp,
    )

    broker.inject_fill(fill)

    result = engine.process_broker_fill(fill)

    assert result == fill

    assert engine.has_open_position("MRL1")

    position = engine.get_position("MRL1")

    assert position is not None
    assert position.entry_price == 25000.0

    assert engine.get_order(order.order_id).status is OrderStatus.FILLED


def test_engine_paper_broker_full_long_lifecycle():
    from src.execution.paper_broker import PaperBroker

    broker = PaperBroker()
    engine = ExecutionEngine(broker=broker)

    entry_time = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    exit_time = datetime(
        2026,
        9,
        23,
        14,
        5,
        tzinfo=timezone.utc,
    )

    entry_decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    entry_intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=entry_decision,
        timestamp=entry_time,
    )

    entry_order = engine.submit_entry(
        entry_intent,
        quantity=1,
    )

    entry_fill = Fill(
        fill_id="paper-entry",
        order_id=entry_order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
        timestamp=entry_time,
    )

    broker.inject_fill(entry_fill)

    engine.process_broker_fill(entry_fill)

    assert engine.has_open_position("MRL1")

    exit_decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="take_profit",
    )

    exit_intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=exit_decision,
        timestamp=exit_time,
    )

    exit_order = engine.submit_exit(exit_intent)

    assert broker.get_order_status(exit_order.order_id) is OrderStatus.SUBMITTED

    exit_fill = Fill(
        fill_id="paper-exit",
        order_id=exit_order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25025.0,
        timestamp=exit_time,
    )

    broker.inject_fill(exit_fill)

    result = engine.process_broker_fill(exit_fill)

    assert result[0] == exit_fill
    assert result[1].points == 25.0

    assert not engine.has_open_position("MRL1")

    closed = engine.get_closed_position("MRL1")

    assert closed is not None
    assert closed.points == 25.0


def test_direct_and_broker_entry_long_have_same_state():
    from src.execution.paper_broker import PaperBroker

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    # --------------------------------------------------------------
    # DIRECT PATH
    # --------------------------------------------------------------

    direct = ExecutionEngine()

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = direct.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    direct_order = direct.submit_entry(
        intent,
        quantity=1,
    )

    direct.process_entry_fill(
        order_id=direct_order.order_id,
        fill_price=25000.0,
        timestamp=timestamp,
    )

    direct_position = direct.get_position("MRL1")

    # --------------------------------------------------------------
    # BROKER PATH
    # --------------------------------------------------------------

    broker = PaperBroker()
    broker_engine = ExecutionEngine(broker=broker)

    broker_intent = broker_engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    broker_order = broker_engine.submit_entry(
        broker_intent,
        quantity=1,
    )

    broker_fill = Fill(
        fill_id="parity-long-entry",
        order_id=broker_order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
        timestamp=timestamp,
    )

    broker.inject_fill(broker_fill)

    broker_engine.process_broker_fill(broker_fill)

    broker_position = broker_engine.get_position("MRL1")

    # --------------------------------------------------------------
    # PARITY
    # --------------------------------------------------------------

    assert direct_position is not None
    assert broker_position is not None

    assert direct_position.side is broker_position.side
    assert direct_position.quantity == broker_position.quantity
    assert direct_position.entry_price == broker_position.entry_price
    assert direct_position.entry_timestamp == broker_position.entry_timestamp

    assert direct.has_open_position("MRL1")
    assert broker_engine.has_open_position("MRL1")


def test_direct_and_broker_full_long_lifecycle_have_same_result():
    from src.execution.paper_broker import PaperBroker

    entry_time = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    exit_time = datetime(
        2026,
        9,
        23,
        14,
        5,
        tzinfo=timezone.utc,
    )

    decision_entry = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    decision_exit = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="take_profit",
    )

    # --------------------------------------------------------------
    # DIRECT PATH
    # --------------------------------------------------------------

    direct = ExecutionEngine()

    entry_intent = direct.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision_entry,
        timestamp=entry_time,
    )

    entry_order = direct.submit_entry(
        entry_intent,
        quantity=1,
    )

    direct.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=25000.0,
        timestamp=entry_time,
    )

    exit_intent = direct.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision_exit,
        timestamp=exit_time,
    )

    exit_order = direct.submit_exit(exit_intent)

    _, direct_closed = direct.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=25025.0,
        timestamp=exit_time,
    )

    # --------------------------------------------------------------
    # BROKER PATH
    # --------------------------------------------------------------

    broker = PaperBroker()
    broker_engine = ExecutionEngine(broker=broker)

    broker_entry_intent = broker_engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision_entry,
        timestamp=entry_time,
    )

    broker_entry_order = broker_engine.submit_entry(
        broker_entry_intent,
        quantity=1,
    )

    broker_entry_fill = Fill(
        fill_id="parity-long-entry-full",
        order_id=broker_entry_order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
        timestamp=entry_time,
    )

    broker.inject_fill(broker_entry_fill)

    broker_engine.process_broker_fill(broker_entry_fill)

    broker_exit_intent = broker_engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision_exit,
        timestamp=exit_time,
    )

    broker_exit_order = broker_engine.submit_exit(broker_exit_intent)

    broker_exit_fill = Fill(
        fill_id="parity-long-exit-full",
        order_id=broker_exit_order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25025.0,
        timestamp=exit_time,
    )

    broker.inject_fill(broker_exit_fill)

    _, broker_closed = broker_engine.process_broker_fill(broker_exit_fill)

    # --------------------------------------------------------------
    # PARITY
    # --------------------------------------------------------------

    assert not direct.has_open_position("MRL1")
    assert not broker_engine.has_open_position("MRL1")

    assert direct_closed.side is broker_closed.side
    assert direct_closed.quantity == broker_closed.quantity
    assert direct_closed.entry_price == broker_closed.entry_price
    assert direct_closed.exit_price == broker_closed.exit_price
    assert direct_closed.entry_timestamp == broker_closed.entry_timestamp
    assert direct_closed.exit_timestamp == broker_closed.exit_timestamp
    assert direct_closed.points == broker_closed.points


def test_direct_and_broker_full_short_lifecycle_have_same_result():
    from src.execution.paper_broker import PaperBroker

    entry_time = datetime(
        2026,
        9,
        23,
        15,
        0,
        tzinfo=timezone.utc,
    )

    exit_time = datetime(
        2026,
        9,
        23,
        15,
        5,
        tzinfo=timezone.utc,
    )

    decision_entry = StrategyDecision(
        signal=StrategySignal.SHORT,
        action=StrategyAction.ENTER,
    )

    decision_exit = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="take_profit",
    )

    # --------------------------------------------------------------
    # DIRECT PATH
    # --------------------------------------------------------------

    direct = ExecutionEngine()

    entry_intent = direct.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=decision_entry,
        timestamp=entry_time,
    )

    entry_order = direct.submit_entry(
        entry_intent,
        quantity=1,
    )

    direct.process_entry_fill(
        order_id=entry_order.order_id,
        fill_price=25000.0,
        timestamp=entry_time,
    )

    exit_intent = direct.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=decision_exit,
        timestamp=exit_time,
    )

    exit_order = direct.submit_exit(exit_intent)

    _, direct_closed = direct.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=24975.0,
        timestamp=exit_time,
    )

    # --------------------------------------------------------------
    # BROKER PATH
    # --------------------------------------------------------------

    broker = PaperBroker()
    broker_engine = ExecutionEngine(broker=broker)

    broker_entry_intent = broker_engine.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=decision_entry,
        timestamp=entry_time,
    )

    broker_entry_order = broker_engine.submit_entry(
        broker_entry_intent,
        quantity=1,
    )

    broker_entry_fill = Fill(
        fill_id="parity-short-entry",
        order_id=broker_entry_order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25000.0,
        timestamp=entry_time,
    )

    broker.inject_fill(broker_entry_fill)

    broker_engine.process_broker_fill(broker_entry_fill)

    broker_exit_intent = broker_engine.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=decision_exit,
        timestamp=exit_time,
    )

    broker_exit_order = broker_engine.submit_exit(broker_exit_intent)

    broker_exit_fill = Fill(
        fill_id="parity-short-exit",
        order_id=broker_exit_order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.LONG,
        quantity=1,
        price=24975.0,
        timestamp=exit_time,
    )

    broker.inject_fill(broker_exit_fill)

    _, broker_closed = broker_engine.process_broker_fill(broker_exit_fill)

    # --------------------------------------------------------------
    # PARITY
    # --------------------------------------------------------------

    assert not direct.has_open_position("MRS2")
    assert not broker_engine.has_open_position("MRS2")

    assert direct_closed.side is broker_closed.side
    assert direct_closed.quantity == broker_closed.quantity
    assert direct_closed.entry_price == broker_closed.entry_price
    assert direct_closed.exit_price == broker_closed.exit_price
    assert direct_closed.entry_timestamp == broker_closed.entry_timestamp
    assert direct_closed.exit_timestamp == broker_closed.exit_timestamp
    assert direct_closed.points == broker_closed.points

    # SHORT: 25000 -> 24975 = +25 points
    assert direct_closed.points == 25.0


def test_engine_accumulates_partial_entry_fills():
    from src.execution.paper_broker import PaperBroker

    broker = PaperBroker()
    engine = ExecutionEngine(broker=broker)

    timestamp1 = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    timestamp2 = datetime(
        2026,
        9,
        23,
        14,
        1,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp1,
    )

    order = engine.submit_entry(
        intent,
        quantity=5,
    )

    fill1 = Fill(
        fill_id="engine-partial-1",
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=2,
        price=25000.0,
        timestamp=timestamp1,
    )

    broker.inject_fill(fill1)
    engine.process_broker_fill(fill1)

    updated_order = engine.get_order(order.order_id)

    assert updated_order.filled_quantity == 2
    assert updated_order.remaining_quantity == 3
    assert updated_order.status is OrderStatus.PARTIALLY_FILLED

    position = engine.get_position("MRL1")

    assert position is not None
    assert position.quantity == 2
    assert position.entry_price == 25000.0

    fill2 = Fill(
        fill_id="engine-partial-2",
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=3,
        price=25010.0,
        timestamp=timestamp2,
    )

    broker.inject_fill(fill2)
    engine.process_broker_fill(fill2)

    final_order = engine.get_order(order.order_id)

    assert final_order.filled_quantity == 5
    assert final_order.remaining_quantity == 0
    assert final_order.status is OrderStatus.FILLED

    expected_average = (2 * 25000.0 + 3 * 25010.0) / 5

    assert final_order.average_fill_price == expected_average

    position = engine.get_position("MRL1")

    assert position is not None
    assert position.quantity == 5
    assert position.entry_price == expected_average


def test_engine_rejects_duplicate_broker_fill():
    from src.execution.paper_broker import PaperBroker

    broker = PaperBroker()
    engine = ExecutionEngine(broker=broker)

    timestamp = datetime(
        2026,
        9,
        23,
        14,
        0,
        tzinfo=timezone.utc,
    )

    decision = StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
    )

    intent = engine.build_intent(
        strategy_name="MRL1",
        strategy_version="1.0.0",
        decision=decision,
        timestamp=timestamp,
    )

    order = engine.submit_entry(
        intent,
        quantity=2,
    )

    fill = Fill(
        fill_id="duplicate-fill",
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
        timestamp=timestamp,
    )

    broker.inject_fill(fill)

    engine.process_broker_fill(fill)

    with pytest.raises(RuntimeError, match="Duplicate fill"):
        engine.process_broker_fill(fill)
