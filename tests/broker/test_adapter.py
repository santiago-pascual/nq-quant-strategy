from __future__ import annotations

import pytest

from src.broker import (
    BrokerOrderRequest,
    BrokerOrderStatus,
    InMemoryBrokerAdapter,
)
from src.strategies.base import StrategySignal


STRATEGY = "MRS2"
VERSION = "1.0"
SYMBOL = "MNQ.v.0"


def make_request(
    signal: StrategySignal = StrategySignal.SHORT,
    quantity: int = 2,
) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        strategy_name=STRATEGY,
        strategy_version=VERSION,
        signal=signal,
        quantity=quantity,
    )


def test_broker_starts_disconnected():
    broker = InMemoryBrokerAdapter()

    assert not broker.is_connected()


def test_connect():
    broker = InMemoryBrokerAdapter()

    broker.connect()

    assert broker.is_connected()


def test_disconnect():
    broker = InMemoryBrokerAdapter()

    broker.connect()
    broker.disconnect()

    assert not broker.is_connected()


def test_submit_requires_connection():
    broker = InMemoryBrokerAdapter()

    with pytest.raises(RuntimeError, match="not connected"):
        broker.submit_order(make_request())


def test_submit_order():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request())

    assert order.broker_order_id == "SIM-00000001"
    assert order.status == BrokerOrderStatus.SUBMITTED
    assert order.request.quantity == 2
    assert order.request.signal == StrategySignal.SHORT


def test_order_ids_are_deterministic():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    first = broker.submit_order(make_request())
    second = broker.submit_order(make_request())

    assert first.broker_order_id == "SIM-00000001"
    assert second.broker_order_id == "SIM-00000002"


def test_get_order():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request())

    assert broker.get_order(order.broker_order_id) == order


def test_unknown_order_returns_none():
    broker = InMemoryBrokerAdapter()

    assert broker.get_order("UNKNOWN") is None


def test_cancel_order():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request())
    cancelled = broker.cancel_order(order.broker_order_id)

    assert cancelled.status == BrokerOrderStatus.CANCELLED
    assert broker.get_order(order.broker_order_id) == cancelled


def test_cancel_unknown_order():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    with pytest.raises(KeyError, match="Unknown broker order"):
        broker.cancel_order("UNKNOWN")


def test_full_fill():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request(quantity=3))

    fill = broker.process_fill(
        order.broker_order_id,
        quantity=3,
        price=100.0,
    )

    assert fill.quantity == 3
    assert fill.price == 100.0
    assert fill.signal == StrategySignal.SHORT

    updated = broker.get_order(order.broker_order_id)

    assert updated is not None
    assert updated.status == BrokerOrderStatus.FILLED


def test_partial_fill():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request(quantity=3))

    broker.process_fill(
        order.broker_order_id,
        quantity=1,
        price=100.0,
    )

    updated = broker.get_order(order.broker_order_id)

    assert updated is not None
    assert updated.status == BrokerOrderStatus.PARTIALLY_FILLED


def test_partial_then_full_fill():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request(quantity=3))

    broker.process_fill(
        order.broker_order_id,
        quantity=1,
        price=100.0,
    )

    broker.process_fill(
        order.broker_order_id,
        quantity=2,
        price=101.0,
    )

    updated = broker.get_order(order.broker_order_id)

    assert updated is not None
    assert updated.status == BrokerOrderStatus.FILLED


def test_overfill_rejected():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request(quantity=2))

    with pytest.raises(
        ValueError,
        match="exceeds remaining",
    ):
        broker.process_fill(
            order.broker_order_id,
            quantity=3,
            price=100.0,
        )


def test_fill_cancelled_order_rejected():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(make_request())

    broker.cancel_order(order.broker_order_id)

    with pytest.raises(
        ValueError,
        match="cancelled or rejected",
    ):
        broker.process_fill(
            order.broker_order_id,
            quantity=2,
            price=100.0,
        )


def test_long_position_created():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(
        make_request(
            signal=StrategySignal.LONG,
            quantity=2,
        )
    )

    broker.process_fill(
        order.broker_order_id,
        quantity=2,
        price=100.0,
    )

    positions = broker.get_positions()

    assert len(positions) == 1
    assert positions[0].strategy_name == STRATEGY
    assert positions[0].signal == StrategySignal.LONG
    assert positions[0].quantity == 2
    assert positions[0].average_price == 100.0


def test_short_position_created():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(
        make_request(
            signal=StrategySignal.SHORT,
            quantity=2,
        )
    )

    broker.process_fill(
        order.broker_order_id,
        quantity=2,
        price=100.0,
    )

    positions = broker.get_positions()

    assert len(positions) == 1
    assert positions[0].signal == StrategySignal.SHORT
    assert positions[0].quantity == 2


def test_same_side_partial_fills_use_weighted_average():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    order = broker.submit_order(
        make_request(
            signal=StrategySignal.LONG,
            quantity=3,
        )
    )

    broker.process_fill(
        order.broker_order_id,
        quantity=1,
        price=100.0,
    )

    broker.process_fill(
        order.broker_order_id,
        quantity=2,
        price=101.0,
    )

    position = broker.get_positions()[0]

    assert position.quantity == 3
    assert position.average_price == pytest.approx(100.6666666667)


def test_opposite_fill_reduces_position():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    entry = broker.submit_order(
        make_request(
            signal=StrategySignal.LONG,
            quantity=3,
        )
    )

    broker.process_fill(
        entry.broker_order_id,
        quantity=3,
        price=100.0,
    )

    exit_order = broker.submit_order(
        make_request(
            signal=StrategySignal.SHORT,
            quantity=1,
        )
    )

    broker.process_fill(
        exit_order.broker_order_id,
        quantity=1,
        price=102.0,
    )

    position = broker.get_positions()[0]

    assert position.signal == StrategySignal.LONG
    assert position.quantity == 2
    assert position.average_price == 100.0


def test_opposite_fill_closes_position():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    entry = broker.submit_order(
        make_request(
            signal=StrategySignal.LONG,
            quantity=2,
        )
    )

    broker.process_fill(
        entry.broker_order_id,
        quantity=2,
        price=100.0,
    )

    exit_order = broker.submit_order(
        make_request(
            signal=StrategySignal.SHORT,
            quantity=2,
        )
    )

    broker.process_fill(
        exit_order.broker_order_id,
        quantity=2,
        price=102.0,
    )

    assert broker.get_positions() == ()


def test_oversized_exit_rejected():
    broker = InMemoryBrokerAdapter()
    broker.connect()

    entry = broker.submit_order(
        make_request(
            signal=StrategySignal.LONG,
            quantity=2,
        )
    )

    broker.process_fill(
        entry.broker_order_id,
        quantity=2,
        price=100.0,
    )

    exit_order = broker.submit_order(
        make_request(
            signal=StrategySignal.SHORT,
            quantity=3,
        )
    )

    with pytest.raises(
        ValueError,
        match="exceeds open position",
    ):
        broker.process_fill(
            exit_order.broker_order_id,
            quantity=3,
            price=102.0,
        )


def test_positions_empty_initially():
    broker = InMemoryBrokerAdapter()

    assert broker.get_positions() == ()


def test_invalid_order_quantity_rejected():
    with pytest.raises(ValueError, match="quantity must be > 0"):
        make_request(quantity=0)


def test_flat_order_signal_rejected():
    with pytest.raises(
        ValueError,
        match="must be LONG or SHORT",
    ):
        BrokerOrderRequest(
            strategy_name=STRATEGY,
            strategy_version=VERSION,
            signal=StrategySignal.FLAT,
            quantity=1,
        )
