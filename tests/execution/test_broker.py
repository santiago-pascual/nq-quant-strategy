from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.execution.paper_broker import PaperBroker
from src.execution.types import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
)
from src.strategies.base import StrategySignal


def make_order(
    *,
    order_id: str | None = None,
) -> Order:
    return Order(
        order_id=order_id or str(uuid4()),
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        order_type=OrderType.MARKET,
        quantity=1,
        created_at=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
        status=OrderStatus.SUBMITTED,
    )


def test_paper_broker_accepts_submitted_order():
    broker = PaperBroker()

    order = make_order()

    broker.submit_order(order)

    assert broker.get_order_status(order.order_id) is OrderStatus.SUBMITTED

    assert len(broker.get_open_orders()) == 1


def test_paper_broker_rejects_duplicate_order_id():
    broker = PaperBroker()

    order = make_order()

    broker.submit_order(order)

    with pytest.raises(RuntimeError, match="already exists"):
        broker.submit_order(order)


def test_paper_broker_can_cancel_order():
    broker = PaperBroker()

    order = make_order()

    broker.submit_order(order)
    broker.cancel_order(order.order_id)

    assert broker.get_order_status(order.order_id) is OrderStatus.CANCELLED

    assert broker.get_open_orders() == []


def test_paper_broker_unknown_order_fails():
    broker = PaperBroker()

    with pytest.raises(KeyError, match="Unknown order"):
        broker.get_order_status("does-not-exist")


def test_paper_broker_injects_fill():
    broker = PaperBroker()

    order = make_order()

    broker.submit_order(order)

    fill = Fill(
        fill_id=str(uuid4()),
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=1,
        price=25000.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    broker.inject_fill(fill)

    assert broker.get_order_status(order.order_id) is OrderStatus.FILLED

    fills = broker.get_fills()

    assert len(fills) == 1
    assert fills[0].price == 25000.0


def test_paper_broker_rejects_overfill():
    broker = PaperBroker()

    order = make_order()

    broker.submit_order(order)

    fill = Fill(
        fill_id=str(uuid4()),
        order_id=order.order_id,
        strategy_name="MRL1",
        side=StrategySignal.LONG,
        quantity=2,
        price=25000.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(
        ValueError,
        match="exceeds order quantity",
    ):
        broker.inject_fill(fill)


def test_partial_fill_updates_order_state():
    broker = PaperBroker()

    order = Order(
        order_id="partial-1",
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        order_type=OrderType.MARKET,
        quantity=5,
        created_at=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
        status=OrderStatus.SUBMITTED,
    )

    broker.submit_order(order)

    fill = Fill(
        fill_id="fill-1",
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=2,
        price=25000.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    updated = broker.inject_fill(fill)

    assert updated.status is OrderStatus.PARTIALLY_FILLED
    assert updated.filled_quantity == 2
    assert updated.remaining_quantity == 3
    assert updated.average_fill_price == 25000.0


def test_multiple_partial_fills_reach_full_fill():
    broker = PaperBroker()

    order = Order(
        order_id="partial-2",
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        order_type=OrderType.MARKET,
        quantity=5,
        created_at=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
        status=OrderStatus.SUBMITTED,
    )

    broker.submit_order(order)

    fill1 = Fill(
        fill_id="fill-1",
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=2,
        price=25000.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    fill2 = Fill(
        fill_id="fill-2",
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=1,
        price=25001.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            1,
            tzinfo=timezone.utc,
        ),
    )

    fill3 = Fill(
        fill_id="fill-3",
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=2,
        price=24999.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            2,
            tzinfo=timezone.utc,
        ),
    )

    broker.inject_fill(fill1)
    broker.inject_fill(fill2)
    final_order = broker.inject_fill(fill3)

    assert final_order.status is OrderStatus.FILLED
    assert final_order.filled_quantity == 5
    assert final_order.remaining_quantity == 0

    expected_average = (2 * 25000.0 + 1 * 25001.0 + 2 * 24999.0) / 5

    assert final_order.average_fill_price == expected_average


def test_partial_fill_cannot_exceed_remaining_quantity():
    broker = PaperBroker()

    order = Order(
        order_id="partial-3",
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        order_type=OrderType.MARKET,
        quantity=3,
        created_at=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
        status=OrderStatus.SUBMITTED,
    )

    broker.submit_order(order)

    fill = Fill(
        fill_id="fill-over",
        order_id=order.order_id,
        strategy_name="MRS2",
        side=StrategySignal.SHORT,
        quantity=4,
        price=25000.0,
        timestamp=datetime(
            2026,
            9,
            23,
            14,
            0,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(
        ValueError,
        match="remaining order quantity",
    ):
        broker.inject_fill(fill)
