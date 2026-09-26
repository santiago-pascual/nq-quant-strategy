from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.risk import RiskEngine
from src.risk.types import RiskLimits
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


class LifecycleStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "TEST_LIFECYCLE"

    @property
    def version(self) -> str:
        return "test-1.0"

    def generate_signal(self, market_data):
        return StrategySignal.LONG

    def evaluate(self, market_data):
        return StrategyDecision(
            signal=StrategySignal.LONG,
            action=StrategyAction.ENTER,
            reason="test entry",
        )

    def get_risk_stop_price(
        self,
        *,
        entry_price: float,
        signal: StrategySignal,
        market_data=None,
    ) -> float | None:
        return entry_price - 10.0


@dataclass
class FakePosition:
    strategy_name: str
    quantity: int
    entry_price: float
    side: StrategySignal


@dataclass
class FakeSubmission:
    execution_intent: object


class FakeExecutionEngine:
    def __init__(self):
        self.positions = {}

    def get_position(self, strategy_name):
        return self.positions.get(strategy_name)

    def get_positions(self):
        return tuple(self.positions.values())

    def build_intent(
        self,
        *,
        strategy_name,
        strategy_version,
        decision,
        timestamp,
    ):
        return SimpleNamespace(
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            action=decision.action,
            signal=decision.signal,
            timestamp=timestamp,
        )


class FakeConflictResult:
    approved = True
    reason = "approved"


class FakeConflictEngine:
    def evaluate(self, entry_request):
        return FakeConflictResult()

    def register_entry(self, entry_request):
        return None

    def register_exit(self, strategy_name):
        return None


class FakeLogger:
    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None

        return noop


class FakeBrokerOrder:
    def __init__(self, broker_order_id):
        self.broker_order_id = broker_order_id


class FakeBrokerExecution:
    def __init__(self, execution):
        self.execution = execution
        self.submissions = {}
        self.next_id = 1

    def connect(self):
        return None

    def disconnect(self):
        return None

    def submit_entry(self, intent, *, quantity):
        order_id = f"ENTRY-{self.next_id}"
        self.next_id += 1

        order = FakeBrokerOrder(order_id)

        self.submissions[order_id] = FakeSubmission(
            execution_intent=intent,
        )

        return order

    def submit_exit(self, intent):
        order_id = f"EXIT-{self.next_id}"
        self.next_id += 1

        order = FakeBrokerOrder(order_id)

        self.submissions[order_id] = FakeSubmission(
            execution_intent=intent,
        )

        return order

    def submission(self, broker_order_id):
        return self.submissions.get(broker_order_id)

    def process_broker_fill(self, broker_fill, *, timestamp):
        submission = self.submission(broker_fill.broker_order_id)

        if submission is None:
            raise KeyError("Unknown fake broker order")

        intent = submission.execution_intent
        strategy_name = intent.strategy_name

        if intent.action is StrategyAction.ENTER:
            self.execution.positions[strategy_name] = FakePosition(
                strategy_name=strategy_name,
                quantity=broker_fill.quantity,
                entry_price=broker_fill.price,
                side=broker_fill.signal,
            )

        elif intent.action is StrategyAction.EXIT:
            self.execution.positions.pop(strategy_name, None)

        return broker_fill


class FakeBroker:
    def __init__(self):
        self.orders = {}

    def get_order(self, broker_order_id):
        return self.orders.get(broker_order_id)


class FakeBrokerFill:
    def __init__(
        self,
        *,
        broker_order_id,
        fill_id,
        quantity,
        price,
        signal,
    ):
        self.broker_order_id = broker_order_id
        self.fill_id = fill_id
        self.quantity = quantity
        self.price = price
        self.signal = signal


def make_engine():
    strategy = LifecycleStrategy()

    execution = FakeExecutionEngine()

    limits = RiskLimits(
        risk_per_trade=100.0,
        max_total_risk=1000.0,
        max_daily_loss=1000.0,
        max_concurrent_positions=5,
        max_daily_trades=10,
        max_contracts=10,
    )

    risk = RiskEngine(limits)

    conflict = FakeConflictEngine()
    broker = FakeBroker()
    logger = FakeLogger()

    engine = PaperTradingEngine(
        strategies=[strategy],
        execution=execution,
        risk=risk,
        conflict=conflict,
        broker=broker,
        logger=logger,
        config=PaperEngineConfig(point_value=2.0),
    )

    broker_execution = FakeBrokerExecution(execution)
    engine.broker_execution = broker_execution

    return engine, execution, broker, broker_execution


def make_bar(timestamp, close):
    return {
        "timestamp": timestamp,
        "symbol": "MNQ",
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000,
    }


def test_entry_fill_creates_execution_position():
    engine, execution, broker, broker_execution = make_engine()

    engine.connect()

    timestamp = datetime(
        2026,
        9,
        26,
        14,
        30,
        tzinfo=timezone.utc,
    )

    step = engine.process_bar(
        make_bar(timestamp, 100.0),
        account_equity=50_000.0,
    )

    assert len(step.submitted_orders) == 1

    order = step.submitted_orders[0]

    broker.orders[order.broker_order_id] = SimpleNamespace(
        status=SimpleNamespace(name="FILLED")
    )

    fill = FakeBrokerFill(
        broker_order_id=order.broker_order_id,
        fill_id="FILL-ENTRY",
        quantity=5,
        price=100.0,
        signal=StrategySignal.LONG,
    )

    engine.process_broker_fill(
        fill,
        timestamp=timestamp,
    )

    position = execution.get_position("TEST_LIFECYCLE")

    assert position is not None
    assert position.quantity == 5
    assert position.entry_price == pytest.approx(100.0)
    assert position.side is StrategySignal.LONG


def test_exit_fill_closes_position_and_releases_risk():
    engine, execution, broker, broker_execution = make_engine()

    engine.connect()

    timestamp = datetime(
        2026,
        9,
        26,
        14,
        30,
        tzinfo=timezone.utc,
    )

    step = engine.process_bar(
        make_bar(timestamp, 100.0),
        account_equity=50_000.0,
    )

    entry_order = step.submitted_orders[0]

    broker.orders[entry_order.broker_order_id] = SimpleNamespace(
        status=SimpleNamespace(name="FILLED")
    )

    entry_fill = FakeBrokerFill(
        broker_order_id=entry_order.broker_order_id,
        fill_id="FILL-ENTRY",
        quantity=5,
        price=100.0,
        signal=StrategySignal.LONG,
    )

    engine.process_broker_fill(
        entry_fill,
        timestamp=timestamp,
    )

    position = execution.get_position("TEST_LIFECYCLE")

    assert position is not None
    assert position.quantity == 5
    assert position.entry_price == pytest.approx(100.0)

    strategy = engine._strategy("TEST_LIFECYCLE")

    exit_decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="test exit",
    )

    exit_timestamp = timestamp + timedelta(minutes=1)

    exit_order = engine._try_exit(
        strategy,
        exit_decision,
        exit_timestamp,
    )

    assert exit_order is not None

    broker.orders[exit_order.broker_order_id] = SimpleNamespace(
        status=SimpleNamespace(name="FILLED")
    )

    exit_fill = FakeBrokerFill(
        broker_order_id=exit_order.broker_order_id,
        fill_id="FILL-EXIT",
        quantity=5,
        price=110.0,
        signal=StrategySignal.FLAT,
    )

    execution_fill = engine.process_broker_fill(
        exit_fill,
        timestamp=exit_timestamp,
    )

    assert execution_fill is exit_fill

    assert execution.get_position("TEST_LIFECYCLE") is None

    assert engine.risk.open_position_count == 0
    assert engine.risk.open_risk == pytest.approx(0.0)

    # LONG: (110 - 100) * 5 contracts * $2/point = $100.
    expected_realized_pnl = (110.0 - 100.0) * 5 * engine.config.point_value

    assert expected_realized_pnl == pytest.approx(100.0)
