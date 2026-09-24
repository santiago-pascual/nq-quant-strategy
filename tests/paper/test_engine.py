from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.engine import (
    PaperEngineConfig,
    PaperTradingEngine,
)
from src.paper.logger import PaperEventLogger
from src.portfolio.conflict import (
    PortfolioConflictEngine,
)
from src.risk.engine import RiskEngine
from src.risk.types import RiskLimits
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


# ======================================================================
# TEST TIMESTAMPS
# ======================================================================

TIMESTAMP_1 = datetime(
    2026,
    1,
    5,
    14,
    30,
    tzinfo=timezone.utc,
)

TIMESTAMP_2 = datetime(
    2026,
    1,
    5,
    14,
    31,
    tzinfo=timezone.utc,
)

TIMESTAMP_3 = datetime(
    2026,
    1,
    5,
    14,
    32,
    tzinfo=timezone.utc,
)


# ======================================================================
# BASIC TEST STRATEGY
# ======================================================================


class StubLongStrategy(BaseStrategy):
    """
    Minimal strategy used by the basic PaperTradingEngine tests.
    """

    @property
    def name(self) -> str:
        return "STUB"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_signal(self, market_data):
        return StrategySignal.LONG


class FakeLifecycleStrategy(BaseStrategy):
    """
    Test strategy used to validate the complete paper lifecycle.

    Behavior:
        flat -> ENTER LONG
        open -> HOLD until emit_exit=True
        open + emit_exit=True -> EXIT
    """

    @property
    def name(self) -> str:
        return "FAKE_LIFECYCLE"

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self) -> None:
        self.evaluate_calls = 0
        self.lifecycle_calls = 0
        self.fill_calls = 0
        self.exit_calls = 0
        self.entry_calls = 0
        self.emit_exit = False

    def generate_signal(self, market_data):
        self.evaluate_calls += 1
        self.entry_calls += 1

        return StrategySignal.LONG

    def on_market_data(
        self,
        market_data,
        position,
    ) -> StrategyDecision:
        self.lifecycle_calls += 1

        if self.emit_exit:
            return StrategyDecision(
                signal=StrategySignal.FLAT,
                action=StrategyAction.EXIT,
                reason="test_exit",
            )

        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.HOLD,
            reason="test_hold",
        )

    def on_fill(
        self,
        *,
        market_data,
        position,
    ) -> None:
        self.fill_calls += 1

    def on_exit(self) -> None:
        self.exit_calls += 1


# ======================================================================
# ENGINE FACTORY
# ======================================================================


def make_engine(
    tmp_path,
    *,
    strategies=None,
    max_contracts=1,
):
    logger = PaperEventLogger(
        tmp_path / "paper.jsonl",
    )

    execution = ExecutionEngine()

    broker = InMemoryBrokerAdapter()

    risk = RiskEngine(
        RiskLimits(
            risk_per_trade=250.0,
            max_total_risk=500.0,
            max_daily_loss=500.0,
            max_concurrent_positions=2,
            max_daily_trades=10,
            max_contracts=max_contracts,
        )
    )

    conflict = PortfolioConflictEngine(
        max_concurrent_positions=2,
    )

    if strategies is None:
        strategies = [
            StubLongStrategy(),
        ]

    engine = PaperTradingEngine(
        strategies=strategies,
        execution=execution,
        risk=risk,
        conflict=conflict,
        broker=broker,
        logger=logger,
        config=PaperEngineConfig(
            point_value=2.0,
        ),
    )

    return engine, execution, broker, logger


# ======================================================================
# MARKET DATA FACTORY
# ======================================================================


def bar(
    timestamp,
    *,
    close=100.0,
    stop=95.0,
):
    return {
        "timestamp": timestamp,
        "symbol": "MNQ.v.0",
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 100,
        "risk_stop_price": stop,
    }


# ======================================================================
# CONSTRUCTOR
# ======================================================================


def test_engine_requires_strategy(tmp_path):
    logger = PaperEventLogger(
        tmp_path / "paper.jsonl",
    )

    with pytest.raises(
        ValueError,
        match="At least one strategy",
    ):
        PaperTradingEngine(
            strategies=[],
            execution=ExecutionEngine(),
            risk=RiskEngine(
                RiskLimits(
                    risk_per_trade=250.0,
                    max_total_risk=500.0,
                    max_daily_loss=500.0,
                    max_concurrent_positions=2,
                    max_daily_trades=10,
                    max_contracts=20,
                )
            ),
            conflict=PortfolioConflictEngine(
                max_concurrent_positions=2,
            ),
            broker=InMemoryBrokerAdapter(),
            logger=logger,
        )


# ======================================================================
# CONNECTION / BASIC PROCESSING
# ======================================================================


def test_engine_connects_and_runs(tmp_path):

    engine, _, _, _ = make_engine(tmp_path)

    assert not engine.running

    engine.connect()

    assert engine.running

    result = engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    assert result.timestamp == TIMESTAMP_1
    assert "STUB" in result.decisions
    assert result.decisions["STUB"].action is StrategyAction.ENTER
    assert len(result.submitted_orders) == 1


def test_market_data_is_logged(tmp_path):

    engine, _, _, logger = make_engine(tmp_path)

    engine.connect()

    engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    events = logger.read_all()

    assert any(event.event_type.value == "market_data" for event in events)


def test_strategy_decision_is_logged(tmp_path):

    engine, _, _, logger = make_engine(tmp_path)

    engine.connect()

    engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    events = logger.read_all()

    assert any(event.event_type.value == "strategy_decision" for event in events)


def test_order_reaches_broker(tmp_path):

    engine, _, broker, _ = make_engine(tmp_path)

    engine.connect()

    result = engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    assert len(result.submitted_orders) == 1

    submission = result.submitted_orders[0]

    broker_order = broker.get_order(submission.broker_order_id)

    assert broker_order is not None
    assert broker_order.request.strategy_name == "STUB"


# ======================================================================
# TIMESTAMP VALIDATION
# ======================================================================


def test_chronology_is_enforced(tmp_path):

    engine, _, _, _ = make_engine(tmp_path)

    engine.connect()

    engine.process_bar(
        bar(TIMESTAMP_2),
        account_equity=50_000.0,
    )

    with pytest.raises(
        ValueError,
        match="strictly chronological",
    ):
        engine.process_bar(
            bar(TIMESTAMP_1),
            account_equity=50_000.0,
        )


def test_naive_timestamp_is_rejected(tmp_path):

    engine, _, _, _ = make_engine(tmp_path)

    engine.connect()

    naive = datetime(
        2026,
        1,
        5,
        14,
        30,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        engine.process_bar(
            bar(naive),
            account_equity=50_000.0,
        )


# ======================================================================
# ENGINE STATE
# ======================================================================


def test_stop_stops_processing(tmp_path):

    engine, _, _, _ = make_engine(tmp_path)

    engine.connect()

    engine.stop()

    with pytest.raises(
        RuntimeError,
        match="not running",
    ):
        engine.process_bar(
            bar(TIMESTAMP_1),
            account_equity=50_000.0,
        )


def test_disconnect_stops_engine(tmp_path):

    engine, _, _, _ = make_engine(tmp_path)

    engine.connect()

    assert engine.running

    engine.disconnect()

    assert not engine.running


# ======================================================================
# SERIALIZATION
# ======================================================================


def test_market_data_datetime_is_serialized_as_isoformat(tmp_path):

    engine, _, _, logger = make_engine(tmp_path)

    engine.connect()

    engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    events = logger.read_all()

    market_events = [
        event for event in events if event.event_type.value == "market_data"
    ]

    assert len(market_events) == 1

    payload = market_events[0].payload

    assert payload["timestamp"] == TIMESTAMP_1.isoformat()


# ======================================================================
# LIFECYCLE CONTROLLER
# ======================================================================


def test_active_position_uses_lifecycle_controller(
    tmp_path,
):
    strategy = FakeLifecycleStrategy()

    engine, execution, broker, logger = make_engine(
        tmp_path,
        strategies=[strategy],
    )

    engine.connect()

    # --------------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------------

    first = engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    assert len(first.submitted_orders) == 1

    broker_submission = first.submitted_orders[0]

    broker_order = broker.get_order(broker_submission.broker_order_id)

    assert broker_order is not None

    fill = broker.process_fill(
        broker_order_id=broker_order.broker_order_id,
        quantity=broker_order.request.quantity,
        price=100.0,
    )

    engine.process_broker_fill(
        fill,
        timestamp=TIMESTAMP_1,
    )

    assert execution.get_position(strategy.name) is not None

    # --------------------------------------------------------------
    # ACTIVE POSITION
    # --------------------------------------------------------------

    second = engine.process_bar(
        bar(TIMESTAMP_2),
        account_equity=50_000.0,
    )

    assert strategy.lifecycle_calls == 1
    assert strategy.entry_calls == 1


# ======================================================================
# FULL ENTER -> FILL -> EXIT -> FILL LIFECYCLE
# ======================================================================


def test_full_paper_lifecycle_enter_exit(
    tmp_path,
):
    strategy = FakeLifecycleStrategy()

    engine, execution, broker, logger = make_engine(
        tmp_path,
        strategies=[strategy],
    )

    engine.connect()

    # ==============================================================
    # ENTER
    # ==============================================================

    result = engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    assert len(result.submitted_orders) == 1

    entry_submission = result.submitted_orders[0]

    entry_broker_order = broker.get_order(entry_submission.broker_order_id)

    assert entry_broker_order is not None

    entry_fill = broker.process_fill(
        broker_order_id=entry_broker_order.broker_order_id,
        quantity=entry_broker_order.request.quantity,
        price=100.0,
    )

    engine.process_broker_fill(
        entry_fill,
        timestamp=TIMESTAMP_1,
    )

    position = execution.get_position(strategy.name)

    assert position is not None
    assert position.quantity == 1

    assert strategy.fill_calls == 1

    # ==============================================================
    # EXIT SIGNAL
    # ==============================================================

    strategy.emit_exit = True

    result = engine.process_bar(
        bar(TIMESTAMP_2),
        account_equity=50_000.0,
    )

    assert len(result.submitted_orders) == 1

    exit_submission = result.submitted_orders[0]

    exit_broker_order = broker.get_order(exit_submission.broker_order_id)

    assert exit_broker_order is not None
    assert exit_broker_order.request.quantity == 1

    # ==============================================================
    # EXIT FILL
    # ==============================================================

    exit_fill = broker.process_fill(
        broker_order_id=exit_broker_order.broker_order_id,
        quantity=1,
        price=101.0,
    )

    engine.process_broker_fill(
        exit_fill,
        timestamp=TIMESTAMP_2,
    )

    assert execution.get_position(strategy.name) is None

    assert strategy.exit_calls == 1


# ======================================================================
# PARTIAL ENTRY FILLS
# ======================================================================


def test_partial_entry_fill_then_update_then_exit(
    tmp_path,
):
    strategy = FakeLifecycleStrategy()

    engine, execution, broker, logger = make_engine(
        tmp_path,
        strategies=[strategy],
        max_contracts=2,
    )

    engine.connect()

    result = engine.process_bar(
        bar(TIMESTAMP_1),
        account_equity=50_000.0,
    )

    assert len(result.submitted_orders) == 1

    submission = result.submitted_orders[0]

    broker_order = broker.get_order(submission.broker_order_id)

    assert broker_order is not None

    # --------------------------------------------------------------
    # PARTIAL FILL 1
    # --------------------------------------------------------------

    first_fill = broker.process_fill(
        broker_order_id=broker_order.broker_order_id,
        quantity=1,
        price=100.0,
    )

    engine.process_broker_fill(
        first_fill,
        timestamp=TIMESTAMP_1,
    )

    position = execution.get_position(strategy.name)

    assert position is not None
    assert position.quantity == 1

    # --------------------------------------------------------------
    # PARTIAL FILL 2
    # --------------------------------------------------------------

    second_fill = broker.process_fill(
        broker_order_id=broker_order.broker_order_id,
        quantity=1,
        price=100.25,
    )

    engine.process_broker_fill(
        second_fill,
        timestamp=TIMESTAMP_2,
    )

    position = execution.get_position(strategy.name)

    assert position is not None
    assert position.quantity == 2

    # --------------------------------------------------------------
    # EXIT
    # --------------------------------------------------------------

    strategy.emit_exit = True

    result = engine.process_bar(
        bar(TIMESTAMP_3),
        account_equity=50_000.0,
    )

    assert len(result.submitted_orders) == 1

    exit_submission = result.submitted_orders[0]

    exit_order = broker.get_order(exit_submission.broker_order_id)

    assert exit_order is not None
    assert exit_order.request.quantity == 2

    exit_fill = broker.process_fill(
        broker_order_id=exit_order.broker_order_id,
        quantity=2,
        price=101.0,
    )

    engine.process_broker_fill(
        exit_fill,
        timestamp=TIMESTAMP_3,
    )

    assert execution.get_position(strategy.name) is None

    assert strategy.exit_calls == 1
