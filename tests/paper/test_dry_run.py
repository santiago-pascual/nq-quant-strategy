from __future__ import annotations

from datetime import datetime, timezone

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.dry_run import DryRunEngine
from src.paper.logger import PaperEventLogger
from src.portfolio.conflict import PortfolioConflictEngine
from src.risk.engine import RiskEngine
from src.risk.types import RiskDecision, RiskLimits, RiskResult
from src.strategies.base import (
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


TIMESTAMP = datetime(
    2026,
    1,
    5,
    14,
    30,
    tzinfo=timezone.utc,
)


def make_engine(tmp_path, risk=None):
    execution = ExecutionEngine()

    if risk is None:
        risk = RiskEngine(
            RiskLimits(
                risk_per_trade=250,
                max_total_risk=500,
                max_daily_loss=500,
                max_concurrent_positions=2,
                max_daily_trades=10,
                max_contracts=20,
            )
        )

    conflict = PortfolioConflictEngine(
        max_concurrent_positions=2,
    )

    broker = InMemoryBrokerAdapter()

    logger = PaperEventLogger(
        tmp_path / "dry_run.jsonl",
    )

    engine = DryRunEngine(
        execution=execution,
        risk=risk,
        conflict=conflict,
        broker=broker,
        logger=logger,
    )

    return engine, execution, broker, logger


def long_decision() -> StrategyDecision:
    return StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
        reason="dry-run-entry",
    )


def flat_decision() -> StrategyDecision:
    return StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.HOLD,
        reason="no-signal",
    )


class RejectingRiskEngine(RiskEngine):
    """
    Test-only Risk Engine that deterministically rejects every request.

    This verifies that DryRunEngine propagates a genuine risk rejection
    before execution/broker submission.
    """

    def evaluate(self, request, *, trading_day):
        return RiskResult(
            decision=RiskDecision.REJECTED,
            strategy_name=request.strategy_name,
            quantity=0,
            risk_per_contract=0.0,
            total_risk=0.0,
            reason="test-risk-rejection",
        )


def test_dry_run_entry_reaches_broker(tmp_path):
    engine, _, broker, logger = make_engine(tmp_path)

    engine.connect()

    result = engine.run_entry(
        decision=long_decision(),
        strategy_name="MRS2",
        strategy_version="1.0",
        timestamp=TIMESTAMP,
        entry_price=100.0,
        stop_price=95.0,
        point_value=2.0,
        account_equity=50_000.0,
    )

    assert result.risk_approved
    assert result.risk_quantity == 20
    assert result.broker_order_id is not None

    order = broker.get_order(result.broker_order_id)

    assert order is not None
    assert order.request.quantity == 20
    assert order.request.strategy_name == "MRS2"
    assert order.request.signal == StrategySignal.LONG

    events = logger.read_all()
    event_types = [event.event_type.value for event in events]

    assert "strategy_decision" in event_types
    assert "risk_request" in event_types
    assert "risk_decision" in event_types
    assert "order_submitted" in event_types


def test_dry_run_fill_opens_position(tmp_path):
    engine, execution, broker, logger = make_engine(tmp_path)

    engine.connect()

    result = engine.run_entry(
        decision=long_decision(),
        strategy_name="MRS2",
        strategy_version="1.0",
        timestamp=TIMESTAMP,
        entry_price=100.0,
        stop_price=95.0,
        point_value=2.0,
        account_equity=50_000.0,
    )

    order_id = result.broker_order_id
    assert order_id is not None

    order = broker.get_order(order_id)
    assert order is not None

    broker_fill = broker.process_fill(
        broker_order_id=order_id,
        quantity=order.request.quantity,
        price=100.25,
    )

    fill_result = engine.process_fill(
        broker_fill=broker_fill,
        timestamp=TIMESTAMP,
    )

    assert fill_result.fill_id == broker_fill.fill_id
    assert fill_result.position_open

    positions = execution.get_positions()

    assert len(positions) == 1
    assert positions[0].strategy_name == "MRS2"
    assert positions[0].quantity == 20
    assert positions[0].entry_price == 100.25

    events = logger.read_all()
    event_types = [event.event_type.value for event in events]

    assert "fill" in event_types
    assert "position_opened" in event_types


def test_dry_run_logs_position_open(tmp_path):
    engine, execution, broker, logger = make_engine(tmp_path)

    engine.connect()

    result = engine.run_entry(
        decision=long_decision(),
        strategy_name="MRS2",
        strategy_version="1.0",
        timestamp=TIMESTAMP,
        entry_price=100.0,
        stop_price=95.0,
        point_value=2.0,
        account_equity=50_000.0,
    )

    order_id = result.broker_order_id
    assert order_id is not None

    order = broker.get_order(order_id)
    assert order is not None

    broker_fill = broker.process_fill(
        broker_order_id=order_id,
        quantity=order.request.quantity,
        price=100.0,
    )

    engine.process_fill(
        broker_fill=broker_fill,
        timestamp=TIMESTAMP,
    )

    positions = execution.get_positions()

    assert len(positions) == 1
    assert positions[0].strategy_name == "MRS2"

    events = logger.read_all()

    position_events = [
        event for event in events if event.event_type.value == "position_opened"
    ]

    assert len(position_events) == 1

    payload = position_events[0].payload

    assert payload["strategy_name"] == "MRS2"
    assert payload["quantity"] == 20
    assert payload["entry_price"] == 100.0


def test_flat_strategy_does_not_reach_broker(tmp_path):
    engine, execution, broker, logger = make_engine(tmp_path)

    engine.connect()

    result = engine.run_entry(
        decision=flat_decision(),
        strategy_name="MRS2",
        strategy_version="1.0",
        timestamp=TIMESTAMP,
        entry_price=100.0,
        stop_price=95.0,
        point_value=2.0,
        account_equity=50_000.0,
    )

    assert not result.risk_approved
    assert result.risk_quantity == 0
    assert result.broker_order_id is None
    assert result.fill_id is None
    assert not result.position_open

    assert broker.get_positions() == ()
    assert execution.get_positions() == []

    events = logger.read_all()
    event_types = [event.event_type.value for event in events]

    assert "strategy_decision" in event_types
    assert "risk_request" not in event_types
    assert "risk_decision" not in event_types
    assert "order_submitted" not in event_types


def test_risk_rejection_stops_pipeline(tmp_path):
    risk = RejectingRiskEngine(
        RiskLimits(
            risk_per_trade=250,
            max_total_risk=500,
            max_daily_loss=500,
            max_concurrent_positions=2,
            max_daily_trades=10,
            max_contracts=20,
        )
    )

    engine, execution, broker, logger = make_engine(
        tmp_path,
        risk=risk,
    )

    engine.connect()

    result = engine.run_entry(
        decision=long_decision(),
        strategy_name="MRS2",
        strategy_version="1.0",
        timestamp=TIMESTAMP,
        entry_price=100.0,
        stop_price=95.0,
        point_value=2.0,
        account_equity=50_000.0,
    )

    assert not result.risk_approved
    assert result.risk_quantity == 0
    assert result.broker_order_id is None
    assert result.fill_id is None
    assert not result.position_open

    assert broker.get_positions() == ()
    assert execution.get_positions() == []

    events = logger.read_all()
    event_types = [event.event_type.value for event in events]

    assert "strategy_decision" in event_types
    assert "risk_request" in event_types
    assert "risk_decision" in event_types
    assert "order_submitted" not in event_types


def test_multiple_fills_update_position(tmp_path):
    engine, execution, broker, logger = make_engine(tmp_path)

    engine.connect()

    result = engine.run_entry(
        decision=long_decision(),
        strategy_name="MRS2",
        strategy_version="1.0",
        timestamp=TIMESTAMP,
        entry_price=100.0,
        stop_price=95.0,
        point_value=2.0,
        account_equity=50_000.0,
    )

    order_id = result.broker_order_id
    assert order_id is not None

    order = broker.get_order(order_id)
    assert order is not None
    assert order.request.quantity == 20

    first_fill = broker.process_fill(
        broker_order_id=order_id,
        quantity=10,
        price=100.0,
    )

    first_result = engine.process_fill(
        broker_fill=first_fill,
        timestamp=TIMESTAMP,
    )

    assert first_result.position_open

    positions = execution.get_positions()

    assert len(positions) == 1
    assert positions[0].quantity == 10
    assert positions[0].entry_price == 100.0

    second_fill = broker.process_fill(
        broker_order_id=order_id,
        quantity=10,
        price=101.0,
    )

    second_result = engine.process_fill(
        broker_fill=second_fill,
        timestamp=TIMESTAMP,
    )

    assert second_result.position_open

    positions = execution.get_positions()

    assert len(positions) == 1
    assert positions[0].quantity == 20
    assert positions[0].entry_price == 100.5

    events = logger.read_all()

    fill_events = [event for event in events if event.event_type.value == "fill"]

    position_events = [
        event
        for event in events
        if event.event_type.value
        in {
            "position_opened",
            "position_updated",
        }
    ]

    assert len(fill_events) == 2
    assert len(position_events) == 2
