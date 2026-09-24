from __future__ import annotations

from datetime import date, datetime, timezone

from src.execution.engine import ExecutionEngine
from src.risk.engine import RiskEngine
from src.risk.types import RiskDecision, RiskLimits
from src.strategies.base import StrategyAction, StrategyDecision, StrategySignal
from src.portfolio.risk_execution import RiskExecutionCoordinator


TRADING_DAY = date(2026, 9, 23)
ENTRY_TIME = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)


def make_coordinator(
    *,
    risk_per_trade: float = 250.0,
    max_total_risk: float = 500.0,
    max_daily_loss: float = 500.0,
    max_concurrent_positions: int = 2,
    max_daily_trades: int = 10,
    max_contracts: int = 20,
):
    risk = RiskEngine(
        limits=RiskLimits(
            risk_per_trade=risk_per_trade,
            max_total_risk=max_total_risk,
            max_daily_loss=max_daily_loss,
            max_concurrent_positions=max_concurrent_positions,
            max_daily_trades=max_daily_trades,
            max_contracts=max_contracts,
        )
    )
    execution = ExecutionEngine()
    return RiskExecutionCoordinator(
        risk_engine=risk,
        execution_engine=execution,
    )


def long_decision() -> StrategyDecision:
    return StrategyDecision(
        signal=StrategySignal.LONG,
        action=StrategyAction.ENTER,
        reason="test entry",
    )


def test_approved_entry_reaches_execution_with_risk_sized_quantity():
    coordinator = make_coordinator()

    approval = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert approval.approved
    assert approval.risk_result.decision is RiskDecision.APPROVED
    assert approval.risk_result.quantity == 5
    assert approval.order is not None
    assert approval.order.quantity == 5
    assert approval.order.strategy_name == "MRS2"


def test_rejected_risk_never_creates_execution_order():
    coordinator = make_coordinator(risk_per_trade=40.0)

    approval = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert not approval.approved
    assert approval.order is None
    assert len(coordinator.execution_engine._orders) == 0


def test_execution_order_is_not_registered_as_risk_until_full_fill():
    coordinator = make_coordinator()

    approval = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert approval.order is not None
    assert coordinator.risk_engine.open_position_count == 0

    fill = coordinator.confirm_entry_fill(
        order_id=approval.order.order_id,
        fill_price=25000.0,
        timestamp=ENTRY_TIME,
    )

    assert fill.quantity == approval.order.quantity
    assert coordinator.execution_engine.has_open_position("MRS2")
    assert coordinator.risk_engine.open_position_count == 0

    coordinator.register_approved_entry(approval.risk_result)

    assert coordinator.risk_engine.open_position_count == 1
    assert coordinator.risk_engine.open_risk == 250.0


def test_rejected_trade_does_not_consume_daily_trade_slot():
    coordinator = make_coordinator(
        risk_per_trade=40.0,
        max_daily_trades=1,
    )

    rejected = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert not rejected.approved
    assert coordinator.risk_engine.daily_trade_count == 0


def test_exit_releases_risk_after_execution_lifecycle():
    coordinator = make_coordinator()

    approval = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert approval.order is not None

    coordinator.confirm_entry_fill(
        order_id=approval.order.order_id,
        fill_price=25000.0,
        timestamp=ENTRY_TIME,
    )
    coordinator.register_approved_entry(approval.risk_result)

    assert coordinator.risk_engine.open_risk == 250.0

    exit_time = datetime(2026, 9, 23, 14, 5, tzinfo=timezone.utc)
    exit_decision = StrategyDecision(
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        reason="take profit",
    )

    exit_intent = coordinator.execution_engine.build_intent(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=exit_decision,
        timestamp=exit_time,
    )
    exit_order = coordinator.execution_engine.submit_exit(exit_intent)

    coordinator.execution_engine.process_exit_fill(
        order_id=exit_order.order_id,
        fill_price=25025.0,
        timestamp=exit_time,
    )

    coordinator.register_exit(
        strategy_name="MRS2",
        realized_pnl=50.0,
    )

    assert coordinator.risk_engine.open_position_count == 0
    assert coordinator.risk_engine.open_risk == 0.0
    assert coordinator.risk_engine.daily_realized_pnl == 50.0


def test_daily_loss_block_propagates_before_execution():
    coordinator = make_coordinator(
        max_daily_loss=100.0,
        risk_per_trade=50.0,
    )

    first = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )
    assert first.approved

    coordinator.confirm_entry_fill(
        order_id=first.order.order_id,
        fill_price=25000.0,
        timestamp=ENTRY_TIME,
    )
    coordinator.register_approved_entry(first.risk_result)
    coordinator.register_exit(
        strategy_name="MRS2",
        realized_pnl=-100.0,
    )

    second = coordinator.authorize_entry(
        strategy_name="S2R",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert not second.approved
    assert second.order is None
    assert len(coordinator.execution_engine._orders) == 1


def test_same_strategy_second_entry_is_blocked_before_execution():
    coordinator = make_coordinator()

    first = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )
    assert first.approved

    coordinator.confirm_entry_fill(
        order_id=first.order.order_id,
        fill_price=25000.0,
        timestamp=ENTRY_TIME,
    )
    coordinator.register_approved_entry(first.risk_result)

    second = coordinator.authorize_entry(
        strategy_name="MRS2",
        strategy_version="1.0.0",
        decision=long_decision(),
        timestamp=ENTRY_TIME,
        entry_price=25000.0,
        stop_price=24975.0,
        point_value=2.0,
        account_equity=50_000.0,
        trading_day=TRADING_DAY,
    )

    assert not second.approved
    assert second.order is None
