from __future__ import annotations

from datetime import date

import pytest

from src.risk.engine import RiskEngine
from src.risk.types import RiskDecision, RiskLimits, RiskRequest


TRADING_DAY = date(2026, 9, 23)


def make_engine(
    *,
    risk_per_trade: float = 250.0,
    max_total_risk: float = 500.0,
    max_daily_loss: float = 500.0,
    max_concurrent_positions: int = 2,
    max_daily_trades: int = 10,
    max_contracts: int = 20,
) -> RiskEngine:
    return RiskEngine(
        limits=RiskLimits(
            risk_per_trade=risk_per_trade,
            max_total_risk=max_total_risk,
            max_daily_loss=max_daily_loss,
            max_concurrent_positions=max_concurrent_positions,
            max_daily_trades=max_daily_trades,
            max_contracts=max_contracts,
        )
    )


def make_request(
    *,
    strategy_name: str = "MRS2",
    entry_price: float = 25000.0,
    stop_price: float = 24975.0,
    point_value: float = 2.0,
    account_equity: float = 50_000.0,
) -> RiskRequest:
    return RiskRequest(
        strategy_name=strategy_name,
        entry_price=entry_price,
        stop_price=stop_price,
        point_value=point_value,
        account_equity=account_equity,
    )


def test_sizes_mnq_position_from_stop_distance():
    engine = make_engine(risk_per_trade=250.0)

    result = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.APPROVED
    assert result.quantity == 5
    assert result.risk_per_contract == 50.0
    assert result.total_risk == 250.0


def test_short_and_long_have_same_risk_for_same_stop_distance():
    engine = make_engine(risk_per_trade=250.0)

    long_result = engine.evaluate(
        make_request(entry_price=25000.0, stop_price=24975.0),
        trading_day=TRADING_DAY,
    )
    short_result = engine.evaluate(
        make_request(
            strategy_name="MRS2_SHORT",
            entry_price=25000.0,
            stop_price=25025.0,
        ),
        trading_day=TRADING_DAY,
    )

    assert long_result.quantity == short_result.quantity == 5
    assert long_result.total_risk == short_result.total_risk == 250.0


def test_trade_is_rejected_when_one_contract_exceeds_risk_budget():
    engine = make_engine(risk_per_trade=40.0)

    result = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.REJECTED
    assert result.quantity == 0
    assert "too large" in result.reason


def test_max_contracts_caps_position_size():
    engine = make_engine(
        risk_per_trade=1000.0,
        max_contracts=3,
    )

    result = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )

    assert result.approved
    assert result.quantity == 3
    assert result.total_risk == 150.0


def test_duplicate_strategy_position_is_rejected():
    engine = make_engine()

    result = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )
    engine.register_entry(result)

    second = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )

    assert second.decision is RiskDecision.REJECTED
    assert "already has an open position" in second.reason


def test_aggregate_open_risk_limit_is_enforced():
    engine = make_engine(
        risk_per_trade=250.0,
        max_total_risk=300.0,
        max_concurrent_positions=3,
    )

    first = engine.evaluate(
        make_request(strategy_name="MRL1"),
        trading_day=TRADING_DAY,
    )
    assert first.approved
    engine.register_entry(first)

    second = engine.evaluate(
        make_request(strategy_name="MRS2"),
        trading_day=TRADING_DAY,
    )

    assert second.decision is RiskDecision.REJECTED
    assert "aggregate open risk" in second.reason


def test_concurrent_position_limit_is_enforced():
    engine = make_engine(
        risk_per_trade=100.0,
        max_total_risk=500.0,
        max_concurrent_positions=2,
    )

    for strategy in ("MRL1", "S2R"):
        result = engine.evaluate(
            make_request(strategy_name=strategy),
            trading_day=TRADING_DAY,
        )
        assert result.approved
        engine.register_entry(result)

    third = engine.evaluate(
        make_request(strategy_name="MRS2"),
        trading_day=TRADING_DAY,
    )

    assert third.decision is RiskDecision.REJECTED
    assert "concurrent positions" in third.reason


def test_daily_trade_limit_is_enforced():
    engine = make_engine(
        risk_per_trade=100.0,
        max_daily_trades=2,
        max_concurrent_positions=3,
    )

    first = engine.evaluate(
        make_request(strategy_name="MRL1"),
        trading_day=TRADING_DAY,
    )
    engine.register_entry(first)
    engine.register_exit("MRL1", realized_pnl=50.0)

    second = engine.evaluate(
        make_request(strategy_name="S2R"),
        trading_day=TRADING_DAY,
    )
    engine.register_entry(second)
    engine.register_exit("S2R", realized_pnl=50.0)

    third = engine.evaluate(
        make_request(strategy_name="MRS2"),
        trading_day=TRADING_DAY,
    )

    assert third.decision is RiskDecision.REJECTED
    assert "daily trades" in third.reason


def test_daily_loss_limit_blocks_new_entries():
    engine = make_engine(
        max_daily_loss=100.0,
        risk_per_trade=50.0,
    )

    first = engine.evaluate(
        make_request(strategy_name="MRL1"),
        trading_day=TRADING_DAY,
    )
    engine.register_entry(first)
    engine.register_exit("MRL1", realized_pnl=-100.0)

    result = engine.evaluate(
        make_request(strategy_name="S2R"),
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.REJECTED
    assert "daily loss limit" in result.reason


def test_daily_state_resets_on_new_trading_day():
    engine = make_engine(
        max_daily_loss=100.0,
        risk_per_trade=50.0,
    )

    first = engine.evaluate(
        make_request(strategy_name="MRL1"),
        trading_day=TRADING_DAY,
    )
    engine.register_entry(first)
    engine.register_exit("MRL1", realized_pnl=-100.0)

    blocked = engine.evaluate(
        make_request(strategy_name="S2R"),
        trading_day=TRADING_DAY,
    )
    assert blocked.decision is RiskDecision.REJECTED

    next_day = engine.evaluate(
        make_request(strategy_name="S2R"),
        trading_day=date(2026, 9, 24),
    )

    assert next_day.approved
    assert engine.daily_realized_pnl == 0.0
    assert engine.daily_trade_count == 0


def test_rejected_result_cannot_be_registered():
    engine = make_engine(risk_per_trade=40.0)

    result = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )

    assert not result.approved

    with pytest.raises(ValueError, match="rejected"):
        engine.register_entry(result)


def test_exit_requires_existing_position():
    engine = make_engine()

    with pytest.raises(RuntimeError, match="no open position"):
        engine.register_exit("MRS2", realized_pnl=10.0)


def test_open_risk_is_removed_after_exit():
    engine = make_engine(risk_per_trade=250.0)

    result = engine.evaluate(
        make_request(),
        trading_day=TRADING_DAY,
    )
    engine.register_entry(result)

    assert engine.open_position_count == 1
    assert engine.open_risk == 250.0

    engine.register_exit("MRS2", realized_pnl=125.0)

    assert engine.open_position_count == 0
    assert engine.open_risk == 0.0
    assert engine.daily_realized_pnl == 125.0
