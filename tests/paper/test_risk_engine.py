from __future__ import annotations

from datetime import date

import pytest

from src.risk.engine import RiskEngine
from src.risk.types import (
    RiskDecision,
    RiskLimits,
    RiskRequest,
)


TRADING_DAY = date(2026, 1, 5)


def _limits(
    *,
    risk_per_trade: float = 125.0,
    max_total_risk: float = 500.0,
    max_daily_loss: float = 500.0,
    max_concurrent_positions: int = 2,
    max_daily_trades: int = 10,
    max_contracts: int = 20,
) -> RiskLimits:
    return RiskLimits(
        risk_per_trade=risk_per_trade,
        max_total_risk=max_total_risk,
        max_daily_loss=max_daily_loss,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_trades=max_daily_trades,
        max_contracts=max_contracts,
    )


def _request(
    *,
    strategy_name: str = "MRL1",
    entry_price: float = 100.0,
    stop_price: float = 75.0,
    point_value: float = 1.0,
    account_equity: float = 50_000.0,
) -> RiskRequest:
    return RiskRequest(
        strategy_name=strategy_name,
        entry_price=entry_price,
        stop_price=stop_price,
        point_value=point_value,
        account_equity=account_equity,
    )


def test_risk_engine_approves_and_sizes_position():
    engine = RiskEngine(_limits())

    request = _request(
        entry_price=100.0,
        stop_price=75.0,
        point_value=1.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.APPROVED
    assert result.approved is True
    assert result.strategy_name == "MRL1"

    # 25 points of risk per contract.
    # 125 / 25 = 5 contracts.
    assert result.risk_per_contract == pytest.approx(25.0)
    assert result.quantity == 5
    assert result.total_risk == pytest.approx(125.0)


def test_risk_engine_respects_max_contracts():
    engine = RiskEngine(
        _limits(
            risk_per_trade=1_000.0,
            max_contracts=3,
        )
    )

    request = _request(
        entry_price=100.0,
        stop_price=90.0,
        point_value=1.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.approved is True
    assert result.quantity == 3
    assert result.risk_per_contract == pytest.approx(10.0)
    assert result.total_risk == pytest.approx(30.0)


def test_risk_engine_rejects_when_stop_is_too_large_for_risk_budget():
    engine = RiskEngine(
        _limits(
            risk_per_trade=125.0,
        )
    )

    request = _request(
        entry_price=100.0,
        stop_price=50.0,
        point_value=10.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    # 50 points * $10 = $500 risk per contract.
    # $125 budget cannot authorize even one contract.
    assert result.decision is RiskDecision.REJECTED
    assert result.approved is False
    assert result.quantity == 0
    assert result.risk_per_contract == pytest.approx(500.0)
    assert result.total_risk == pytest.approx(0.0)
    assert "stop distance" in result.reason


def test_risk_engine_rejects_duplicate_strategy_position():
    engine = RiskEngine(_limits())

    request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    first = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert first.approved is True

    engine.register_entry(first)

    second = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert second.decision is RiskDecision.REJECTED
    assert "already has an open position" in second.reason


def test_risk_engine_registers_actual_entry_fill_not_authorized_quantity():
    engine = RiskEngine(_limits())

    request = _request(
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.quantity == 5

    # Only 2 contracts actually fill.
    engine.register_entry_fill(
        result,
        fill_quantity=2,
    )

    assert engine.open_position_count == 1
    assert engine.open_risk == pytest.approx(50.0)
    assert engine.daily_trade_count == 1


def test_risk_engine_supports_partial_entry_fills():
    engine = RiskEngine(_limits())

    request = _request(
        strategy_name="MRS2",
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.quantity == 5

    engine.register_entry_fill(
        result,
        fill_quantity=2,
    )

    assert engine.open_risk == pytest.approx(50.0)

    engine.register_entry_fill(
        result,
        fill_quantity=1,
    )

    assert engine.open_risk == pytest.approx(75.0)

    assert engine.open_position_count == 1
    assert engine.daily_trade_count == 1


def test_risk_engine_releases_risk_only_as_exit_fills_occur():
    engine = RiskEngine(_limits())

    request = _request(
        strategy_name="MRS2",
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    engine.register_entry_fill(
        result,
        fill_quantity=4,
    )

    assert engine.open_risk == pytest.approx(100.0)

    engine.register_exit_fill(
        "MRS2",
        fill_quantity=1,
        realized_pnl=10.0,
    )

    assert engine.open_position_count == 1
    assert engine.open_risk == pytest.approx(75.0)
    assert engine.daily_realized_pnl == pytest.approx(0.0)

    engine.register_exit_fill(
        "MRS2",
        fill_quantity=3,
        realized_pnl=40.0,
    )

    assert engine.open_position_count == 0
    assert engine.open_risk == pytest.approx(0.0)
    assert engine.daily_realized_pnl == pytest.approx(40.0)


def test_risk_engine_daily_trade_count_increments_on_first_fill():
    engine = RiskEngine(_limits())

    request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    engine.register_entry_fill(
        result,
        fill_quantity=1,
    )

    assert engine.daily_trade_count == 1

    engine.register_entry_fill(
        result,
        fill_quantity=1,
    )

    assert engine.daily_trade_count == 1


def test_risk_engine_rejects_when_max_concurrent_positions_is_reached():
    engine = RiskEngine(
        _limits(
            max_concurrent_positions=1,
        )
    )

    first_request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    first_result = engine.evaluate(
        first_request,
        trading_day=TRADING_DAY,
    )

    assert first_result.approved is True

    engine.register_entry(first_result)

    second_request = _request(
        strategy_name="MRS2",
        entry_price=100.0,
        stop_price=75.0,
    )

    second_result = engine.evaluate(
        second_request,
        trading_day=TRADING_DAY,
    )

    assert second_result.decision is RiskDecision.REJECTED
    assert "concurrent positions" in second_result.reason


def test_risk_engine_rejects_when_max_total_risk_is_reached():
    engine = RiskEngine(
        _limits(
            risk_per_trade=125.0,
            max_total_risk=100.0,
        )
    )

    request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    # Authorized quantity is 5, but total authorized risk would be
    # 125, exceeding the 100 maximum aggregate risk.
    assert result.decision is RiskDecision.REJECTED
    assert "aggregate open risk" in result.reason


def test_risk_engine_rejects_when_daily_trade_limit_is_reached():
    engine = RiskEngine(
        _limits(
            max_daily_trades=1,
        )
    )

    first_request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    first_result = engine.evaluate(
        first_request,
        trading_day=TRADING_DAY,
    )

    assert first_result.approved is True

    engine.register_entry(first_result)
    engine.register_exit(
        "MRL1",
        realized_pnl=25.0,
    )

    second_request = _request(
        strategy_name="MRS2",
        entry_price=100.0,
        stop_price=75.0,
    )

    second_result = engine.evaluate(
        second_request,
        trading_day=TRADING_DAY,
    )

    assert second_result.decision is RiskDecision.REJECTED
    assert "daily trades" in second_result.reason


def test_risk_engine_rejects_when_daily_loss_limit_is_reached():
    engine = RiskEngine(
        _limits(
            max_daily_loss=100.0,
        )
    )

    request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.approved is True

    engine.register_entry(result)
    engine.register_exit(
        "MRL1",
        realized_pnl=-100.0,
    )

    next_request = _request(
        strategy_name="MRS2",
        entry_price=100.0,
        stop_price=75.0,
    )

    next_result = engine.evaluate(
        next_request,
        trading_day=TRADING_DAY,
    )

    assert next_result.decision is RiskDecision.REJECTED
    assert "daily loss limit" in next_result.reason


def test_risk_engine_resets_daily_limits_on_new_trading_day():
    engine = RiskEngine(
        _limits(
            max_daily_trades=1,
            max_daily_loss=100.0,
        )
    )

    request = _request(
        strategy_name="MRL1",
        entry_price=100.0,
        stop_price=75.0,
    )

    first_result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert first_result.approved is True

    engine.register_entry(first_result)
    engine.register_exit(
        "MRL1",
        realized_pnl=-100.0,
    )

    next_day = date(2026, 1, 6)

    next_result = engine.evaluate(
        _request(
            strategy_name="MRS2",
            entry_price=100.0,
            stop_price=75.0,
        ),
        trading_day=next_day,
    )

    assert next_result.approved is True
    assert engine.daily_trade_count == 0
    assert engine.daily_realized_pnl == pytest.approx(0.0)


def test_risk_engine_rejects_entry_fill_above_authorized_quantity():
    engine = RiskEngine(_limits())

    request = _request(
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.quantity == 5

    with pytest.raises(
        ValueError,
        match="entry fill exceeds risk-authorized quantity",
    ):
        engine.register_entry_fill(
            result,
            fill_quantity=6,
        )


def test_risk_engine_rejects_exit_fill_above_current_position():
    engine = RiskEngine(_limits())

    request = _request(
        entry_price=100.0,
        stop_price=75.0,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    engine.register_entry_fill(
        result,
        fill_quantity=2,
    )

    with pytest.raises(
        ValueError,
        match="exit fill exceeds current risk position quantity",
    ):
        engine.register_exit_fill(
            "MRL1",
            fill_quantity=3,
        )
