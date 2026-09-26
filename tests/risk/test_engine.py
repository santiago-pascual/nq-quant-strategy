from __future__ import annotations

from datetime import date

import pytest

from src.risk.engine import RiskEngine
from src.risk.types import RiskDecision, RiskLimits, RiskRequest


TRADING_DAY = date(2026, 1, 5)


def make_risk_engine(
    *,
    risk_per_trade: float = 125.0,
    max_total_risk: float = 375.0,
    max_contracts: int = 20,
) -> RiskEngine:
    return RiskEngine(
        RiskLimits(
            risk_per_trade=risk_per_trade,
            max_total_risk=max_total_risk,
            max_daily_loss=500.0,
            max_concurrent_positions=3,
            max_daily_trades=10,
            max_contracts=max_contracts,
        )
    )


def make_request(
    *,
    entry_price: float,
    stop_price: float,
    point_value: float = 2.0,
    strategy_name: str = "TEST",
) -> RiskRequest:
    return RiskRequest(
        strategy_name=strategy_name,
        entry_price=entry_price,
        stop_price=stop_price,
        point_value=point_value,
        account_equity=50_000.0,
    )


def test_fractional_quantity_below_one_is_rejected():
    """
    If the risk budget allows less than one indivisible MNQ contract,
    the trade must be rejected rather than rounded up.

    Example:
        risk budget = $125
        risk/contract = $166.67
        theoretical size = 0.75 contracts
        executable size = 0
        result = REJECT
    """
    engine = make_risk_engine()

    request = make_request(
        entry_price=100.0,
        stop_price=16.6666666667,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.REJECTED
    assert not result.approved
    assert result.quantity == 0
    assert result.risk_per_contract == pytest.approx(
        166.6666666666,
        rel=1e-9,
    )
    assert result.total_risk == pytest.approx(0.0)
    assert "stop distance" in result.reason


def test_fractional_quantity_1_55_is_floored_to_one():
    """
    A theoretical size of 1.55 MNQ contracts must become exactly
    1 contract, never 2.
    """
    engine = make_risk_engine()

    # $125 / ($80.64516129 per contract) = 1.55 contracts.
    risk_per_contract = 125.0 / 1.55
    stop_distance = risk_per_contract / 2.0

    request = make_request(
        entry_price=100.0,
        stop_price=100.0 - stop_distance,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.APPROVED
    assert result.approved
    assert result.quantity == 1
    assert result.total_risk <= 125.0


def test_fractional_quantity_2_99_is_floored_to_two():
    """
    A theoretical size of 2.99 MNQ contracts must become exactly
    2 contracts.
    """
    engine = make_risk_engine()

    # $125 / ($41.80602007 per contract) = 2.99 contracts.
    risk_per_contract = 125.0 / 2.99
    stop_distance = risk_per_contract / 2.0

    request = make_request(
        entry_price=100.0,
        stop_price=100.0 - stop_distance,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.APPROVED
    assert result.approved
    assert result.quantity == 2
    assert result.total_risk <= 125.0


@pytest.mark.parametrize(
    ("theoretical_quantity", "expected_quantity"),
    [
        (0.25, 0),
        (0.50, 0),
        (0.75, 0),
        (0.99, 0),
        (1.00, 1),
        (1.01, 1),
        (1.49, 1),
        (1.50, 1),
        (1.55, 1),
        (1.99, 1),
        (2.00, 2),
        (2.01, 2),
        (2.99, 2),
        (3.00, 3),
    ],
)
def test_contract_sizing_always_floors(
    theoretical_quantity: float,
    expected_quantity: int,
):
    """
    Contract sizing must always satisfy:

        executable_quantity <= theoretical_quantity

    and must never round upward.
    """
    engine = make_risk_engine()

    risk_per_contract = 125.0 / theoretical_quantity
    stop_distance = risk_per_contract / 2.0

    request = make_request(
        entry_price=100.0,
        stop_price=100.0 - stop_distance,
    )

    result = engine.evaluate(
        request,
        trading_day=TRADING_DAY,
    )

    assert result.quantity == expected_quantity

    if expected_quantity == 0:
        assert result.decision is RiskDecision.REJECTED
    else:
        assert result.decision is RiskDecision.APPROVED
        assert result.total_risk <= 125.0 + 1e-9
