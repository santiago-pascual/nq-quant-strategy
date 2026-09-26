from __future__ import annotations

import pytest

from src.risk.policy import (
    ProductionRiskPolicy,
    XFA_50K_PRODUCTION_POLICY,
)


def test_production_policy_is_50k_xfa():
    policy = XFA_50K_PRODUCTION_POLICY

    assert policy.account_size == 50_000.0
    assert policy.risk_fraction == 0.0025


def test_risk_per_trade_is_125_dollars():
    policy = XFA_50K_PRODUCTION_POLICY

    assert policy.risk_per_trade == pytest.approx(125.0)


def test_max_total_risk_is_375_dollars():
    policy = XFA_50K_PRODUCTION_POLICY

    assert policy.max_total_risk == pytest.approx(375.0)


def test_three_concurrent_positions_are_allowed():
    policy = XFA_50K_PRODUCTION_POLICY

    assert policy.max_concurrent_positions == 3


def test_policy_converts_to_existing_risk_limits():
    policy = XFA_50K_PRODUCTION_POLICY
    limits = policy.to_risk_limits()

    assert limits.risk_per_trade == pytest.approx(125.0)
    assert limits.max_total_risk == pytest.approx(375.0)
    assert limits.max_daily_loss == pytest.approx(500.0)
    assert limits.max_concurrent_positions == 3
    assert limits.max_daily_trades == 10
    assert limits.max_contracts == 20


def test_policy_is_not_using_test_fixture_values():
    policy = XFA_50K_PRODUCTION_POLICY
    limits = policy.to_risk_limits()

    assert limits.risk_per_trade != 250.0
    assert limits.max_total_risk != 500.0


@pytest.mark.parametrize(
    "risk_fraction",
    [
        0.0,
        -0.001,
    ],
)
def test_invalid_risk_fraction_is_rejected(risk_fraction):
    with pytest.raises(ValueError):
        ProductionRiskPolicy(
            risk_fraction=risk_fraction,
        )


def test_invalid_account_size_is_rejected():
    with pytest.raises(ValueError):
        ProductionRiskPolicy(
            account_size=0.0,
        )


def test_max_total_risk_cannot_be_below_single_trade_risk():
    with pytest.raises(ValueError):
        ProductionRiskPolicy(
            risk_fraction=0.01,
            max_total_risk_fraction=0.005,
        )
