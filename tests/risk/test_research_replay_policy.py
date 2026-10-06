from __future__ import annotations

from datetime import date

import pytest

from src.risk.engine import RiskEngine
from src.risk.policy import (
    RESEARCH_REPLAY_RISK_POLICY,
    XFA_50K_PRODUCTION_POLICY,
)
from src.risk.types import RiskDecision, RiskRequest


TRADING_DAY = date(2026, 1, 5)


def _request(strategy: str, entry: float, stop: float) -> RiskRequest:
    return RiskRequest(
        strategy_name=strategy,
        entry_price=entry,
        stop_price=stop,
        point_value=2.0,
        account_equity=50_000.0,
    )


def _engine(policy) -> RiskEngine:
    return RiskEngine(policy.to_risk_limits())


def _register_300_dollar_position(engine: RiskEngine) -> None:
    existing = engine.evaluate(
        _request("EXISTING", 250.0, 100.0),
        trading_day=TRADING_DAY,
        adaptive_quantity=1,
        adaptive_risk_limit=300.0,
    )
    assert existing.approved
    assert existing.total_risk == pytest.approx(300.0)
    engine.register_entry_fill(existing, fill_quantity=1)


def test_production_rejects_100_risk_when_300_is_already_open():
    engine = _engine(XFA_50K_PRODUCTION_POLICY)
    _register_300_dollar_position(engine)

    result = engine.evaluate(
        _request("CANDIDATE", 200.0, 150.0),
        trading_day=TRADING_DAY,
    )

    assert result.decision is RiskDecision.REJECTED
    assert result.reason == "maximum aggregate open risk would be exceeded"
    assert engine.limits.max_total_risk == pytest.approx(375.0)


def test_research_profile_allows_100_risk_when_300_is_already_open():
    engine = _engine(RESEARCH_REPLAY_RISK_POLICY)
    _register_300_dollar_position(engine)

    result = engine.evaluate(
        _request("CANDIDATE", 200.0, 150.0),
        trading_day=TRADING_DAY,
    )

    assert result.approved
    assert result.total_risk == pytest.approx(100.0)
    assert engine.open_risk + result.total_risk == pytest.approx(400.0)
    assert engine.limits.max_total_risk is None


def test_register_entry_fill_obeys_each_profile_aggregate_setting():
    production = _engine(XFA_50K_PRODUCTION_POLICY)
    research = _engine(RESEARCH_REPLAY_RISK_POLICY)
    candidate_request = _request("CANDIDATE", 200.0, 150.0)

    production_candidate = production.evaluate(
        candidate_request, trading_day=TRADING_DAY
    )
    research_candidate = research.evaluate(
        candidate_request, trading_day=TRADING_DAY
    )
    assert production_candidate.approved and research_candidate.approved

    _register_300_dollar_position(production)
    _register_300_dollar_position(research)

    with pytest.raises(RuntimeError, match="maximum aggregate open risk"):
        production.register_entry_fill(production_candidate, fill_quantity=1)

    research.register_entry_fill(research_candidate, fill_quantity=1)
    assert research.open_risk == pytest.approx(400.0)
