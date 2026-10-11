from __future__ import annotations

from datetime import datetime, timezone
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


class ForcedLongStrategy(BaseStrategy):
    """Minimal strategy used only to test the PaperTradingEngine risk path."""

    @property
    def name(self) -> str:
        return "TEST_LONG"

    @property
    def version(self) -> str:
        return "test-1.0"

    def generate_signal(self, market_data):
        return StrategySignal.LONG

    def evaluate(self, market_data):
        return StrategyDecision(
            signal=StrategySignal.LONG,
            action=StrategyAction.ENTER,
            reason="forced test entry",
        )

    def get_risk_stop_price(
        self,
        *,
        entry_price: float,
        signal: StrategySignal,
        market_data=None,
    ) -> float | None:
        return entry_price - 10.0


class RecordingRiskEngine(RiskEngine):
    """Risk engine that records the exact RiskRequest received."""

    def __init__(self, limits: RiskLimits):
        super().__init__(limits)
        self.requests = []

    def evaluate(self, request, *, trading_day, risk_per_trade_budget=None,
                 adaptive_quantity=None, adaptive_risk_limit=None):
        self.requests.append(request)
        return super().evaluate(
            request,
            trading_day=trading_day,
            risk_per_trade_budget=risk_per_trade_budget,
            adaptive_quantity=adaptive_quantity,
            adaptive_risk_limit=adaptive_risk_limit,
        )


class DummyExecutionEngine:
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


class DummyConflictResult:
    approved = True
    reason = "approved"


class DummyConflictEngine:
    def evaluate(self, entry_request):
        return DummyConflictResult()


class DummyLogger:
    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None

        return noop


class DummyBrokerOrder:
    broker_order_id = "TEST-ORDER-001"


class DummyBrokerExecution:
    def connect(self):
        return None

    def disconnect(self):
        return None

    def submit_entry(self, intent, *, quantity):
        return DummyBrokerOrder()

    def submit_exit(self, intent):
        return DummyBrokerOrder()

    def submission(self, broker_order_id):
        return None


class DummyBroker:
    def get_order(self, broker_order_id):
        return None


def make_engine(risk):
    strategy = ForcedLongStrategy()

    execution = DummyExecutionEngine()
    conflict = DummyConflictEngine()
    broker = DummyBroker()
    logger = DummyLogger()

    engine = PaperTradingEngine(
        strategies=[strategy],
        execution=execution,
        risk=risk,
        conflict=conflict,
        broker=broker,
        logger=logger,
        config=PaperEngineConfig(point_value=2.0),
    )

    engine.broker_execution = DummyBrokerExecution()

    return engine


def test_paper_engine_passes_strategy_stop_into_risk_request():
    limits = RiskLimits(
        risk_per_trade=100.0,
        max_total_risk=1000.0,
        max_daily_loss=1000.0,
        max_concurrent_positions=5,
        max_daily_trades=10,
        max_contracts=10,
    )

    risk = RecordingRiskEngine(limits)
    engine = make_engine(risk)

    engine.connect()

    timestamp = datetime(
        2026,
        9,
        26,
        14,
        30,
        tzinfo=timezone.utc,
    )

    market_data = {
        "timestamp": timestamp,
        "symbol": "MNQ",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
    }

    result = engine.process_bar(
        market_data,
        account_equity=50_000.0,
    )

    assert len(risk.requests) == 1

    request = risk.requests[0]

    assert request.strategy_name == "TEST_LONG"
    assert request.entry_price == pytest.approx(100.0)
    assert request.stop_price == pytest.approx(90.0)

    assert request.entry_price != request.stop_price

    assert result.submitted_orders


def test_paper_engine_risk_uses_stop_distance():
    limits = RiskLimits(
        risk_per_trade=100.0,
        max_total_risk=1000.0,
        max_daily_loss=1000.0,
        max_concurrent_positions=5,
        max_daily_trades=10,
        max_contracts=10,
    )

    risk = RecordingRiskEngine(limits)
    engine = make_engine(risk)

    engine.connect()

    timestamp = datetime(
        2026,
        9,
        26,
        14,
        31,
        tzinfo=timezone.utc,
    )

    market_data = {
        "timestamp": timestamp,
        "symbol": "MNQ",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
    }

    engine.process_bar(
        market_data,
        account_equity=50_000.0,
    )

    request = risk.requests[0]

    # 10 points * $2/point = $20 risk per contract.
    assert request.stop_price == pytest.approx(90.0)

    result = risk.evaluate(
        request,
        trading_day=timestamp.date(),
    )

    assert result.approved
    assert result.risk_per_contract == pytest.approx(20.0)
    assert result.quantity == 5
    assert result.total_risk == pytest.approx(100.0)
