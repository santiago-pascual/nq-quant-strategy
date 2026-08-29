from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.backtesting import BacktestEngine, MarketReplay
from src.domain import MarketData, OrderSide, Position
from src.execution import BacktestExecutionEngine
from src.infrastructure import FixedClock
from src.risk import RiskDecision, RiskManager
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


class DummyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "DummyStrategy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_signal(self, market_data) -> StrategySignal:
        if market_data["close"] > market_data["open"]:
            return StrategySignal.LONG

        if market_data["close"] < market_data["open"]:
            return StrategySignal.SHORT

        return StrategySignal.FLAT


class DummyRiskManager(RiskManager):
    @property
    def name(self) -> str:
        return "DummyRiskManager"

    @property
    def version(self) -> str:
        return "1.0.0"

    def evaluate(
        self,
        decision: StrategyDecision,
        position: Position | None,
    ) -> RiskDecision:

        if decision.action is StrategyAction.HOLD:
            return RiskDecision(
                approved=True,
                order=None,
                reason="No position change required.",
            )

        side = (
            OrderSide.BUY if decision.signal is StrategySignal.LONG else OrderSide.SELL
        )

        from src.domain import Order

        order = Order(
            symbol="MNQ",
            side=side,
            quantity=1,
        )

        return RiskDecision(
            approved=True,
            order=order,
            reason="Dummy risk approval.",
        )

    def validate_order(self, order) -> RiskDecision:
        return RiskDecision(
            approved=True,
            order=order,
            reason="Dummy order validation.",
        )


def build_test_data() -> list[MarketData]:
    base_time = datetime(
        2026,
        1,
        1,
        14,
        30,
        tzinfo=timezone.utc,
    )

    return [
        MarketData(
            timestamp=base_time,
            symbol="MNQ",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        ),
        MarketData(
            timestamp=base_time + timedelta(minutes=1),
            symbol="MNQ",
            open=100.5,
            high=101.5,
            low=100.0,
            close=100.0,
        ),
        MarketData(
            timestamp=base_time + timedelta(minutes=2),
            symbol="MNQ",
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
        ),
    ]


def build_engine() -> BacktestEngine:
    clock = FixedClock(
        datetime(
            2026,
            1,
            1,
            14,
            30,
            tzinfo=timezone.utc,
        )
    )

    execution = BacktestExecutionEngine(
        market_prices={"MNQ": 100.0},
        clock=clock,
    )

    return BacktestEngine(
        strategy=DummyStrategy(),
        risk_manager=DummyRiskManager(),
        execution_engine=execution,
    )


def test_end_to_end_backtest_pipeline() -> None:
    replay = MarketReplay(build_test_data())
    engine = build_engine()

    results = engine.run(replay)

    assert len(results) == 3

    first = results[0]

    assert first.strategy_decision.signal is StrategySignal.LONG
    assert first.risk_decision.approved is True
    assert first.risk_decision.order is not None

    assert len(first.fills) == 1
    assert first.fills[0].side is OrderSide.BUY
    assert first.fills[0].price == 100.5

    assert first.position is not None
    assert first.position.quantity == 1


def test_execution_price_follows_replay() -> None:
    replay = MarketReplay(build_test_data())
    engine = build_engine()

    results = engine.run(replay)

    assert results[0].fills[0].price == 100.5
    assert results[1].fills[0].price == 100.0
    assert results[2].fills == ()


def test_replay_is_deterministic() -> None:
    market_data = build_test_data()

    first_run = build_engine().run(MarketReplay(market_data))

    second_run = build_engine().run(MarketReplay(market_data))

    assert first_run == second_run
