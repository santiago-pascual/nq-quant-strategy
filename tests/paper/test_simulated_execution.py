from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.paper.logger import PaperEventLogger
from src.portfolio.conflict import PortfolioConflictEngine
from src.risk import RiskEngine, RiskLimits
from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategyDecision,
    StrategySignal,
)


class OneEntryThenExit(BaseStrategy):
    @property
    def name(self) -> str:
        return "TEST"

    @property
    def version(self) -> str:
        return "1"

    def generate_signal(self, market_data):
        return StrategySignal.LONG

    def get_risk_stop_price(self, *, entry_price, signal, market_data=None):
        return entry_price - 10.0

    def on_market_data(self, market_data, position):
        return StrategyDecision(
            signal=StrategySignal.FLAT,
            action=StrategyAction.EXIT,
            reason="test exit",
        )


def _bar(timestamp: datetime, open_price: float, close: float | None = None):
    close = open_price if close is None else close
    return {
        "timestamp": timestamp,
        "open": open_price,
        "high": max(open_price, close) + 1.0,
        "low": min(open_price, close) - 1.0,
        "close": close,
        "volume": 1_000,
    }


def _engine(tmp_path):
    broker = InMemoryBrokerAdapter()
    engine = PaperTradingEngine(
        strategies=[OneEntryThenExit()],
        execution=ExecutionEngine(),
        risk=RiskEngine(
            RiskLimits(
                risk_per_trade=20,
                max_total_risk=100,
                max_daily_loss=100,
                max_concurrent_positions=1,
                max_daily_trades=10,
                max_contracts=1,
            )
        ),
        conflict=PortfolioConflictEngine(max_concurrent_positions=1),
        broker=broker,
        logger=PaperEventLogger(tmp_path / "simulated.jsonl"),
        config=PaperEngineConfig(
            point_value=2,
            initial_equity=10_000,
            tick_size=0.25,
            price_offset=0.5,
            commission_per_contract=2.5,
            automatic_simulated_fills=True,
        ),
    )
    engine.connect()
    return engine


def test_next_bar_open_fills_pending_orders_once_and_tracks_equity(tmp_path):
    engine = _engine(tmp_path)
    start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)

    signal_bar = engine.process_bar(_bar(start, 100))
    assert len(signal_bar.submitted_orders) == 1
    assert signal_bar.fills == ()
    assert engine.position("TEST") is None

    entry_bar = engine.process_bar(_bar(start + timedelta(minutes=1), 101))
    assert len(entry_bar.fills) == 1
    position = engine.position("TEST")
    assert position is not None
    assert position.entry_price == 101.5
    assert position.entry_timestamp == start + timedelta(minutes=1)
    assert entry_bar.account_equity == 9_997.5

    exit_signal_bar = engine.process_bar(_bar(start + timedelta(minutes=2), 102))
    assert len(exit_signal_bar.submitted_orders) == 1
    assert exit_signal_bar.fills == ()

    exit_fill_bar = engine.process_bar(_bar(start + timedelta(minutes=3), 104))
    assert len(exit_fill_bar.fills) == 1
    assert engine.position("TEST") is None
    assert exit_fill_bar.realized_pnl == 4.0
    assert exit_fill_bar.commissions == 5.0
    assert exit_fill_bar.account_equity == 9_999.0
