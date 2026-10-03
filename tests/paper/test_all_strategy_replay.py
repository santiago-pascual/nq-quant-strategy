from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.autonomous_runner import AutonomousPaperRunner
from src.paper.context_adapter import PaperMarketContextAdapter
from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.paper.logger import PaperEventLogger
from src.portfolio.conflict import PortfolioConflictEngine
from src.risk import RiskEngine, RiskLimits
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy
from src.strategies.orb.strategy import ORBStrategy
from src.strategies.s2r.fitting import S2FittedModel
from src.strategies.s2r.signal import BASE_FEATURES, S2SignalModel
from src.strategies.s2r.strategy import S2RStrategy


class SyntheticReplayContext:
    def __init__(self):
        self.bars_seen = 0
        self.windowed_hmm = None


class SyntheticReplayContextAdapter(PaperMarketContextAdapter):
    def __init__(self):
        self.context = SyntheticReplayContext()

    def configure_research_windows(self, windows):
        self.context.windowed_hmm = SimpleNamespace(windows=windows)

    def update(self, market_data):
        index = self.context.bars_seen
        self.context.bars_seen += 1
        enriched = dict(market_data)
        enriched.update(
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
            realized_vol_30=50.0,
            **{feature: 1.0 for feature in BASE_FEATURES},
        )
        if index == 32:
            enriched.update(hmm_state=1, vol_percentile=30.0, zscore=-2.6)
        if index == 35:
            enriched.update(hmm_state=2, vol_percentile=90.0, zscore=2.1)
            enriched.update({feature: -1.0 for feature in BASE_FEATURES})
        return enriched


class CountingPaperTradingEngine(PaperTradingEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_fills = 0

    def process_bar(self, market_data, **kwargs):
        result = super().process_bar(market_data, **kwargs)
        self.total_fills += len(result.fills)
        return result


def test_one_session_replay_processes_all_four_strategy_lifecycles(tmp_path):
    signal_model = S2SignalModel(
        thresholds={feature: 0.0 for feature in BASE_FEATURES},
        scales={feature: 1.0 for feature in BASE_FEATURES},
    )
    s2r = S2RStrategy(
        S2FittedModel(
            signal_model=signal_model,
            volatility_reference=tuple(float(value) for value in range(100)),
        )
    )
    strategies = [
        MeanReversionStrategy(MRL1_CONFIG),
        MeanReversionStrategy(MRS2_CONFIG),
        s2r,
        ORBStrategy(),
    ]
    broker = InMemoryBrokerAdapter()
    context_adapter = SyntheticReplayContextAdapter()
    engine = CountingPaperTradingEngine(
        strategies=strategies,
        execution=ExecutionEngine(),
        risk=RiskEngine(
            RiskLimits(
                risk_per_trade=250.0,
                max_total_risk=10_000.0,
                max_daily_loss=5_000.0,
                max_concurrent_positions=4,
                max_daily_trades=10,
                max_contracts=1,
            )
        ),
        conflict=PortfolioConflictEngine(max_concurrent_positions=4),
        broker=broker,
        logger=PaperEventLogger(tmp_path / "four-strategy.jsonl"),
        context_adapter=context_adapter,
        config=PaperEngineConfig(
            initial_equity=50_000.0,
            point_value=2.0,
            commission_per_contract=1.0,
            automatic_simulated_fills=True,
        ),
    )
    engine.connect()
    start = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
    bars = []
    for index in range(390):
        close = 100.0
        high = close + 0.5
        low = close - 0.5
        if index < 30:
            high, low = 101.0, 99.0
        elif index == 30:
            close, high, low = 101.0, 101.0, 100.0
        elif index == 31:
            close, high, low = 104.0, 105.0, 100.0
        elif index == 34:
            high, low = 126.0, 99.0
        elif index == 37:
            high, low = 118.0, 72.5

        bars.append(
            {
                "timestamp": start + timedelta(minutes=index),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
            }
        )

    runner = AutonomousPaperRunner(
        paper_engine=engine,
        context_adapter=context_adapter,
    )
    replay_stats = runner.run(pd.DataFrame(bars))

    assert replay_stats.processed_bars == 390
    assert context_adapter.context.bars_seen == 390
    assert engine.total_fills == 8
    assert {strategy.name for strategy in strategies} == {
        "MRL1",
        "MRS2",
        "S2R",
        "ORB",
    }
    order_names = {order.strategy_name for order in engine.execution.get_orders()}
    assert order_names == {"MRL1", "MRS2", "S2R", "ORB"}
    assert engine.execution.get_positions() == []
    assert engine.execution.get_closed_positions()
    assert engine.commissions == 8.0
    assert engine.realized_pnl == 113.0
    assert engine.account_equity == 50_105.0
    assert s2r.in_trade is False


def test_orb_reenters_once_after_next_session_reset(tmp_path):
    orb = ORBStrategy()
    engine = PaperTradingEngine(
        strategies=[orb],
        execution=ExecutionEngine(),
        risk=RiskEngine(
            RiskLimits(
                risk_per_trade=500.0,
                max_total_risk=10_000.0,
                max_daily_loss=5_000.0,
                max_concurrent_positions=1,
                max_daily_trades=10,
                max_contracts=1,
            )
        ),
        conflict=PortfolioConflictEngine(max_concurrent_positions=1),
        broker=InMemoryBrokerAdapter(),
        logger=PaperEventLogger(tmp_path / "orb-session-reset.jsonl"),
        config=PaperEngineConfig(
            initial_equity=50_000.0,
            point_value=2.0,
            commission_per_contract=0.0,
            automatic_simulated_fills=True,
        ),
    )
    engine.connect()
    start = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
    entries = []
    closed_positions = {}

    try:
        for session_index, opening_range_base in enumerate((100.0, 200.0)):
            session_start = start + timedelta(days=session_index)
            for minute in range(32):
                timestamp = session_start + timedelta(minutes=minute)
                if minute < 30:
                    high = opening_range_base + 1.0
                    low = opening_range_base - 1.0
                    close = opening_range_base
                elif minute == 30:
                    high = opening_range_base + 2.0
                    low = opening_range_base
                    close = opening_range_base + 1.0
                else:
                    high = opening_range_base + 5.0
                    low = opening_range_base + 1.0
                    close = opening_range_base + 4.0

                result = engine.process_bar(
                    {
                        "timestamp": timestamp,
                        "open": close,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": 1_000.0,
                    }
                )
                if result.decisions["ORB"].action.value == "enter":
                    entries.append(timestamp)
                for position in engine.execution.get_closed_positions():
                    closed_positions[id(position)] = position

        assert entries == [
            start + timedelta(minutes=30),
            start + timedelta(days=1, minutes=30),
        ]
        assert len(closed_positions) == 2
        assert orb.in_trade is False
    finally:
        engine.disconnect()


@pytest.mark.parametrize(
    ("strategy", "context"),
    [
        (
            MeanReversionStrategy(MRL1_CONFIG),
            {"hmm_state": 1, "vol_percentile": 30.0, "zscore": -3.0},
        ),
        (
            S2RStrategy(
                S2FittedModel(
                    signal_model=S2SignalModel(
                        thresholds={feature: 0.0 for feature in BASE_FEATURES},
                        scales={feature: 1.0 for feature in BASE_FEATURES},
                    ),
                    volatility_reference=tuple(float(value) for value in range(100)),
                )
            ),
            {
                "hmm_state": 2,
                "realized_vol_30": 50.0,
                **{feature: -1.0 for feature in BASE_FEATURES},
            },
        ),
    ],
)
def test_close_based_entries_fill_on_the_signal_bar(tmp_path, strategy, context):
    engine = PaperTradingEngine(
        strategies=[strategy],
        execution=ExecutionEngine(),
        risk=RiskEngine(
            RiskLimits(
                risk_per_trade=500.0,
                max_total_risk=10_000.0,
                max_daily_loss=5_000.0,
                max_concurrent_positions=1,
                max_daily_trades=10,
                max_contracts=1,
            )
        ),
        conflict=PortfolioConflictEngine(max_concurrent_positions=1),
        broker=InMemoryBrokerAdapter(),
        logger=PaperEventLogger(tmp_path / f"{strategy.name}-same-bar.jsonl"),
        config=PaperEngineConfig(
            initial_equity=50_000.0,
            point_value=2.0,
            commission_per_contract=0.0,
            automatic_simulated_fills=True,
        ),
    )
    timestamp = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
    result = None

    engine.connect()
    try:
        result = engine.process_bar(
            {
                "timestamp": timestamp,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000.0,
                **context,
            }
        )
    finally:
        engine.disconnect()

    assert result is not None
    assert result.timestamp == timestamp
    assert result.decisions[strategy.name].action.value == "enter"
    assert len(result.submitted_orders) == 1
    assert len(result.fills) == 1
    position = engine.execution.get_position(strategy.name)
    assert position is not None
    assert position.entry_price == 100.0
