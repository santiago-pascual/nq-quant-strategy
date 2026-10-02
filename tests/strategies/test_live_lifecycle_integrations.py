from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.strategies.base import StrategyAction, StrategySignal
from src.strategies.mean_reversion.config import MRL1_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy
from src.strategies.orb.strategy import ORBStrategy
from src.strategies.s2r.config import S2RConfig
from src.strategies.s2r.fitting import S2FittedModel
from src.strategies.s2r.signal import BASE_FEATURES, S2SignalModel
from src.strategies.s2r.strategy import S2RStrategy


def test_mean_reversion_strategy_emits_active_target_exit():
    strategy = MeanReversionStrategy(MRL1_CONFIG)
    position = SimpleNamespace(
        entry_price=100.0,
        side=StrategySignal.LONG,
    )
    strategy.on_fill(market_data={}, position=position)

    decision = strategy.on_market_data(
        {"high": 125.0, "low": 99.0, "close": 120.0},
        position,
    )
    assert decision.action is StrategyAction.EXIT
    assert decision.reason == "mean-reversion target"
    strategy.on_exit()
    assert strategy._trade_state is None


def test_s2r_uses_training_fitted_volatility_scale_not_context_percentile():
    signal_model = S2SignalModel(
        thresholds={feature: 0.0 for feature in BASE_FEATURES},
        scales={feature: 1.0 for feature in BASE_FEATURES},
    )
    fitted = S2FittedModel(
        signal_model=signal_model,
        volatility_reference=(10.0, 20.0, 30.0, 40.0, 50.0),
    )
    strategy = S2RStrategy(fitted, S2RConfig())
    market_data = {
        "hmm_state": 2,
        "vol_percentile": 95.0,
        "realized_vol_30": 20.0,
        **{feature: -1.0 for feature in BASE_FEATURES},
    }

    assert strategy.generate_signal(market_data) is StrategySignal.SHORT


def test_s2r_recovery_lifecycle_exits_at_recovery_deadline():
    signal_model = S2SignalModel(
        thresholds={feature: 0.0 for feature in BASE_FEATURES},
        scales={feature: 1.0 for feature in BASE_FEATURES},
    )
    strategy = S2RStrategy(
        S2FittedModel(signal_model, tuple(float(i) for i in range(1, 101)))
    )
    position = SimpleNamespace(
        entry_price=100.0,
        side=StrategySignal.SHORT,
    )
    strategy.on_fill(market_data={}, position=position)

    first = strategy.on_market_data(
        {"high": 118.0, "low": 100.0, "close": 100.0},
        position,
    )
    assert first.action is StrategyAction.HOLD
    last = first
    for _ in range(6):
        last = strategy.on_market_data(
            {"high": 100.0, "low": 100.0, "close": 100.0},
            position,
        )
    assert last.action is StrategyAction.EXIT
    assert "failed_to_recover" in (last.reason or "")


def test_orb_live_path_uses_new_york_time_and_explicit_touch_lifecycle():
    strategy = ORBStrategy()
    timestamp = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)

    def bar(index: int, *, high: float, low: float, close: float) -> dict:
        return {
            "timestamp": timestamp + timedelta(minutes=index),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        }

    for index in range(30):
        decision = strategy.evaluate(
            bar(index, high=101.0, low=99.0, close=100.0)
        )
        assert decision.action is StrategyAction.HOLD

    entry_bar = bar(30, high=101.0, low=100.0, close=101.0)
    decision = strategy.evaluate(entry_bar)
    assert decision.action is StrategyAction.ENTER
    assert strategy.get_entry_fill_price(
        signal=decision.signal,
        market_data=entry_bar,
    ) == 101.0

    position = SimpleNamespace(
        side=StrategySignal.LONG,
        entry_price=101.0,
    )
    strategy.on_fill(market_data=entry_bar, position=position)
    exit_bar = bar(31, high=105.0, low=100.0, close=104.0)
    exit_decision = strategy.on_market_data(exit_bar, position)
    assert exit_decision.action is StrategyAction.EXIT
    assert strategy.get_exit_fill_price(market_data=exit_bar) == 105.0
