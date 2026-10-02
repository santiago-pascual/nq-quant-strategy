from zoneinfo import ZoneInfo

import pandas as pd

from src.strategies.base import StrategyAction, StrategySignal
from src.strategies.s2r import S2RStrategy, fit_s2_model
from src.strategies.s2r.recovery import RecoveryState


def fitted_model():
    rows = [
        {
            "hmm_state": 2,
            "past_return_30": float(value),
            "directional_pressure_30": float(value) * 0.5,
            "close_location_30": float(value) * 0.25,
            "normalized_momentum_30": float(value) * 0.1,
            "realized_vol_30": float(value),
        }
        for value in range(-100, 101)
    ]

    return fit_s2_model(rows)


def test_recovery_exit_after_recovery():
    strategy = S2RStrategy(
        fitted_model=fitted_model(),
    )

    strategy.start_trade(
        entry_price=100.0,
        entry_bar=0,
    )

    result = strategy.update_trade(
        bar_index=0,
        close_r=-0.10,
        mae_r=0.70,
    )

    assert result.state is RecoveryState.ADVERSE

    result = strategy.update_trade(
        bar_index=1,
        close_r=0.20,
        mae_r=0.70,
    )

    assert result.state is RecoveryState.RECOVERED

    decision = strategy.evaluate({})

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.EXIT


def test_recovery_exit_after_deadline():
    strategy = S2RStrategy(
        fitted_model=fitted_model(),
    )

    strategy.start_trade(
        entry_price=100.0,
        entry_bar=0,
    )

    strategy.update_trade(
        bar_index=0,
        close_r=-0.10,
        mae_r=0.70,
    )

    for bar in range(1, 7):
        strategy.update_trade(
            bar_index=bar,
            close_r=0.0,
            mae_r=0.70,
        )

    assert strategy.recovery_state is RecoveryState.FAILED_TO_RECOVER

    decision = strategy.evaluate({})

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.EXIT


def test_active_recovery_blocks_new_entry():
    strategy = S2RStrategy(
        fitted_model=fitted_model(),
    )

    strategy.start_trade(
        entry_price=100.0,
        entry_bar=0,
    )

    decision = strategy.evaluate(
        {
            "hmm_state": 2,
            "past_return_30": -100.0,
            "directional_pressure_30": -50.0,
            "close_location_30": -25.0,
            "normalized_momentum_30": -10.0,
            "realized_vol_30": 50.0,
        }
    )

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.HOLD


def test_s2r_baseline_times_out_after_20_bars_on_reference_session():
    strategy = S2RStrategy(fitted_model=fitted_model())
    entry_price = 13_532.5
    strategy.start_trade(entry_price=entry_price, entry_bar=0)
    timestamps = pd.date_range(
        "2021-05-06 13:15",
        periods=20,
        freq="min",
        tz=ZoneInfo("America/New_York"),
    )
    reference_bars = [
        (13_541.0, 13_532.25, 13_538.5),
        (13_538.5, 13_532.75, 13_534.75),
        (13_538.0, 13_528.5, 13_532.0),
        (13_532.75, 13_522.5, 13_529.0),
        (13_530.25, 13_520.5, 13_525.25),
        (13_528.75, 13_522.5, 13_524.25),
        (13_530.25, 13_521.75, 13_529.0),
        (13_534.75, 13_528.75, 13_530.0),
        (13_530.5, 13_519.5, 13_522.0),
        (13_524.75, 13_516.25, 13_523.0),
        (13_530.5, 13_521.25, 13_524.0),
        (13_533.5, 13_522.75, 13_532.5),
        (13_536.5, 13_531.5, 13_534.0),
        (13_536.25, 13_530.0, 13_530.25),
        (13_532.5, 13_527.5, 13_529.5),
        (13_535.25, 13_527.25, 13_532.75),
        (13_535.0, 13_530.0, 13_532.5),
        (13_535.75, 13_528.0, 13_534.5),
        (13_539.0, 13_532.25, 13_538.0),
        (13_542.75, 13_534.5, 13_540.25),
    ]

    for index, (timestamp, (high, low, close)) in enumerate(
        zip(timestamps, reference_bars, strict=True),
        start=1,
    ):
        decision = strategy.on_market_data(
            {
                "timestamp": timestamp.to_pydatetime(),
                "high": high,
                "low": low,
                "close": close,
            },
            position=object(),
        )

        if index < strategy.config.horizon_bars:
            assert decision.action is StrategyAction.HOLD
        else:
            assert decision.action is StrategyAction.EXIT
            assert timestamp == pd.Timestamp(
                "2021-05-06 13:34",
                tz=ZoneInfo("America/New_York"),
            )
            assert decision.reason == (
                "S2 baseline timeout "
                "(NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)"
            )
            assert strategy.get_exit_fill_price(market_data={"close": close}) == close
            assert close == 13_540.25
            assert strategy.recovery_state is RecoveryState.INITIAL


def test_s2r_baseline_stop_and_target_exit_at_frozen_prices():
    cases = [
        (
            {"high": 125.0, "low": 99.0, "close": 124.0},
            125.0,
            "S2 baseline stop",
        ),
        (
            {"high": 101.0, "low": 56.25, "close": 60.0},
            56.25,
            "S2 baseline target",
        ),
    ]

    for market_data, expected_price, expected_reason in cases:
        strategy = S2RStrategy(fitted_model=fitted_model())
        strategy.start_trade(entry_price=100.0, entry_bar=0)

        decision = strategy.on_market_data(market_data, position=object())

        assert decision.action is StrategyAction.EXIT
        assert decision.reason.startswith(expected_reason)
        assert strategy.get_exit_fill_price(market_data=market_data) == expected_price


def test_s2r_baseline_uses_stop_when_stop_and_target_hit_together():
    strategy = S2RStrategy(fitted_model=fitted_model())
    strategy.start_trade(entry_price=100.0, entry_bar=0)
    market_data = {"high": 125.0, "low": 56.25, "close": 100.0}

    decision = strategy.on_market_data(market_data, position=object())

    assert decision.action is StrategyAction.EXIT
    assert decision.reason.startswith("S2 baseline both-hit conservative stop")
    assert strategy.get_exit_fill_price(market_data=market_data) == 125.0


def test_s2r_recovery_exit_keeps_precedence_over_baseline_exit():
    strategy = S2RStrategy(fitted_model=fitted_model())
    strategy.start_trade(entry_price=100.0, entry_bar=0)

    adverse = strategy.on_market_data(
        {"high": 117.5, "low": 99.0, "close": 110.0},
        position=object(),
    )
    recovered = strategy.on_market_data(
        {"high": 125.0, "low": 94.0, "close": 94.0},
        position=object(),
    )

    assert adverse.action is StrategyAction.HOLD
    assert recovered.action is StrategyAction.EXIT
    assert recovered.reason == "S2R recovery recovered"
    assert strategy.get_exit_fill_price(market_data={"close": 94.0}) == 94.0
    assert strategy.recovery_state is RecoveryState.RECOVERED
