from __future__ import annotations

from src.strategies.base import (
    StrategyAction,
    StrategySignal,
)
from src.strategies.s2r.fitting import S2FittedModel
from src.strategies.s2r.recovery import RecoveryState
from src.strategies.s2r.signal import S2SignalModel
from src.strategies.s2r.strategy import S2RStrategy


def _fitted_s2_model() -> S2FittedModel:
    """
    Minimal frozen S2 model used by lifecycle tests.

    These tests exercise recovery behavior, not model fitting.
    The signal model therefore only needs valid frozen thresholds/scales.
    """

    thresholds = {
        "past_return_30": 0.0,
        "directional_pressure_30": 0.0,
        "close_location_30": 0.0,
        "normalized_momentum_30": 0.0,
    }

    scales = {
        "past_return_30": 1.0,
        "directional_pressure_30": 1.0,
        "close_location_30": 1.0,
        "normalized_momentum_30": 1.0,
    }

    return S2FittedModel(
        signal_model=S2SignalModel(
            thresholds=thresholds,
            scales=scales,
        ),
        volatility_reference=(10.0, 20.0, 30.0, 40.0, 50.0),
    )


def test_s2r_emits_exit_after_recovery() -> None:
    strategy = S2RStrategy(
        fitted_model=_fitted_s2_model(),
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

    strategy.update_trade(
        bar_index=1,
        close_r=0.20,
        mae_r=0.70,
    )

    decision = strategy.evaluate({})

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.EXIT
    assert strategy.recovery_state is RecoveryState.RECOVERED


def test_s2r_emits_exit_after_recovery_deadline() -> None:
    strategy = S2RStrategy(
        fitted_model=_fitted_s2_model(),
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

    decision = strategy.evaluate({})

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.EXIT
    assert strategy.recovery_state is RecoveryState.FAILED_TO_RECOVER


def test_s2r_does_not_enter_while_recovery_trade_is_active() -> None:
    strategy = S2RStrategy(
        fitted_model=_fitted_s2_model(),
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
