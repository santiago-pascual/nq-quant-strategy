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
