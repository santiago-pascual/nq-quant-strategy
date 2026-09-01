from __future__ import annotations

import pytest

from src.strategies.base import (
    BaseStrategy,
    StrategyAction,
    StrategySignal,
)

from src.strategies.mean_reversion import (
    MRL1_CONFIG,
    MRL2_CONFIG,
    MRS2_CONFIG,
    MeanReversionStrategy,
)


def valid_context(
    *,
    hmm_state: int = 2,
    vol_percentile: float = 90.0,
    zscore: float = 2.5,
) -> dict:
    return {
        "hmm_state": hmm_state,
        "vol_percentile": vol_percentile,
        "zscore": zscore,
    }


# ============================================================================
# CONTRACT
# ============================================================================


def test_mean_reversion_implements_base_strategy_contract() -> None:
    strategy = MeanReversionStrategy()

    assert isinstance(strategy, BaseStrategy)
    assert strategy.name == "MRS2"
    assert strategy.version == "1.0.0"


# ============================================================================
# MRS2
# ============================================================================


def test_mrs2_generates_short_for_valid_context() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.0,
        )
    )

    assert decision.signal is StrategySignal.SHORT
    assert decision.action is StrategyAction.ENTER


def test_mrs2_rejects_wrong_hmm_state() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=1,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.HOLD


def test_mrs2_accepts_lower_volatility_boundary() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=80.0,
            zscore=2.0,
        )
    )

    assert decision.signal is StrategySignal.SHORT


def test_mrs2_rejects_upper_volatility_boundary() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=100.0,
            zscore=2.0,
        )
    )

    assert decision.signal is StrategySignal.FLAT


def test_mrs2_accepts_exact_zscore_threshold() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.0,
        )
    )

    assert decision.signal is StrategySignal.SHORT


def test_mrs2_rejects_zscore_below_threshold() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=90.0,
            zscore=1.999,
        )
    )

    assert decision.signal is StrategySignal.FLAT


# ============================================================================
# MRL1
# ============================================================================


def test_mrl1_generates_long_for_valid_context() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-2.5,
        )
    )

    assert decision.signal is StrategySignal.LONG
    assert decision.action is StrategyAction.ENTER


def test_mrl1_rejects_wrong_hmm_state() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=30.0,
            zscore=-3.0,
        )
    )

    assert decision.signal is StrategySignal.FLAT


def test_mrl1_rejects_wrong_volatility_bucket() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=1,
            vol_percentile=50.0,
            zscore=-3.0,
        )
    )

    assert decision.signal is StrategySignal.FLAT


def test_mrl1_accepts_exact_zscore_threshold() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-2.5,
        )
    )

    assert decision.signal is StrategySignal.LONG


def test_mrl1_rejects_zscore_above_threshold() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-2.499,
        )
    )

    assert decision.signal is StrategySignal.FLAT


# ============================================================================
# MRL2
# ============================================================================


def test_mrl2_generates_long_for_valid_context() -> None:
    strategy = MeanReversionStrategy(MRL2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=70.0,
            zscore=-3.5,
        )
    )

    assert decision.signal is StrategySignal.LONG
    assert decision.action is StrategyAction.ENTER


def test_mrl2_rejects_wrong_hmm_state() -> None:
    strategy = MeanReversionStrategy(MRL2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=1,
            vol_percentile=70.0,
            zscore=-4.0,
        )
    )

    assert decision.signal is StrategySignal.FLAT


def test_mrl2_rejects_wrong_volatility_bucket() -> None:
    strategy = MeanReversionStrategy(MRL2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=50.0,
            zscore=-4.0,
        )
    )

    assert decision.signal is StrategySignal.FLAT


def test_mrl2_accepts_exact_zscore_threshold() -> None:
    strategy = MeanReversionStrategy(MRL2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=70.0,
            zscore=-3.5,
        )
    )

    assert decision.signal is StrategySignal.LONG


def test_mrl2_rejects_zscore_above_threshold() -> None:
    strategy = MeanReversionStrategy(MRL2_CONFIG)

    decision = strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=70.0,
            zscore=-3.499,
        )
    )

    assert decision.signal is StrategySignal.FLAT


# ============================================================================
# CONFIGURATION INTEGRITY
# ============================================================================


def test_frozen_parameters_are_exact() -> None:
    assert MRS2_CONFIG.candidate_id == "C01"
    assert MRS2_CONFIG.hmm_state == 2
    assert MRS2_CONFIG.side == "SHORT"
    assert MRS2_CONFIG.volatility_low == 80.0
    assert MRS2_CONFIG.volatility_high == 100.0
    assert MRS2_CONFIG.zscore_threshold == 2.0
    assert MRS2_CONFIG.target_points == 5.0
    assert MRS2_CONFIG.stop_points == 2.0
    assert MRS2_CONFIG.horizon_bars == 5

    assert MRL1_CONFIG.candidate_id == "C02"
    assert MRL1_CONFIG.hmm_state == 1
    assert MRL1_CONFIG.side == "LONG"
    assert MRL1_CONFIG.volatility_low == 20.0
    assert MRL1_CONFIG.volatility_high == 40.0
    assert MRL1_CONFIG.zscore_threshold == 2.5
    assert MRL1_CONFIG.target_points == 5.0
    assert MRL1_CONFIG.stop_points == 2.0
    assert MRL1_CONFIG.horizon_bars == 20

    assert MRL2_CONFIG.candidate_id == "C06"
    assert MRL2_CONFIG.hmm_state == 2
    assert MRL2_CONFIG.side == "LONG"
    assert MRL2_CONFIG.volatility_low == 60.0
    assert MRL2_CONFIG.volatility_high == 80.0
    assert MRL2_CONFIG.zscore_threshold == 3.5
    assert MRL2_CONFIG.target_points == 5.0
    assert MRL2_CONFIG.stop_points == 2.0
    assert MRL2_CONFIG.horizon_bars == 2


def test_frozen_rr_is_2_5() -> None:
    assert MRS2_CONFIG.rr == 2.5
    assert MRL1_CONFIG.rr == 2.5
    assert MRL2_CONFIG.rr == 2.5


# ============================================================================
# MISSING / INVALID DATA
# ============================================================================


@pytest.mark.parametrize(
    "field",
    [
        "hmm_state",
        "vol_percentile",
        "zscore",
    ],
)
def test_missing_required_feature_returns_flat(field: str) -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    context = valid_context()

    del context[field]

    decision = strategy.evaluate(context)

    assert decision.signal is StrategySignal.FLAT
    assert decision.action is StrategyAction.HOLD


def test_boolean_hmm_state_is_rejected() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    context = valid_context()
    context["hmm_state"] = True

    decision = strategy.evaluate(context)

    assert decision.signal is StrategySignal.FLAT


# ============================================================================
# NO STATE / NO PARAMETER MUTATION
# ============================================================================


def test_strategy_does_not_mutate_frozen_configuration() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    before = strategy.config

    strategy.evaluate(
        valid_context(
            hmm_state=2,
            vol_percentile=90.0,
            zscore=3.0,
        )
    )

    after = strategy.config

    assert before == after


def test_repeated_evaluation_is_deterministic() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    context = valid_context(
        hmm_state=1,
        vol_percentile=30.0,
        zscore=-3.0,
    )

    first = strategy.evaluate(context)
    second = strategy.evaluate(context)

    assert first == second
