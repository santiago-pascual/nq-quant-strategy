from __future__ import annotations

import pytest

from src.strategies.base import StrategySignal
from src.strategies.mean_reversion import (
    FROZEN_CONFIGS,
    MRL1_CONFIG,
    MRS2_CONFIG,
    MeanReversionCandidate,
    MeanReversionConfig,
    MeanReversionStrategy,
)


# =============================================================================
# CONFIGURATION CONTRACT
# =============================================================================


def test_frozen_configs_contains_only_frozen_candidates():
    assert set(FROZEN_CONFIGS.keys()) == {
        MeanReversionCandidate.MRS2,
        MeanReversionCandidate.MRL1,
    }


def test_mrs2_configuration():
    assert MRS2_CONFIG.candidate_id == "MRS2_NEW"
    assert MRS2_CONFIG.name == "MRS2"
    assert MRS2_CONFIG.side == "SHORT"
    assert MRS2_CONFIG.hmm_state == 2
    assert MRS2_CONFIG.volatility_low == 80.0
    assert MRS2_CONFIG.volatility_high == 100.0
    assert MRS2_CONFIG.zscore_threshold == 2.0
    assert MRS2_CONFIG.target_points == 27.5
    assert MRS2_CONFIG.stop_points == 25.0
    assert MRS2_CONFIG.horizon_bars == 30
    assert MRS2_CONFIG.rr == pytest.approx(1.10)


def test_mrl1_configuration():
    assert MRL1_CONFIG.candidate_id == "MRL1_NEW"
    assert MRL1_CONFIG.name == "MRL1"
    assert MRL1_CONFIG.side == "LONG"
    assert MRL1_CONFIG.hmm_state == 1
    assert MRL1_CONFIG.volatility_low == 20.0
    assert MRL1_CONFIG.volatility_high == 40.0
    assert MRL1_CONFIG.zscore_threshold == 2.5
    assert MRL1_CONFIG.target_points == 25.0
    assert MRL1_CONFIG.stop_points == 37.5
    assert MRL1_CONFIG.horizon_bars == 8
    assert MRL1_CONFIG.rr == pytest.approx(25.0 / 37.5)


def test_configuration_rr_is_target_divided_by_stop():
    config = MeanReversionConfig(
        candidate_id="TEST",
        name="TEST",
        side="LONG",
        hmm_state=1,
        volatility_low=20.0,
        volatility_high=40.0,
        zscore_threshold=2.0,
        target_points=30.0,
        stop_points=20.0,
        horizon_bars=10,
    )

    assert config.rr == pytest.approx(1.5)


def test_configuration_rejects_non_positive_stop():
    config = MeanReversionConfig(
        candidate_id="TEST",
        name="TEST",
        side="LONG",
        hmm_state=1,
        volatility_low=20.0,
        volatility_high=40.0,
        zscore_threshold=2.0,
        target_points=30.0,
        stop_points=0.0,
        horizon_bars=10,
    )

    with pytest.raises(ValueError, match="stop_points must be positive"):
        _ = config.rr


def test_configurations_are_frozen():
    with pytest.raises(AttributeError):
        MRS2_CONFIG.target_points = 100.0


# =============================================================================
# STRATEGY CONSTRUCTION
# =============================================================================


def test_mrs2_strategy_constructs():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    assert strategy.config == MRS2_CONFIG


def test_mrl1_strategy_constructs():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    assert strategy.config == MRL1_CONFIG


# =============================================================================
# SIGNAL TEST HELPERS
# =============================================================================


def make_market_data(
    *,
    hmm_state: int = 2,
    vol_percentile: float = 90.0,
    zscore: float = 0.0,
) -> dict:
    return {
        "timestamp": "2026-01-01T14:30:00+00:00",
        "symbol": "MNQ",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
        "hmm_state": hmm_state,
        "vol_percentile": vol_percentile,
        "zscore": zscore,
    }


# =============================================================================
# MRS2 SIGNAL
# =============================================================================


def test_mrs2_generates_short_signal_when_all_conditions_match():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=90.0,
        zscore=2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.SHORT


def test_mrs2_requires_positive_zscore_threshold():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=90.0,
        zscore=-2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


def test_mrs2_rejects_wrong_hmm_state():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=90.0,
        zscore=2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


def test_mrs2_rejects_wrong_volatility_bucket():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=70.0,
        zscore=2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


def test_mrs2_accepts_lower_volatility_boundary():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=80.0,
        zscore=2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.SHORT


def test_mrs2_rejects_upper_volatility_boundary():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=100.0,
        zscore=2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


# =============================================================================
# MRL1 SIGNAL
# =============================================================================


def test_mrl1_generates_long_signal_when_all_conditions_match():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=30.0,
        zscore=-3.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.LONG


def test_mrl1_requires_negative_zscore_threshold():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=30.0,
        zscore=3.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


def test_mrl1_rejects_wrong_hmm_state():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=30.0,
        zscore=-3.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


def test_mrl1_rejects_wrong_volatility_bucket():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=50.0,
        zscore=-3.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


def test_mrl1_accepts_lower_volatility_boundary():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=20.0,
        zscore=-3.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.LONG


def test_mrl1_rejects_upper_volatility_boundary():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=40.0,
        zscore=-3.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


# =============================================================================
# Z-SCORE BOUNDARIES
# =============================================================================


def test_mrs2_accepts_exact_zscore_threshold():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=90.0,
        zscore=2.0,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.SHORT


def test_mrl1_accepts_exact_zscore_threshold():
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    data = make_market_data(
        hmm_state=1,
        vol_percentile=30.0,
        zscore=-2.5,
    )

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.LONG


# =============================================================================
# MISSING FEATURES
# =============================================================================


@pytest.mark.parametrize(
    "missing_feature",
    [
        "hmm_state",
        "vol_percentile",
        "zscore",
    ],
)
def test_missing_required_feature_returns_flat(missing_feature):
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=90.0,
        zscore=2.5,
    )

    del data[missing_feature]

    signal = strategy.generate_signal(data)

    assert signal is StrategySignal.FLAT


# =============================================================================
# DETERMINISM
# =============================================================================


def test_signal_generation_is_deterministic():
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    data = make_market_data(
        hmm_state=2,
        vol_percentile=90.0,
        zscore=2.5,
    )

    signal_1 = strategy.generate_signal(data)
    signal_2 = strategy.generate_signal(data)

    assert signal_1 == signal_2
