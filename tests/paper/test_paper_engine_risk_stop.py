from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.paper.engine import PaperTradingEngine
from src.strategies.base import StrategyAction, StrategyDecision, StrategySignal
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy


def test_mrl1_risk_stop_price() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    stop = strategy.get_risk_stop_price(
        entry_price=100.0,
        signal=StrategySignal.LONG,
        market_data={},
    )

    assert stop == pytest.approx(62.5)


def test_mrs2_risk_stop_price() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    stop = strategy.get_risk_stop_price(
        entry_price=100.0,
        signal=StrategySignal.SHORT,
        market_data={},
    )

    assert stop == pytest.approx(125.0)


def test_flat_signal_has_no_risk_stop() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    stop = strategy.get_risk_stop_price(
        entry_price=100.0,
        signal=StrategySignal.FLAT,
        market_data={},
    )

    assert stop is None


def test_mrl1_stop_moves_below_entry() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    entry_price = 18500.0

    stop = strategy.get_risk_stop_price(
        entry_price=entry_price,
        signal=StrategySignal.LONG,
        market_data={},
    )

    assert stop == pytest.approx(18462.5)
    assert stop < entry_price
    assert entry_price - stop == pytest.approx(37.5)


def test_mrs2_stop_moves_above_entry() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    entry_price = 18500.0

    stop = strategy.get_risk_stop_price(
        entry_price=entry_price,
        signal=StrategySignal.SHORT,
        market_data={},
    )

    assert stop == pytest.approx(18525.0)
    assert stop > entry_price
    assert stop - entry_price == pytest.approx(25.0)


def test_negative_entry_price_is_rejected() -> None:
    strategy = MeanReversionStrategy(MRL1_CONFIG)

    with pytest.raises(ValueError, match="entry_price must be positive"):
        strategy.get_risk_stop_price(
            entry_price=-100.0,
            signal=StrategySignal.LONG,
            market_data={},
        )


def test_zero_entry_price_is_rejected() -> None:
    strategy = MeanReversionStrategy(MRS2_CONFIG)

    with pytest.raises(ValueError, match="entry_price must be positive"):
        strategy.get_risk_stop_price(
            entry_price=0.0,
            signal=StrategySignal.SHORT,
            market_data={},
        )


def test_stop_geometry_matches_frozen_configs() -> None:
    mrl1 = MeanReversionStrategy(MRL1_CONFIG)
    mrs2 = MeanReversionStrategy(MRS2_CONFIG)

    entry = 18000.0

    mrl1_stop = mrl1.get_risk_stop_price(
        entry_price=entry,
        signal=StrategySignal.LONG,
        market_data={},
    )

    mrs2_stop = mrs2.get_risk_stop_price(
        entry_price=entry,
        signal=StrategySignal.SHORT,
        market_data={},
    )

    assert entry - mrl1_stop == pytest.approx(MRL1_CONFIG.stop_points)
    assert mrs2_stop - entry == pytest.approx(MRS2_CONFIG.stop_points)
