from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.paper.market_context import PaperMarketContextEngine
from src.strategies.base import StrategyAction, StrategySignal
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy


ROOT = Path(__file__).resolve().parents[2]


def test_mrl1_consumes_causal_market_context():
    raw = load_data()
    context_engine = PaperMarketContextEngine(raw)

    first_ready = context_engine.find_first_hmm_ready_index()
    assert first_ready is not None

    strategy = MeanReversionStrategy(config=MRL1_CONFIG)

    context = context_engine.process_bar(first_ready)

    decision = strategy.evaluate(context)

    assert decision is not None
    assert decision.signal in {
        StrategySignal.FLAT,
        StrategySignal.LONG,
        StrategySignal.SHORT,
    }
    assert decision.action in {
        StrategyAction.HOLD,
        StrategyAction.ENTER,
    }


def test_mrs2_consumes_causal_market_context():
    raw = load_data()
    context_engine = PaperMarketContextEngine(raw)

    first_ready = context_engine.find_first_hmm_ready_index()
    assert first_ready is not None

    strategy = MeanReversionStrategy(config=MRS2_CONFIG)

    context = context_engine.process_bar(first_ready)

    decision = strategy.evaluate(context)

    assert decision is not None
    assert decision.signal in {
        StrategySignal.FLAT,
        StrategySignal.LONG,
        StrategySignal.SHORT,
    }
    assert decision.action in {
        StrategyAction.HOLD,
        StrategyAction.ENTER,
    }


def test_mean_reversion_signal_conditions_are_respected():
    raw = load_data()
    context_engine = PaperMarketContextEngine(raw)

    first_ready = context_engine.find_first_hmm_ready_index()
    assert first_ready is not None

    mrl1 = MeanReversionStrategy(config=MRL1_CONFIG)
    mrs2 = MeanReversionStrategy(config=MRS2_CONFIG)

    checked = 0

    for index in range(first_ready, first_ready + 100):
        context = context_engine.process_bar(index)

        hmm_state = context["hmm_state"]
        vol_percentile = context["vol_percentile"]
        zscore = context["zscore"]

        assert pd.notna(hmm_state)
        assert pd.notna(vol_percentile)
        assert pd.notna(zscore)

        mrl1_decision = mrl1.evaluate(context)
        mrs2_decision = mrs2.evaluate(context)

        # MRL1 is LONG-only.
        if mrl1_decision.signal is StrategySignal.LONG:
            assert int(hmm_state) == MRL1_CONFIG.hmm_state
            assert (
                MRL1_CONFIG.volatility_low
                <= float(vol_percentile)
                < MRL1_CONFIG.volatility_high
            )
            assert float(zscore) <= -float(MRL1_CONFIG.zscore_threshold)

        assert mrl1_decision.signal is not StrategySignal.SHORT

        # MRS2 is SHORT-only.
        if mrs2_decision.signal is StrategySignal.SHORT:
            assert int(hmm_state) == MRS2_CONFIG.hmm_state
            assert (
                MRS2_CONFIG.volatility_low
                <= float(vol_percentile)
                < MRS2_CONFIG.volatility_high
            )
            assert float(zscore) >= float(MRS2_CONFIG.zscore_threshold)

        assert mrs2_decision.signal is not StrategySignal.LONG

        checked += 1

    assert checked == 100
