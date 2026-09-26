from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.models.regime import HMM_FEATURES
from src.paper.market_context import (
    CausalMarketContext,
    MarketContextConfig,
)


def make_bar(
    timestamp,
    close,
    volume=1000.0,
):
    return {
        "timestamp": timestamp,
        "symbol": "MNQ",
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
    }


def make_bars(n=700):
    timestamp = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    bars = []

    for i in range(n):
        close = 100.0 + 0.05 * i + 2.0 * np.sin(i / 17.0) + 0.5 * np.sin(i / 5.0)

        bars.append(
            make_bar(
                timestamp + timedelta(minutes=i),
                close,
                volume=1000.0 + (i % 100),
            )
        )

    return bars


def make_feature_frame(n=1200):
    """
    Deterministic but non-degenerate HMM feature fixture.

    The six HMM variables deliberately have different scales,
    frequencies, amplitudes and nonlinear components so that the
    covariance matrices are well-conditioned enough for hmmlearn.
    """

    bars = make_bars(n)

    frame = pd.DataFrame(bars)

    frame["canonical_timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )

    x = np.arange(n, dtype=float)

    # --------------------------------------------------------------
    # Independent deterministic volatility-like processes
    # --------------------------------------------------------------

    frame["realized_vol_5"] = (
        0.010
        + 0.0015 * np.sin(x / 13.0)
        + 0.0008 * np.cos(x / 31.0)
        + 0.0002 * np.sin(x / 3.0)
    )

    frame["realized_vol_15"] = (
        0.020
        + 0.0020 * np.cos(x / 19.0)
        + 0.0010 * np.sin(x / 47.0)
        + 0.0003 * np.cos(x / 7.0)
    )

    frame["realized_vol_30"] = (
        0.030
        + 0.0025 * np.sin(x / 29.0)
        + 0.0012 * np.cos(x / 61.0)
        + 0.0004 * np.sin(x / 11.0)
    )

    frame["realized_vol_60"] = (
        0.040
        + 0.0030 * np.cos(x / 37.0)
        + 0.0014 * np.sin(x / 71.0)
        + 0.0005 * np.cos(x / 9.0)
    )

    frame["variance_ratio_5_30"] = (
        0.80
        + 0.08 * np.sin(x / 17.0)
        + 0.04 * np.cos(x / 43.0)
        + 0.015 * np.sin(x / 5.0)
    )

    frame["variance_ratio_5_60"] = (
        0.90
        + 0.07 * np.cos(x / 23.0)
        + 0.05 * np.sin(x / 53.0)
        + 0.012 * np.cos(x / 8.0)
    )

    return frame


def test_online_context_warms_up():
    context = CausalMarketContext(
        MarketContextConfig(
            hmm_min_train_valid=100,
            zscore_window=30,
            volatility_percentile_window=200,
        )
    )

    bars = make_bars(150)

    first = context.update(bars[0])

    assert first["hmm_state"] is None
    assert first["market_context_ready"] is False

    for bar in bars[1:]:
        result = context.update(bar)

    assert result["zscore"] is not None


def test_online_context_produces_complete_context():
    context = CausalMarketContext(
        MarketContextConfig(
            hmm_min_train_valid=100,
            zscore_window=30,
            volatility_percentile_window=200,
        )
    )

    last = None

    for bar in make_bars(700):
        last = context.update(bar)

    assert last is not None

    assert last["zscore"] is not None
    assert np.isfinite(last["zscore"])

    assert last["vol_percentile"] is not None
    assert 0.0 <= last["vol_percentile"] <= 100.0

    assert last["hmm_state"] in {0, 1, 2}

    assert last["market_context_ready"] is True

    for feature in HMM_FEATURES:
        assert feature in last
        assert last[feature] is not None
        assert np.isfinite(last[feature])


def test_research_window_uses_strict_pre_oos_training():
    context = CausalMarketContext(
        MarketContextConfig(
            hmm_min_train_valid=500,
        )
    )

    features = make_feature_frame(900)

    oos_start = features["canonical_timestamp"].iloc[700]

    oos_end = features["canonical_timestamp"].iloc[799]

    result = context.fit_research_window(
        features,
        window=1,
        oos_start=oos_start,
        oos_end=oos_end,
    )

    assert not result.empty

    assert result["window"].nunique() == 1
    assert result["window"].iloc[0] == 1

    assert result["timestamp"].min() >= oos_start
    assert result["timestamp"].max() <= oos_end

    assert set(result["hmm_state"].unique()).issubset({0, 1, 2})

    assert 1 in context.research_models


def test_research_windows_fit_independent_models():
    context = CausalMarketContext(
        MarketContextConfig(
            hmm_min_train_valid=500,
        )
    )

    features = make_feature_frame(1200)

    timestamps = features["canonical_timestamp"]

    windows = pd.DataFrame(
        {
            "window": [1, 2],
            "oos_start": [
                timestamps.iloc[700],
                timestamps.iloc[900],
            ],
            "oos_end": [
                timestamps.iloc[799],
                timestamps.iloc[999],
            ],
        }
    )

    result = context.fit_research_windows(
        features,
        windows,
    )

    assert not result.empty

    assert set(result["window"].unique()) == {1, 2}

    assert set(context.research_models.keys()) == {1, 2}

    assert context.research_models[1] is not context.research_models[2]


def test_research_window_rejects_insufficient_history():
    context = CausalMarketContext(
        MarketContextConfig(
            hmm_min_train_valid=500,
        )
    )

    features = make_feature_frame(600)

    oos_start = features["canonical_timestamp"].iloc[300]

    oos_end = features["canonical_timestamp"].iloc[399]

    try:
        context.fit_research_window(
            features,
            window=1,
            oos_start=oos_start,
            oos_end=oos_end,
        )
    except ValueError as exc:
        assert "insufficient causal HMM training data" in str(exc)
    else:
        raise AssertionError("Expected insufficient-training-history failure.")
