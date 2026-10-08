from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engine import add_return_features, add_volatility_features
from src.paper.market_context import (
    CausalMarketContext,
    CausalOnlineHMMProvider,
    build_causal_context_features,
)
from src.paper.research_replay import research_causal_expanding_percentile
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)
from src.session_engine import add_session_information


def test_mr_context_matches_research_rth_features_with_eth_present():
    eastern = "America/New_York"
    timestamps = list(pd.date_range("2024-01-08 08:30", periods=60, freq="min", tz=eastern))
    timestamps += list(pd.date_range("2024-01-08 09:30", periods=35, freq="min", tz=eastern))
    close = 17000 + np.cumsum(np.sin(np.arange(len(timestamps)) / 4) * 0.7 + 0.05)
    raw = pd.DataFrame(
        {
            "timestamp ET": timestamps,
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.arange(len(close)) + 1,
        }
    )
    canonical = add_session_information(raw.copy())
    canonical = add_return_features(canonical)
    canonical = add_volatility_features(canonical)
    rth = canonical.loc[canonical["market_period"].eq("RTH")].copy()
    mr = build_mean_reversion_features(rth.copy())
    percentile = research_causal_expanding_percentile(rth["realized_vol_30"])
    expected_z = float(mr["zscore_30"].iloc[-1])
    expected_percentile = float(percentile.iloc[-1] * 100.0)

    context = CausalMarketContext()
    result = None
    for _, row in raw.iterrows():
        ts = pd.Timestamp(row["timestamp ET"]).tz_convert("UTC")
        result = context.update(
            {
                "timestamp": ts,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
        )

    assert result is not None
    assert result["market_period"] == "RTH"
    assert result["zscore"] == pytest.approx(expected_z)
    assert result["vol_percentile"] == pytest.approx(expected_percentile)


def test_eth_bars_do_not_produce_mr_features():
    context = CausalMarketContext()
    result = context.update(
        {
            "timestamp": pd.Timestamp("2024-01-08 14:29", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    )
    assert result["market_period"] == "ETH"
    assert result["zscore"] is None
    assert result["vol_percentile"] is None


def test_paper_context_uses_non_test_only_online_hmm_provider_by_default():
    context = CausalMarketContext()
    assert isinstance(context.hmm_provider, CausalOnlineHMMProvider)
    assert context.hmm_provider.TEST_ONLY is False


def test_indexed_context_features_use_rth_only_zscore():
    eastern = "America/New_York"
    timestamps = list(
        pd.date_range("2024-01-08 08:45", periods=45, freq="min", tz=eastern)
    )
    timestamps += list(
        pd.date_range("2024-01-08 09:30", periods=35, freq="min", tz=eastern)
    )
    close = 17000 + np.cumsum(np.cos(np.arange(len(timestamps)) / 5) * 0.4)
    raw = pd.DataFrame(
        {
            "timestamp ET": timestamps,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1,
        }
    )
    features = build_causal_context_features(raw)
    rth_closes = raw["close"].iloc[45:]
    expected = (rth_closes.iloc[-1] - rth_closes.tail(30).mean()) / rth_closes.tail(30).std()

    assert features["zscore_30"].iloc[-1] == pytest.approx(expected)
    assert features["zscore_30"].iloc[0:45].isna().all()
