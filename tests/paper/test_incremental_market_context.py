from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engine import add_return_features, add_volatility_features
from src.paper.market_context import CausalMarketContext, MarketContextConfig


def test_incremental_features_match_batch_features_at_current_bar():
    start = pd.Timestamp("2026-01-05 09:30", tz="America/New_York")
    bars = []
    for index in range(140):
        close = 100.0 + index * 0.02 + np.sin(index / 4.0)
        bars.append(
            {
                "timestamp": (start + pd.Timedelta(minutes=index)).to_pydatetime(),
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000 + index,
            }
        )

    batch = add_volatility_features(add_return_features(pd.DataFrame(bars)))
    context = CausalMarketContext(
        MarketContextConfig(hmm_min_train_valid=10_000)
    )
    result = None
    for bar in bars:
        result = context.update(bar)

    assert result is not None
    for name in (
        "realized_vol_5",
        "realized_vol_15",
        "realized_vol_30",
        "realized_vol_60",
        "variance_ratio_5_30",
        "variance_ratio_5_60",
        "past_return_30",
    ):
        assert result[name] == pytest.approx(batch.iloc[-1][name], rel=1e-12)
    prior_volatility = batch["realized_vol_30"].iloc[:-1].dropna()
    current_volatility = batch["realized_vol_30"].iloc[-1]
    expected_percentile = (
        100.0
        * float((prior_volatility <= current_volatility).sum())
        / len(prior_volatility)
    )
    assert result["vol_percentile"] == pytest.approx(expected_percentile)
    recent_close = batch["close"].tail(30)
    expected_zscore = (
        (recent_close.iloc[-1] - recent_close.mean()) / recent_close.std(ddof=1)
    )
    assert result["zscore"] == pytest.approx(expected_zscore)
    assert context.bars_seen == len(bars)
