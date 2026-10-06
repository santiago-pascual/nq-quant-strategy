from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.feature_engine import add_return_features, add_volatility_features
from src.paper.research_replay import _fit_s2r_models
from src.research.direction_features import add_directional_features
from src.session_engine import add_session_information
from src.strategies.s2r.signal import BASE_FEATURES
from src.strategies.s2r.strategy import S2RStrategy


def test_20210728_s2r_window_model_matches_validated_entry_attribution():
    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "mnq" / "ohlcv_1m"
    files = [
        next(data_dir.glob(f"*{year}0101*{year}1231*.csv.zst"), None)
        for year in (2019, 2020, 2021)
    ]
    if any(path is None for path in files):
        pytest.skip("Canonical 2019–2021 MNQ source bars are unavailable.")

    frames = []
    for path in files:
        assert path is not None
        frame = pd.read_csv(
            path,
            compression="zstd",
            usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
        )
        frame["timestamp ET"] = pd.to_datetime(
            frame["ts_event"], unit="ns", utc=True
        ).dt.tz_convert("America/New_York")
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] / 1_000_000_000
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values("timestamp ET", kind="mergesort").reset_index(drop=True)
    market = add_directional_features(
        add_volatility_features(
            add_return_features(add_session_information(raw))
        )
    )
    market = market.loc[market["market_period"].eq("RTH")].copy()
    market["timestamp"] = market["timestamp ET"].dt.tz_convert("UTC")
    market = market[
        [
            "timestamp",
            "realized_vol_5",
            "realized_vol_15",
            "realized_vol_30",
            "realized_vol_60",
            "variance_ratio_5_30",
            "variance_ratio_5_60",
            *BASE_FEATURES,
        ]
    ]
    schedule = pd.DataFrame(
        [
            {
                "window": 1,
                "train_start": "2019-05-06T13:30:00Z",
                "train_end": "2021-05-06T13:30:00Z",
                "oos_start": "2021-05-06T13:30:00Z",
                "oos_end": "2021-08-06T13:30:00Z",
            }
        ]
    )

    models, state_map = _fit_s2r_models(market, schedule)
    timestamp = pd.Timestamp("2021-07-28T20:05:00Z")
    row = market.loc[market["timestamp"].eq(timestamp)].iloc[0]
    mapped = state_map.loc[state_map["timestamp"].eq(timestamp)].iloc[0]
    strategy = S2RStrategy()
    strategy.set_fitted_models_for_bar({1: models[1]})
    context = {
        "hmm_state": 0,  # Shared 08B provider is deliberately irrelevant to S2R.
        "s2r_hmm_states": mapped["s2r_hmm_states"],
        "realized_vol_30": float(row["realized_vol_30"]),
        **{feature: float(row[feature]) for feature in BASE_FEATURES},
    }

    evaluation = strategy.evaluate_entry_window(
        context, model=models[1], window=1
    )

    assert mapped["s2r_hmm_states"] == ((1, 2),)
    assert evaluation["hmm_state"] == 2
    assert evaluation["qualifies"] is True
    assert evaluation["quality"] == pytest.approx(1.0, abs=1e-12)
    assert evaluation["volatility_percentile"] == pytest.approx(
        0.4143328550283135, abs=1e-12
    )
