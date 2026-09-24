from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from src.market_data import (
    MarketDataBar,
    ReplayMarketDataAdapter,
)


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-09-23T14:30:00Z",
                    "2026-09-23T14:31:00Z",
                    "2026-09-23T14:32:00Z",
                ]
            ),
            "symbol": [
                "MNQ",
                "MNQ",
                "MNQ",
            ],
            "open": [
                20000.0,
                20002.0,
                20001.0,
            ],
            "high": [
                20005.0,
                20006.0,
                20004.0,
            ],
            "low": [
                19998.0,
                20000.0,
                19999.0,
            ],
            "close": [
                20002.0,
                20001.0,
                20003.0,
            ],
            "volume": [
                100,
                150,
                120,
            ],
        }
    )


# ============================================================================
# MarketDataBar
# ============================================================================


def test_market_data_bar_accepts_valid_data():
    bar = MarketDataBar(
        timestamp=pd.Timestamp("2026-09-23T14:30:00Z").to_pydatetime(),
        symbol="MNQ",
        open=20000.0,
        high=20005.0,
        low=19998.0,
        close=20002.0,
        volume=100,
    )

    assert bar.symbol == "MNQ"
    assert bar.close == 20002.0
    assert bar.volume == 100


def test_market_data_bar_requires_timezone():
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        MarketDataBar(
            timestamp=pd.Timestamp("2026-09-23 14:30:00").to_pydatetime(),
            symbol="MNQ",
            open=20000.0,
            high=20005.0,
            low=19998.0,
            close=20002.0,
            volume=100,
        )


def test_market_data_bar_rejects_invalid_high():
    with pytest.raises(
        ValueError,
        match="high",
    ):
        MarketDataBar(
            timestamp=pd.Timestamp("2026-09-23T14:30:00Z").to_pydatetime(),
            symbol="MNQ",
            open=20000.0,
            high=19999.0,
            low=19998.0,
            close=20002.0,
            volume=100,
        )


def test_market_data_bar_rejects_invalid_low():
    with pytest.raises(
        ValueError,
        match="low",
    ):
        MarketDataBar(
            timestamp=pd.Timestamp("2026-09-23T14:30:00Z").to_pydatetime(),
            symbol="MNQ",
            open=20000.0,
            high=20005.0,
            low=20003.0,
            close=20002.0,
            volume=100,
        )


# ============================================================================
# Adapter initialization
# ============================================================================


def test_adapter_loads_dataframe():
    adapter = ReplayMarketDataAdapter(make_data())

    assert adapter.size == 3
    assert adapter.position == 0
    assert not adapter.exhausted


def test_adapter_requires_dataframe():
    with pytest.raises(TypeError):
        ReplayMarketDataAdapter(
            [
                {
                    "timestamp": "2026-09-23T14:30:00Z",
                }
            ]
        )


def test_missing_columns_are_rejected():
    data = make_data().drop(columns=["volume"])

    with pytest.raises(
        ValueError,
        match="Missing market-data columns",
    ):
        ReplayMarketDataAdapter(data)


# ============================================================================
# Chronology
# ============================================================================


def test_duplicate_timestamps_are_rejected():
    data = make_data()

    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]

    with pytest.raises(
        ValueError,
        match="Duplicate market-data timestamps",
    ):
        ReplayMarketDataAdapter(data)


def test_non_chronological_data_is_rejected():
    data = make_data()

    data = data.iloc[[1, 0, 2]].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="chronological",
    ):
        ReplayMarketDataAdapter(data)


# ============================================================================
# Sequential replay
# ============================================================================


def test_next_bar_returns_bars_in_order():
    adapter = ReplayMarketDataAdapter(make_data())

    first = adapter.next_bar()
    second = adapter.next_bar()
    third = adapter.next_bar()
    fourth = adapter.next_bar()

    assert isinstance(first, MarketDataBar)
    assert isinstance(second, MarketDataBar)
    assert isinstance(third, MarketDataBar)

    assert fourth is None

    assert first.close == 20002.0
    assert second.close == 20001.0
    assert third.close == 20003.0

    assert adapter.exhausted


def test_iterator_replays_all_bars():
    adapter = ReplayMarketDataAdapter(make_data())

    bars = list(adapter)

    assert len(bars) == 3

    assert [bar.close for bar in bars] == [
        20002.0,
        20001.0,
        20003.0,
    ]


def test_reset_restarts_replay():
    adapter = ReplayMarketDataAdapter(make_data())

    first = adapter.next_bar()

    assert first is not None
    assert first.close == 20002.0

    adapter.reset()

    again = adapter.next_bar()

    assert again is not None
    assert again.close == 20002.0
    assert adapter.position == 1


# ============================================================================
# Callback replay
# ============================================================================


def test_replay_callback_receives_every_bar():
    adapter = ReplayMarketDataAdapter(make_data())

    received = []

    count = adapter.replay(lambda bar: received.append(bar))

    assert count == 3
    assert len(received) == 3

    assert all(isinstance(bar, MarketDataBar) for bar in received)


def test_replay_callback_preserves_order():
    adapter = ReplayMarketDataAdapter(make_data())

    timestamps = []

    adapter.replay(lambda bar: timestamps.append(bar.timestamp))

    assert timestamps == sorted(timestamps)


# ============================================================================
# Historical inspection
# ============================================================================


def test_bars_returns_immutable_tuple():
    adapter = ReplayMarketDataAdapter(make_data())

    bars = adapter.bars()

    assert isinstance(bars, tuple)
    assert len(bars) == 3


def test_dataframe_returns_copy():
    adapter = ReplayMarketDataAdapter(make_data())

    data = adapter.dataframe()

    data.loc[0, "close"] = 999999.0

    original = adapter.dataframe()

    assert original.loc[0, "close"] == 20002.0


# ============================================================================
# Validation
# ============================================================================


def test_negative_volume_is_rejected():
    data = make_data()

    data.loc[0, "volume"] = -1

    with pytest.raises(
        ValueError,
        match="volume",
    ):
        ReplayMarketDataAdapter(data)


def test_non_positive_price_is_rejected():
    data = make_data()

    data.loc[0, "close"] = 0

    with pytest.raises(
        ValueError,
        match="close",
    ):
        ReplayMarketDataAdapter(data)


def test_replay_requires_callable():
    adapter = ReplayMarketDataAdapter(make_data())

    with pytest.raises(TypeError):
        adapter.replay("not callable")


# ============================================================================
# UTC contract
# ============================================================================


def test_replayed_timestamp_is_utc():
    adapter = ReplayMarketDataAdapter(make_data())

    bar = adapter.next_bar()

    assert bar is not None
    assert bar.timestamp.tzinfo is not None
    assert bar.timestamp.utcoffset() == timezone.utc.utcoffset(bar.timestamp)
