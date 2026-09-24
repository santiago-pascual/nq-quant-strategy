from __future__ import annotations

import pandas as pd
import pytest

from src.market_data.canonical import (
    CANONICAL_COLUMNS,
    CANONICAL_SYMBOL,
    CanonicalMarketDataAdapter,
)
from src.market_data.types import MarketDataBar


def make_dataframe(rows: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-05 14:30:00+00:00",
        periods=rows,
        freq="min",
    )

    return pd.DataFrame(
        {
            "timestamp ET": timestamps,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [10.0 + i for i in range(rows)],
            "symbol": [CANONICAL_SYMBOL] * rows,
        }
    )


def test_canonical_adapter_accepts_valid_dataframe():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    assert adapter.row_count == 3
    assert adapter.position == 0
    assert not adapter.exhausted


def test_canonical_adapter_contract():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    contract = adapter.contract

    assert contract.row_count == 3
    assert contract.symbol == CANONICAL_SYMBOL
    assert contract.first_timestamp == pd.Timestamp("2026-01-05 14:30:00+00:00")
    assert contract.last_timestamp == pd.Timestamp("2026-01-05 14:32:00+00:00")


def test_first_bar_preserves_ohlcv():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    bar = adapter.first_bar()

    assert isinstance(bar, MarketDataBar)
    assert bar.timestamp == pd.Timestamp("2026-01-05 14:30:00+00:00")
    assert bar.symbol == CANONICAL_SYMBOL
    assert bar.open == 100.0
    assert bar.high == 101.0
    assert bar.low == 99.0
    assert bar.close == 100.5
    assert bar.volume == 10.0


def test_last_bar_preserves_ohlcv():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    bar = adapter.last_bar()

    assert bar.timestamp == pd.Timestamp("2026-01-05 14:32:00+00:00")
    assert bar.open == 102.0
    assert bar.high == 103.0
    assert bar.low == 101.0
    assert bar.close == 102.5
    assert bar.volume == 12.0


def test_next_bar_advances_cursor():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    first = adapter.next_bar()
    second = adapter.next_bar()

    assert first is not None
    assert second is not None

    assert first.timestamp < second.timestamp
    assert adapter.position == 2


def test_exhaustion():
    adapter = CanonicalMarketDataAdapter(make_dataframe(2))

    assert adapter.next_bar() is not None
    assert adapter.next_bar() is not None
    assert adapter.next_bar() is None
    assert adapter.exhausted


def test_iterator_returns_all_bars_in_order():
    adapter = CanonicalMarketDataAdapter(make_dataframe(5))

    bars = list(adapter)

    assert len(bars) == 5
    assert [bar.timestamp for bar in bars] == sorted(bar.timestamp for bar in bars)


def test_reset_rewinds_adapter():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    first = adapter.next_bar()
    assert first is not None

    adapter.reset()

    again = adapter.next_bar()

    assert again is not None
    assert again == first
    assert adapter.position == 1


def test_bars_returns_tuple():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    bars = adapter.bars()

    assert isinstance(bars, tuple)
    assert len(bars) == 3


def test_dataframe_returns_copy():
    adapter = CanonicalMarketDataAdapter(make_dataframe())

    df = adapter.dataframe()

    df.loc[0, "close"] = 999999.0

    original = adapter.first_bar()

    assert original.close == 100.5


def test_missing_column_rejected():
    df = make_dataframe().drop(columns=["volume"])

    with pytest.raises(ValueError, match="missing columns"):
        CanonicalMarketDataAdapter(df)


def test_duplicate_timestamp_rejected():
    df = make_dataframe()

    df.loc[1, "timestamp ET"] = df.loc[0, "timestamp ET"]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        CanonicalMarketDataAdapter(df)


def test_nonchronological_data_rejected():
    df = make_dataframe()

    df.loc[2, "timestamp ET"] = df.loc[0, "timestamp ET"] - pd.Timedelta(minutes=1)

    with pytest.raises(ValueError, match="chronological"):
        CanonicalMarketDataAdapter(df)


def test_wrong_symbol_rejected():
    df = make_dataframe()

    df["symbol"] = "ES.v.0"

    with pytest.raises(ValueError, match="Unexpected canonical symbol"):
        CanonicalMarketDataAdapter(df)


def test_multiple_symbols_rejected():
    df = make_dataframe()

    df.loc[1, "symbol"] = "MNQ.v.1"

    with pytest.raises(ValueError, match="exactly one symbol"):
        CanonicalMarketDataAdapter(df)


def test_invalid_ohlc_rejected():
    df = make_dataframe()

    df.loc[0, "high"] = 98.0

    with pytest.raises(ValueError, match="OHLC relationships"):
        CanonicalMarketDataAdapter(df)


def test_negative_volume_rejected():
    df = make_dataframe()

    df.loc[0, "volume"] = -1.0

    with pytest.raises(ValueError, match="volume cannot be negative"):
        CanonicalMarketDataAdapter(df)


def test_nonpositive_price_rejected():
    df = make_dataframe()

    df.loc[0, "close"] = 0.0

    with pytest.raises(ValueError, match="prices must be positive"):
        CanonicalMarketDataAdapter(df)


def test_required_columns_constant_matches_contract():
    assert CANONICAL_COLUMNS == [
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
    ]


def test_bar_conversion_is_deterministic():
    df = make_dataframe()

    adapter_a = CanonicalMarketDataAdapter(df)
    adapter_b = CanonicalMarketDataAdapter(df)

    assert adapter_a.bars() == adapter_b.bars()
