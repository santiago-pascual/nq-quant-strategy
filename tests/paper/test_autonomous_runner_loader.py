from __future__ import annotations

import pandas as pd
import pytest

from src.paper import autonomous_runner


def test_canonical_loader_returns_utc_timestamp_column_without_changing_bars(
    monkeypatch,
):
    timestamps = pd.date_range(
        "2024-01-02 09:30",
        periods=3,
        freq="min",
        tz="America/New_York",
    )
    source = pd.DataFrame(
        {
            "timestamp ET": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10, 20, 30],
            "symbol": ["MNQ.v.0"] * 3,
        }
    )
    monkeypatch.setattr(autonomous_runner, "load_databento_mnq", lambda: source)

    result = autonomous_runner.load_canonical_raw_mnq()

    assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(
        result.columns
    )
    assert isinstance(result["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(result["timestamp"].dt.tz) == "UTC"
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert len(result) == len(source)
    pd.testing.assert_frame_equal(
        result[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
        source[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
    )
    assert result["symbol"].equals(source["symbol"])


def test_canonical_loader_normalizes_datetime_index(monkeypatch):
    timestamps = pd.date_range(
        "2024-01-02 14:30",
        periods=2,
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    source = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100, 101],
            "volume": [10, 20],
        },
        index=timestamps,
    )
    monkeypatch.setattr(autonomous_runner, "load_databento_mnq", lambda: source)

    result = autonomous_runner.load_canonical_raw_mnq()

    assert result["timestamp"].tolist() == list(timestamps)
    assert len(result) == len(source)


def test_canonical_loader_rejects_duplicate_timestamps(monkeypatch):
    timestamp = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    source = pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
            "volume": [10, 10],
        }
    )
    monkeypatch.setattr(autonomous_runner, "load_databento_mnq", lambda: source)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        autonomous_runner.load_canonical_raw_mnq()


def test_oos_loader_preserves_validated_instrument_mapping(monkeypatch):
    timestamp = pd.Timestamp("2026-08-27 00:00", tz="UTC")
    source = pd.DataFrame({
        "timestamp ET": [timestamp.tz_convert("America/New_York")],
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.0], "volume": [10], "symbol": ["MNQ.v.0"],
        "instrument_id": [42004946],
    })
    called = []

    def loader(*, include_instrument_id=False):
        called.append(include_instrument_id)
        return source.copy()

    monkeypatch.setattr(autonomous_runner, "load_databento_mnq", loader)
    result = autonomous_runner.load_canonical_raw_mnq(include_contract_metadata=True)
    assert called == [True]
    assert result["instrument_id"].tolist() == [42004946]
    assert result["symbol"].tolist() == ["MNQ.v.0"]
