from __future__ import annotations

import numpy as np
import pandas as pd

from src.paper.context_adapter import PaperMarketContextAdapter


class FakeContext:
    """
    Minimal deterministic context used only to test the adapter contract.
    """

    def __init__(self) -> None:
        self.calls = []

    def update(self, frame: pd.DataFrame):
        self.calls.append(frame.copy())

        row = frame.iloc[-1]

        return pd.Series(
            {
                "hmm_state": 2,
                "vol_percentile": 0.85,
                "zscore": 2.25,
                "realized_vol_5": 0.01,
                "realized_vol_15": 0.02,
                "realized_vol_30": 0.03,
                "realized_vol_60": 0.04,
            },
            name=row["timestamp"],
        )


def make_bar(timestamp="2026-01-05 14:30:00"):
    return {
        "timestamp": pd.Timestamp(timestamp),
        "open": 20000.0,
        "high": 20005.0,
        "low": 19995.0,
        "close": 20002.0,
        "volume": 1000.0,
    }


def test_adapter_preserves_raw_market_data_and_adds_context():
    context = FakeContext()
    adapter = PaperMarketContextAdapter(context=context)

    bar = make_bar()

    enriched = adapter.update(bar)

    # Original market data is preserved.
    assert enriched["timestamp"] == bar["timestamp"]
    assert enriched["open"] == bar["open"]
    assert enriched["high"] == bar["high"]
    assert enriched["low"] == bar["low"]
    assert enriched["close"] == bar["close"]
    assert enriched["volume"] == bar["volume"]

    # Causal context is added.
    assert enriched["hmm_state"] == 2
    assert enriched["vol_percentile"] == 0.85
    assert enriched["zscore"] == 2.25

    # Context received exactly one bar.
    assert len(context.calls) == 1
    assert len(context.calls[0]) == 1


def test_adapter_is_stateful():
    context = FakeContext()
    adapter = PaperMarketContextAdapter(context=context)

    adapter.update(make_bar("2026-01-05 14:30:00"))
    adapter.update(make_bar("2026-01-05 14:31:00"))
    adapter.update(make_bar("2026-01-05 14:32:00"))

    assert len(context.calls) == 3

    assert context.calls[0]["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 14:30:00")

    assert context.calls[1]["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 14:31:00")

    assert context.calls[2]["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 14:32:00")


def test_adapter_rejects_missing_columns():
    context = FakeContext()
    adapter = PaperMarketContextAdapter(context=context)

    bar = make_bar()
    del bar["close"]

    try:
        adapter.update(bar)
    except ValueError as exc:
        assert "close" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing close column.")


def test_adapter_rejects_non_numeric_price():
    context = FakeContext()
    adapter = PaperMarketContextAdapter(context=context)

    bar = make_bar()
    bar["close"] = "not-a-price"

    try:
        adapter.update(bar)
    except ValueError as exc:
        assert "close" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-numeric close.")
