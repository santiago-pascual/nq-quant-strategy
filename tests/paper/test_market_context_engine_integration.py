from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.paper.market_context import PaperMarketContextEngine


ROOT = Path(__file__).resolve().parents[2]


def test_market_context_produces_strategy_ready_context():
    """
    Minimal integration test.

    Validates that the causal market-context engine produces a context
    containing everything required by the downstream strategies.
    """

    raw = load_data()

    engine = PaperMarketContextEngine(raw)

    first_ready = engine.find_first_hmm_ready_index()

    assert first_ready is not None
    assert first_ready >= 0

    context = engine.process_bar(first_ready)

    required = {
        "timestamp",
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hmm_state",
        "vol_percentile",
        "zscore",
        "realized_vol_30",
        "past_return_30",
        "directional_pressure_30",
        "close_location_30",
        "normalized_momentum_30",
    }

    missing = required.difference(context)

    assert not missing, (
        "Market context is missing required downstream fields: "
        + ", ".join(sorted(missing))
    )

    assert pd.notna(context["hmm_state"])
    assert pd.notna(context["vol_percentile"])
    assert pd.notna(context["zscore"])
    assert pd.notna(context["realized_vol_30"])

    assert pd.notna(context["past_return_30"])
    assert pd.notna(context["directional_pressure_30"])
    assert pd.notna(context["close_location_30"])
    assert pd.notna(context["normalized_momentum_30"])


def test_market_context_is_chronological():
    """
    Process a small sequence of ready bars and verify strict chronology.
    """

    raw = load_data()

    engine = PaperMarketContextEngine(raw)

    first_ready = engine.find_first_hmm_ready_index()

    assert first_ready is not None

    contexts = []

    for index in range(first_ready, first_ready + 10):
        contexts.append(engine.process_bar(index))

    timestamps = [context["timestamp"] for context in contexts]

    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))


def test_market_context_zscore_matches_canonical_feature():
    """
    The strategy-facing `zscore` field must be exactly the canonical
    `zscore_30` value from the feature engine.
    """

    raw = load_data()

    engine = PaperMarketContextEngine(raw)

    first_ready = engine.find_first_hmm_ready_index()

    assert first_ready is not None

    context = engine.process_bar(first_ready)

    canonical_value = engine.features.iloc[first_ready]["zscore_30"]

    assert pd.notna(canonical_value)
    assert context["zscore"] == canonical_value
