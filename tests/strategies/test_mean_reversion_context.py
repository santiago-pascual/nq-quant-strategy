from __future__ import annotations

from src.strategies.mean_reversion.context import (
    MeanReversionContextBuilder,
)


def test_context_builder_preserves_required_fields() -> None:
    builder = MeanReversionContextBuilder()

    market_data = {
        "hmm_state": 2,
        "vol_percentile": 85.0,
        "zscore": 2.4,
        "close": 21000.0,
    }

    context = builder.build(market_data)

    assert context["hmm_state"] == 2
    assert context["vol_percentile"] == 85.0
    assert context["zscore"] == 2.4
    assert context["close"] == 21000.0


def test_context_builder_adds_missing_required_fields() -> None:
    builder = MeanReversionContextBuilder()

    context = builder.build(
        {
            "close": 21000.0,
        }
    )

    assert context["hmm_state"] is None
    assert context["vol_percentile"] is None
    assert context["zscore"] is None


def test_context_builder_does_not_mutate_input() -> None:
    builder = MeanReversionContextBuilder()

    market_data = {
        "hmm_state": 1,
        "vol_percentile": 30.0,
        "zscore": -2.7,
    }

    original = market_data.copy()

    builder.build(market_data)

    assert market_data == original
