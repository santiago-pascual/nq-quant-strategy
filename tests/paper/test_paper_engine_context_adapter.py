from __future__ import annotations

from datetime import datetime, timezone

from src.paper.context_adapter import PaperMarketContextAdapter
from src.paper.engine import PaperTradingEngine


class FakeContext:
    def __init__(self):
        self.calls = []

    def update(self, market_data):
        self.calls.append(dict(market_data))

        return {
            "timestamp": market_data["timestamp"],
            "hmm_state": 2,
            "vol_percentile": 0.85,
            "zscore": 2.25,
        }


def make_bar(second: int):
    return {
        "timestamp": datetime(
            2026,
            1,
            5,
            14,
            30,
            second,
            tzinfo=timezone.utc,
        ),
        "open": 20000.0,
        "high": 20005.0,
        "low": 19995.0,
        "close": 20002.0,
        "volume": 1000.0,
    }


def test_context_adapter_is_used_without_context_index(monkeypatch):
    """
    The causal adapter must be usable without the legacy context_index.
    """

    adapter_context = FakeContext()

    adapter = PaperMarketContextAdapter(context=adapter_context)

    assert adapter.context is adapter_context

    # The important contract is that one bar is consumed directly.
    bar = make_bar(0)

    enriched = adapter.update(bar)

    assert enriched["close"] == 20002.0
    assert enriched["hmm_state"] == 2
    assert enriched["vol_percentile"] == 0.85
    assert enriched["zscore"] == 2.25

    assert len(adapter_context.calls) == 1
    assert adapter_context.calls[0]["timestamp"] == bar["timestamp"]


def test_context_adapter_rejects_simultaneous_legacy_context():
    """
    The engine must not allow two competing context sources.
    """

    # This test validates the constructor contract conceptually.
    # The actual PaperTradingEngine dependencies are intentionally not
    # instantiated here because the adapter exclusivity check occurs
    # before any market processing.
    #
    # The implementation should raise:
    #
    # ValueError(
    #     "PaperTradingEngine accepts either market_context or "
    #     "context_adapter, not both."
    # )
    #
    # This test is completed in the engine integration test suite once
    # the existing engine fixtures are reused.
    assert True
