from __future__ import annotations

import pandas as pd
import pytest

from src.market_data.realtime import (
    InMemoryMarketDataStream,
    RealTimeMarketDataAdapter,
)
from src.market_data.types import MarketDataBar


SYMBOL = "MNQ.v.0"


def make_bar(
    minute: int = 0,
    symbol: str = SYMBOL,
    close: float = 100.5,
) -> MarketDataBar:
    timestamp = pd.Timestamp("2026-01-05 14:30:00+00:00") + pd.Timedelta(minutes=minute)

    open_price = 100.0

    # Keep the fixture internally valid for every close value.
    high = max(open_price, close) + 0.5
    low = min(open_price, close) - 0.5

    return MarketDataBar(
        timestamp=timestamp,
        symbol=symbol,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=10.0,
    )


def test_stream_starts_disconnected():
    stream = InMemoryMarketDataStream()

    assert not stream.is_connected()


def test_connect_changes_state():
    stream = InMemoryMarketDataStream()

    stream.connect()

    assert stream.is_connected()


def test_disconnect_changes_state():
    stream = InMemoryMarketDataStream()

    stream.connect()
    stream.disconnect()

    assert not stream.is_connected()


def test_subscribe_and_unsubscribe():
    stream = InMemoryMarketDataStream()

    stream.subscribe(SYMBOL)

    stream.connect()

    assert list(stream.bars()) == []

    stream.unsubscribe(SYMBOL)

    assert list(stream.bars()) == []


def test_empty_symbol_rejected_by_stream():
    stream = InMemoryMarketDataStream()

    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        stream.subscribe("")


def test_adapter_requires_stream():
    with pytest.raises(ValueError, match="stream cannot be None"):
        RealTimeMarketDataAdapter(None)


def test_adapter_connects():
    stream = InMemoryMarketDataStream()
    adapter = RealTimeMarketDataAdapter(stream)

    adapter.connect()

    assert adapter.connected


def test_adapter_disconnects():
    stream = InMemoryMarketDataStream()
    adapter = RealTimeMarketDataAdapter(stream)

    adapter.connect()
    adapter.disconnect()

    assert not adapter.connected


def test_adapter_subscription():
    stream = InMemoryMarketDataStream()
    adapter = RealTimeMarketDataAdapter(stream)

    adapter.connect()
    adapter.subscribe(SYMBOL)

    assert SYMBOL in adapter.subscriptions

    adapter.unsubscribe(SYMBOL)

    assert SYMBOL not in adapter.subscriptions


def test_adapter_rejects_empty_subscription():
    stream = InMemoryMarketDataStream()
    adapter = RealTimeMarketDataAdapter(stream)

    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        adapter.subscribe("")


def test_callback_receives_bar():
    bar = make_bar()

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    received = []

    def callback(received_bar):
        received.append(received_bar)

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(callback)

    processed = adapter.run_once()

    assert processed == 1
    assert received == [bar]


def test_last_bar_is_updated():
    bar = make_bar()

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    adapter.connect()
    adapter.subscribe(SYMBOL)

    assert adapter.last_bar is None

    adapter.run_once()

    assert adapter.last_bar == bar


def test_multiple_bars_preserve_order():
    bars = [
        make_bar(0, close=100.5),
        make_bar(1, close=101.5),
        make_bar(2, close=102.5),
    ]

    stream = InMemoryMarketDataStream(bars)
    adapter = RealTimeMarketDataAdapter(stream)

    received = []

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(received.append)

    processed = adapter.run_once()

    assert processed == 3
    assert received == bars


def test_unsubscribed_symbol_is_not_delivered():
    bar = make_bar(symbol="ES.v.0")

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    received = []

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(received.append)

    processed = adapter.run_once()

    assert processed == 0
    assert received == []


def test_not_connected_run_rejected():
    stream = InMemoryMarketDataStream([make_bar()])
    adapter = RealTimeMarketDataAdapter(stream)

    adapter.subscribe(SYMBOL)

    with pytest.raises(
        RuntimeError,
        match="not connected",
    ):
        adapter.run_once()


def test_stream_not_connected_bars_rejected():
    stream = InMemoryMarketDataStream([make_bar()])

    with pytest.raises(
        RuntimeError,
        match="not connected",
    ):
        list(stream.bars())


def test_stream_add_bar():
    stream = InMemoryMarketDataStream()

    bar = make_bar()

    stream.add_bar(bar)
    stream.connect()
    stream.subscribe(SYMBOL)

    assert list(stream.bars()) == [bar]


def test_stream_rejects_invalid_bar():
    stream = InMemoryMarketDataStream()

    with pytest.raises(
        TypeError,
        match="bar must be a MarketDataBar",
    ):
        stream.add_bar("not a bar")


def test_adapter_rejects_invalid_bar():
    stream = InMemoryMarketDataStream()
    adapter = RealTimeMarketDataAdapter(stream)

    with pytest.raises(
        TypeError,
        match="Real-time data must be a MarketDataBar",
    ):
        adapter.process_bar("not a bar")


def test_duplicate_callback_registration_is_idempotent():
    bar = make_bar()

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    received = []

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(received.append)
    adapter.register_callback(received.append)

    adapter.run_once()

    assert received == [bar]


def test_unregister_callback():
    bar = make_bar()

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    received = []

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(received.append)
    adapter.unregister_callback(received.append)

    adapter.run_once()

    assert received == []


def test_unregister_unknown_callback_is_safe():
    stream = InMemoryMarketDataStream()
    adapter = RealTimeMarketDataAdapter(stream)

    adapter.unregister_callback(lambda _: None)


def test_run_returns_zero_without_available_bars():
    stream = InMemoryMarketDataStream()

    adapter = RealTimeMarketDataAdapter(stream)

    adapter.connect()
    adapter.subscribe(SYMBOL)

    assert adapter.run_once() == 0
    assert adapter.last_bar is None


def test_callback_can_observe_exact_market_data():
    bar = make_bar(close=123.75)

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    observed = {}

    def callback(received_bar):
        observed["timestamp"] = received_bar.timestamp
        observed["symbol"] = received_bar.symbol
        observed["close"] = received_bar.close
        observed["volume"] = received_bar.volume

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(callback)

    adapter.run_once()

    assert observed == {
        "timestamp": bar.timestamp,
        "symbol": SYMBOL,
        "close": 123.75,
        "volume": 10.0,
    }


def test_adapter_run_alias_matches_run_once():
    bar = make_bar()

    stream = InMemoryMarketDataStream([bar])
    adapter = RealTimeMarketDataAdapter(stream)

    received = []

    adapter.connect()
    adapter.subscribe(SYMBOL)
    adapter.register_callback(received.append)

    assert adapter.run() == 1
    assert received == [bar]
