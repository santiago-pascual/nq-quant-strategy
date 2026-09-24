from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from threading import Lock
from typing import Protocol

from .types import MarketDataBar


MarketDataCallback = Callable[[MarketDataBar], None]


class MarketDataStream(Protocol):
    """
    Provider-neutral interface for a real-time market-data source.

    A concrete implementation can later wrap:
    - Topstep-compatible market data
    - broker APIs
    - WebSocket feeds
    - another market-data provider
    """

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def subscribe(self, symbol: str) -> None: ...

    def unsubscribe(self, symbol: str) -> None: ...

    def bars(self) -> Iterator[MarketDataBar]: ...


class InMemoryMarketDataStream:
    """
    Deterministic provider used for development and testing.

    It intentionally contains no broker/API/network logic.
    """

    def __init__(
        self,
        bars: Iterable[MarketDataBar] = (),
    ) -> None:
        self._bars = list(bars)
        self._connected = False
        self._subscriptions: set[str] = set()
        self._lock = Lock()

    def connect(self) -> None:
        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def subscribe(self, symbol: str) -> None:
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        with self._lock:
            self._subscriptions.add(symbol)

    def unsubscribe(self, symbol: str) -> None:
        with self._lock:
            self._subscriptions.discard(symbol)

    def add_bar(self, bar: MarketDataBar) -> None:
        if not isinstance(bar, MarketDataBar):
            raise TypeError("bar must be a MarketDataBar")

        with self._lock:
            self._bars.append(bar)

    def bars(self) -> Iterator[MarketDataBar]:
        with self._lock:
            connected = self._connected
            subscriptions = set(self._subscriptions)
            bars = list(self._bars)

        if not connected:
            raise RuntimeError("Market data stream is not connected")

        for bar in bars:
            if bar.symbol in subscriptions:
                yield bar


class RealTimeMarketDataAdapter:
    """
    Provider-neutral real-time market-data adapter.

    Responsibilities:
    - manage stream lifecycle
    - manage subscriptions
    - validate incoming MarketDataBar objects
    - deliver bars to registered callbacks
    - expose connection state

    Non-responsibilities:
    - strategy logic
    - signal generation
    - risk management
    - order execution
    - position management
    - broker authentication
    """

    def __init__(self, stream: MarketDataStream) -> None:
        if stream is None:
            raise ValueError("stream cannot be None")

        self._stream = stream
        self._callbacks: list[MarketDataCallback] = []
        self._subscriptions: set[str] = set()
        self._last_bar: MarketDataBar | None = None
        self._lock = Lock()

    @property
    def connected(self) -> bool:
        return self._stream.is_connected()

    @property
    def last_bar(self) -> MarketDataBar | None:
        with self._lock:
            return self._last_bar

    @property
    def subscriptions(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._subscriptions)

    def connect(self) -> None:
        self._stream.connect()

    def disconnect(self) -> None:
        self._stream.disconnect()

    def subscribe(self, symbol: str) -> None:
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        self._stream.subscribe(symbol)

        with self._lock:
            self._subscriptions.add(symbol)

    def unsubscribe(self, symbol: str) -> None:
        self._stream.unsubscribe(symbol)

        with self._lock:
            self._subscriptions.discard(symbol)

    def register_callback(self, callback: MarketDataCallback) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")

        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: MarketDataCallback) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def process_bar(self, bar: MarketDataBar) -> None:
        """
        Process one incoming real-time bar.

        The adapter only accepts MarketDataBar instances and forwards
        them unchanged to registered callbacks.
        """

        if not isinstance(bar, MarketDataBar):
            raise TypeError("Real-time data must be a MarketDataBar")

        with self._lock:
            self._last_bar = bar
            callbacks = list(self._callbacks)

        for callback in callbacks:
            callback(bar)

    def run_once(self) -> int:
        """
        Consume currently available bars from the stream once.

        Returns:
            Number of bars processed.
        """

        if not self.connected:
            raise RuntimeError("Market data adapter is not connected")

        processed = 0

        for bar in self._stream.bars():
            self.process_bar(bar)
            processed += 1

        return processed

    def run(self) -> int:
        """
        Consume the currently available stream.

        Provider-specific implementations can later replace the stream
        with a true blocking/event-driven feed.
        """

        return self.run_once()
