"""Provider-neutral completed-bar interface for realtime Paper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class CanonicalBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    completed: bool = True
    provider: str | None = None
    source_id: str | None = None
    is_session_final: bool | None = None
    contract_symbol: str | None = None
    bid: float | None = None
    ask: float | None = None
    provider_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("bar symbol is required")
        stamp = pd.Timestamp(self.timestamp)
        if stamp.tzinfo is None:
            raise ValueError("bar timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", stamp.tz_convert("UTC").to_pydatetime())
        if not self.completed:
            raise ValueError("only completed bars may enter Paper")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if any(pd.isna(value) or not pd.api.types.is_number(value) or not math.isfinite(float(value)) for value in values):
            raise ValueError("OHLCV values must be finite numbers")
        if self.volume < 0 or min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("bar prices must be positive and volume non-negative")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("bar high is inconsistent with OHLC")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("bar low is inconsistent with OHLC")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be supplied together")
        if self.bid is not None and (not math.isfinite(float(self.bid)) or not math.isfinite(float(self.ask)) or self.bid <= 0 or self.ask < self.bid):
            raise ValueError("bid/ask must be finite, positive, and non-crossed")
        if self.provider_timestamp is not None:
            provider_stamp=pd.Timestamp(self.provider_timestamp)
            if provider_stamp.tzinfo is None:
                raise ValueError("provider timestamp must be timezone-aware")
            object.__setattr__(self,"provider_timestamp",provider_stamp.tz_convert("UTC").to_pydatetime())

    def to_market_data(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "provider": self.provider,
            "source_id": self.source_id,
            "is_session_final": self.is_session_final,
            "contract_symbol": self.contract_symbol,
            "bid": float(self.bid) if self.bid is not None else None,
            "ask": float(self.ask) if self.ask is not None else None,
            "provider_timestamp": self.provider_timestamp,
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, symbol: str = "MNQ") -> "CanonicalBar":
        return cls(
            symbol=str(row.get("symbol", symbol)),
            timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]),
            completed=bool(row.get("completed", True)),
            provider=row.get("provider"), source_id=row.get("source_id"),
            is_session_final=row.get("is_session_final"),
            contract_symbol=row.get("contract_symbol"), bid=row.get("bid"), ask=row.get("ask"),
            provider_timestamp=(pd.Timestamp(row["provider_timestamp"]).to_pydatetime() if row.get("provider_timestamp") else None),
        )


@runtime_checkable
class MarketDataSource(Protocol):
    """Contract for a future vendor adapter or deterministic replay source.

    ``subscribe(..., after_timestamp=t)`` must first emit every available
    completed bar strictly after ``t`` (gap catch-up), then continue live.
    """

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def subscribe(
        self, symbols: tuple[str, ...], *, after_timestamp: datetime | None = None
    ) -> None: ...
    def bars(self) -> Iterator[CanonicalBar]: ...
    def health(self) -> Mapping[str, Any]: ...
    def backfill(
        self, *, after_timestamp: datetime, before_timestamp: datetime
    ) -> Iterable[CanonicalBar]: ...


class ReplayMarketDataSource:
    """Deterministic historical source using the realtime bar contract."""

    name = "deterministic_replay"

    def __init__(self, bars: Iterable[Mapping[str, Any] | CanonicalBar], *, symbol: str = "MNQ",
                 backfill_bars: Iterable[Mapping[str, Any] | CanonicalBar] | None = None) -> None:
        self._input = bars
        self._backfill_input = backfill_bars
        self._symbol = symbol
        self._started = False
        self._stopped = False
        self._emitted = 0
        self._after_timestamp: pd.Timestamp | None = None

    def start(self) -> None:
        if self._started and not self._stopped:
            return
        self._started = True
        self._stopped = False

    def subscribe(
        self, symbols: tuple[str, ...], *, after_timestamp: datetime | None = None
    ) -> None:
        if symbols != (self._symbol,):
            raise ValueError(f"Replay source is configured for {self._symbol} only")
        self._after_timestamp = (
            pd.Timestamp(after_timestamp).tz_convert("UTC")
            if after_timestamp is not None else None
        )

    def stop(self) -> None:
        self._stopped = True

    def bars(self) -> Iterator[CanonicalBar]:
        if not self._started:
            raise RuntimeError("market data source must be started before iteration")
        for row in self._input:
            if self._stopped:
                break
            if isinstance(row, CanonicalBar):
                bar = row
            else:
                values = dict(row)
                raw_symbol = str(values.get("symbol", self._symbol))
                # The canonical raw Databento MNQ continuous symbol is
                # MNQ.v.0. Normalize only matching root-symbol aliases.
                if raw_symbol != self._symbol and raw_symbol.split(".", 1)[0] != self._symbol:
                    raise ValueError(f"Replay source received unexpected symbol {raw_symbol!r}")
                values["symbol"] = self._symbol
                bar = CanonicalBar.from_mapping(values, symbol=self._symbol)
            if self._after_timestamp is not None and pd.Timestamp(bar.timestamp) <= self._after_timestamp:
                continue
            self._emitted += 1
            yield bar

    def backfill(self, *, after_timestamp: datetime, before_timestamp: datetime) -> Iterator[CanonicalBar]:
        """Return cached replay bars strictly inside the requested interval."""
        rows = self._backfill_input
        if rows is None:
            rows = self._input if isinstance(self._input, (list, tuple)) else ()
        after = pd.Timestamp(after_timestamp).tz_convert("UTC")
        before = pd.Timestamp(before_timestamp).tz_convert("UTC")
        for row in rows:
            if isinstance(row, CanonicalBar):
                bar = row
            else:
                values = dict(row)
                raw_symbol = str(values.get("symbol", self._symbol))
                if raw_symbol != self._symbol and raw_symbol.split(".", 1)[0] != self._symbol:
                    raise ValueError(f"Replay backfill received unexpected symbol {raw_symbol!r}")
                values["symbol"] = self._symbol
                bar = CanonicalBar.from_mapping(values, symbol=self._symbol)
            stamp = pd.Timestamp(bar.timestamp)
            if after < stamp < before:
                yield bar

    def health(self) -> Mapping[str, Any]:
        return {
            "connected": self._started and not self._stopped,
            "stale": False,
            "source": self.name,
            "bars_emitted": self._emitted,
        }


class IterableMarketDataSource(ReplayMarketDataSource):
    """Alias for deterministic test and adapter-development streams."""


def mark_replay_session_final_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark the final available ORB RTH bar for each New York session date.

    This requires the full replay dataset, so the replay runner calls it before
    clipping the requested interval. A live provider must instead supply its
    authoritative session-final marker.
    """
    required = {"timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Replay session marking requires columns: {sorted(missing)}")
    from src.strategies.orb.config import ORBConfig

    config = ORBConfig()
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    local = timestamps.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    start = config.rth_start_hour * 60 + config.rth_start_minute
    end = config.rth_end_hour * 60 + config.rth_end_minute
    in_rth = minute.ge(start) & minute.lt(end)
    final = pd.Series(False, index=result.index, dtype=bool)
    if in_rth.any():
        session_dates = local.dt.date
        selected = pd.DataFrame({"timestamp": timestamps, "session_date": session_dates}, index=result.index)
        selected = selected.loc[in_rth]
        last_rows = selected.groupby("session_date", sort=False)["timestamp"].idxmax()
        final.loc[last_rows] = True
    result["is_session_final"] = final
    return result
