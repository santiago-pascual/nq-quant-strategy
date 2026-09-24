from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd

from .types import MarketDataBar


class ReplayMarketDataAdapter:
    """
    Historical market-data adapter.

    Converts a normalized DataFrame into MarketDataBar objects and
    optionally replays them through a callback.

    This adapter deliberately contains no strategy, risk, execution,
    broker, or optimization logic.
    """

    REQUIRED_COLUMNS = (
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self._data = self._normalize(data)
        self._index = 0

    # ------------------------------------------------------------------
    # Validation / normalization
    # ------------------------------------------------------------------

    @classmethod
    def _normalize(
        cls,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        missing = set(cls.REQUIRED_COLUMNS) - set(data.columns)

        if missing:
            raise ValueError(f"Missing market-data columns: {sorted(missing)}")

        df = data.loc[
            :,
            cls.REQUIRED_COLUMNS,
        ].copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

        numeric_columns = (
            "open",
            "high",
            "low",
            "close",
            "volume",
        )

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )

        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate market-data timestamps")

        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("Market data must be chronological")

        if df["symbol"].isna().any():
            raise ValueError("symbol must not contain null values")

        if (df["symbol"].astype(str).str.len() == 0).any():
            raise ValueError("symbol must not be empty")

        for column in numeric_columns:
            if not pd.Series(df[column]).map(lambda x: pd.notna(x)).all():
                raise ValueError(f"{column} contains null values")

        if (df["open"] <= 0).any():
            raise ValueError("open must be > 0")

        if (df["high"] <= 0).any():
            raise ValueError("high must be > 0")

        if (df["low"] <= 0).any():
            raise ValueError("low must be > 0")

        if (df["close"] <= 0).any():
            raise ValueError("close must be > 0")

        if (df["volume"] < 0).any():
            raise ValueError("volume must be >= 0")

        invalid_high = df["high"] < df[["open", "close"]].max(axis=1)

        if invalid_high.any():
            raise ValueError("high must be >= open and close")

        invalid_low = df["low"] > df[["open", "close"]].min(axis=1)

        if invalid_low.any():
            raise ValueError("low must be <= open and close")

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Adapter state
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def position(self) -> int:
        return self._index

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._data)

    def reset(self) -> None:
        self._index = 0

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_bar(row) -> MarketDataBar:
        timestamp = row.timestamp

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")

        timestamp = timestamp.tz_convert("UTC")

        return MarketDataBar(
            timestamp=timestamp.to_pydatetime(),
            symbol=str(row.symbol),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )

    # ------------------------------------------------------------------
    # Sequential access
    # ------------------------------------------------------------------

    def next_bar(self) -> MarketDataBar | None:
        if self.exhausted:
            return None

        row = self._data.iloc[self._index]

        self._index += 1

        return self._row_to_bar(row)

    def __iter__(self) -> Iterator[MarketDataBar]:
        while not self.exhausted:
            bar = self.next_bar()

            if bar is not None:
                yield bar

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        callback: Callable[[MarketDataBar], None],
    ) -> int:

        if not callable(callback):
            raise TypeError("callback must be callable")

        count = 0

        for bar in self:
            callback(bar)
            count += 1

        return count

    # ------------------------------------------------------------------
    # Historical inspection
    # ------------------------------------------------------------------

    def bars(self) -> tuple[MarketDataBar, ...]:
        return tuple(
            self._row_to_bar(row) for row in self._data.itertuples(index=False)
        )

    def dataframe(self) -> pd.DataFrame:
        return self._data.copy()
