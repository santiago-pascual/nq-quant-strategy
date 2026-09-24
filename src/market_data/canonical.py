from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.data_loader import load_databento_mnq

from .types import MarketDataBar


CANONICAL_COLUMNS = [
    "timestamp ET",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
]

CANONICAL_SYMBOL = "MNQ.v.0"


@dataclass(frozen=True)
class CanonicalMarketDataContract:
    row_count: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    symbol: str


class CanonicalMarketDataAdapter:
    """
    Adapter between the frozen canonical market-data pipeline and the
    execution/paper-trading MarketDataBar contract.

    Important:
    - Uses the canonical loader.
    - Does not resample.
    - Does not modify prices.
    - Does not generate signals.
    - Does not perform risk management.
    - Does not perform execution.
    """

    def __init__(self, dataframe: pd.DataFrame | None = None) -> None:
        if dataframe is None:
            dataframe = load_databento_mnq()

        self._data = self._validate_and_normalize(dataframe)
        self._cursor = 0

    @staticmethod
    def _validate_and_normalize(dataframe: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Canonical market data must be a pandas DataFrame")

        missing = [
            column for column in CANONICAL_COLUMNS if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(f"Canonical market data missing columns: {missing}")

        df = dataframe[CANONICAL_COLUMNS].copy()

        timestamp = pd.to_datetime(
            df["timestamp ET"],
            utc=True,
            errors="raise",
        )

        if timestamp.isna().any():
            raise ValueError("Canonical timestamps contain null values")

        if timestamp.duplicated().any():
            raise ValueError("Canonical market data contains duplicate timestamps")

        if not timestamp.is_monotonic_increasing:
            raise ValueError("Canonical market data timestamps must be chronological")

        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(df[column], errors="raise")

        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise ValueError("Canonical market data contains null OHLCV values")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Canonical OHLC prices must be positive")

        if (df["volume"] < 0).any():
            raise ValueError("Canonical volume cannot be negative")

        invalid_ohlc = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
            df["low"] > df[["open", "close", "high"]].min(axis=1)
        )

        if invalid_ohlc.any():
            raise ValueError("Canonical OHLC relationships are invalid")

        symbols = df["symbol"].astype(str)

        if symbols.isna().any() or (symbols.str.len() == 0).any():
            raise ValueError("Canonical symbol values cannot be empty")

        unique_symbols = symbols.unique()

        if len(unique_symbols) != 1:
            raise ValueError(
                f"Canonical market data must contain exactly one symbol; "
                f"found {list(unique_symbols)}"
            )

        if unique_symbols[0] != CANONICAL_SYMBOL:
            raise ValueError(
                f"Unexpected canonical symbol: {unique_symbols[0]!r}; "
                f"expected {CANONICAL_SYMBOL!r}"
            )

        df["timestamp ET"] = timestamp
        df["symbol"] = symbols

        df.reset_index(drop=True, inplace=True)

        return df

    @property
    def row_count(self) -> int:
        return len(self._data)

    @property
    def position(self) -> int:
        return self._cursor

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._data)

    @property
    def contract(self) -> CanonicalMarketDataContract:
        if self._data.empty:
            raise ValueError("Canonical market data is empty")

        return CanonicalMarketDataContract(
            row_count=len(self._data),
            first_timestamp=self._data.iloc[0]["timestamp ET"],
            last_timestamp=self._data.iloc[-1]["timestamp ET"],
            symbol=str(self._data.iloc[0]["symbol"]),
        )

    def reset(self) -> None:
        self._cursor = 0

    def next_bar(self) -> MarketDataBar | None:
        if self.exhausted:
            return None

        row = self._data.iloc[self._cursor]
        self._cursor += 1

        return MarketDataBar(
            timestamp=row["timestamp ET"],
            symbol=str(row["symbol"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def __iter__(self) -> Iterator[MarketDataBar]:
        while not self.exhausted:
            bar = self.next_bar()

            if bar is not None:
                yield bar

    def bars(self) -> tuple[MarketDataBar, ...]:
        return tuple(self)

    def dataframe(self) -> pd.DataFrame:
        return self._data.copy()

    def first_bar(self) -> MarketDataBar:
        if self._data.empty:
            raise ValueError("Canonical market data is empty")

        row = self._data.iloc[0]

        return MarketDataBar(
            timestamp=row["timestamp ET"],
            symbol=str(row["symbol"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def last_bar(self) -> MarketDataBar:
        if self._data.empty:
            raise ValueError("Canonical market data is empty")

        row = self._data.iloc[-1]

        return MarketDataBar(
            timestamp=row["timestamp ET"],
            symbol=str(row["symbol"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
