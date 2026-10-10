from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "mnq" / "ohlcv_1m"


PRICE_SCALE = 1_000_000_000


def load_databento_mnq(*, include_instrument_id: bool = False) -> pd.DataFrame:
    """
    Load the raw Databento MNQ 1-minute dataset.

    Returns a standardized DataFrame compatible with the
    existing research pipeline.
    """

    files = sorted(DATA_DIR.glob("*.csv.zst"))

    if not files:
        raise FileNotFoundError(f"No Databento files found in {DATA_DIR}")

    frames = []
    usecols = [
        "ts_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
    ]
    if include_instrument_id:
        usecols.append("instrument_id")

    for path in files:
        df = pd.read_csv(
            path,
            compression="zstd",
            usecols=usecols,
        )

        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    data["timestamp ET"] = pd.to_datetime(
        data["ts_event"],
        unit="ns",
        utc=True,
    ).dt.tz_convert("America/New_York")

    # --------------------------------------------------------
    # PRICE CONVERSION
    # --------------------------------------------------------

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in price_columns:
        data[column] = data[column] / PRICE_SCALE

    # --------------------------------------------------------
    # STANDARDIZE ORDER
    # --------------------------------------------------------

    ordered_columns = [
        "timestamp ET", "open", "high", "low", "close", "volume", "symbol",
    ]
    if include_instrument_id:
        ordered_columns.append("instrument_id")
    data = data[ordered_columns]

    if include_instrument_id:
        data["instrument_id"] = pd.to_numeric(data["instrument_id"], errors="raise")
        data["instrument_id"] = data["instrument_id"].astype("int64")
        data["symbol"] = data["symbol"].astype(str)

    data = data.sort_values("timestamp ET").reset_index(drop=True)

    return data
