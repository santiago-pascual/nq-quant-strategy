from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "mnq" / "ohlcv_1m"


PRICE_SCALE = 1_000_000_000


def load_databento_mnq() -> pd.DataFrame:
    """
    Load the raw Databento MNQ 1-minute dataset.

    Returns a standardized DataFrame compatible with the
    existing research pipeline.
    """

    files = sorted(DATA_DIR.glob("*.csv.zst"))

    if not files:
        raise FileNotFoundError(f"No Databento files found in {DATA_DIR}")

    frames = []

    for path in files:
        df = pd.read_csv(
            path,
            compression="zstd",
            usecols=[
                "ts_event",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
            ],
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

    data = data[
        [
            "timestamp ET",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ]
    ]

    data = data.sort_values("timestamp ET").reset_index(drop=True)

    return data
