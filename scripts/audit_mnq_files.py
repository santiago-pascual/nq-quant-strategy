from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "mnq" / "ohlcv_1m"


def audit_file(path: Path) -> dict:

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

    timestamps = pd.to_datetime(
        df["ts_event"],
        unit="ns",
        utc=True,
    )

    prices = (
        df[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        / 1_000_000_000
    )

    return {
        "file": path.name,
        "rows": len(df),
        "first_utc": timestamps.min(),
        "last_utc": timestamps.max(),
        "symbols": df["symbol"].unique().tolist(),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "out_of_order": int((timestamps.diff().dropna() < pd.Timedelta(0)).sum()),
        "invalid_ohlc": int(
            (
                (prices["high"] < prices["low"])
                | (prices["open"] > prices["high"])
                | (prices["open"] < prices["low"])
                | (prices["close"] > prices["high"])
                | (prices["close"] < prices["low"])
            ).sum()
        ),
        "negative_volume": int((df["volume"] < 0).sum()),
        "zero_volume": int((df["volume"] == 0).sum()),
        "min_price": prices["low"].min(),
        "max_price": prices["high"].max(),
    }


def main() -> None:

    files = sorted(DATA_DIR.glob("*.csv.zst"))

    if not files:
        raise FileNotFoundError(f"No .csv.zst files found in {DATA_DIR}")

    results = []

    for path in files:
        print(f"Auditing {path.name}...")

        results.append(audit_file(path))

    report = pd.DataFrame(results)

    pd.set_option(
        "display.max_columns",
        None,
    )

    print()
    print("=" * 150)
    print("MNQ FILE-LEVEL DATA AUDIT")
    print("=" * 150)

    print(report.to_string(index=False))

    print()
    print("=" * 150)
    print("TOTAL ROWS")
    print("=" * 150)

    print(f"{report['rows'].sum():,}")


if __name__ == "__main__":
    main()
