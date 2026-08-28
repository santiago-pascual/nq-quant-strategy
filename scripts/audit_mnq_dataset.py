from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mnq" / "ohlcv_1m"


REQUIRED_COLUMNS = {
    "ts_event",
    "rtype",
    "publisher_id",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_metadata() -> None:
    metadata_path = DATA_DIR / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query = metadata["query"]
    customizations = metadata["customizations"]

    print("METADATA")
    print("-" * 70)
    print(f"Dataset       : {query['dataset']}")
    print(f"Schema        : {query['schema']}")
    print(f"Symbols       : {query['symbols']}")
    print(f"Input type    : {query['stype_in']}")
    print(f"Output type   : {query['stype_out']}")
    print(f"Encoding      : {query['encoding']}")
    print(f"Compression   : {query['compression']}")
    print(f"Split         : {customizations['split_duration']}")
    print()


def audit_file(path: Path) -> dict:
    print("=" * 70)
    print(f"FILE: {path.name}")
    print("=" * 70)

    df = pd.read_csv(
        path,
        compression="zstd",
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["ts_event"],
        unit="ns",
        utc=True,
    )

    timestamp_min = df["timestamp"].min()
    timestamp_max = df["timestamp"].max()

    duplicate_timestamps = int(df["timestamp"].duplicated().sum())

    out_of_order = int((~df["timestamp"].diff().dropna().ge(pd.Timedelta(0))).sum())

    # --------------------------------------------------------
    # PRICE VALIDATION
    # --------------------------------------------------------

    invalid_high = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())

    invalid_low = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())

    invalid_ohlc = int((df["high"] < df["low"]).sum())

    invalid_prices = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())

    invalid_volume = int((df["volume"] < 0).sum())

    # --------------------------------------------------------
    # GAPS
    # --------------------------------------------------------

    deltas = df["timestamp"].sort_values().diff().dropna()

    gaps = deltas[deltas > pd.Timedelta(minutes=1)]

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("TIMESTAMP")
    print("-" * 70)
    print(f"First UTC timestamp : {timestamp_min}")
    print(f"Last UTC timestamp  : {timestamp_max}")
    print(f"Duplicate timestamps: {duplicate_timestamps:,}")
    print(f"Out-of-order rows   : {out_of_order:,}")

    print()
    print("OHLC / VOLUME")
    print("-" * 70)
    print(f"Invalid HIGH rows   : {invalid_high:,}")
    print(f"Invalid LOW rows    : {invalid_low:,}")
    print(f"HIGH < LOW rows     : {invalid_ohlc:,}")
    print(f"Invalid prices      : {invalid_prices:,}")
    print(f"Negative volume     : {invalid_volume:,}")

    print()
    print("GAPS")
    print("-" * 70)
    print(f"Gaps > 1 minute    : {len(gaps):,}")

    if len(gaps):
        print("Largest gaps:")

        largest = gaps.sort_values(ascending=False).head(10)

        for timestamp, delta in largest.items():
            print(f"  {delta} after row timestamp")

    print()

    return {
        "file": path.name,
        "rows": len(df),
        "first_timestamp": str(timestamp_min),
        "last_timestamp": str(timestamp_max),
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order": out_of_order,
        "invalid_high": invalid_high,
        "invalid_low": invalid_low,
        "invalid_ohlc": invalid_ohlc,
        "invalid_prices": invalid_prices,
        "invalid_volume": invalid_volume,
        "gaps_over_1m": len(gaps),
    }


def main() -> None:

    print("=" * 70)
    print("MNQ RAW DATA AUDIT")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found:\n{DATA_DIR}")

    load_metadata()

    files = sorted(DATA_DIR.glob("*.csv.zst"))

    if not files:
        raise FileNotFoundError("No .csv.zst files found.")

    print(f"Found {len(files)} yearly data files.")
    print()

    results = []

    for path in files:
        results.append(audit_file(path))

    summary = pd.DataFrame(results)

    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print()
    print(f"TOTAL ROWS: {summary['rows'].sum():,}")


if __name__ == "__main__":
    main()
