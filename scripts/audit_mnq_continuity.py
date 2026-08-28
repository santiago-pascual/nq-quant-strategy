from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mnq" / "ohlcv_1m"


def main() -> None:

    files = sorted(DATA_DIR.glob("*.csv.zst"))

    frames = []

    for path in files:
        print(f"Reading {path.name}...")

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

        df["timestamp"] = pd.to_datetime(
            df["ts_event"],
            unit="ns",
            utc=True,
        )

        # Databento price representation
        for column in [
            "open",
            "high",
            "low",
            "close",
        ]:
            df[column] = df[column] / 1_000_000_000

        frames.append(
            df[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                ]
            ]
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data = data.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    data["return"] = data["close"].pct_change()

    data["point_change"] = data["close"].diff()

    # --------------------------------------------------------
    # LARGE MOVES
    # --------------------------------------------------------

    large_moves = data.loc[data["point_change"].abs() >= 100].copy()

    print()
    print("=" * 100)
    print("MNQ CONTINUITY AUDIT")
    print("=" * 100)

    print()
    print(f"Total bars: {len(data):,}")

    print()
    print("Price range:")

    print(f"Min: {data['low'].min():,.2f}")

    print(f"Max: {data['high'].max():,.2f}")

    print()
    print("Large 1-minute moves (>= 100 points):")

    print(len(large_moves))

    if not large_moves.empty:
        print()

        print(
            large_moves[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "point_change",
                ]
            ]
            .sort_values(
                "point_change",
                key=lambda x: x.abs(),
                ascending=False,
            )
            .head(50)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # DAILY CLOSE-TO-OPEN MOVES
    # --------------------------------------------------------

    local = data.copy()

    local["ny_timestamp"] = local["timestamp"].dt.tz_convert("America/New_York")

    local["date"] = local["ny_timestamp"].dt.date

    daily = local.groupby("date").agg(
        first_price=(
            "open",
            "first",
        ),
        last_price=(
            "close",
            "last",
        ),
    )

    daily["daily_range"] = daily["last_price"] - daily["first_price"]

    print()
    print("=" * 100)
    print("DAILY PRICE CONTINUITY")
    print("=" * 100)

    print(f"Trading dates: {len(daily):,}")

    print()
    print("Largest daily absolute moves:")

    print(
        daily.loc[daily["daily_range"].abs().nlargest(20).index]
        .sort_values(
            "daily_range",
            key=lambda x: x.abs(),
            ascending=False,
        )
        .to_string()
    )

    print()
    print("=" * 100)
    print("CONTINUITY AUDIT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
