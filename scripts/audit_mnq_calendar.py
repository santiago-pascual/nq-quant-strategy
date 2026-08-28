from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mnq" / "ohlcv_1m"


# ============================================================
# MARKET SESSION
# ============================================================

NY_TIMEZONE = "America/New_York"

RTH_START = pd.Timestamp("09:30").time()
RTH_END = pd.Timestamp("16:00").time()


# ============================================================
# HELPERS
# ============================================================


def load_timestamps(path: Path) -> pd.DatetimeIndex:
    """
    Read only the timestamp column from one Databento file.

    We do not need OHLCV for this audit.
    """

    df = pd.read_csv(
        path,
        compression="zstd",
        usecols=["ts_event"],
    )

    timestamps = pd.to_datetime(
        df["ts_event"],
        unit="ns",
        utc=True,
    )

    return timestamps


def is_rth(timestamp: pd.Timestamp) -> bool:
    """
    Return True when a timestamp falls inside
    the regular trading hours window.
    """

    local_time = timestamp.tz_convert(NY_TIMEZONE).time()

    return RTH_START <= local_time < RTH_END


# ============================================================
# DAILY COVERAGE
# ============================================================


def build_daily_coverage(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:

    local = timestamps.tz_convert(NY_TIMEZONE)

    df = pd.DataFrame(
        {
            "timestamp": local,
        }
    )

    df["date"] = df["timestamp"].dt.date

    df["is_rth"] = df["timestamp"].map(is_rth)

    daily = (
        df.groupby("date")
        .agg(
            total_bars=(
                "timestamp",
                "size",
            ),
            first_timestamp=(
                "timestamp",
                "min",
            ),
            last_timestamp=(
                "timestamp",
                "max",
            ),
            rth_bars=(
                "is_rth",
                "sum",
            ),
        )
        .reset_index()
    )

    daily["weekday"] = pd.to_datetime(daily["date"]).dt.day_name()

    return daily


# ============================================================
# RTH GAPS
# ============================================================


def find_rth_gaps(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:

    local = timestamps.tz_convert(NY_TIMEZONE)

    df = pd.DataFrame(
        {
            "timestamp": local,
        }
    )

    df["is_rth"] = df["timestamp"].map(is_rth)

    rth = df.loc[df["is_rth"]].copy()

    rth["date"] = rth["timestamp"].dt.date

    gaps = []

    for date, day in rth.groupby(
        "date",
        sort=True,
    ):
        day = day.sort_values("timestamp")

        delta = day["timestamp"].diff()

        mask = delta > pd.Timedelta(minutes=1)

        for timestamp, gap in zip(
            day.loc[mask, "timestamp"],
            delta.loc[mask],
        ):
            gaps.append(
                {
                    "date": date,
                    "timestamp": timestamp,
                    "gap": gap,
                }
            )

    return pd.DataFrame(gaps)


# ============================================================
# MAIN AUDIT
# ============================================================


def main() -> None:

    print("=" * 80)
    print("MNQ CALENDAR / SESSION AUDIT")
    print("=" * 80)

    print(f"Data directory: {DATA_DIR}")

    files = sorted(DATA_DIR.glob("*.csv.zst"))

    if not files:
        raise FileNotFoundError("No Databento .csv.zst files found.")

    print()
    print(f"Found {len(files)} data files.")

    # --------------------------------------------------------
    # LOAD TIMESTAMPS
    # --------------------------------------------------------

    timestamp_parts = []

    for path in files:
        print(f"Reading timestamps: {path.name}")

        timestamps = load_timestamps(path)

        timestamp_parts.append(timestamps)

    timestamps = pd.DatetimeIndex(
        pd.concat(
            [pd.Series(x) for x in timestamp_parts],
            ignore_index=True,
        )
    )

    timestamps = timestamps.sort_values()

    print()
    print("-" * 80)
    print("GLOBAL COVERAGE")
    print("-" * 80)

    print(f"First UTC timestamp: {timestamps.min()}")

    print(f"Last UTC timestamp : {timestamps.max()}")

    print(f"Total bars         : {len(timestamps):,}")

    # --------------------------------------------------------
    # DAILY COVERAGE
    # --------------------------------------------------------

    daily = build_daily_coverage(timestamps)

    print()
    print("-" * 80)
    print("DAILY COVERAGE")
    print("-" * 80)

    print(f"Trading dates: {len(daily):,}")

    print(f"Average bars/day: {daily['total_bars'].mean():.1f}")

    print(f"Median bars/day: {daily['total_bars'].median():.1f}")

    print(f"Minimum bars/day: {daily['total_bars'].min()}")

    print(f"Maximum bars/day: {daily['total_bars'].max()}")

    # --------------------------------------------------------
    # LOW-COVERAGE DAYS
    # --------------------------------------------------------

    low_coverage = daily.loc[daily["total_bars"] < 100].copy()

    print()
    print("-" * 80)
    print("LOW-COVERAGE DAYS (<100 BARS)")
    print("-" * 80)

    print(f"Count: {len(low_coverage)}")

    if not low_coverage.empty:
        print(
            low_coverage[
                [
                    "date",
                    "weekday",
                    "total_bars",
                    "rth_bars",
                    "first_timestamp",
                    "last_timestamp",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # RTH COVERAGE
    # --------------------------------------------------------

    rth_days = daily.loc[daily["rth_bars"] > 0].copy()

    print()
    print("-" * 80)
    print("RTH COVERAGE")
    print("-" * 80)

    print(f"Days with RTH bars: {len(rth_days):,}")

    print(f"Average RTH bars/day: {rth_days['rth_bars'].mean():.1f}")

    print(f"Median RTH bars/day: {rth_days['rth_bars'].median():.1f}")

    print(f"Minimum RTH bars/day: {rth_days['rth_bars'].min()}")

    print(f"Maximum RTH bars/day: {rth_days['rth_bars'].max()}")

    # --------------------------------------------------------
    # RTH GAPS
    # --------------------------------------------------------

    rth_gaps = find_rth_gaps(timestamps)

    print()
    print("-" * 80)
    print("GAPS INSIDE RTH")
    print("-" * 80)

    print(f"RTH gaps > 1 minute: {len(rth_gaps)}")

    if not rth_gaps.empty:
        largest = rth_gaps.sort_values(
            "gap",
            ascending=False,
        ).head(20)

        print()
        print("Largest RTH gaps:")

        print(largest.to_string(index=False))

    # --------------------------------------------------------
    # WEEKDAY DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("TRADING DAYS BY WEEKDAY")
    print("-" * 80)

    weekday_counts = (
        daily["weekday"]
        .value_counts()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            fill_value=0,
        )
    )

    print(weekday_counts.to_string())

    # --------------------------------------------------------
    # SAVE AUDIT
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT / "src" / "research" / "results" / "mnq_calendar_audit.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("CALENDAR AUDIT COMPLETE")
    print("=" * 80)

    print(f"Saved daily audit to:")

    print(output_path)


if __name__ == "__main__":
    main()
