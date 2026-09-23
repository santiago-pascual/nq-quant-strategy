from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]

MARKET_DATA_DIR = ROOT / "data" / "raw" / "mnq" / "ohlcv_1m"

MR_08AA_PATH = (
    ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08aa_modular_reproduction_trades.csv"
)

S2R_PATH = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv"
)

ORB_PATH = (
    ROOT / "src" / "research" / "results" / "orb" / "orb_reconciliation_trades.csv"
)

OUTPUT_DIR = ROOT / "src" / "research" / "results" / "portfolio_4_strategy"

PNG_DIR = OUTPUT_DIR / "png"


# =============================================================================
# FROZEN FULL-SAMPLE COUNTS
# =============================================================================

EXPECTED_COUNTS = {
    "MRL1": 483,
    "S2R": 537,
    "MRS2": 1052,
    "ORB": 1747,
}

EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


# =============================================================================
# COMMON OOS
# =============================================================================

COMMON_OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="UTC",
)

COMMON_OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59",
    tz="UTC",
)

EXPECTED_OOS_COUNTS = {
    "ORB": 1442,
}


# =============================================================================
# HELPERS
# =============================================================================


def banner(title: str) -> None:

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def normalize_timestamp_series(
    series: pd.Series,
    *,
    assume_naive_timezone: str = "UTC",
) -> pd.Series:

    raw = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if raw.isna().any():
        raise ValueError("Found invalid timestamps while normalizing.")

    return raw


def max_drawdown(
    series: pd.Series,
) -> float:

    if len(series) == 0:
        return 0.0

    equity = series.cumsum()

    running_max = equity.cummax()

    dd = equity - running_max

    return float(dd.min())


def streak_stats(
    r_values: pd.Series,
) -> Dict[str, int]:

    values = pd.to_numeric(
        r_values,
        errors="coerce",
    ).dropna()

    longest_win = 0
    longest_loss = 0

    current_win = 0
    current_loss = 0

    for value in values:
        if value > 0:
            current_win += 1
            current_loss = 0

            longest_win = max(
                longest_win,
                current_win,
            )

        elif value < 0:
            current_loss += 1
            current_win = 0

            longest_loss = max(
                longest_loss,
                current_loss,
            )

        else:
            current_win = 0
            current_loss = 0

    return {
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
    }


# =============================================================================
# TRADE METRICS
# =============================================================================


def trade_metrics(
    df: pd.DataFrame,
) -> Dict[str, float]:

    if df.empty:
        return {
            "trades": 0,
            "total_R": 0.0,
            "expectancy_R": np.nan,
            "median_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_win_R": np.nan,
            "avg_loss_R": np.nan,
            "payoff_ratio": np.nan,
            "max_DD_R": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "t_stat": np.nan,
            "best_trade_R": np.nan,
            "worst_trade_R": np.nan,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
        }

    r = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    ).dropna()

    wins = r[r > 0]

    losses = r[r < 0]

    avg_win = float(wins.mean()) if len(wins) else np.nan

    avg_loss = float(losses.mean()) if len(losses) else np.nan

    gross_profit = float(wins.sum()) if len(wins) else 0.0

    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    payoff_ratio = (
        avg_win / abs(avg_loss)
        if (pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0)
        else np.nan
    )

    std = float(r.std(ddof=1)) if len(r) > 1 else np.nan

    if pd.notna(std) and std > 0:
        sharpe = float(r.mean()) / std * math.sqrt(len(r))

        t_stat = sharpe

    else:
        sharpe = np.nan
        t_stat = np.nan

    downside = r[r < 0]

    if len(downside) > 1:
        downside_std = float(downside.std(ddof=1))

        sortino = (
            float(r.mean()) / downside_std * math.sqrt(len(r))
            if downside_std > 0
            else np.nan
        )

    else:
        sortino = np.nan

    streaks = streak_stats(r)

    return {
        "trades": int(len(r)),
        "total_R": float(r.sum()),
        "expectancy_R": float(r.mean()),
        "median_R": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "payoff_ratio": payoff_ratio,
        "max_DD_R": max_drawdown(r),
        "sharpe": sharpe,
        "sortino": sortino,
        "t_stat": t_stat,
        "best_trade_R": float(r.max()),
        "worst_trade_R": float(r.min()),
        "longest_win_streak": streaks["longest_win_streak"],
        "longest_loss_streak": streaks["longest_loss_streak"],
    }


# =============================================================================
# DAILY METRICS
# =============================================================================


def daily_series(
    df: pd.DataFrame,
) -> pd.Series:

    if df.empty:
        return pd.Series(
            dtype=float,
            name="daily_R",
        )

    work = df.copy()

    work["date"] = work["entry_timestamp"].dt.floor("D")

    return work.groupby("date")["r_multiple"].sum().sort_index().rename("daily_R")


def daily_metrics(
    df: pd.DataFrame,
) -> Dict[str, float]:

    daily = daily_series(df)

    if daily.empty:
        return {
            "trading_days": 0,
            "daily_mean_R": np.nan,
            "daily_std_R": np.nan,
            "daily_sharpe": np.nan,
            "daily_sortino": np.nan,
            "daily_best_R": np.nan,
            "daily_worst_R": np.nan,
            "daily_max_DD_R": np.nan,
        }

    mean = float(daily.mean())

    std = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan

    if pd.notna(std) and std > 0:
        sharpe = mean / std * math.sqrt(252)

    else:
        sharpe = np.nan

    downside = daily[daily < 0]

    if len(downside) > 1:
        downside_std = float(downside.std(ddof=1))

        sortino = mean / downside_std * math.sqrt(252) if downside_std > 0 else np.nan

    else:
        sortino = np.nan

    return {
        "trading_days": int(len(daily)),
        "daily_mean_R": mean,
        "daily_std_R": std,
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "daily_best_R": float(daily.max()),
        "daily_worst_R": float(daily.min()),
        "daily_max_DD_R": max_drawdown(daily),
    }


def metrics(
    df: pd.DataFrame,
) -> Dict[str, float]:

    result = trade_metrics(df)

    result.update(daily_metrics(df))

    return result


def print_metrics(
    name: str,
    df: pd.DataFrame,
) -> None:

    m = metrics(df)

    print()
    print(name)
    print("-" * 90)

    print(f"Trades:              {m['trades']:,}")

    print(f"Total R:             {m['total_R']:+.4f}")

    print(f"Expectancy:          {m['expectancy_R']:+.6f} R")

    print(f"Median R:            {m['median_R']:+.6f}")

    print(f"Win rate:            {m['win_rate']:.4%}")

    print(f"Profit factor:       {m['profit_factor']:.6f}")

    print(f"Average win:         {m['avg_win_R']:+.6f} R")

    print(f"Average loss:        {m['avg_loss_R']:+.6f} R")

    print(f"Payoff ratio:        {m['payoff_ratio']:.6f}")

    print(f"Max DD:              {m['max_DD_R']:+.6f} R")

    print(f"Trade Sharpe:        {m['sharpe']:.6f}")

    print(f"Trade Sortino:       {m['sortino']:.6f}")

    print(f"Trade t-stat:        {m['t_stat']:.6f}")

    print(f"Best trade:          {m['best_trade_R']:+.6f} R")

    print(f"Worst trade:         {m['worst_trade_R']:+.6f} R")

    print(f"Longest win streak:  {m['longest_win_streak']}")

    print(f"Longest loss streak: {m['longest_loss_streak']}")

    print(f"Trading days:        {m['trading_days']:,}")

    print(f"Daily Sharpe:        {m['daily_sharpe']:.6f}")

    print(f"Daily Sortino:       {m['daily_sortino']:.6f}")

    print(f"Best day:            {m['daily_best_R']:+.6f} R")

    print(f"Worst day:           {m['daily_worst_R']:+.6f} R")

    print(f"Daily max DD:        {m['daily_max_DD_R']:+.6f} R")


# =============================================================================
# CANONICAL MARKET DATA
# =============================================================================


def load_canonical_market() -> pd.DataFrame:

    banner("LOAD CANONICAL MNQ MARKET DATA")

    files = sorted(MARKET_DATA_DIR.glob("*.csv.zst"))

    print(f"Market files found: {len(files)}")

    if not files:
        raise FileNotFoundError(f"No .csv.zst files found in {MARKET_DATA_DIR}")

    frames = []

    for path in files:
        print(f"  Loading {path.name}")

        frame = pd.read_csv(
            path,
            compression="zstd",
        )

        frames.append(frame)

    market = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"Raw market rows: {len(market):,}")

    timestamp_candidates = [
        "ts_event",
        "timestamp",
        "datetime",
        "time",
    ]

    timestamp_column = None

    for candidate in timestamp_candidates:
        if candidate in market.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        raise ValueError("Could not identify market timestamp column.")

    market["timestamp"] = normalize_timestamp_series(market[timestamp_column])

    rename_map = {}

    for source, target in [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
    ]:
        if source in market.columns:
            rename_map[source] = target

    market = market.rename(columns=rename_map)

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [c for c in required if c not in market.columns]

    if missing:
        raise ValueError(f"Missing market OHLC columns: {missing}")

    for column in required:
        market[column] = pd.to_numeric(
            market[column],
            errors="coerce",
        )

    market = (
        market[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"Canonical rows: {len(market):,}")

    print(f"Start: {market['timestamp'].iloc[0]}")

    print(f"End:   {market['timestamp'].iloc[-1]}")

    return market


# =============================================================================
# LOAD MRL1 + MRS2
# =============================================================================


def load_mr_08aa(
    market: pd.DataFrame,
) -> pd.DataFrame:

    banner("LOAD MRL1 + MRS2 FROM 08AA")

    if not MR_08AA_PATH.exists():
        raise FileNotFoundError(MR_08AA_PATH)

    df = pd.read_csv(MR_08AA_PATH)

    print(f"08AA rows: {len(df):,}")

    expected = EXPECTED_COUNTS["MRL1"] + EXPECTED_COUNTS["MRS2"]

    if len(df) != expected:
        raise ValueError(f"08AA count mismatch. Expected {expected}, got {len(df)}.")

    required = [
        "strategy_name",
        "side",
        "entry_timestamp",
        "entry_price",
        "exit_price",
        "exit_reason",
        "bars_elapsed",
        "r_multiple",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"08AA missing columns: {missing}")

    df["entry_timestamp"] = normalize_timestamp_series(df["entry_timestamp"])

    df["strategy"] = df["strategy_name"].astype(str).str.upper()

    df["direction"] = df["side"].astype(str).str.upper()

    df["entry_price"] = pd.to_numeric(
        df["entry_price"],
        errors="coerce",
    )

    df["exit_price"] = pd.to_numeric(
        df["exit_price"],
        errors="coerce",
    )

    df["bars_elapsed"] = pd.to_numeric(
        df["bars_elapsed"],
        errors="coerce",
    )

    df["r_multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    )

    if df["bars_elapsed"].isna().any():
        raise ValueError("08AA contains missing bars_elapsed.")

    if df["r_multiple"].isna().any():
        raise ValueError("08AA contains missing r_multiple.")

    position_map = pd.Series(
        np.arange(len(market)),
        index=market["timestamp"],
        dtype="int64",
    )

    entry_positions = df["entry_timestamp"].map(position_map)

    exact_matches = entry_positions.notna().sum()

    if exact_matches != len(df):
        raise ValueError(
            "Some MR entry timestamps are not present in canonical market data."
        )

    entry_positions = entry_positions.astype(np.int64).to_numpy()

    bars_elapsed = df["bars_elapsed"].astype(np.int64).to_numpy()

    exit_positions = entry_positions + bars_elapsed

    if (exit_positions < 0).any() or (exit_positions >= len(market)).any():
        raise ValueError("MR reconstructed exit positions fall outside market data.")

    exit_timestamps = pd.Index(market["timestamp"]).take(exit_positions)

    df["exit_timestamp"] = pd.to_datetime(
        exit_timestamps,
        utc=True,
    )

    print()
    print("RECONSTRUCT MRL1/MRS2 EXIT TIMESTAMPS")

    print(f"Reconstructed exits: {df['exit_timestamp'].notna().sum():,}/{len(df):,}")

    print(f"Exact entry timestamp matches: {exact_matches:,}/{len(df):,}")

    return df[
        [
            "strategy",
            "direction",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "r_multiple",
            "exit_reason",
            "bars_elapsed",
        ]
    ].copy()


# =============================================================================
# LOAD S2R
# =============================================================================


def load_s2r() -> pd.DataFrame:

    banner("LOAD AUTHORITATIVE S2R")

    if not S2R_PATH.exists():
        raise FileNotFoundError(S2R_PATH)

    df = pd.read_csv(S2R_PATH)

    print(f"S2R rows: {len(df):,}")

    if len(df) != EXPECTED_COUNTS["S2R"]:
        raise ValueError(
            f"S2R count mismatch. Expected {EXPECTED_COUNTS['S2R']}, got {len(df)}."
        )

    required = [
        "entry_timestamp",
        "exit_timestamp",
        "net_R",
        "exit_reason",
        "holding_bars",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"S2R missing columns: {missing}")

    df["entry_timestamp"] = normalize_timestamp_series(df["entry_timestamp"])

    df["exit_timestamp"] = normalize_timestamp_series(df["exit_timestamp"])

    df["strategy"] = "S2R"

    df["direction"] = "SHORT"

    df["entry_price"] = np.nan

    df["exit_price"] = np.nan

    df["r_multiple"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["bars_elapsed"] = pd.to_numeric(
        df["holding_bars"],
        errors="coerce",
    )

    if df["r_multiple"].isna().any():
        raise ValueError("S2R contains invalid net_R.")

    return df[
        [
            "strategy",
            "direction",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "r_multiple",
            "exit_reason",
            "bars_elapsed",
        ]
    ].copy()


# =============================================================================
# LOAD ORB
# =============================================================================


def load_orb() -> pd.DataFrame:

    banner("LOAD ORB")

    if not ORB_PATH.exists():
        raise FileNotFoundError(ORB_PATH)

    df = pd.read_csv(ORB_PATH)

    print(f"ORB raw rows: {len(df):,}")

    print("ORB source treated as FULL SAMPLE.")

    if len(df) != EXPECTED_COUNTS["ORB"]:
        raise ValueError(
            "ORB full-sample count mismatch. "
            f"Expected {EXPECTED_COUNTS['ORB']}, "
            f"got {len(df)}."
        )

    print("ORB columns:")

    for column in df.columns:
        print(f"  {column}")

    required = [
        "entry_timestamp",
        "exit_timestamp",
        "direction",
        "entry_price",
        "net_R",
        "exit_reason",
        "holding_minutes",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"ORB missing required columns: {missing}")

    df["entry_timestamp"] = normalize_timestamp_series(df["entry_timestamp"])

    df["exit_timestamp"] = normalize_timestamp_series(df["exit_timestamp"])

    df["direction"] = df["direction"].astype(str).str.upper()

    direction_map = {
        "BUY": "LONG",
        "SELL": "SHORT",
        "LONG": "LONG",
        "SHORT": "SHORT",
    }

    df["direction"] = df["direction"].map(direction_map).fillna(df["direction"])

    df["entry_price"] = pd.to_numeric(
        df["entry_price"],
        errors="coerce",
    )

    if "exit_price" in df.columns:
        df["exit_price"] = pd.to_numeric(
            df["exit_price"],
            errors="coerce",
        )

    else:
        df["exit_price"] = np.nan

    df["r_multiple"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["bars_elapsed"] = pd.to_numeric(
        df["holding_minutes"],
        errors="coerce",
    )

    df["strategy"] = "ORB"

    if df["r_multiple"].isna().any():
        raise ValueError("ORB contains invalid net_R values.")

    if df["entry_timestamp"].isna().any():
        raise ValueError("ORB contains invalid entry timestamps.")

    if df["exit_timestamp"].isna().any():
        raise ValueError("ORB contains invalid exit timestamps.")

    if (df["exit_timestamp"] < df["entry_timestamp"]).any():
        raise ValueError("ORB contains exits before entries.")

    return df[
        [
            "strategy",
            "direction",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "r_multiple",
            "exit_reason",
            "bars_elapsed",
        ]
    ].copy()


# =============================================================================
# BUILD FULL PORTFOLIO
# =============================================================================


def build_portfolio() -> pd.DataFrame:

    banner("BUILD FROZEN 4-STRATEGY PORTFOLIO")

    market = load_canonical_market()

    mr = load_mr_08aa(market)

    s2r = load_s2r()

    orb = load_orb()

    portfolio = pd.concat(
        [
            mr,
            s2r,
            orb,
        ],
        ignore_index=True,
    )

    portfolio = portfolio.sort_values(
        [
            "entry_timestamp",
            "strategy",
        ]
    ).reset_index(drop=True)

    banner("FULL-SAMPLE STREAM AUDIT")

    actual_counts = portfolio["strategy"].value_counts().to_dict()

    for strategy in [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]:
        actual = int(
            actual_counts.get(
                strategy,
                0,
            )
        )

        expected = EXPECTED_COUNTS[strategy]

        print(f"{strategy:5s}: {actual:5d} / {expected:5d}")

        if actual != expected:
            raise ValueError(f"{strategy} count mismatch.")

    print(f"TOTAL : {len(portfolio):5d} / {EXPECTED_TOTAL:5d}")

    if len(portfolio) != EXPECTED_TOTAL:
        raise ValueError("Full portfolio total mismatch.")

    duplicate = portfolio.duplicated(
        subset=[
            "entry_timestamp",
            "strategy",
        ],
        keep=False,
    )

    print()
    print(f"Duplicate (entry_timestamp, strategy): {int(duplicate.sum())}")

    if duplicate.any():
        raise ValueError("Duplicate entry timestamp within strategy.")

    return portfolio


# =============================================================================
# COMMON OOS
# =============================================================================


def filter_common_oos(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:

    oos = portfolio[
        (portfolio["entry_timestamp"] >= COMMON_OOS_START)
        & (portfolio["entry_timestamp"] <= COMMON_OOS_END)
    ].copy()

    return oos.sort_values(
        [
            "entry_timestamp",
            "strategy",
        ]
    ).reset_index(drop=True)


def print_oos_counts(
    oos: pd.DataFrame,
) -> None:

    banner("COMMON OOS STREAM AUDIT")

    counts = oos["strategy"].value_counts().to_dict()

    for strategy in [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]:
        print(f"{strategy:5s}: {int(counts.get(strategy, 0)):5d}")

    print(f"TOTAL : {len(oos):5d}")


# =============================================================================
# SUBSETS
# =============================================================================


def subset_strategy(
    df: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:

    return (
        df[df["strategy"] == strategy]
        .copy()
        .sort_values("entry_timestamp")
        .reset_index(drop=True)
    )


def subset_strategies(
    df: pd.DataFrame,
    strategies: Iterable[str],
) -> pd.DataFrame:

    return (
        df[df["strategy"].isin(list(strategies))]
        .copy()
        .sort_values(
            [
                "entry_timestamp",
                "strategy",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# DAILY CORRELATION
# =============================================================================


def daily_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:

    work = df.copy()

    work["date"] = work["entry_timestamp"].dt.floor("D")

    matrix = work.pivot_table(
        index="date",
        columns="strategy",
        values="r_multiple",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    return matrix.corr()


# =============================================================================
# TRADE OVERLAP
# =============================================================================


def calculate_overlap(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    strategies = sorted(df["strategy"].dropna().unique())

    for i, strategy_a in enumerate(strategies):
        a = subset_strategy(
            df,
            strategy_a,
        )

        for strategy_b in strategies[i + 1 :]:
            b = subset_strategy(
                df,
                strategy_b,
            )

            overlaps = 0

            a_overlap = set()

            b_overlap = set()

            for idx_a, trade_a in a.iterrows():
                a_start = trade_a["entry_timestamp"]

                a_end = trade_a["exit_timestamp"]

                mask = (b["entry_timestamp"] <= a_end) & (
                    b["exit_timestamp"] >= a_start
                )

                matched = b.index[mask].tolist()

                if matched:
                    overlaps += len(matched)

                    a_overlap.add(idx_a)

                    b_overlap.update(matched)

            rows.append(
                {
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "overlapping_trade_pairs": overlaps,
                    "trades_a_with_overlap": len(a_overlap),
                    "trades_b_with_overlap": len(b_overlap),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# SAME-DAY INTERACTION
# =============================================================================


def same_day_interaction(
    df: pd.DataFrame,
) -> pd.DataFrame:

    work = df.copy()

    work["date"] = work["entry_timestamp"].dt.floor("D")

    rows = []

    strategies = sorted(work["strategy"].unique())

    for strategy in strategies:
        strategy_days = set(
            work.loc[
                work["strategy"] == strategy,
                "date",
            ]
        )

        for other in strategies:
            if strategy == other:
                continue

            other_days = set(
                work.loc[
                    work["strategy"] == other,
                    "date",
                ]
            )

            shared = strategy_days & other_days

            rows.append(
                {
                    "strategy": strategy,
                    "other_strategy": other,
                    "strategy_days": len(strategy_days),
                    "shared_days": len(shared),
                    "shared_day_pct": (
                        len(shared) / len(strategy_days) if strategy_days else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# STRATEGY CONTRIBUTION
# =============================================================================


def strategy_contribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total_r = float(df["r_multiple"].sum())

    for strategy, group in df.groupby("strategy"):
        strategy_r = float(group["r_multiple"].sum())

        rows.append(
            {
                "strategy": strategy,
                "trades": len(group),
                "total_R": strategy_r,
                "share_of_total_R": (strategy_r / total_r if total_r != 0 else np.nan),
                "share_of_trades": (len(group) / len(df) if len(df) else np.nan),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "total_R",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# PERIOD BREAKDOWNS
# =============================================================================


def period_breakdown(
    df: pd.DataFrame,
    period: str,
) -> pd.DataFrame:

    work = df.copy()

    if period == "year":
        work["period"] = work["entry_timestamp"].dt.year

    elif period == "month":
        work["period"] = work["entry_timestamp"].dt.strftime("%Y-%m")

    else:
        raise ValueError(f"Unknown period: {period}")

    return work.groupby(
        [
            "period",
            "strategy",
        ],
        as_index=False,
    ).agg(
        trades=(
            "r_multiple",
            "size",
        ),
        total_R=(
            "r_multiple",
            "sum",
        ),
        expectancy_R=(
            "r_multiple",
            "mean",
        ),
    )


def combined_period_breakdown(
    df: pd.DataFrame,
    period: str,
) -> pd.DataFrame:

    work = df.copy()

    if period == "year":
        work["period"] = work["entry_timestamp"].dt.year

    elif period == "month":
        work["period"] = work["entry_timestamp"].dt.strftime("%Y-%m")

    else:
        raise ValueError(f"Unknown period: {period}")

    return work.groupby(
        "period",
        as_index=False,
    ).agg(
        trades=(
            "r_multiple",
            "size",
        ),
        total_R=(
            "r_multiple",
            "sum",
        ),
        expectancy_R=(
            "r_multiple",
            "mean",
        ),
    )


# =============================================================================
# COMPARISON DATASETS
# =============================================================================


def build_comparison_sets(
    full: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:

    oos = filter_common_oos(full)

    mr = [
        "MRL1",
        "S2R",
        "MRS2",
    ]

    return {
        "FULL_ALL": full,
        "FULL_MR": subset_strategies(
            full,
            mr,
        ),
        "FULL_ORB": subset_strategy(
            full,
            "ORB",
        ),
        "FULL_MR_PLUS_ORB": full,
        "OOS_ALL": oos,
        "OOS_MR": subset_strategies(
            oos,
            mr,
        ),
        "OOS_ORB": subset_strategy(
            oos,
            "ORB",
        ),
        "OOS_MR_PLUS_ORB": oos,
    }


# =============================================================================
# METRICS TABLE
# =============================================================================


def build_metrics_table(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for name, data in datasets.items():
        rows.append(
            {
                "portfolio": name,
                **metrics(data),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PNG 01 — EQUITY CURVES
# =============================================================================


def generate_equity_curve_png(
    oos: pd.DataFrame,
) -> None:

    work = oos.copy()

    work["date"] = work["entry_timestamp"].dt.floor("D")

    daily = work.pivot_table(
        index="date",
        columns="strategy",
        values="r_multiple",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    mr_columns = [
        c
        for c in [
            "MRL1",
            "MRS2",
            "S2R",
        ]
        if c in daily.columns
    ]

    daily["MR"] = daily[mr_columns].sum(axis=1)

    daily["ORB"] = daily["ORB"] if "ORB" in daily.columns else 0.0

    daily["MR_PLUS_ORB"] = daily["MR"] + daily["ORB"]

    equity = daily[
        [
            "MR",
            "ORB",
            "MR_PLUS_ORB",
        ]
    ].cumsum()

    path = PNG_DIR / "01_equity_curve_oos.png"

    plt.figure(figsize=(15, 8))

    plt.plot(
        equity.index,
        equity["MR"],
        label="MR",
        linewidth=1.8,
    )

    plt.plot(
        equity.index,
        equity["ORB"],
        label="ORB",
        linewidth=1.8,
    )

    plt.plot(
        equity.index,
        equity["MR_PLUS_ORB"],
        label="MR + ORB",
        linewidth=2.4,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("MR + ORB Portfolio — Common OOS Equity Curve")

    plt.xlabel("Date")

    plt.ylabel("Cumulative R")

    plt.legend()

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 02 — DRAWDOWN
# =============================================================================


def generate_drawdown_png(
    oos: pd.DataFrame,
) -> None:

    work = oos.copy()

    work["date"] = work["entry_timestamp"].dt.floor("D")

    daily = work.groupby("date")["r_multiple"].sum().sort_index()

    equity = daily.cumsum()

    running_max = equity.cummax()

    dd = equity - running_max

    path = PNG_DIR / "02_drawdown_oos.png"

    plt.figure(figsize=(15, 7))

    plt.fill_between(
        dd.index,
        dd.values,
        0,
        alpha=0.25,
    )

    plt.plot(
        dd.index,
        dd.values,
        linewidth=1.5,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("4-Strategy Portfolio — Common OOS Drawdown")

    plt.xlabel("Date")

    plt.ylabel("Drawdown (R)")

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 03 — INDIVIDUAL STRATEGY EQUITY
# =============================================================================


def generate_strategy_equity_png(
    oos: pd.DataFrame,
) -> None:

    work = oos.copy()

    work["date"] = work["entry_timestamp"].dt.floor("D")

    daily = work.pivot_table(
        index="date",
        columns="strategy",
        values="r_multiple",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    strategies = [
        "MRL1",
        "MRS2",
        "S2R",
        "ORB",
    ]

    path = PNG_DIR / "03_strategy_equity_oos.png"

    plt.figure(figsize=(15, 8))

    for strategy in strategies:
        if strategy not in daily.columns:
            continue

        plt.plot(
            daily.index,
            daily[strategy].cumsum(),
            label=strategy,
            linewidth=1.7,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("Individual Strategy Equity Curves — Common OOS")

    plt.xlabel("Date")

    plt.ylabel("Cumulative R")

    plt.legend()

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 04 — DAILY RETURNS
# =============================================================================


def generate_daily_returns_png(
    oos: pd.DataFrame,
) -> None:

    daily = daily_series(oos)

    path = PNG_DIR / "04_daily_returns_oos.png"

    plt.figure(figsize=(15, 7))

    plt.bar(
        daily.index,
        daily.values,
        width=1.0,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("4-Strategy Portfolio — Daily R")

    plt.xlabel("Date")

    plt.ylabel("Daily R")

    plt.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 05 — STRATEGY CONTRIBUTION
# =============================================================================


def generate_strategy_contribution_png(
    oos: pd.DataFrame,
) -> None:

    contribution = strategy_contribution(oos).sort_values(
        "total_R",
        ascending=False,
    )

    path = PNG_DIR / "05_strategy_contribution_oos.png"

    plt.figure(figsize=(11, 7))

    plt.bar(
        contribution["strategy"],
        contribution["total_R"],
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("Strategy Contribution — Common OOS")

    plt.xlabel("Strategy")

    plt.ylabel("Total R")

    plt.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 06 — MONTHLY RETURNS
# =============================================================================


def generate_monthly_returns_png(
    oos: pd.DataFrame,
) -> None:

    work = oos.copy()

    work["month"] = work["entry_timestamp"].dt.strftime("%Y-%m")

    monthly = work.groupby("month")["r_multiple"].sum().sort_index()

    path = PNG_DIR / "06_monthly_returns_oos.png"

    plt.figure(figsize=(16, 7))

    plt.bar(
        monthly.index,
        monthly.values,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("4-Strategy Portfolio — Monthly R")

    plt.xlabel("Month")

    plt.ylabel("Monthly R")

    plt.xticks(
        rotation=90,
        fontsize=7,
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 07 — YEARLY RETURNS
# =============================================================================


def generate_yearly_returns_png(
    oos: pd.DataFrame,
) -> None:

    work = oos.copy()

    work["year"] = work["entry_timestamp"].dt.year

    yearly = work.groupby("year")["r_multiple"].sum().sort_index()

    path = PNG_DIR / "07_yearly_returns_oos.png"

    plt.figure(figsize=(12, 7))

    plt.bar(
        yearly.index.astype(str),
        yearly.values,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("4-Strategy Portfolio — Yearly R")

    plt.xlabel("Year")

    plt.ylabel("Yearly R")

    plt.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# PNG 08 — DAILY CORRELATION HEATMAP
# =============================================================================


def generate_correlation_png(
    oos: pd.DataFrame,
) -> None:

    correlation = daily_correlation(oos)

    path = PNG_DIR / "08_daily_correlation_oos.png"

    fig, ax = plt.subplots(figsize=(8, 7))

    matrix = correlation.values

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(correlation.columns)))

    ax.set_yticks(np.arange(len(correlation.index)))

    ax.set_xticklabels(correlation.columns)

    ax.set_yticklabels(correlation.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )

    ax.set_title("Daily Strategy Correlation — Common OOS")

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  PNG: {path.name}")


# =============================================================================
# GENERATE ALL PNGs
# =============================================================================


def generate_all_pngs(
    oos: pd.DataFrame,
) -> None:

    banner("GENERATE PORTFOLIO PNG REPORT")

    PNG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Output directory:")

    print(f"  {PNG_DIR}")

    generate_equity_curve_png(oos)

    generate_drawdown_png(oos)

    generate_strategy_equity_png(oos)

    generate_daily_returns_png(oos)

    generate_strategy_contribution_png(oos)

    generate_monthly_returns_png(oos)

    generate_yearly_returns_png(oos)

    generate_correlation_png(oos)

    print()
    print("PNG report generation complete.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("MR + ORB PORTFOLIO ANALYSIS")

    print("Frozen FULL-SAMPLE streams:")

    print(f"  MRL1 : {EXPECTED_COUNTS['MRL1']}")

    print(f"  S2R  : {EXPECTED_COUNTS['S2R']}")

    print(f"  MRS2 : {EXPECTED_COUNTS['MRS2']}")

    print(f"  ORB  : {EXPECTED_COUNTS['ORB']}")

    print(f"  TOTAL: {EXPECTED_TOTAL}")

    print()

    print("Primary common OOS:")

    print(
        f"  "
        f"{COMMON_OOS_START.strftime('%Y-%m-%d')}"
        f" -> "
        f"{COMMON_OOS_END.strftime('%Y-%m-%d')}"
    )

    print()

    print("No parameter optimization.")

    print("No costs/slippage.")

    # -------------------------------------------------------------------------
    # BUILD FULL FROZEN PORTFOLIO
    # -------------------------------------------------------------------------

    full = build_portfolio()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full.to_csv(
        OUTPUT_DIR / "portfolio_4_strategy_full_sample.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # COMMON OOS
    # -------------------------------------------------------------------------

    oos = filter_common_oos(full)

    print_oos_counts(oos)

    orb_oos = subset_strategy(
        oos,
        "ORB",
    )

    if len(orb_oos) != EXPECTED_OOS_COUNTS["ORB"]:
        raise ValueError(
            "ORB common-OOS count mismatch. "
            f"Expected "
            f"{EXPECTED_OOS_COUNTS['ORB']}, "
            f"got {len(orb_oos)}."
        )

    oos.to_csv(
        OUTPUT_DIR / "portfolio_4_strategy_common_oos.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------

    datasets = build_comparison_sets(full)

    # -------------------------------------------------------------------------
    # FULL SAMPLE
    # -------------------------------------------------------------------------

    banner("FULL-SAMPLE METRICS")

    print_metrics(
        "FULL MR",
        datasets["FULL_MR"],
    )

    print_metrics(
        "FULL ORB",
        datasets["FULL_ORB"],
    )

    print_metrics(
        "FULL MR + ORB",
        datasets["FULL_MR_PLUS_ORB"],
    )

    # -------------------------------------------------------------------------
    # COMMON OOS
    # -------------------------------------------------------------------------

    banner("COMMON OOS METRICS")

    print_metrics(
        "OOS MR",
        datasets["OOS_MR"],
    )

    print_metrics(
        "OOS ORB",
        datasets["OOS_ORB"],
    )

    print_metrics(
        "OOS MR + ORB",
        datasets["OOS_MR_PLUS_ORB"],
    )

    print_metrics(
        "OOS ALL 4 STRATEGIES",
        datasets["OOS_ALL"],
    )

    # -------------------------------------------------------------------------
    # METRICS TABLE
    # -------------------------------------------------------------------------

    metrics_table = build_metrics_table(
        {
            "FULL_MR": datasets["FULL_MR"],
            "FULL_ORB": datasets["FULL_ORB"],
            "FULL_MR_PLUS_ORB": datasets["FULL_MR_PLUS_ORB"],
            "OOS_MR": datasets["OOS_MR"],
            "OOS_ORB": datasets["OOS_ORB"],
            "OOS_MR_PLUS_ORB": datasets["OOS_MR_PLUS_ORB"],
            "OOS_ALL_4": datasets["OOS_ALL"],
        }
    )

    metrics_table.to_csv(
        OUTPUT_DIR / "portfolio_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # DAILY CORRELATION
    # -------------------------------------------------------------------------

    banner("COMMON OOS DAILY CORRELATION")

    correlation = daily_correlation(datasets["OOS_ALL"])

    print(correlation.to_string())

    correlation.to_csv(OUTPUT_DIR / "daily_correlation_oos.csv")

    # -------------------------------------------------------------------------
    # OVERLAP
    # -------------------------------------------------------------------------

    banner("COMMON OOS TRADE OVERLAP")

    overlap = calculate_overlap(datasets["OOS_ALL"])

    print(overlap.to_string(index=False))

    overlap.to_csv(
        OUTPUT_DIR / "trade_overlap_oos.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAME DAY
    # -------------------------------------------------------------------------

    banner("COMMON OOS SAME-DAY INTERACTION")

    same_day = same_day_interaction(datasets["OOS_ALL"])

    print(same_day.to_string(index=False))

    same_day.to_csv(
        OUTPUT_DIR / "same_day_interaction_oos.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # CONTRIBUTION
    # -------------------------------------------------------------------------

    banner("COMMON OOS STRATEGY CONTRIBUTION")

    contribution = strategy_contribution(datasets["OOS_ALL"])

    print(contribution.to_string(index=False))

    contribution.to_csv(
        OUTPUT_DIR / "strategy_contribution_oos.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # YEARLY
    # -------------------------------------------------------------------------

    banner("COMMON OOS YEARLY BREAKDOWN")

    yearly = period_breakdown(
        datasets["OOS_ALL"],
        "year",
    )

    print(yearly.to_string(index=False))

    yearly.to_csv(
        OUTPUT_DIR / "yearly_breakdown_oos.csv",
        index=False,
    )

    yearly_combined = combined_period_breakdown(
        datasets["OOS_ALL"],
        "year",
    )

    yearly_combined.to_csv(
        OUTPUT_DIR / "yearly_combined_oos.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # MONTHLY
    # -------------------------------------------------------------------------

    banner("COMMON OOS MONTHLY BREAKDOWN")

    monthly = period_breakdown(
        datasets["OOS_ALL"],
        "month",
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_breakdown_oos.csv",
        index=False,
    )

    monthly_combined = combined_period_breakdown(
        datasets["OOS_ALL"],
        "month",
    )

    monthly_combined.to_csv(
        OUTPUT_DIR / "monthly_combined_oos.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # DAILY PORTFOLIO COMPARISON
    # -------------------------------------------------------------------------

    daily_mr = daily_series(datasets["OOS_MR"])

    daily_orb = daily_series(datasets["OOS_ORB"])

    daily_all = daily_series(datasets["OOS_ALL"])

    daily_compare = pd.concat(
        [
            daily_mr.rename("MR"),
            daily_orb.rename("ORB"),
            daily_all.rename("MR_PLUS_ORB"),
        ],
        axis=1,
        sort=True,
    ).fillna(0.0)

    daily_compare.to_csv(OUTPUT_DIR / "daily_portfolio_comparison_oos.csv")

    # -------------------------------------------------------------------------
    # EQUITY DATA
    # -------------------------------------------------------------------------

    equity = daily_compare.cumsum()

    equity.to_csv(OUTPUT_DIR / "daily_equity_comparison_oos.csv")

    # -------------------------------------------------------------------------
    # DRAWDOWN DATA
    # -------------------------------------------------------------------------

    drawdown = pd.DataFrame(index=equity.index)

    for column in equity.columns:
        running_max = equity[column].cummax()

        drawdown[column] = equity[column] - running_max

    drawdown.to_csv(OUTPUT_DIR / "daily_drawdown_comparison_oos.csv")

    # -------------------------------------------------------------------------
    # ALL PNG REPORTS
    # -------------------------------------------------------------------------

    generate_all_pngs(datasets["OOS_ALL"])

    # -------------------------------------------------------------------------
    # FINAL AUDIT
    # -------------------------------------------------------------------------

    banner("FINAL AUDIT")

    print("FULL SAMPLE")

    for strategy in [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]:
        print(f"  {strategy}: {len(subset_strategy(full, strategy))}")

    print(f"  TOTAL: {len(full)}")

    print()

    print("COMMON OOS")

    for strategy in [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]:
        print(f"  {strategy}: {len(subset_strategy(oos, strategy))}")

    print(f"  TOTAL: {len(oos)}")

    print()

    print("Output directory:")

    print(f"  {OUTPUT_DIR}")

    print()

    print("PNG directory:")

    print(f"  {PNG_DIR}")

    banner("MR + ORB PORTFOLIO ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
