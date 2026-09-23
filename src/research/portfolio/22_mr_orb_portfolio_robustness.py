"""
22 — FINAL MR + ORB PORTFOLIO ROBUSTNESS ENGINE
================================================

Frozen production portfolio
----------------------------
    MRL1
    S2R
    MRS2
    ORB

Official common OOS
-------------------
    2020-06-23 -> 2026-06-19

Purpose
-------
This is the final system-level robustness layer BEFORE construction of the
production Risk Engine.

It does NOT optimize parameters and does NOT select strategies.

It tests the frozen portfolio under:

    1. Full frozen-stream integrity
    2. Common-OOS integrity
    3. Gross historical performance
    4. TopstepX MNQ transaction costs
    5. Deterministic adverse slippage
    6. Random adverse slippage
    7. Cost + slippage combinations
    8. Break-even execution friction
    9. Trade-sequence permutation Monte Carlo
   10. IID trade bootstrap
   11. IID daily bootstrap
   12. Moving-block daily bootstrap
   13. Strategy-preserving daily bootstrap
   14. Trade-removal / missed-trade stress
   15. Worst-tail degradation
   16. Yearly / monthly / weekly stability
   17. Drawdown episodes
   18. Strategy contribution
   19. Daily correlation
   20. Entry overlap
   21. Concurrent exposure
   22. Concentration
   23. Full PNG diagnostic report

Important execution-model limitations
-------------------------------------
The trade streams contain completed trade outcomes, not tick-level fills.
Therefore:

    - Topstep transaction costs are modeled at the published MNQ round-turn
      fee level used here.
    - Slippage is an explicit adverse stress model, not a claim about the
      average fill actually achieved.
    - Funding/account-rule simulation is deliberately excluded from this
      script. It belongs to the separate funded-account validation stage.

The official common OOS is the primary robustness sample.

ORB's post-OOS observations are reported separately and are NEVER used to
change the frozen system.
"""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PROJECT ROOT
# =============================================================================
# This fixes direct execution:
#
#     python .\src\research\portfolio\22_mr_orb_portfolio_robustness.py
#
# without requiring PYTHONPATH to be configured.
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = PROJECT_ROOT

MR_RESULTS = ROOT / "src" / "research" / "mean_reversion" / "results"

ORB_RESULTS = ROOT / "src" / "research" / "results" / "orb"

OUTPUT_DIR = ROOT / "src" / "research" / "results" / "portfolio_robustness"

PNG_DIR = OUTPUT_DIR / "png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INPUT FILES
# =============================================================================

MR_TRADES_PATH = MR_RESULTS / "research_08aa_modular_reproduction_trades.csv"

# S2R authoritative output lives under the general research results root,
# NOT under mean_reversion/results.
S2R_TRADES_PATH = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv"
)

ORB_TRADES_PATH = ORB_RESULTS / "orb_reconciliation_trades.csv"


# =============================================================================
# FROZEN WINDOWS
# =============================================================================

OOS_START = pd.Timestamp(
    "2020-06-23",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59",
    tz="UTC",
)

POST_OOS_START = pd.Timestamp(
    "2026-06-20",
    tz="UTC",
)

POST_OOS_END = pd.Timestamp(
    "2026-08-26 23:59:59",
    tz="UTC",
)


# =============================================================================
# EXPECTED COUNTS
# =============================================================================

EXPECTED_FULL = {
    "MRL1": 483,
    "S2R": 537,
    "MRS2": 1052,
    "ORB": 1747,
}

EXPECTED_OOS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}


# =============================================================================
# TOPSTEP / MNQ EXECUTION MODEL
# =============================================================================
#
# Current TopstepX published MNQ RT cost:
#
#     Exchange = $0.70
#     Commission = $0.50
#     NFA = $0.02
#     ----------------
#     RT = $1.22
#
# MNQ:
#     tick = 0.25 points
#     tick = $0.50
#     point = $2.00
# =============================================================================

TOPSTEP_MNQ_RT_COST_USD = 1.22

MNQ_TICK_SIZE = 0.25
MNQ_DOLLARS_PER_TICK = 0.50
MNQ_DOLLARS_PER_POINT = 2.00


# =============================================================================
# STRESS GRIDS
# =============================================================================

SLIPPAGE_TICKS_GRID = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    16,
    20,
]

# Random adverse execution:
# each side is sampled independently from 0..N ticks.
RANDOM_SLIPPAGE_MAX_TICKS = [
    1,
    2,
    4,
    6,
    10,
]

# Random missed-trade / execution-failure stress.
TRADE_REMOVAL_PCT = [
    0.01,
    0.02,
    0.05,
    0.10,
]

# Daily moving-block bootstrap lengths.
BLOCK_LENGTHS = [
    5,
    10,
    20,
]

# Monte Carlo.
N_SIMULATIONS = 50_000
RANDOM_SEED = 20260922

# Memory-safe Monte Carlo batch size.
MC_BATCH_SIZE = 500


# =============================================================================
# BASIC HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def ensure_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    name: str,
) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name}: missing columns {missing}\nAvailable columns: {list(df.columns)}"
        )


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    )


def max_drawdown(values: np.ndarray) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    equity = np.cumsum(values)

    peaks = np.maximum.accumulate(
        np.concatenate(
            [
                np.array([0.0]),
                equity,
            ]
        )
    )[1:]

    dd = equity - peaks

    return float(dd.min())


def profit_factor(values: np.ndarray) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    gross_profit = values[values > 0].sum()

    gross_loss = -values[values < 0].sum()

    if gross_loss <= 0:
        return float("inf")

    return float(gross_profit / gross_loss)


def daily_sharpe(
    daily: pd.Series,
) -> float:

    daily = pd.to_numeric(
        daily,
        errors="coerce",
    ).dropna()

    if len(daily) < 2:
        return np.nan

    std = daily.std(ddof=1)

    if std <= 0:
        return np.nan

    return float(daily.mean() / std * np.sqrt(252))


def daily_sortino(
    daily: pd.Series,
) -> float:

    daily = pd.to_numeric(
        daily,
        errors="coerce",
    ).dropna()

    if len(daily) < 2:
        return np.nan

    downside = daily[daily < 0]

    if len(downside) < 2:
        return np.inf

    downside_std = downside.std(ddof=1)

    if downside_std <= 0:
        return np.inf

    return float(daily.mean() / downside_std * np.sqrt(252))


def streak_length(
    values: Iterable[bool],
) -> int:

    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(
                best,
                current,
            )

        else:
            current = 0

    return best


# =============================================================================
# LOAD MRL1 / MRS2
# =============================================================================


def load_mr_trades() -> pd.DataFrame:
    df = pd.read_csv(MR_TRADES_PATH)

    ensure_columns(
        df,
        [
            "strategy_name",
            "entry_timestamp",
            "bars_elapsed",
            "r_multiple",
        ],
        "08AA MR",
    )

    df["strategy"] = df["strategy_name"].astype(str)

    df = df.loc[
        df["strategy"].isin(
            [
                "MRL1",
                "MRS2",
            ]
        )
    ].copy()

    df["entry_timestamp"] = to_utc(df["entry_timestamp"])

    # -------------------------------------------------------------------------
    # Reconstruct exit timestamps from the FULL canonical market sequence.
    # -------------------------------------------------------------------------

    from src.databento_loader import (
        load_databento_mnq,
    )

    market = load_databento_mnq().copy()

    # The canonical loader currently returns the timestamp column as
    # "timestamp ET". Older research outputs may expose it as "timestamp".
    # Normalize both schemas here so this robustness engine is independent
    # of the loader display-column name.
    if "timestamp" not in market.columns:
        if "timestamp ET" in market.columns:
            market = market.rename(columns={"timestamp ET": "timestamp"})
        else:
            raise RuntimeError(
                "canonical market: no usable timestamp column found. "
                f"Available columns: {list(market.columns)}"
            )

    market["timestamp"] = to_utc(market["timestamp"])

    market = (
        market.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Canonical market rows: {len(market):,} "
        "| timestamp column normalized successfully"
    )

    timestamp_index = pd.Series(
        np.arange(
            len(market),
            dtype=np.int64,
        ),
        index=market["timestamp"],
    )

    entry_positions = df["entry_timestamp"].map(timestamp_index)

    if entry_positions.isna().any():
        count = int(entry_positions.isna().sum())

        raise RuntimeError(
            "MR exit reconstruction failed: "
            f"{count} entries do not map to canonical market timestamps."
        )

    bars_elapsed = (
        pd.to_numeric(
            df["bars_elapsed"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    exit_positions = entry_positions.astype(np.int64) + bars_elapsed.to_numpy()

    invalid = (exit_positions < 0) | (exit_positions >= len(market))

    if invalid.any():
        raise RuntimeError(
            "MR exit reconstruction produced out-of-range positions: "
            f"{int(invalid.sum())}"
        )

    exit_timestamps = market.iloc[exit_positions.to_numpy()]["timestamp"].to_numpy()

    df["exit_timestamp"] = pd.to_datetime(
        exit_timestamps,
        utc=True,
    )

    df["R"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    )

    df["risk_points"] = np.select(
        [
            df["strategy"].eq("MRL1"),
            df["strategy"].eq("MRS2"),
        ],
        [
            37.5,
            25.0,
        ],
        default=np.nan,
    )

    df["direction"] = np.select(
        [
            df["strategy"].eq("MRL1"),
            df["strategy"].eq("MRS2"),
        ],
        [
            "LONG",
            "SHORT",
        ],
        default="UNKNOWN",
    )

    return df[
        [
            "strategy",
            "entry_timestamp",
            "exit_timestamp",
            "direction",
            "R",
            "risk_points",
            "exit_reason",
        ]
    ].copy()


# =============================================================================
# LOAD S2R
# =============================================================================


def load_s2r() -> pd.DataFrame:
    if not S2R_TRADES_PATH.exists():
        raise FileNotFoundError(
            "S2R authoritative trade stream not found.\n"
            f"Expected path:\n{S2R_TRADES_PATH}\n\n"
            "Check that the frozen authoritative S2R output exists at "
            "src\\research\\results\\s2_extended\\."
        )

    df = pd.read_csv(S2R_TRADES_PATH)

    ensure_columns(
        df,
        [
            "entry_timestamp",
            "exit_timestamp",
            "net_R",
            "exit_reason",
        ],
        "S2R authoritative",
    )

    df["strategy"] = "S2R"

    df["entry_timestamp"] = to_utc(df["entry_timestamp"])

    df["exit_timestamp"] = to_utc(df["exit_timestamp"])

    df["R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["risk_points"] = 25.0
    df["direction"] = "SHORT"

    return df[
        [
            "strategy",
            "entry_timestamp",
            "exit_timestamp",
            "direction",
            "R",
            "risk_points",
            "exit_reason",
        ]
    ].copy()


# =============================================================================
# LOAD ORB
# =============================================================================


def load_orb() -> pd.DataFrame:
    df = pd.read_csv(ORB_TRADES_PATH)

    ensure_columns(
        df,
        [
            "direction",
            "entry_timestamp",
            "exit_timestamp",
            "risk_points",
            "net_R",
            "exit_reason",
        ],
        "ORB reconciliation",
    )

    df["strategy"] = "ORB"

    df["entry_timestamp"] = to_utc(df["entry_timestamp"])

    df["exit_timestamp"] = to_utc(df["exit_timestamp"])

    df["risk_points"] = pd.to_numeric(
        df["risk_points"],
        errors="coerce",
    )

    df["R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["direction"] = df["direction"].astype(str)

    return df[
        [
            "strategy",
            "entry_timestamp",
            "exit_timestamp",
            "direction",
            "R",
            "risk_points",
            "exit_reason",
        ]
    ].copy()


# =============================================================================
# MASTER LOAD
# =============================================================================


def load_all_trades() -> pd.DataFrame:

    mr = load_mr_trades()
    s2r = load_s2r()
    orb = load_orb()

    full = pd.concat(
        [
            mr,
            s2r,
            orb,
        ],
        ignore_index=True,
    )

    full = (
        full.dropna(
            subset=[
                "strategy",
                "entry_timestamp",
                "exit_timestamp",
                "R",
                "risk_points",
            ]
        )
        .sort_values(
            [
                "entry_timestamp",
                "strategy",
            ]
        )
        .reset_index(drop=True)
    )

    full["trade_id"] = np.arange(
        1,
        len(full) + 1,
        dtype=np.int64,
    )

    # New York trading date.
    full["entry_date"] = (
        full["entry_timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    )

    return full


# =============================================================================
# INTEGRITY AUDIT
# =============================================================================


def audit_full_stream(
    full: pd.DataFrame,
) -> None:

    banner("FULL SAMPLE FROZEN STREAM AUDIT")

    expected_total = sum(EXPECTED_FULL.values())

    if len(full) != expected_total:
        raise RuntimeError(f"Full count mismatch: {len(full)} != {expected_total}")

    for strategy, expected in EXPECTED_FULL.items():
        actual = int(full["strategy"].eq(strategy).sum())

        status = "PASS" if actual == expected else "FAIL"

        print(f"{strategy:>5}: {actual:5d} / {expected:5d} {status}")

        if actual != expected:
            raise RuntimeError(f"{strategy} count mismatch.")

    duplicate_pairs = int(
        full.duplicated(
            [
                "entry_timestamp",
                "strategy",
            ]
        ).sum()
    )

    print(f"\nDuplicate (entry_timestamp, strategy): {duplicate_pairs}")

    if duplicate_pairs:
        raise RuntimeError("Duplicate frozen trade keys detected.")

    if full["R"].isna().any():
        raise RuntimeError("NaN R values detected.")

    if full["risk_points"].isna().any():
        raise RuntimeError("NaN risk_points detected.")

    if (full["risk_points"] <= 0).any():
        raise RuntimeError("Non-positive risk_points detected.")

    if (full["exit_timestamp"] < full["entry_timestamp"]).any():
        raise RuntimeError("Exit timestamp precedes entry timestamp.")

    print(f"TOTAL: {len(full):,} / {expected_total:,} PASS")


# =============================================================================
# SAMPLE FILTER
# =============================================================================


def filter_period(
    full: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:

    return (
        full.loc[(full["entry_timestamp"] >= start) & (full["entry_timestamp"] <= end)]
        .sort_values(
            [
                "entry_timestamp",
                "strategy",
            ]
        )
        .reset_index(drop=True)
    )


def audit_oos(
    oos: pd.DataFrame,
) -> None:

    banner("COMMON OOS AUDIT")

    expected_total = sum(EXPECTED_OOS.values())

    if len(oos) != expected_total:
        raise RuntimeError(f"OOS count mismatch: {len(oos)} != {expected_total}")

    for strategy, expected in EXPECTED_OOS.items():
        actual = int(oos["strategy"].eq(strategy).sum())

        status = "PASS" if actual == expected else "FAIL"

        print(f"{strategy:>5}: {actual:5d} / {expected:5d} {status}")

        if actual != expected:
            raise RuntimeError(f"OOS {strategy} count mismatch.")

    print(f"TOTAL: {len(oos):,} / {expected_total:,} PASS")


# =============================================================================
# METRICS
# =============================================================================


def daily_returns(
    df: pd.DataFrame,
    value_col: str = "R",
) -> pd.Series:

    return df.groupby("entry_date")[value_col].sum().sort_index()


def portfolio_metrics(
    df: pd.DataFrame,
    value_col: str = "R",
) -> dict[str, float]:

    values = (
        pd.to_numeric(
            df[value_col],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(values) == 0:
        return {}

    wins = values[values > 0]

    losses = values[values < 0]

    daily = daily_returns(
        df,
        value_col,
    )

    std = values.std(ddof=1) if len(values) > 1 else np.nan

    return {
        "trades": int(len(values)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float((values > 0).mean()),
        "total_R": float(values.sum()),
        "expectancy_R": float(values.mean()),
        "median_R": float(np.median(values)),
        "PF": profit_factor(values),
        "avg_win_R": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_R": float(losses.mean()) if len(losses) else np.nan,
        "payoff": float(wins.mean() / abs(losses.mean()))
        if (len(wins) and len(losses))
        else np.nan,
        "max_DD_R": max_drawdown(values),
        "trade_sharpe": float(values.mean() / std * np.sqrt(len(values)))
        if (len(values) > 1 and std > 0)
        else np.nan,
        "daily_sharpe": daily_sharpe(daily),
        "daily_sortino": daily_sortino(daily),
        "best_day_R": float(daily.max()),
        "worst_day_R": float(daily.min()),
        "positive_day_rate": float((daily > 0).mean()),
        "daily_max_DD_R": max_drawdown(daily.to_numpy(dtype=float)),
    }


# =============================================================================
# YEAR / MONTH / WEEK
# =============================================================================


def period_table(
    df: pd.DataFrame,
    period: str,
) -> pd.DataFrame:

    x = df.copy()

    ny_time = x["entry_timestamp"].dt.tz_convert("America/New_York")

    if period == "year":
        x["_period"] = ny_time.dt.year

    elif period == "month":
        x["_period"] = ny_time.dt.strftime("%Y-%m")

    elif period == "week":
        x["_period"] = ny_time.dt.to_period("W").astype(str)

    else:
        raise ValueError(f"Unsupported period: {period}")

    rows = []

    for key, group in x.groupby(
        "_period",
        sort=True,
    ):
        rows.append(
            {
                "period": key,
                **portfolio_metrics(group),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# STRATEGY CONTRIBUTION
# =============================================================================


def strategy_contribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    total_r = float(df["R"].sum())

    total_trades = len(df)

    rows = []

    for strategy, group in df.groupby(
        "strategy",
        sort=True,
    ):
        metrics = portfolio_metrics(group)

        rows.append(
            {
                "strategy": strategy,
                "trades": len(group),
                "trade_share_pct": (100 * len(group) / total_trades),
                "total_R": float(group["R"].sum()),
                "R_contribution_pct": (
                    100 * group["R"].sum() / total_r if total_r != 0 else np.nan
                ),
                "expectancy_R": metrics["expectancy_R"],
                "PF": metrics["PF"],
                "max_DD_R": metrics["max_DD_R"],
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
# DAILY CORRELATION
# =============================================================================


def daily_strategy_matrix(
    df: pd.DataFrame,
    value_col: str = "R",
) -> pd.DataFrame:

    matrix = df.pivot_table(
        index="entry_date",
        columns="strategy",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    for strategy in [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]:
        if strategy not in matrix.columns:
            matrix[strategy] = 0.0

    return matrix[
        [
            "MRL1",
            "S2R",
            "MRS2",
            "ORB",
        ]
    ]


# =============================================================================
# OVERLAP / CONCURRENCY
# =============================================================================


def overlap_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    strategies = [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]

    rows = []

    for i, a in enumerate(strategies):
        for b in strategies[i + 1 :]:
            a_times = set(
                df.loc[
                    df["strategy"].eq(a),
                    "entry_timestamp",
                ]
            )

            b_times = set(
                df.loc[
                    df["strategy"].eq(b),
                    "entry_timestamp",
                ]
            )

            a_days = set(
                df.loc[
                    df["strategy"].eq(a),
                    "entry_date",
                ]
            )

            b_days = set(
                df.loc[
                    df["strategy"].eq(b),
                    "entry_date",
                ]
            )

            rows.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "exact_entry_overlap": len(a_times & b_times),
                    "shared_trading_days": len(a_days & b_days),
                }
            )

    return pd.DataFrame(rows)


def concurrency_analysis(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    events = []

    for row in df[
        [
            "strategy",
            "entry_timestamp",
            "exit_timestamp",
        ]
    ].itertuples(index=False):
        events.append(
            (
                row.entry_timestamp,
                1,  # entry
                row.strategy,
            )
        )

        events.append(
            (
                row.exit_timestamp,
                0,  # exit gets processed first
                row.strategy,
            )
        )

    events.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    active = {}
    rows = []

    max_positions = 0
    max_strategies = 0
    max_timestamp = None

    for timestamp, event_type, strategy in events:
        if event_type == 0:
            # Remove one active trade for the strategy.
            if strategy in active:
                active[strategy] -= 1

                if active[strategy] <= 0:
                    del active[strategy]

        else:
            active[strategy] = (
                active.get(
                    strategy,
                    0,
                )
                + 1
            )

        total = int(sum(active.values()))

        strategy_count = len(active)

        if total > max_positions:
            max_positions = total
            max_strategies = strategy_count
            max_timestamp = timestamp

        rows.append(
            {
                "timestamp": timestamp,
                "active_positions": total,
                "active_strategies": strategy_count,
            }
        )

    event_df = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "max_concurrent_positions": max_positions,
                "max_concurrent_strategies": max_strategies,
                "timestamp_of_max_concurrency": max_timestamp,
                "mean_active_positions": (
                    float(event_df["active_positions"].mean()) if len(event_df) else 0.0
                ),
            }
        ]
    )

    return (
        event_df,
        summary,
    )


# =============================================================================
# CONCENTRATION
# =============================================================================


def concentration_analysis(
    df: pd.DataFrame,
) -> dict[str, float]:

    daily = daily_returns(df)

    total = float(daily.sum())

    top5 = float(daily.nlargest(min(5, len(daily))).sum())

    top10 = float(daily.nlargest(min(10, len(daily))).sum())

    worst5 = float(daily.nsmallest(min(5, len(daily))).sum())

    return {
        "total_R": total,
        "top5_days_R": top5,
        "top10_days_R": top10,
        "top5_contribution_pct": (100 * top5 / total if total != 0 else np.nan),
        "top10_contribution_pct": (100 * top10 / total if total != 0 else np.nan),
        "worst5_days_R": worst5,
    }


# =============================================================================
# STREAKS
# =============================================================================


def streak_analysis(
    df: pd.DataFrame,
) -> dict[str, int]:

    daily = daily_returns(df)

    return {
        "longest_positive_day_streak": (streak_length(daily > 0)),
        "longest_negative_day_streak": (streak_length(daily < 0)),
    }


# =============================================================================
# DRAWDOWN EPISODES
# =============================================================================


def drawdown_episodes(
    df: pd.DataFrame,
) -> pd.DataFrame:

    daily = daily_returns(df)

    equity = daily.cumsum()
    peak = equity.cummax()
    dd = equity - peak

    episodes = []

    in_dd = False
    start = None
    trough_date = None
    trough_value = 0.0

    for date, value in dd.items():
        value = float(value)

        if value < 0 and not in_dd:
            in_dd = True
            start = date
            trough_date = date
            trough_value = value

        elif value < 0 and in_dd:
            if value < trough_value:
                trough_value = value
                trough_date = date

        elif value >= 0 and in_dd:
            episodes.append(
                {
                    "start": start,
                    "trough": trough_date,
                    "end": date,
                    "drawdown_R": trough_value,
                    "duration_calendar_days": (date - start).days,
                }
            )

            in_dd = False
            start = None
            trough_date = None
            trough_value = 0.0

    if in_dd:
        episodes.append(
            {
                "start": start,
                "trough": trough_date,
                "end": pd.NaT,
                "drawdown_R": trough_value,
                "duration_calendar_days": (daily.index[-1] - start).days,
            }
        )

    return pd.DataFrame(episodes).sort_values("drawdown_R").reset_index(drop=True)


# =============================================================================
# COST / SLIPPAGE
# =============================================================================


def apply_execution_costs(
    df: pd.DataFrame,
    slippage_ticks_per_side: float,
    include_topstep_cost: bool = True,
) -> pd.DataFrame:

    out = df.copy()

    commission = TOPSTEP_MNQ_RT_COST_USD if include_topstep_cost else 0.0

    slippage = 2.0 * float(slippage_ticks_per_side) * MNQ_DOLLARS_PER_TICK

    total_usd = commission + slippage

    risk_usd = out["risk_points"] * MNQ_DOLLARS_PER_POINT

    out["execution_cost_usd"] = total_usd

    out["execution_cost_R"] = total_usd / risk_usd

    out["R_net"] = out["R"] - out["execution_cost_R"]

    return out


def cost_slippage_grid(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for ticks in SLIPPAGE_TICKS_GRID:
        stressed = apply_execution_costs(
            df,
            slippage_ticks_per_side=ticks,
            include_topstep_cost=True,
        )

        metrics = portfolio_metrics(
            stressed,
            value_col="R_net",
        )

        rows.append(
            {
                "slippage_ticks_per_side": ticks,
                "slippage_rt_usd": (2 * ticks * MNQ_DOLLARS_PER_TICK),
                "topstep_rt_cost_usd": (TOPSTEP_MNQ_RT_COST_USD),
                "total_rt_friction_usd": (
                    TOPSTEP_MNQ_RT_COST_USD + 2 * ticks * MNQ_DOLLARS_PER_TICK
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def break_even_slippage(
    df: pd.DataFrame,
    max_ticks: int = 100,
) -> dict[str, float]:

    previous_total = None

    for ticks in np.arange(
        0.0,
        max_ticks + 0.25,
        0.25,
    ):
        stressed = apply_execution_costs(
            df,
            slippage_ticks_per_side=float(ticks),
            include_topstep_cost=True,
        )

        total = float(stressed["R_net"].sum())

        if total <= 0:
            return {
                "break_even_ticks_per_side": float(ticks),
                "previous_positive_ticks": float(
                    max(
                        0.0,
                        ticks - 0.25,
                    )
                ),
                "break_even_rt_friction_usd": (
                    TOPSTEP_MNQ_RT_COST_USD + 2 * ticks * MNQ_DOLLARS_PER_TICK
                ),
                "total_R_at_break_even_grid": total,
            }

        previous_total = total

    return {
        "break_even_ticks_per_side": np.nan,
        "previous_positive_ticks": np.nan,
        "break_even_rt_friction_usd": np.nan,
        "last_positive_total_R": (previous_total),
    }


# =============================================================================
# TAIL STRESS
# =============================================================================


def tail_degradation(
    df: pd.DataFrame,
) -> pd.DataFrame:

    values = df["R"].to_numpy(dtype=float)

    rows = []

    for pct in [
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ]:
        n = max(
            1,
            int(math.ceil(len(values) * pct / 100.0)),
        )

        stressed = values.copy()

        worst = np.argsort(stressed)[:n]

        # 50% additional damage to the
        # selected worst outcomes.
        stressed[worst] *= 1.5

        rows.append(
            {
                "worst_trade_fraction_pct": pct,
                "trades_stressed": n,
                "tail_loss_multiplier": 1.5,
                "total_R": float(stressed.sum()),
                "expectancy_R": float(stressed.mean()),
                "PF": profit_factor(stressed),
                "max_DD_R": max_drawdown(stressed),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MONTE CARLO SUMMARY
# =============================================================================


def summarize_mc(
    simulation: pd.DataFrame,
    scenario: str,
) -> dict[str, float]:

    final = simulation["final_R"].to_numpy(dtype=float)

    dd = simulation["max_DD_R"].to_numpy(dtype=float)

    return {
        "scenario": scenario,
        "simulations": len(simulation),
        "final_mean_R": float(final.mean()),
        "final_std_R": float(final.std(ddof=1)),
        "final_p01_R": float(
            np.percentile(
                final,
                1,
            )
        ),
        "final_p05_R": float(
            np.percentile(
                final,
                5,
            )
        ),
        "final_p50_R": float(
            np.percentile(
                final,
                50,
            )
        ),
        "final_p95_R": float(
            np.percentile(
                final,
                95,
            )
        ),
        "final_p99_R": float(
            np.percentile(
                final,
                99,
            )
        ),
        "prob_final_negative": float((final < 0).mean()),
        "dd_mean_R": float(dd.mean()),
        "dd_p01_R": float(
            np.percentile(
                dd,
                1,
            )
        ),
        "dd_p05_R": float(
            np.percentile(
                dd,
                5,
            )
        ),
        "dd_p50_R": float(
            np.percentile(
                dd,
                50,
            )
        ),
        "dd_p95_R": float(
            np.percentile(
                dd,
                95,
            )
        ),
        "prob_DD_below_-15R": float((dd < -15).mean()),
        "prob_DD_below_-20R": float((dd < -20).mean()),
        "prob_DD_below_-25R": float((dd < -25).mean()),
        "prob_DD_below_-30R": float((dd < -30).mean()),
        "prob_DD_below_-35R": float((dd < -35).mean()),
        "prob_DD_below_-40R": float((dd < -40).mean()),
    }


# =============================================================================
# MEMORY-SAFE IID BOOTSTRAP
# =============================================================================


def iid_trade_bootstrap(
    values: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    n_trades = len(values)

    final_parts = []
    dd_parts = []

    for start in range(
        0,
        n_sim,
        MC_BATCH_SIZE,
    ):
        batch_n = min(
            MC_BATCH_SIZE,
            n_sim - start,
        )

        sample = rng.choice(
            values,
            size=(
                batch_n,
                n_trades,
            ),
            replace=True,
        )

        finals = sample.sum(axis=1)

        equity = np.cumsum(
            sample,
            axis=1,
        )

        peaks = np.maximum.accumulate(
            equity,
            axis=1,
        )

        dds = (equity - peaks).min(axis=1)

        final_parts.append(finals)

        dd_parts.append(dds)

    return pd.DataFrame(
        {
            "final_R": np.concatenate(final_parts),
            "max_DD_R": np.concatenate(dd_parts),
        }
    )


# =============================================================================
# MEMORY-SAFE DAILY IID BOOTSTRAP
# =============================================================================


def iid_daily_bootstrap(
    daily_values: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    n_days = len(daily_values)

    final_parts = []
    dd_parts = []

    for start in range(
        0,
        n_sim,
        MC_BATCH_SIZE,
    ):
        batch_n = min(
            MC_BATCH_SIZE,
            n_sim - start,
        )

        sample = rng.choice(
            daily_values,
            size=(
                batch_n,
                n_days,
            ),
            replace=True,
        )

        finals = sample.sum(axis=1)

        equity = np.cumsum(
            sample,
            axis=1,
        )

        peaks = np.maximum.accumulate(
            equity,
            axis=1,
        )

        dds = (equity - peaks).min(axis=1)

        final_parts.append(finals)

        dd_parts.append(dds)

    return pd.DataFrame(
        {
            "final_R": np.concatenate(final_parts),
            "max_DD_R": np.concatenate(dd_parts),
        }
    )


# =============================================================================
# PERMUTATION MC
# =============================================================================


def permutation_mc(
    values: np.ndarray,
    n_sim: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    final = np.empty(
        n_sim,
        dtype=float,
    )

    dd = np.empty(
        n_sim,
        dtype=float,
    )

    for i in range(n_sim):
        path = rng.permutation(values)

        final[i] = path.sum()
        dd[i] = max_drawdown(path)

    return pd.DataFrame(
        {
            "final_R": final,
            "max_DD_R": dd,
        }
    )


# =============================================================================
# MOVING BLOCK BOOTSTRAP
# =============================================================================


def block_bootstrap(
    daily_values: np.ndarray,
    n_sim: int,
    block_length: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    n_days = len(daily_values)

    if n_days == 0:
        return pd.DataFrame()

    starts = np.arange(
        n_days,
        dtype=np.int64,
    )

    n_blocks = math.ceil(n_days / block_length)

    final = np.empty(
        n_sim,
        dtype=float,
    )

    dd = np.empty(
        n_sim,
        dtype=float,
    )

    for i in range(n_sim):
        selected = rng.choice(
            starts,
            size=n_blocks,
            replace=True,
        )

        path = np.concatenate(
            [
                daily_values[start : start + block_length]
                if start + block_length <= n_days
                else np.concatenate(
                    [
                        daily_values[start:],
                        daily_values[: (start + block_length - n_days)],
                    ]
                )
                for start in selected
            ]
        )[:n_days]

        final[i] = path.sum()
        dd[i] = max_drawdown(path)

    return pd.DataFrame(
        {
            "final_R": final,
            "max_DD_R": dd,
        }
    )


# =============================================================================
# STRATEGY-PRESERVING DAILY BOOTSTRAP
# =============================================================================


def strategy_day_bootstrap(
    df: pd.DataFrame,
    n_sim: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    matrix = daily_strategy_matrix(df)

    arrays = {
        strategy: matrix[strategy].to_numpy(dtype=float) for strategy in matrix.columns
    }

    n_days = len(matrix)

    final = np.empty(
        n_sim,
        dtype=float,
    )

    dd = np.empty(
        n_sim,
        dtype=float,
    )

    for i in range(n_sim):
        path = np.zeros(
            n_days,
            dtype=float,
        )

        for values in arrays.values():
            path += rng.choice(
                values,
                size=n_days,
                replace=True,
            )

        final[i] = path.sum()
        dd[i] = max_drawdown(path)

    return pd.DataFrame(
        {
            "final_R": final,
            "max_DD_R": dd,
        }
    )


# =============================================================================
# RANDOM EXECUTION MC
# =============================================================================


def random_execution_mc(
    df: pd.DataFrame,
    n_sim: int,
    max_slippage_ticks: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Memory-safe randomized execution simulation.

    For each trade:
        entry slippage ~ Uniform integer [0, max_ticks]
        exit slippage  ~ Uniform integer [0, max_ticks]

    Topstep $1.22 RT cost is applied to every trade.
    """

    gross = df["R"].to_numpy(dtype=float)

    risk_usd = df["risk_points"].to_numpy(dtype=float) * MNQ_DOLLARS_PER_POINT

    final_parts = []
    dd_parts = []

    n_trades = len(df)

    for start in range(
        0,
        n_sim,
        MC_BATCH_SIZE,
    ):
        batch_n = min(
            MC_BATCH_SIZE,
            n_sim - start,
        )

        entry_ticks = rng.integers(
            0,
            max_slippage_ticks + 1,
            size=(
                batch_n,
                n_trades,
            ),
        )

        exit_ticks = rng.integers(
            0,
            max_slippage_ticks + 1,
            size=(
                batch_n,
                n_trades,
            ),
        )

        friction_usd = (
            TOPSTEP_MNQ_RT_COST_USD + (entry_ticks + exit_ticks) * MNQ_DOLLARS_PER_TICK
        )

        net = gross[None, :] - (friction_usd / risk_usd[None, :])

        finals = net.sum(axis=1)

        equity = np.cumsum(
            net,
            axis=1,
        )

        peaks = np.maximum.accumulate(
            equity,
            axis=1,
        )

        dds = (equity - peaks).min(axis=1)

        final_parts.append(finals)

        dd_parts.append(dds)

    return pd.DataFrame(
        {
            "final_R": np.concatenate(final_parts),
            "max_DD_R": np.concatenate(dd_parts),
        }
    )


# =============================================================================
# TRADE REMOVAL MC
# =============================================================================


def trade_removal_mc(
    values: np.ndarray,
    n_sim: int,
    removal_pct: float,
    rng: np.random.Generator,
) -> pd.DataFrame:

    n_trades = len(values)

    final = np.empty(
        n_sim,
        dtype=float,
    )

    dd = np.empty(
        n_sim,
        dtype=float,
    )

    keep_probability = 1.0 - removal_pct

    for i in range(n_sim):
        keep = rng.random(n_trades) < keep_probability

        path = values[keep]

        final[i] = path.sum()

        dd[i] = max_drawdown(path)

    return pd.DataFrame(
        {
            "final_R": final,
            "max_DD_R": dd,
        }
    )


# =============================================================================
# ROBUSTNESS SCORECARD
# =============================================================================


def make_scorecard(
    gross: dict,
    stress: dict,
    iid: dict,
    block20: dict,
    random10: dict,
    removal5: dict,
    breakeven: dict,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "gross_expectancy_R": gross["expectancy_R"],
                "gross_PF": gross["PF"],
                "gross_max_DD_R": gross["max_DD_R"],
                "topstep_2tick_expectancy_R": stress["expectancy_R"],
                "topstep_2tick_PF": stress["PF"],
                "topstep_2tick_max_DD_R": stress["max_DD_R"],
                "iid_prob_final_negative": iid["prob_final_negative"],
                "iid_prob_DD_below_-20R": iid["prob_DD_below_-20R"],
                "iid_prob_DD_below_-25R": iid["prob_DD_below_-25R"],
                "iid_prob_DD_below_-30R": iid["prob_DD_below_-30R"],
                "block20_prob_final_negative": block20["prob_final_negative"],
                "block20_prob_DD_below_-20R": block20["prob_DD_below_-20R"],
                "random_0_to_10_prob_final_negative": random10["prob_final_negative"],
                "random_0_to_10_prob_DD_below_-20R": random10["prob_DD_below_-20R"],
                "removal_5pct_prob_final_negative": removal5["prob_final_negative"],
                "removal_5pct_prob_DD_below_-20R": removal5["prob_DD_below_-20R"],
                "break_even_ticks_per_side": breakeven["break_even_ticks_per_side"],
            }
        ]
    )


# =============================================================================
# PLOTS
# =============================================================================


def save_equity_plot(
    df: pd.DataFrame,
    value_col: str,
    filename: str,
    title: str,
) -> None:

    daily = daily_returns(
        df,
        value_col,
    )

    equity = daily.cumsum()

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        equity.index,
        equity.values,
    )

    ax.set_title(title)
    ax.set_ylabel("Cumulative R")
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / filename,
        dpi=180,
    )
    plt.close(fig)


def save_drawdown_plot(
    df: pd.DataFrame,
    value_col: str,
    filename: str,
    title: str,
) -> None:

    daily = daily_returns(
        df,
        value_col,
    )

    equity = daily.cumsum()
    dd = equity - equity.cummax()

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.fill_between(
        dd.index,
        dd.values,
        0,
    )

    ax.set_title(title)
    ax.set_ylabel("Drawdown R")
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / filename,
        dpi=180,
    )
    plt.close(fig)


def save_strategy_equity(
    df: pd.DataFrame,
) -> None:

    matrix = daily_strategy_matrix(df)

    fig, ax = plt.subplots(figsize=(15, 7))

    for strategy in matrix.columns:
        ax.plot(
            matrix.index,
            matrix[strategy].cumsum(),
            label=strategy,
        )

    ax.set_title("Frozen Strategy Equity Curves — Common OOS")
    ax.set_ylabel("Cumulative R")
    ax.legend()
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "03_strategy_equity_oos.png",
        dpi=180,
    )
    plt.close(fig)


def save_daily_distribution(
    df: pd.DataFrame,
) -> None:

    daily = daily_returns(df)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        daily.values,
        bins=50,
    )

    ax.axvline(
        0,
        linestyle="--",
    )

    ax.set_title("Daily Portfolio R Distribution — Common OOS")
    ax.set_xlabel("Daily R")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "04_daily_distribution_oos.png",
        dpi=180,
    )
    plt.close(fig)


def save_monthly_heatmap(
    df: pd.DataFrame,
) -> None:

    x = df.copy()

    ny = x["entry_timestamp"].dt.tz_convert("America/New_York")

    x["year"] = ny.dt.year
    x["month"] = ny.dt.month

    table = (
        x.groupby(
            [
                "year",
                "month",
            ]
        )["R"]
        .sum()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(13, 8))

    image = ax.imshow(
        table.values,
        aspect="auto",
    )

    ax.set_title("Monthly Portfolio R — Common OOS")

    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    ax.set_xticks(range(len(table.columns)))

    ax.set_xticklabels([str(x) for x in table.columns])

    ax.set_yticks(range(len(table.index)))

    ax.set_yticklabels([str(x) for x in table.index])

    fig.colorbar(
        image,
        ax=ax,
        label="R",
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "05_monthly_heatmap_oos.png",
        dpi=180,
    )
    plt.close(fig)


def save_correlation(
    df: pd.DataFrame,
) -> None:

    corr = daily_strategy_matrix(df).corr()

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        corr.values,
        vmin=-1,
        vmax=1,
    )

    strategies = list(corr.columns)

    ax.set_xticks(range(len(strategies)))

    ax.set_xticklabels(strategies)

    ax.set_yticks(range(len(strategies)))

    ax.set_yticklabels(strategies)

    ax.set_title("Daily Strategy Correlation — Common OOS")

    for i in range(len(strategies)):
        for j in range(len(strategies)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "06_daily_correlation_oos.png",
        dpi=180,
    )
    plt.close(fig)


def save_cost_curve(
    grid: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(
        grid["slippage_ticks_per_side"],
        grid["total_R"],
        marker="o",
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_title("Topstep Cost + Slippage Survival")

    ax.set_xlabel("Adverse slippage — ticks / side")

    ax.set_ylabel("Net Total R")

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "07_topstep_cost_slippage_survival.png",
        dpi=180,
    )
    plt.close(fig)


def save_pf_curve(
    grid: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(
        grid["slippage_ticks_per_side"],
        grid["PF"],
        marker="o",
    )

    ax.axhline(
        1,
        linestyle="--",
    )

    ax.set_title("Profit Factor Under Topstep Execution Stress")

    ax.set_xlabel("Adverse slippage — ticks / side")

    ax.set_ylabel("PF")

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "08_topstep_pf_stress.png",
        dpi=180,
    )
    plt.close(fig)


def save_mc_distribution(
    simulation: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
    zero_line: float = 0.0,
) -> None:

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.hist(
        simulation[column],
        bins=80,
    )

    ax.axvline(
        zero_line,
        linestyle="--",
    )

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / filename,
        dpi=180,
    )
    plt.close(fig)


def save_mc_probability(
    summaries: pd.DataFrame,
) -> None:

    selected = summaries[
        [
            "scenario",
            "prob_DD_below_-20R",
        ]
    ].copy()

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(
        np.arange(len(selected)),
        selected["prob_DD_below_-20R"],
    )

    ax.set_title("Monte Carlo Probability of DD < -20R")

    ax.set_ylabel("Probability")

    ax.set_xticks(np.arange(len(selected)))

    ax.set_xticklabels(
        selected["scenario"],
        rotation=65,
        ha="right",
    )

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "11_mc_dd_probability.png",
        dpi=180,
    )
    plt.close(fig)


def save_yearly_plot(
    yearly: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.bar(
        yearly["period"].astype(str),
        yearly["total_R"],
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_title("Yearly Portfolio R — Common OOS")

    ax.set_ylabel("Total R")

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "12_yearly_returns_oos.png",
        dpi=180,
    )
    plt.close(fig)


def save_contribution_plot(
    contribution: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(
        contribution["strategy"],
        contribution["total_R"],
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_title("Strategy Contribution — Common OOS")

    ax.set_ylabel("Total R")

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "13_strategy_contribution_oos.png",
        dpi=180,
    )
    plt.close(fig)


def save_drawdown_duration(
    episodes: pd.DataFrame,
) -> None:

    if episodes.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 7))

    ordered = episodes.sort_values("drawdown_R").head(20)

    ax.bar(
        np.arange(len(ordered)),
        ordered["drawdown_R"],
    )

    ax.set_title("Worst Historical Drawdown Episodes")

    ax.set_ylabel("Drawdown R")

    fig.tight_layout()
    fig.savefig(
        PNG_DIR / "14_drawdown_episodes.png",
        dpi=180,
    )
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    rng = np.random.default_rng(RANDOM_SEED)

    banner("FINAL SYSTEM ROBUSTNESS — MRL1 + S2R + MRS2 + ORB")

    print(f"Project root: {ROOT}")

    print(f"Official OOS: {OOS_START.date()} -> {OOS_END.date()}")

    print(f"Monte Carlo: {N_SIMULATIONS:,}")

    print(f"Topstep MNQ RT cost: ${TOPSTEP_MNQ_RT_COST_USD:.2f}")

    print("Funding / Topstep account simulation: EXCLUDED — separate stage")

    # =========================================================================
    # 1. LOAD
    # =========================================================================

    banner("1. LOAD FROZEN TRADE STREAMS")

    full = load_all_trades()

    print(f"MRL1: {int(full.strategy.eq('MRL1').sum()):,}")

    print(f"S2R : {int(full.strategy.eq('S2R').sum()):,}")

    print(f"MRS2: {int(full.strategy.eq('MRS2').sum()):,}")

    print(f"ORB : {int(full.strategy.eq('ORB').sum()):,}")

    print(f"TOTAL: {len(full):,}")

    # =========================================================================
    # 2. FULL AUDIT
    # =========================================================================

    audit_full_stream(full)

    # =========================================================================
    # 3. OFFICIAL OOS
    # =========================================================================

    oos = filter_period(
        full,
        OOS_START,
        OOS_END,
    )

    audit_oos(oos)

    # =========================================================================
    # 4. POST-OOS REPORT ONLY
    # =========================================================================

    post_oos = filter_period(
        full,
        POST_OOS_START,
        POST_OOS_END,
    )

    banner("POST-OOS HOLDOUT — REPORT ONLY")

    print("IMPORTANT: Post-OOS data is NOT used for optimization.")

    if len(post_oos):
        post_metrics = portfolio_metrics(post_oos)

        print(pd.Series(post_metrics).to_string())

        post_oos.to_csv(
            OUTPUT_DIR / "00_post_oos_holdout_trades.csv",
            index=False,
        )

    # =========================================================================
    # 5. GROSS BASELINE
    # =========================================================================

    banner("5. GROSS COMMON-OOS BASELINE")

    gross_metrics = portfolio_metrics(oos)

    print(pd.Series(gross_metrics).to_string())

    pd.DataFrame([gross_metrics]).to_csv(
        OUTPUT_DIR / "01_gross_oos_baseline.csv",
        index=False,
    )

    # =========================================================================
    # 6. COST + SLIPPAGE
    # =========================================================================

    banner("6. TOPSTEP COST + SLIPPAGE GRID")

    cost_grid = cost_slippage_grid(oos)

    print(
        cost_grid[
            [
                "slippage_ticks_per_side",
                "slippage_rt_usd",
                "topstep_rt_cost_usd",
                "total_rt_friction_usd",
                "total_R",
                "expectancy_R",
                "PF",
                "max_DD_R",
                "daily_sharpe",
                "daily_sortino",
            ]
        ].to_string(index=False)
    )

    cost_grid.to_csv(
        OUTPUT_DIR / "02_topstep_cost_slippage_grid.csv",
        index=False,
    )

    # =========================================================================
    # 7. BREAK EVEN
    # =========================================================================

    banner("7. BREAK-EVEN EXECUTION FRICTION")

    breakeven = break_even_slippage(oos)

    print(pd.Series(breakeven).to_string())

    pd.DataFrame([breakeven]).to_csv(
        OUTPUT_DIR / "03_break_even_execution.csv",
        index=False,
    )

    # =========================================================================
    # 8. TOPSTEP 2-TICK STRESS BASELINE
    # =========================================================================

    stressed_2 = apply_execution_costs(
        oos,
        slippage_ticks_per_side=2,
        include_topstep_cost=True,
    )

    banner("8. TOPSTEP + 2 TICKS/SIDE STRESS BASELINE")

    stress_metrics = portfolio_metrics(
        stressed_2,
        value_col="R_net",
    )

    print(pd.Series(stress_metrics).to_string())

    pd.DataFrame([stress_metrics]).to_csv(
        OUTPUT_DIR / "04_topstep_2tick_baseline.csv",
        index=False,
    )

    # =========================================================================
    # 9. CORRELATION
    # =========================================================================

    banner("9. DAILY STRATEGY CORRELATION")

    corr = daily_strategy_matrix(oos).corr()

    print(corr.to_string(float_format=lambda x: f"{x:.6f}"))

    corr.to_csv(OUTPUT_DIR / "05_daily_correlation.csv")

    # =========================================================================
    # 10. OVERLAP
    # =========================================================================

    banner("10. TRADE OVERLAP")

    overlap = overlap_table(oos)

    print(overlap.to_string(index=False))

    overlap.to_csv(
        OUTPUT_DIR / "06_trade_overlap.csv",
        index=False,
    )

    # =========================================================================
    # 11. CONCURRENCY
    # =========================================================================

    banner("11. CONCURRENT EXPOSURE")

    concurrency_events, concurrency_summary = concurrency_analysis(oos)

    print(concurrency_summary.to_string(index=False))

    concurrency_events.to_csv(
        OUTPUT_DIR / "07_concurrency_events.csv",
        index=False,
    )

    concurrency_summary.to_csv(
        OUTPUT_DIR / "08_concurrency_summary.csv",
        index=False,
    )

    # =========================================================================
    # 12. CONTRIBUTION
    # =========================================================================

    banner("12. STRATEGY CONTRIBUTION")

    contribution = strategy_contribution(oos)

    print(contribution.to_string(index=False))

    contribution.to_csv(
        OUTPUT_DIR / "09_strategy_contribution.csv",
        index=False,
    )

    # =========================================================================
    # 13. TIME STABILITY
    # =========================================================================

    yearly = period_table(
        oos,
        "year",
    )

    monthly = period_table(
        oos,
        "month",
    )

    weekly = period_table(
        oos,
        "week",
    )

    yearly.to_csv(
        OUTPUT_DIR / "10_yearly_stability.csv",
        index=False,
    )

    monthly.to_csv(
        OUTPUT_DIR / "11_monthly_stability.csv",
        index=False,
    )

    weekly.to_csv(
        OUTPUT_DIR / "12_weekly_stability.csv",
        index=False,
    )

    banner("13. YEARLY STABILITY")

    print(yearly.to_string(index=False))

    # =========================================================================
    # 14. STREAKS
    # =========================================================================

    streaks = streak_analysis(oos)

    print()
    print("Streak analysis:")

    print(pd.Series(streaks).to_string())

    pd.DataFrame([streaks]).to_csv(
        OUTPUT_DIR / "13_streak_analysis.csv",
        index=False,
    )

    # =========================================================================
    # 15. CONCENTRATION
    # =========================================================================

    concentration = concentration_analysis(oos)

    banner("14. RETURN CONCENTRATION")

    print(pd.Series(concentration).to_string())

    pd.DataFrame([concentration]).to_csv(
        OUTPUT_DIR / "14_return_concentration.csv",
        index=False,
    )

    # =========================================================================
    # 16. DRAWDOWN EPISODES
    # =========================================================================

    episodes = drawdown_episodes(oos)

    banner("15. HISTORICAL DRAWDOWN EPISODES")

    if len(episodes):
        print(episodes.head(20).to_string(index=False))

    episodes.to_csv(
        OUTPUT_DIR / "15_drawdown_episodes.csv",
        index=False,
    )

    # =========================================================================
    # 17. TAIL STRESS
    # =========================================================================

    tail = tail_degradation(oos)

    banner("16. WORST-TAIL TRADE DEGRADATION")

    print(tail.to_string(index=False))

    tail.to_csv(
        OUTPUT_DIR / "16_tail_degradation.csv",
        index=False,
    )

    # =========================================================================
    # 18. MONTE CARLO — PERMUTATION
    # =========================================================================

    banner("17. MONTE CARLO — TRADE SEQUENCE PERMUTATION")

    values = oos["R"].to_numpy(dtype=float)

    mc_perm = permutation_mc(
        values,
        N_SIMULATIONS,
        rng,
    )

    summary_perm = summarize_mc(
        mc_perm,
        "PERMUTATION",
    )

    print(pd.Series(summary_perm).to_string())

    mc_perm.to_csv(
        OUTPUT_DIR / "17_mc_permutation.csv",
        index=False,
    )

    # =========================================================================
    # 19. MONTE CARLO — IID TRADE
    # =========================================================================

    banner("18. MONTE CARLO — IID TRADE BOOTSTRAP")

    mc_iid_trade = iid_trade_bootstrap(
        values,
        N_SIMULATIONS,
        rng,
    )

    summary_iid_trade = summarize_mc(
        mc_iid_trade,
        "IID_TRADE",
    )

    print(pd.Series(summary_iid_trade).to_string())

    mc_iid_trade.to_csv(
        OUTPUT_DIR / "18_mc_iid_trade.csv",
        index=False,
    )

    # =========================================================================
    # 20. MONTE CARLO — IID DAILY
    # =========================================================================

    banner("19. MONTE CARLO — IID DAILY BOOTSTRAP")

    daily_values = daily_returns(oos).to_numpy(dtype=float)

    mc_iid_daily = iid_daily_bootstrap(
        daily_values,
        N_SIMULATIONS,
        rng,
    )

    summary_iid_daily = summarize_mc(
        mc_iid_daily,
        "IID_DAILY",
    )

    print(pd.Series(summary_iid_daily).to_string())

    mc_iid_daily.to_csv(
        OUTPUT_DIR / "19_mc_iid_daily.csv",
        index=False,
    )

    # =========================================================================
    # 21. MOVING BLOCK BOOTSTRAPS
    # =========================================================================

    block_summaries = []

    for block_length in BLOCK_LENGTHS:
        banner(f"20. MOVING-BLOCK DAILY BOOTSTRAP — {block_length} DAYS")

        mc_block = block_bootstrap(
            daily_values,
            N_SIMULATIONS,
            block_length,
            rng,
        )

        summary = summarize_mc(
            mc_block,
            f"BLOCK_{block_length}D",
        )

        block_summaries.append(summary)

        print(pd.Series(summary).to_string())

        mc_block.to_csv(
            OUTPUT_DIR / (f"20_mc_block_{block_length}d.csv"),
            index=False,
        )

    # =========================================================================
    # 22. STRATEGY-PRESERVING BOOTSTRAP
    # =========================================================================

    banner("21. STRATEGY-PRESERVING DAILY BOOTSTRAP")

    mc_strategy = strategy_day_bootstrap(
        oos,
        N_SIMULATIONS,
        rng,
    )

    summary_strategy = summarize_mc(
        mc_strategy,
        "STRATEGY_DAY",
    )

    print(pd.Series(summary_strategy).to_string())

    mc_strategy.to_csv(
        OUTPUT_DIR / "21_mc_strategy_day.csv",
        index=False,
    )

    # =========================================================================
    # 23. RANDOM EXECUTION STRESS
    # =========================================================================

    execution_summaries = []

    for max_ticks in RANDOM_SLIPPAGE_MAX_TICKS:
        banner(f"22. RANDOM TOPSTEP EXECUTION STRESS — 0..{max_ticks} TICKS/SIDE")

        mc_exec = random_execution_mc(
            oos,
            N_SIMULATIONS,
            max_ticks,
            rng,
        )

        summary = summarize_mc(
            mc_exec,
            f"RANDOM_SLIP_0_TO_{max_ticks}",
        )

        execution_summaries.append(summary)

        print(pd.Series(summary).to_string())

        mc_exec.to_csv(
            OUTPUT_DIR / (f"22_mc_random_slip_{max_ticks}ticks.csv"),
            index=False,
        )

    # =========================================================================
    # 24. TRADE REMOVAL STRESS
    # =========================================================================

    removal_summaries = []

    for pct in TRADE_REMOVAL_PCT:
        banner(f"23. MISSED-TRADE STRESS — {pct:.0%}")

        mc_removed = trade_removal_mc(
            values,
            N_SIMULATIONS,
            pct,
            rng,
        )

        summary = summarize_mc(
            mc_removed,
            f"TRADE_REMOVAL_{pct:.0%}",
        )

        removal_summaries.append(summary)

        print(pd.Series(summary).to_string())

        mc_removed.to_csv(
            OUTPUT_DIR / (f"23_mc_trade_removal_{int(pct * 100)}pct.csv"),
            index=False,
        )

    # =========================================================================
    # 24. MONTE CARLO SUMMARY TABLE
    # =========================================================================

    all_mc_summaries = [
        summary_perm,
        summary_iid_trade,
        summary_iid_daily,
        summary_strategy,
        *block_summaries,
        *execution_summaries,
        *removal_summaries,
    ]

    mc_summary = pd.DataFrame(all_mc_summaries)

    mc_summary.to_csv(
        OUTPUT_DIR / "24_monte_carlo_summary.csv",
        index=False,
    )

    # =========================================================================
    # 25. FINAL SCORECARD
    # =========================================================================

    block20 = next(row for row in block_summaries if row["scenario"] == "BLOCK_20D")

    random10 = next(
        row for row in execution_summaries if row["scenario"] == "RANDOM_SLIP_0_TO_10"
    )

    removal5 = next(
        row for row in removal_summaries if row["scenario"] == "TRADE_REMOVAL_5%"
    )

    scorecard = make_scorecard(
        gross_metrics,
        stress_metrics,
        summary_iid_daily,
        block20,
        random10,
        removal5,
        breakeven,
    )

    scorecard.to_csv(
        OUTPUT_DIR / "25_FINAL_ROBUSTNESS_SCORECARD.csv",
        index=False,
    )

    print()
    print(scorecard.to_string(index=False))

    # =========================================================================
    # 26. PNG REPORT
    # =========================================================================

    banner("26. GENERATING PNG REPORT")

    save_equity_plot(
        oos,
        "R",
        "01_equity_gross_oos.png",
        "Frozen Portfolio — Gross OOS Equity",
    )

    save_equity_plot(
        stressed_2,
        "R_net",
        "02_equity_topstep_2tick_oos.png",
        "Frozen Portfolio — Topstep + 2 Ticks/Side",
    )

    save_strategy_equity(oos)

    save_daily_distribution(oos)

    save_monthly_heatmap(oos)

    save_correlation(oos)

    save_cost_curve(cost_grid)

    save_pf_curve(cost_grid)

    save_drawdown_plot(
        oos,
        "R",
        "09_drawdown_gross_oos.png",
        "Frozen Portfolio — Gross OOS Drawdown",
    )

    save_drawdown_plot(
        stressed_2,
        "R_net",
        "10_drawdown_topstep_2tick_oos.png",
        "Frozen Portfolio — Topstep + 2 Ticks/Side Drawdown",
    )

    save_mc_probability(mc_summary)

    save_yearly_plot(yearly)

    save_contribution_plot(contribution)

    save_drawdown_duration(episodes)

    save_mc_distribution(
        mc_iid_daily,
        "final_R",
        "15_mc_iid_daily_final_R.png",
        "IID Daily Bootstrap — Final R",
    )

    save_mc_distribution(
        mc_iid_daily,
        "max_DD_R",
        "16_mc_iid_daily_drawdown.png",
        "IID Daily Bootstrap — Maximum Drawdown",
        zero_line=0,
    )

    save_mc_distribution(
        mc_strategy,
        "final_R",
        "17_mc_strategy_day_final_R.png",
        "Strategy-Preserving Bootstrap — Final R",
    )

    save_mc_distribution(
        mc_strategy,
        "max_DD_R",
        "18_mc_strategy_day_drawdown.png",
        "Strategy-Preserving Bootstrap — Maximum Drawdown",
        zero_line=0,
    )

    # =========================================================================
    # 27. FINAL AUDIT
    # =========================================================================

    banner("27. FINAL ROBUSTNESS AUDIT")

    print(f"Frozen full sample : {len(full):,}")

    print(f"Official OOS      : {len(oos):,}")

    print(f"Post-OOS holdout  : {len(post_oos):,}")

    print(f"Monte Carlo       : {N_SIMULATIONS:,}")

    print("Strategies        : MRL1 + S2R + MRS2 + ORB")

    print("Optimization      : NONE")

    print("Post-OOS tuning   : NONE")

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("PNG directory:")
    print(PNG_DIR)

    print()
    print("FINAL SYSTEM ROBUSTNESS COMPLETE — FUNDING TEST EXCLUDED.")


if __name__ == "__main__":
    main()
