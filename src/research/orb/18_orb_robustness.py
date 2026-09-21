"""
18_orb_robustness.py

ORB EXACT-OOS ROBUSTNESS / TOPSTEP DESTRUCTION TEST

Purpose
-------
Try to destroy the already-frozen ORB specification using ONLY the exact
external OOS period used for reconciliation.

Frozen ORB specification:
    Opening Range : 30 minutes
    RR            : 2.0
    Entry cutoff  : 11:00 ET
    One trade/day
    Breakout      : touch
    Data           : canonical Databento MNQ 1m

Exact external OOS:
    2020-06-23 -> 2026-06-19 inclusive

Important:
    - PRE-OOS trades are excluded.
    - POST-OOS trades are excluded.
    - No parameters are optimized.
    - POST-OOS is reported separately only as an untouched holdout.
    - Permutation Monte Carlo is used for path/DD risk.
    - IID bootstrap is used for final-result uncertainty.
    - Topstep MNQ RT commission = $1.22.
    - Slippage scenarios are assumptions, not official Topstep slippage.

Authoritative frozen input:
    src/research/results/orb/orb_reconciliation_trades.csv

Outputs:
    src/research/results/orb/robustness_oos/
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "orb"
    / "orb_reconciliation_trades.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb" / "robustness_oos"

PNG_DIR = OUTPUT_DIR / "png"

# ---------------------------------------------------------------------
# EXACT EXTERNAL OOS
# ---------------------------------------------------------------------

OOS_START = pd.Timestamp("2020-06-23", tz="America/New_York")
OOS_END = pd.Timestamp("2026-06-19 23:59:59", tz="America/New_York")

# Untouched post-OOS holdout.
POST_OOS_START = pd.Timestamp("2026-06-20", tz="America/New_York")
POST_OOS_END = pd.Timestamp("2026-08-26 23:59:59", tz="America/New_York")

# ---------------------------------------------------------------------
# TOPSTEP / MNQ EXECUTION MODEL
# ---------------------------------------------------------------------

TOPSTEP_RT_COMMISSION_USD = 1.22

MNQ_TICK_SIZE = 0.25
MNQ_TICK_VALUE_USD = 0.50
MNQ_POINT_VALUE_USD = MNQ_TICK_VALUE_USD / MNQ_TICK_SIZE

# Per-side slippage scenarios.
SLIPPAGE_TICKS_PER_SIDE = [
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
]

# ---------------------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------------------

MC_SIMULATIONS = 50_000
RNG_SEED = 42

# ---------------------------------------------------------------------
# TRADE DEGRADATION
# ---------------------------------------------------------------------

DEGRADATION_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

# =============================================================================
# UTILS
# =============================================================================


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


# =============================================================================
# LOAD
# =============================================================================


def load_trades() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Frozen ORB reconciliation file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("ORB reconciliation file is empty.")

    # -----------------------------------------------------------------
    # Timestamp discovery
    # -----------------------------------------------------------------

    timestamp_candidates = [
        "entry_timestamp",
        "timestamp",
        "entry_time",
        "datetime",
        "date",
    ]

    timestamp_col = None

    for col in timestamp_candidates:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(
            "Could not find an entry timestamp column. "
            f"Available columns:\n{list(df.columns)}"
        )

    df["entry_timestamp"] = pd.to_datetime(
        df[timestamp_col],
        utc=True,
        errors="coerce",
    )

    if df["entry_timestamp"].isna().any():
        raise ValueError("Some entry timestamps could not be parsed.")

    df["entry_timestamp_et"] = df["entry_timestamp"].dt.tz_convert("America/New_York")

    # -----------------------------------------------------------------
    # R multiple discovery
    # -----------------------------------------------------------------

    r_candidates = [
        "net_R",
        "r_multiple",
        "R",
        "r",
        "result_R",
        "trade_R",
    ]

    r_col = None

    for col in r_candidates:
        if col in df.columns:
            r_col = col
            break

    if r_col is None:
        raise ValueError(
            f"Could not find trade R column. Available columns:\n{list(df.columns)}"
        )

    df["net_R"] = pd.to_numeric(
        df[r_col],
        errors="coerce",
    )

    if df["net_R"].isna().any():
        raise ValueError("Some R values could not be parsed.")

    # -----------------------------------------------------------------
    # Stop/risk distance
    # -----------------------------------------------------------------

    df["stop_points"] = find_stop_points(df)

    # -----------------------------------------------------------------
    # Sort
    # -----------------------------------------------------------------

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Period classification
    # -----------------------------------------------------------------

    df["period"] = "OTHER"

    df.loc[
        (df["entry_timestamp_et"] >= OOS_START) & (df["entry_timestamp_et"] <= OOS_END),
        "period",
    ] = "EXACT_OOS"

    df.loc[
        (df["entry_timestamp_et"] >= POST_OOS_START)
        & (df["entry_timestamp_et"] <= POST_OOS_END),
        "period",
    ] = "POST_OOS"

    print(f"Input:")
    print(INPUT_FILE)
    print(f"All trades loaded: {len(df):,}")

    print(
        f"All-data range: "
        f"{df['entry_timestamp'].min()} -> "
        f"{df['entry_timestamp'].max()}"
    )

    print(f"Exact OOS range: {OOS_START} -> {OOS_END}")

    print(f"Exact OOS trades: {(df['period'] == 'EXACT_OOS').sum():,}")

    print(f"Post-OOS holdout trades: {(df['period'] == 'POST_OOS').sum():,}")

    return df


# =============================================================================
# STOP DISTANCE
# =============================================================================


def find_stop_points(df: pd.DataFrame) -> pd.Series:
    """
    Discover the frozen ORB risk distance.

    Priority:
        1. stop_points
        2. risk_points
        3. stop_distance
        4. or_width
        5. opening_range_width
        6. entry/stop price difference
    """

    candidates = [
        "stop_points",
        "risk_points",
        "stop_distance",
        "or_width",
        "opening_range_width",
    ]

    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if values.notna().all() and (values > 0).all():
                print(f"Risk distance source: {col}")
                return values.astype(float)

    # -----------------------------------------------------------------
    # Entry / stop price fallback
    # -----------------------------------------------------------------

    entry_candidates = [
        "entry_price",
        "entry",
        "fill_price",
    ]

    stop_candidates = [
        "stop_price",
        "stop",
        "sl_price",
    ]

    entry_col = next(
        (c for c in entry_candidates if c in df.columns),
        None,
    )

    stop_col = next(
        (c for c in stop_candidates if c in df.columns),
        None,
    )

    if entry_col and stop_col:
        entry = pd.to_numeric(
            df[entry_col],
            errors="coerce",
        )

        stop = pd.to_numeric(
            df[stop_col],
            errors="coerce",
        )

        risk = (entry - stop).abs()

        if risk.notna().all() and (risk > 0).all():
            print(f"Risk distance source: |{entry_col} - {stop_col}|")
            return risk.astype(float)

    raise ValueError(
        "\nCould not determine stop/risk distance.\n"
        "Available columns:\n"
        f"{list(df.columns)}\n\n"
        "The robustness test needs the actual frozen ORB "
        "risk distance to convert Topstep dollar costs into R."
    )


# =============================================================================
# FILTER EXACT OOS
# =============================================================================


def split_periods(
    all_trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    exact_oos = all_trades.loc[all_trades["period"] == "EXACT_OOS"].copy()

    post_oos = all_trades.loc[all_trades["period"] == "POST_OOS"].copy()

    pre_other = all_trades.loc[all_trades["period"] == "OTHER"].copy()

    if len(exact_oos) != 1442:
        warnings.warn(
            f"Expected 1,442 exact OOS trades from the "
            f"previous reconciliation, but found {len(exact_oos):,}."
        )

    if exact_oos.empty:
        raise ValueError("Exact OOS dataset is empty.")

    exact_oos = exact_oos.sort_values("entry_timestamp").reset_index(drop=True)

    return exact_oos, post_oos, pre_other


# =============================================================================
# TOPSTEP COST MODEL
# =============================================================================


def add_execution_friction(
    df: pd.DataFrame,
    slippage_ticks_per_side: int,
) -> pd.DataFrame:

    out = df.copy()

    # Commission is round-trip.
    commission_usd = TOPSTEP_RT_COMMISSION_USD

    # Slippage:
    # ticks_per_side * $0.50 * 2 sides
    slippage_usd = slippage_ticks_per_side * MNQ_TICK_VALUE_USD * 2.0

    # R conversion depends on each trade's actual stop distance.
    risk_usd = out["stop_points"] * MNQ_POINT_VALUE_USD

    out["commission_R"] = commission_usd / risk_usd

    out["slippage_R"] = slippage_usd / risk_usd

    out["friction_R"] = out["commission_R"] + out["slippage_R"]

    out["stressed_R"] = out["net_R"] - out["friction_R"]

    return out


# =============================================================================
# METRICS
# =============================================================================


def max_drawdown(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)

    if len(x) == 0:
        return 0.0

    equity = np.cumsum(x)
    peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]

    dd = equity - peak

    return float(dd.min())


def profit_factor(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)

    gross_profit = x[x > 0].sum()
    gross_loss = -x[x < 0].sum()

    if gross_loss <= 0:
        return float("inf")

    return float(gross_profit / gross_loss)


def daily_returns(
    df: pd.DataFrame,
    r_col: str = "net_R",
) -> pd.Series:

    temp = df.copy()

    temp["date_et"] = temp["entry_timestamp_et"].dt.date

    daily = temp.groupby("date_et")[r_col].sum()

    return daily


def daily_sharpe(
    df: pd.DataFrame,
    r_col: str = "net_R",
) -> float:

    daily = daily_returns(df, r_col)

    if len(daily) < 2:
        return float("nan")

    std = daily.std(ddof=1)

    if std == 0:
        return float("nan")

    return float(daily.mean() / std * math.sqrt(252))


def daily_sortino(
    df: pd.DataFrame,
    r_col: str = "net_R",
) -> float:

    daily = daily_returns(df, r_col)

    if len(daily) < 2:
        return float("nan")

    downside = daily[daily < 0]

    if len(downside) == 0:
        return float("inf")

    downside_std = downside.std(ddof=1)

    if downside_std == 0:
        return float("nan")

    return float(daily.mean() / downside_std * math.sqrt(252))


def calculate_metrics(
    df: pd.DataFrame,
    r_col: str = "net_R",
) -> dict:

    r = pd.to_numeric(
        df[r_col],
        errors="coerce",
    ).dropna()

    wins = r[r > 0]
    losses = r[r < 0]

    result = {
        "trades": int(len(r)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": (float((r > 0).mean()) if len(r) else float("nan")),
        "total_R": float(r.sum()),
        "expectancy_R": (float(r.mean()) if len(r) else float("nan")),
        "median_R": (float(r.median()) if len(r) else float("nan")),
        "profit_factor": profit_factor(r),
        "avg_win_R": (float(wins.mean()) if len(wins) else float("nan")),
        "avg_loss_R": (float(losses.mean()) if len(losses) else float("nan")),
        "payoff": (
            float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses)
            else float("nan")
        ),
        "max_dd_R": max_drawdown(r),
        "daily_sharpe": daily_sharpe(
            df,
            r_col,
        ),
        "daily_sortino": daily_sortino(
            df,
            r_col,
        ),
    }

    return result


def print_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        print(f"{key:30s}: {value}")


# =============================================================================
# BASELINE
# =============================================================================


def baseline_analysis(
    oos: pd.DataFrame,
) -> dict:

    print_section("EXACT OOS BASELINE — BEFORE TOPSTEP COST")

    metrics = calculate_metrics(
        oos,
        "net_R",
    )

    print_metrics(metrics)

    return metrics


# =============================================================================
# TOPSTEP STRESS
# =============================================================================


def topstep_stress(
    oos: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:

    print_header("TOPSTEP + SLIPPAGE STRESS — EXACT OOS ONLY")

    rows = []
    stressed_datasets = {}

    for ticks in SLIPPAGE_TICKS_PER_SIDE:
        stressed = add_execution_friction(
            oos,
            ticks,
        )

        stressed_datasets[ticks] = stressed

        metrics = calculate_metrics(
            stressed,
            "stressed_R",
        )

        average_friction = stressed["friction_R"].mean()

        slip_rt_usd = ticks * MNQ_TICK_VALUE_USD * 2

        print(
            f"\n"
            f"{ticks:2d} ticks/side | "
            f"slip RT=${slip_rt_usd:.2f} | "
            f"friction={average_friction:.5f}R | "
            f"total={metrics['total_R']:.2f}R | "
            f"exp={metrics['expectancy_R']:.5f} | "
            f"PF={metrics['profit_factor']:.4f} | "
            f"DD={metrics['max_dd_R']:.2f}R"
        )

        row = {
            "ticks_per_side": ticks,
            "slippage_rt_usd": slip_rt_usd,
            "average_friction_R": average_friction,
            **metrics,
        }

        rows.append(row)

    return pd.DataFrame(rows), stressed_datasets


# =============================================================================
# BREAK-EVEN
# =============================================================================


def calculate_break_even(
    stress_df: pd.DataFrame,
) -> dict:

    print_header("BREAK-EVEN SLIPPAGE")

    non_positive = stress_df.loc[stress_df["total_R"] <= 0]

    if non_positive.empty:
        print("No tested slippage level reached a non-positive result.")

        return {
            "break_even_ticks_per_side": np.nan,
            "break_even_rt_slippage_usd": np.nan,
        }

    first = non_positive.iloc[0]

    ticks = int(first["ticks_per_side"])

    rt_usd = float(first["slippage_rt_usd"])

    print(f"First non-positive tested result: {ticks} ticks/side")

    print(f"Approx RT slippage: ${rt_usd:.2f}")

    return {
        "break_even_ticks_per_side": ticks,
        "break_even_rt_slippage_usd": rt_usd,
    }


# =============================================================================
# COST GRID
# =============================================================================


def cost_slippage_grid(
    oos: pd.DataFrame,
) -> pd.DataFrame:

    print_header("TOPSTEP COST + SLIPPAGE GRID — EXACT OOS")

    rows = []

    commission_levels = [
        0.0,
        0.61,
        1.22,
        1.83,
        2.44,
    ]

    for commission in commission_levels:
        for ticks in SLIPPAGE_TICKS_PER_SIDE:
            temp = oos.copy()

            risk_usd = temp["stop_points"] * MNQ_POINT_VALUE_USD

            commission_R = commission / risk_usd

            slippage_usd = ticks * MNQ_TICK_VALUE_USD * 2

            slippage_R = slippage_usd / risk_usd

            temp["stress_R"] = temp["net_R"] - commission_R - slippage_R

            metrics = calculate_metrics(
                temp,
                "stress_R",
            )

            rows.append(
                {
                    "commission_rt_usd": commission,
                    "ticks_per_side": ticks,
                    "slippage_rt_usd": slippage_usd,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PERMUTATION MONTE CARLO
# =============================================================================


def permutation_monte_carlo(
    oos: pd.DataFrame,
    simulations: int = MC_SIMULATIONS,
    seed: int = RNG_SEED,
) -> dict:

    print_header("MONTE CARLO — SEQUENCE PERMUTATION")

    rng = np.random.default_rng(seed)

    values = oos["net_R"].to_numpy(dtype=float)

    n = len(values)

    dd_values = np.empty(
        simulations,
        dtype=float,
    )

    # Save representative equity curves.
    sample_curves = []

    for i in range(simulations):
        shuffled = rng.permutation(values)

        equity = np.cumsum(shuffled)

        peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]

        dd = equity - peak

        dd_values[i] = dd.min()

        if i < 100:
            sample_curves.append(equity)

    final_R = float(values.sum())

    print(f"Historical final R: {final_R:.6f}")

    print("Permutation final R is invariant because the trades are only reordered.")

    print(f"DD mean: {np.mean(dd_values):.4f}R")

    print(f"DD P5: {np.percentile(dd_values, 5):.4f}R")

    print(f"DD P50: {np.percentile(dd_values, 50):.4f}R")

    print(f"DD P95: {np.percentile(dd_values, 95):.4f}R")

    print(f"P(DD < -20R): {np.mean(dd_values < -20):.4f}")

    print(f"P(DD < -25R): {np.mean(dd_values < -25):.4f}")

    print(f"P(DD < -30R): {np.mean(dd_values < -30):.4f}")

    return {
        "dd_values": dd_values,
        "sample_curves": sample_curves,
        "final_R": final_R,
    }


# =============================================================================
# IID BOOTSTRAP
# =============================================================================


def bootstrap_final_result(
    oos: pd.DataFrame,
    simulations: int = MC_SIMULATIONS,
    seed: int = RNG_SEED,
) -> dict:

    print_header("MONTE CARLO — IID BOOTSTRAP")

    rng = np.random.default_rng(seed)

    values = oos["net_R"].to_numpy(dtype=float)

    n = len(values)

    final_values = np.empty(
        simulations,
        dtype=float,
    )

    dd_values = np.empty(
        simulations,
        dtype=float,
    )

    for i in range(simulations):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        final_values[i] = sample.sum()

        dd_values[i] = max_drawdown(sample)

    result = {
        "final_values": final_values,
        "dd_values": dd_values,
    }

    print(f"simulations          : {simulations}")
    print(f"final_mean_R         : {final_values.mean()}")
    print(f"final_std_R          : {final_values.std(ddof=1)}")
    print(f"final_p1_R           : {np.percentile(final_values, 1)}")
    print(f"final_p5_R           : {np.percentile(final_values, 5)}")
    print(f"final_p50_R          : {np.percentile(final_values, 50)}")
    print(f"final_p95_R          : {np.percentile(final_values, 95)}")
    print(f"final_p99_R          : {np.percentile(final_values, 99)}")
    print(f"prob_final_negative  : {np.mean(final_values < 0)}")
    print(f"dd_mean_R            : {dd_values.mean()}")
    print(f"dd_p5_R              : {np.percentile(dd_values, 5)}")
    print(f"dd_p50_R             : {np.percentile(dd_values, 50)}")
    print(f"dd_p95_R             : {np.percentile(dd_values, 95)}")
    print(f"prob_dd_below_-20R   : {np.mean(dd_values < -20)}")
    print(f"prob_dd_below_-25R   : {np.mean(dd_values < -25)}")
    print(f"prob_dd_below_-30R   : {np.mean(dd_values < -30)}")

    return result


# =============================================================================
# BOOTSTRAP UNDER EXECUTION STRESS
# =============================================================================


def bootstrap_execution_stress(
    stressed_datasets: dict[int, pd.DataFrame],
    simulations: int = MC_SIMULATIONS,
    seed: int = RNG_SEED + 100,
) -> pd.DataFrame:

    print_header("BOOTSTRAP UNDER TOPSTEP EXECUTION STRESS")

    rng = np.random.default_rng(seed)

    rows = []

    for ticks, df in stressed_datasets.items():
        values = df["stressed_R"].to_numpy(dtype=float)

        n = len(values)

        negative_count = 0
        dd20_count = 0
        dd25_count = 0

        for _ in range(simulations):
            sample = rng.choice(
                values,
                size=n,
                replace=True,
            )

            if sample.sum() < 0:
                negative_count += 1

            dd = max_drawdown(sample)

            if dd < -20:
                dd20_count += 1

            if dd < -25:
                dd25_count += 1

        p_loss = negative_count / simulations

        p_dd20 = dd20_count / simulations

        p_dd25 = dd25_count / simulations

        print(
            f"{ticks:2d} ticks/side | "
            f"P(loss)={p_loss:.4f} | "
            f"P(DD<-20R)={p_dd20:.4f} | "
            f"P(DD<-25R)={p_dd25:.4f}"
        )

        rows.append(
            {
                "ticks_per_side": ticks,
                "prob_final_negative": p_loss,
                "prob_dd_below_-20R": p_dd20,
                "prob_dd_below_-25R": p_dd25,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TRADE-LEVEL DEGRADATION
# =============================================================================


def trade_level_degradation(
    oos: pd.DataFrame,
) -> pd.DataFrame:

    print_header("TRADE-LEVEL DEGRADATION")

    base = oos["net_R"].to_numpy(dtype=float)

    rows = []

    for degradation in DEGRADATION_LEVELS:
        stressed = base.copy()

        winners = stressed > 0
        losers = stressed < 0

        # Winners lose a percentage of their profit.
        stressed[winners] *= 1.0 - degradation

        # Losers become proportionally worse.
        stressed[losers] *= 1.0 + degradation

        temp = oos.copy()
        temp["degraded_R"] = stressed

        metrics = calculate_metrics(
            temp,
            "degraded_R",
        )

        rows.append(
            {
                "degradation": degradation,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# YEARLY STRESS
# =============================================================================


def yearly_stress(
    stressed_datasets: dict[int, pd.DataFrame],
) -> pd.DataFrame:

    print_header("YEARLY TOPSTEP STRESS — EXACT OOS")

    rows = []

    for ticks, df in stressed_datasets.items():
        temp = df.copy()

        temp["year"] = temp["entry_timestamp_et"].dt.year

        for year, group in temp.groupby("year"):
            metrics = calculate_metrics(
                group,
                "stressed_R",
            )

            rows.append(
                {
                    "ticks_per_side": ticks,
                    "year": year,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# POST-OOS REPORT
# =============================================================================


def post_oos_report(
    post_oos: pd.DataFrame,
) -> pd.DataFrame:

    print_header("POST-OOS HOLDOUT — REPORT ONLY")

    if post_oos.empty:
        print("No post-OOS trades found.")
        return pd.DataFrame()

    metrics = calculate_metrics(
        post_oos,
        "net_R",
    )

    print(
        "IMPORTANT: this dataset was NOT used for optimization or robustness decisions."
    )

    print_metrics(metrics)

    return pd.DataFrame([metrics])


# =============================================================================
# PLOTS
# =============================================================================


def savefig(name: str) -> None:
    path = PNG_DIR / name
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_baseline_equity(
    oos: pd.DataFrame,
) -> None:

    plt.figure(figsize=(12, 6))

    equity = np.cumsum(oos["net_R"].to_numpy())

    plt.plot(
        equity,
        linewidth=1.2,
    )

    plt.title("ORB Exact OOS — Baseline Equity Curve")
    plt.xlabel("Trade")
    plt.ylabel("Cumulative R")
    plt.grid(alpha=0.25)

    savefig("01_oos_baseline_equity.png")


def plot_trade_distribution(
    oos: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.hist(
        oos["net_R"],
        bins=60,
    )

    plt.title("ORB Exact OOS — Trade R Distribution")
    plt.xlabel("R")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)

    savefig("02_oos_trade_distribution.png")


def plot_expectancy_vs_slippage(
    stress_df: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        stress_df["ticks_per_side"],
        stress_df["expectancy_R"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.title("ORB Exact OOS — Expectancy vs Slippage")
    plt.xlabel("Slippage (ticks per side)")
    plt.ylabel("Expectancy (R)")
    plt.grid(alpha=0.25)

    savefig("03_oos_expectancy_vs_slippage.png")


def plot_pf_vs_slippage(
    stress_df: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        stress_df["ticks_per_side"],
        stress_df["profit_factor"],
        marker="o",
    )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.title("ORB Exact OOS — Profit Factor vs Slippage")
    plt.xlabel("Slippage (ticks per side)")
    plt.ylabel("Profit Factor")
    plt.grid(alpha=0.25)

    savefig("04_oos_pf_vs_slippage.png")


def plot_stress_curves(
    stressed_datasets: dict[int, pd.DataFrame],
) -> None:

    plt.figure(figsize=(12, 6))

    for ticks, df in stressed_datasets.items():
        if ticks not in [
            0,
            1,
            2,
            5,
            10,
            12,
        ]:
            continue

        equity = np.cumsum(df["stressed_R"].to_numpy())

        plt.plot(
            equity,
            label=f"{ticks} ticks/side",
        )

    plt.title("ORB Exact OOS — Topstep Execution Stress")
    plt.xlabel("Trade")
    plt.ylabel("Cumulative R")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("05_oos_stress_equity_curves.png")


def plot_heatmap(
    grid: pd.DataFrame,
) -> None:

    pivot = grid.pivot(
        index="commission_rt_usd",
        columns="ticks_per_side",
        values="expectancy_R",
    )

    plt.figure(figsize=(12, 6))

    plt.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(label="Expectancy (R)")

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
    )

    plt.yticks(
        range(len(pivot.index)),
        [f"${x:.2f}" for x in pivot.index],
    )

    plt.xlabel("Slippage (ticks/side)")

    plt.ylabel("Commission ($ RT)")

    plt.title("ORB Exact OOS — Cost / Slippage Expectancy Heatmap")

    savefig("06_oos_cost_slippage_heatmap.png")


def plot_permutation_dd(
    permutation: dict,
) -> None:

    dd_values = permutation["dd_values"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        dd_values,
        bins=60,
    )

    plt.axvline(
        -20,
        linestyle="--",
        label="-20R",
    )

    plt.axvline(
        -25,
        linestyle="--",
        label="-25R",
    )

    plt.axvline(
        -30,
        linestyle="--",
        label="-30R",
    )

    plt.title("Permutation Monte Carlo — Maximum Drawdown")
    plt.xlabel("Maximum Drawdown (R)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("07_oos_permutation_max_dd.png")


def plot_bootstrap_final(
    bootstrap: dict,
) -> None:

    values = bootstrap["final_values"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=60,
    )

    plt.axvline(
        0,
        linestyle="--",
        label="0R",
    )

    plt.axvline(
        np.percentile(values, 5),
        linestyle="--",
        label="P5",
    )

    plt.title("IID Bootstrap — Final R Distribution")
    plt.xlabel("Final R")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("08_oos_bootstrap_final_R.png")


def plot_bootstrap_dd(
    bootstrap: dict,
) -> None:

    values = bootstrap["dd_values"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=60,
    )

    plt.axvline(
        -20,
        linestyle="--",
        label="-20R",
    )

    plt.axvline(
        -25,
        linestyle="--",
        label="-25R",
    )

    plt.title("IID Bootstrap — Maximum Drawdown")
    plt.xlabel("Maximum Drawdown (R)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("09_oos_bootstrap_max_dd.png")


def plot_bootstrap_stress(
    stress_bootstrap: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        stress_bootstrap["ticks_per_side"],
        stress_bootstrap["prob_final_negative"],
        marker="o",
        label="P(final < 0)",
    )

    plt.plot(
        stress_bootstrap["ticks_per_side"],
        stress_bootstrap["prob_dd_below_-20R"],
        marker="o",
        label="P(DD < -20R)",
    )

    plt.xlabel("Slippage (ticks/side)")
    plt.ylabel("Probability")
    plt.title("Bootstrap — Topstep Execution Stress")

    plt.legend()
    plt.grid(alpha=0.25)

    savefig("10_oos_bootstrap_stress_probabilities.png")


def plot_degradation(
    degradation: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 6))

    plt.plot(
        degradation["degradation"] * 100,
        degradation["expectancy_R"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.title("Trade-Level Winner/Loser Degradation")

    plt.xlabel("Degradation (%)")

    plt.ylabel("Expectancy (R)")

    plt.grid(alpha=0.25)

    savefig("11_oos_trade_degradation.png")


def plot_yearly_stress(
    yearly: pd.DataFrame,
) -> None:

    selected = yearly.loc[yearly["ticks_per_side"].isin([0, 2, 5, 10, 12])]

    pivot = selected.pivot(
        index="year",
        columns="ticks_per_side",
        values="total_R",
    )

    plt.figure(figsize=(12, 6))

    pivot.plot(
        kind="bar",
        ax=plt.gca(),
    )

    plt.title("ORB Exact OOS — Yearly Stress")

    plt.xlabel("Year")
    plt.ylabel("Total R")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    savefig("12_oos_yearly_stress.png")


# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================


def robustness_checks(
    baseline: dict,
    stress_df: pd.DataFrame,
    bootstrap: dict,
) -> pd.DataFrame:

    checks = []

    def add(
        name: str,
        condition: bool,
        value,
    ):
        status = "PASS" if condition else "FAIL"

        print(f"{status:4s} | {name:50s} | value={value}")

        checks.append(
            {
                "check": name,
                "pass": bool(condition),
                "value": value,
            }
        )

    print_header("ROBUSTNESS CHECKS — EXACT OOS")

    add(
        "OOS_trade_count_is_1442",
        baseline["trades"] == 1442,
        baseline["trades"],
    )

    add(
        "baseline_positive_expectancy",
        baseline["expectancy_R"] > 0,
        baseline["expectancy_R"],
    )

    add(
        "baseline_PF_gt_1",
        baseline["profit_factor"] > 1,
        baseline["profit_factor"],
    )

    fee_row = stress_df.loc[stress_df["ticks_per_side"] == 0].iloc[0]

    add(
        "topstep_fees_preserve_positive_expectancy",
        fee_row["expectancy_R"] > 0,
        fee_row["expectancy_R"],
    )

    for ticks in [1, 2]:
        row = stress_df.loc[stress_df["ticks_per_side"] == ticks].iloc[0]

        add(
            f"survives_{ticks}_tick_per_side",
            row["expectancy_R"] > 0,
            row["expectancy_R"],
        )

    bootstrap_negative = np.mean(bootstrap["final_values"] < 0)

    add(
        "bootstrap_negative_probability_lt_5pct",
        bootstrap_negative < 0.05,
        bootstrap_negative,
    )

    return pd.DataFrame(checks)


# =============================================================================
# SAVE MASTER SUMMARY
# =============================================================================


def save_master_summary(
    baseline: dict,
    stress_df: pd.DataFrame,
    break_even: dict,
    bootstrap: dict,
    checks: pd.DataFrame,
    post_oos: pd.DataFrame,
) -> pd.DataFrame:

    zero_slip = stress_df.loc[stress_df["ticks_per_side"] == 0].iloc[0]

    row_2 = stress_df.loc[stress_df["ticks_per_side"] == 2].iloc[0]

    bootstrap_negative = np.mean(bootstrap["final_values"] < 0)

    summary = pd.DataFrame(
        [
            {
                "oos_start": OOS_START,
                "oos_end": OOS_END,
                "trades": baseline["trades"],
                "baseline_total_R": baseline["total_R"],
                "baseline_expectancy_R": baseline["expectancy_R"],
                "baseline_profit_factor": baseline["profit_factor"],
                "baseline_max_dd_R": baseline["max_dd_R"],
                "topstep_rt_commission_usd": (TOPSTEP_RT_COMMISSION_USD),
                "topstep_expectancy_R": (zero_slip["expectancy_R"]),
                "topstep_profit_factor": (zero_slip["profit_factor"]),
                "topstep_max_dd_R": (zero_slip["max_dd_R"]),
                "two_tick_expectancy_R": (row_2["expectancy_R"]),
                "two_tick_profit_factor": (row_2["profit_factor"]),
                "two_tick_max_dd_R": (row_2["max_dd_R"]),
                "break_even_slippage_ticks_per_side": (
                    break_even["break_even_ticks_per_side"]
                ),
                "break_even_rt_slippage_usd": (
                    break_even["break_even_rt_slippage_usd"]
                ),
                "bootstrap_probability_negative": (bootstrap_negative),
                "bootstrap_final_p5_R": (
                    np.percentile(
                        bootstrap["final_values"],
                        5,
                    )
                ),
                "bootstrap_final_p50_R": (
                    np.percentile(
                        bootstrap["final_values"],
                        50,
                    )
                ),
                "bootstrap_final_p95_R": (
                    np.percentile(
                        bootstrap["final_values"],
                        95,
                    )
                ),
                "bootstrap_dd_p5": (
                    np.percentile(
                        bootstrap["dd_values"],
                        5,
                    )
                ),
                "bootstrap_dd_p50": (
                    np.percentile(
                        bootstrap["dd_values"],
                        50,
                    )
                ),
                "bootstrap_dd_p95": (
                    np.percentile(
                        bootstrap["dd_values"],
                        95,
                    )
                ),
                "post_oos_trades_reported_only": len(post_oos),
                "all_checks_pass": bool(checks["pass"].all()),
            }
        ]
    )

    return summary


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    ensure_dirs()

    print_header("ORB EXACT-OOS ROBUSTNESS / TOPSTEP DESTRUCTION TEST")

    all_trades = load_trades()

    exact_oos, post_oos, pre_other = split_periods(all_trades)

    # -----------------------------------------------------------------
    # IMPORTANT AUDIT
    # -----------------------------------------------------------------

    print_section("DATA PARTITION AUDIT")

    print(f"Pre/other excluded trades : {len(pre_other):,}")

    print(f"Exact OOS trades          : {len(exact_oos):,}")

    print(f"Post-OOS holdout          : {len(post_oos):,}")

    print(
        f"Total                    : "
        f"{len(pre_other) + len(exact_oos) + len(post_oos):,}"
    )

    print("\nRobustness calculations use EXACT OOS ONLY.")

    print("Post-OOS is NOT used for optimization or robustness decisions.")

    # -----------------------------------------------------------------
    # STOP DISTANCE
    # -----------------------------------------------------------------

    print_section("STOP DISTANCE SUMMARY")

    print(exact_oos["stop_points"].describe())

    # -----------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------

    baseline = baseline_analysis(exact_oos)

    # -----------------------------------------------------------------
    # TOPSTEP STRESS
    # -----------------------------------------------------------------

    stress_df, stressed_datasets = topstep_stress(exact_oos)

    # -----------------------------------------------------------------
    # BREAK EVEN
    # -----------------------------------------------------------------

    break_even = calculate_break_even(stress_df)

    # -----------------------------------------------------------------
    # COST GRID
    # -----------------------------------------------------------------

    grid = cost_slippage_grid(exact_oos)

    # -----------------------------------------------------------------
    # PERMUTATION MC
    # -----------------------------------------------------------------

    permutation = permutation_monte_carlo(exact_oos)

    # -----------------------------------------------------------------
    # IID BOOTSTRAP
    # -----------------------------------------------------------------

    bootstrap = bootstrap_final_result(exact_oos)

    # -----------------------------------------------------------------
    # BOOTSTRAP STRESS
    # -----------------------------------------------------------------

    stress_bootstrap = bootstrap_execution_stress(stressed_datasets)

    # -----------------------------------------------------------------
    # DEGRADATION
    # -----------------------------------------------------------------

    degradation = trade_level_degradation(exact_oos)

    # -----------------------------------------------------------------
    # YEARLY
    # -----------------------------------------------------------------

    yearly = yearly_stress(stressed_datasets)

    # -----------------------------------------------------------------
    # POST-OOS REPORT ONLY
    # -----------------------------------------------------------------

    post_oos_summary = post_oos_report(post_oos)

    # -----------------------------------------------------------------
    # CHECKS
    # -----------------------------------------------------------------

    checks = robustness_checks(
        baseline,
        stress_df,
        bootstrap,
    )

    # -----------------------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------------------

    stress_df.to_csv(
        OUTPUT_DIR / "orb_exact_oos_topstep_stress.csv",
        index=False,
    )

    grid.to_csv(
        OUTPUT_DIR / "orb_exact_oos_cost_slippage_grid.csv",
        index=False,
    )

    stress_bootstrap.to_csv(
        OUTPUT_DIR / "orb_exact_oos_bootstrap_stress.csv",
        index=False,
    )

    degradation.to_csv(
        OUTPUT_DIR / "orb_exact_oos_trade_degradation.csv",
        index=False,
    )

    yearly.to_csv(
        OUTPUT_DIR / "orb_exact_oos_yearly_stress.csv",
        index=False,
    )

    checks.to_csv(
        OUTPUT_DIR / "orb_exact_oos_robustness_checks.csv",
        index=False,
    )

    if not post_oos_summary.empty:
        post_oos_summary.to_csv(
            OUTPUT_DIR / "orb_post_oos_holdout_report.csv",
            index=False,
        )

    # Monte Carlo DD
    pd.DataFrame(
        {
            "max_dd_R": permutation["dd_values"],
        }
    ).to_csv(
        OUTPUT_DIR / "orb_exact_oos_permutation_dd.csv",
        index=False,
    )

    # Bootstrap final
    pd.DataFrame(
        {
            "final_R": bootstrap["final_values"],
            "max_dd_R": bootstrap["dd_values"],
        }
    ).to_csv(
        OUTPUT_DIR / "orb_exact_oos_bootstrap_distribution.csv",
        index=False,
    )

    # Master summary
    master_summary = save_master_summary(
        baseline,
        stress_df,
        break_even,
        bootstrap,
        checks,
        post_oos,
    )

    master_summary.to_csv(
        OUTPUT_DIR / "orb_exact_oos_robustness_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------

    plot_baseline_equity(exact_oos)

    plot_trade_distribution(exact_oos)

    plot_expectancy_vs_slippage(stress_df)

    plot_pf_vs_slippage(stress_df)

    plot_stress_curves(stressed_datasets)

    plot_heatmap(grid)

    plot_permutation_dd(permutation)

    plot_bootstrap_final(bootstrap)

    plot_bootstrap_dd(bootstrap)

    plot_bootstrap_stress(stress_bootstrap)

    plot_degradation(degradation)

    plot_yearly_stress(yearly)

    # -----------------------------------------------------------------
    # MASTER OUTPUT
    # -----------------------------------------------------------------

    print_header("EXACT OOS ROBUSTNESS COMPLETE")

    print(f"OOS trades                         : {baseline['trades']:,}")

    print(f"Baseline total R                   : {baseline['total_R']:.4f}")

    print(f"Baseline expectancy                : {baseline['expectancy_R']:.6f}R")

    print(f"Baseline PF                        : {baseline['profit_factor']:.4f}")

    print(f"Baseline max DD                    : {baseline['max_dd_R']:.4f}R")

    print(f"Topstep RT commission              : ${TOPSTEP_RT_COMMISSION_USD:.2f}")

    print(
        f"Topstep expectancy, 0 slip         : {stress_df.iloc[0]['expectancy_R']:.6f}R"
    )

    row_2 = stress_df.loc[stress_df["ticks_per_side"] == 2].iloc[0]

    print(f"Expectancy, 2 ticks/side           : {row_2['expectancy_R']:.6f}R")

    row_5 = stress_df.loc[stress_df["ticks_per_side"] == 5].iloc[0]

    print(f"Expectancy, 5 ticks/side           : {row_5['expectancy_R']:.6f}R")

    row_10 = stress_df.loc[stress_df["ticks_per_side"] == 10].iloc[0]

    print(f"Expectancy, 10 ticks/side          : {row_10['expectancy_R']:.6f}R")

    print(
        f"Break-even tested slippage         : "
        f"{break_even['break_even_ticks_per_side']} "
        f"ticks/side"
    )

    print(
        f"Bootstrap P(final < 0)             : "
        f"{np.mean(bootstrap['final_values'] < 0):.6f}"
    )

    print(
        f"Bootstrap DD P5                    : "
        f"{np.percentile(bootstrap['dd_values'], 5):.4f}R"
    )

    print(
        f"Bootstrap DD P50                   : "
        f"{np.percentile(bootstrap['dd_values'], 50):.4f}R"
    )

    print(
        f"Bootstrap DD P95                   : "
        f"{np.percentile(bootstrap['dd_values'], 95):.4f}R"
    )

    print(f"Post-OOS holdout trades            : {len(post_oos):,}")

    print(f"All checks pass                    : {checks['pass'].all()}")

    print("\nCSV output:")
    print(OUTPUT_DIR)

    print("\nPNG output:")
    print(PNG_DIR)

    print("\nGenerated PNGs:")

    for path in sorted(PNG_DIR.glob("*.png")):
        print(f"  - {path.name}")

    print_header("NEXT STEP")

    if checks["pass"].all():
        print("EXACT OOS ROBUSTNESS PASSED.")

        print(
            "The ORB can now proceed to the "
            "funded simulation without using "
            "the post-OOS holdout for optimization."
        )

    else:
        print("ONE OR MORE ROBUSTNESS CHECKS FAILED.")

        print("Review the CSV outputs before proceeding to funded simulation.")


if __name__ == "__main__":
    main()
