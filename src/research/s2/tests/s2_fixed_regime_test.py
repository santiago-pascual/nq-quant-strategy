from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION — FROZEN HYPOTHESIS
# ============================================================

QUALITY_THRESHOLD = 0.75
RR = 1.30
STOP_POINTS = 20
HORIZON_BARS = 15

TARGET_VOL_BUCKET = "40-60"
CANDIDATE = "B"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "s2_extended"

INPUT_FILE = RESULTS_DIR / "s2_selective_execution_B_trades.csv"

OUTPUT_BY_WINDOW = RESULTS_DIR / "s2_fixed_regime_40_60_by_window.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "s2_fixed_regime_40_60_summary.csv"


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(group: pd.DataFrame) -> dict:

    if group.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "median_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "longest_losing_streak": 0,
        }

    pnl = group["net_R"].astype(float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()

    max_drawdown = drawdown.min()

    longest_losing_streak = 0
    current_streak = 0

    for value in pnl:
        if value < 0:
            current_streak += 1
            longest_losing_streak = max(
                longest_losing_streak,
                current_streak,
            )

        else:
            current_streak = 0

    return {
        "trades": len(pnl),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": (pnl > 0).mean(),
        "mean_R": pnl.mean(),
        "median_R": pnl.median(),
        "total_R": pnl.sum(),
        "profit_factor": profit_factor,
        "max_drawdown_R": max_drawdown,
        "longest_losing_streak": longest_losing_streak,
    }


# ============================================================
# LOAD TRADES
# ============================================================


def load_trades() -> pd.DataFrame:

    df = pd.read_csv(INPUT_FILE)

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
    )

    df["exit_timestamp"] = pd.to_datetime(
        df["exit_timestamp"],
        utc=True,
    )

    df["window"] = df["window"].astype(int)

    df["vol_bucket"] = df["vol_bucket"].astype(str)

    df["net_R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["quality"] = pd.to_numeric(
        df["quality"],
        errors="coerce",
    )

    return df.dropna(subset=["net_R"])


# ============================================================
# FILTER FROZEN REGIME
# ============================================================


def filter_frozen_scope(
    df: pd.DataFrame,
) -> pd.DataFrame:

    filtered = df[
        (df["candidate"] == CANDIDATE)
        & (df["quality"] >= QUALITY_THRESHOLD)
        & (df["vol_bucket"] == TARGET_VOL_BUCKET)
    ].copy()

    return filtered


# ============================================================
# WINDOW ANALYSIS
# ============================================================


def analyze_windows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    windows = sorted(df["window"].unique())

    for window in windows:
        group = df[df["window"] == window]

        metrics = calculate_metrics(group)

        records.append(
            {
                "window": window,
                "vol_bucket": TARGET_VOL_BUCKET,
                **metrics,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# OVERALL OOS
# ============================================================


def analyze_overall(
    df: pd.DataFrame,
    by_window: pd.DataFrame,
) -> pd.DataFrame:

    metrics = calculate_metrics(df)

    active_windows = by_window[by_window["trades"] > 0]

    positive_windows = (active_windows["total_R"] > 0).sum()

    negative_windows = (active_windows["total_R"] < 0).sum()

    return pd.DataFrame(
        [
            {
                **metrics,
                "windows_with_trades": len(active_windows),
                "positive_windows": positive_windows,
                "negative_windows": negative_windows,
                "positive_window_pct": (
                    positive_windows / len(active_windows)
                    if len(active_windows) > 0
                    else np.nan
                ),
                "mean_window_R": (
                    active_windows["total_R"].mean()
                    if len(active_windows) > 0
                    else np.nan
                ),
                "median_window_R": (
                    active_windows["total_R"].median()
                    if len(active_windows) > 0
                    else np.nan
                ),
                "best_window_R": (
                    active_windows["total_R"].max()
                    if len(active_windows) > 0
                    else np.nan
                ),
                "worst_window_R": (
                    active_windows["total_R"].min()
                    if len(active_windows) > 0
                    else np.nan
                ),
            }
        ]
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 FIXED REGIME TEST")
    print("=" * 110)

    print()
    print("FROZEN HYPOTHESIS")
    print("-" * 110)

    print(f"Candidate        : {CANDIDATE}")
    print(f"Quality >=       : {QUALITY_THRESHOLD}")
    print(f"RR               : {RR}")
    print(f"Stop             : {STOP_POINTS} points")
    print(f"Horizon          : {HORIZON_BARS} bars")
    print(f"Volatility       : {TARGET_VOL_BUCKET}")

    print()
    print("Input:")
    print(INPUT_FILE)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_trades()

    print()
    print(f"Original B trades: {len(df):,}")

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    filtered = filter_frozen_scope(df)

    print(f"Trades in fixed 40-60 scope: {len(filtered):,}")

    if filtered.empty:
        print()
        print("ERROR: No trades found in frozen scope.")
        return

    # --------------------------------------------------------
    # WINDOW METRICS
    # --------------------------------------------------------

    by_window = analyze_windows(filtered)

    by_window.to_csv(
        OUTPUT_BY_WINDOW,
        index=False,
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    summary = analyze_overall(
        filtered,
        by_window,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT WINDOW RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("OOS RESULTS BY WINDOW")
    print("=" * 110)

    print(by_window.to_string(index=False))

    # --------------------------------------------------------
    # PRINT OVERALL
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("COMBINED OOS RESULTS")
    print("=" * 110)

    print(summary.to_string(index=False))

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(OUTPUT_BY_WINDOW)
    print(OUTPUT_SUMMARY)

    print()
    print("S2 FIXED REGIME TEST COMPLETE")


if __name__ == "__main__":
    main()
