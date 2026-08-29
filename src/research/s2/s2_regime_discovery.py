from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "s2_extended"

A_TRADES_FILE = RESULTS_DIR / "s2_selective_execution_A_trades.csv"
B_TRADES_FILE = RESULTS_DIR / "s2_selective_execution_B_trades.csv"

OUTPUT_BY_WINDOW = RESULTS_DIR / "s2_regime_discovery_by_window.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "s2_regime_discovery_summary.csv"


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(group: pd.DataFrame) -> dict:

    if group.empty:
        return {
            "trades": 0,
            "WR": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "PF": np.nan,
            "max_DD_R": np.nan,
            "worst_streak": 0,
        }

    pnl = group["net_R"].astype(float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = np.inf if gross_profit > 0 else np.nan

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    max_dd = drawdown.min()

    worst_streak = 0
    current_streak = 0

    for value in pnl:
        if value < 0:
            current_streak += 1
            worst_streak = max(
                worst_streak,
                current_streak,
            )
        else:
            current_streak = 0

    return {
        "trades": len(pnl),
        "WR": (pnl > 0).mean(),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "PF": pf,
        "max_DD_R": max_dd,
        "worst_streak": worst_streak,
    }


# ============================================================
# LOAD
# ============================================================


def load_trades(path: Path, candidate: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
    )

    df["window"] = df["window"].astype(int)

    df["vol_bucket"] = df["vol_bucket"].astype(str)

    df["net_R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df = df.dropna(subset=["net_R"])

    df["candidate"] = candidate

    return df


# ============================================================
# REGIME × WINDOW
# ============================================================


def analyze_by_window(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    grouped = trades.groupby(
        ["candidate", "window", "vol_bucket"],
        sort=True,
    )

    for (
        candidate,
        window,
        vol_bucket,
    ), group in grouped:
        metrics = calculate_metrics(group)

        records.append(
            {
                "candidate": candidate,
                "window": window,
                "vol_bucket": vol_bucket,
                **metrics,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# SUMMARY BY REGIME
# ============================================================


def analyze_regime_summary(
    by_window: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for (
        candidate,
        vol_bucket,
    ), group in by_window.groupby(
        ["candidate", "vol_bucket"],
        sort=True,
    ):
        active = group[group["trades"] > 0]

        if active.empty:
            continue

        positive_windows = (active["total_R"] > 0).sum()

        records.append(
            {
                "candidate": candidate,
                "vol_bucket": vol_bucket,
                "windows_with_trades": len(active),
                "positive_windows": positive_windows,
                "positive_window_pct": (positive_windows / len(active)),
                "total_trades": active["trades"].sum(),
                "mean_window_R": active["mean_R"].mean(),
                "median_window_R": active["mean_R"].median(),
                "mean_total_R": active["total_R"].mean(),
                "median_total_R": active["total_R"].median(),
                "median_PF": active["PF"].median(),
                "best_window_R": active["total_R"].max(),
                "worst_window_R": active["total_R"].min(),
                "mean_max_DD_R": active["max_DD_R"].mean(),
                "worst_max_DD_R": active["max_DD_R"].min(),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 REGIME DISCOVERY — EXPANDED DATASET")
    print("=" * 110)

    print()
    print("Results directory:")
    print(RESULTS_DIR)

    print()
    print("Loading A...")
    a = load_trades(
        A_TRADES_FILE,
        "A",
    )

    print(f"A trades: {len(a):,}")

    print()
    print("Loading B...")
    b = load_trades(
        B_TRADES_FILE,
        "B",
    )

    print(f"B trades: {len(b):,}")

    trades = pd.concat(
        [a, b],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # BY WINDOW
    # --------------------------------------------------------

    by_window = analyze_by_window(trades)

    by_window = by_window.sort_values(
        [
            "candidate",
            "window",
            "vol_bucket",
        ]
    )

    by_window.to_csv(
        OUTPUT_BY_WINDOW,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = analyze_regime_summary(by_window)

    summary = summary.sort_values(
        [
            "candidate",
            "vol_bucket",
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("REGIME SUMMARY")
    print("=" * 110)

    print(summary.to_string(index=False))

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(OUTPUT_BY_WINDOW)
    print(OUTPUT_SUMMARY)

    print()
    print("S2 REGIME DISCOVERY COMPLETE")


if __name__ == "__main__":
    main()
