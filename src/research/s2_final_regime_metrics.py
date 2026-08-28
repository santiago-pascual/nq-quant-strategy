from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2"

INPUT_FILE = RESULTS_DIR / "s2_selective_execution_B_trades.csv"

OUTPUT_FILE = RESULTS_DIR / "s2_final_regime_metrics.csv"

OUTPUT_BY_REGIME_FILE = RESULTS_DIR / "s2_final_regime_metrics_by_regime.csv"


# ============================================================
# FROZEN S2
# ============================================================

STRATEGY_NAME = "S2 B-SELECTIVE"

QUALITY_THRESHOLD = 0.75
RR = 1.30
TAIL_PERCENT = 17.5
STOP_POINTS = 20.0
HORIZON = 15


# ============================================================
# FROZEN ROUTING DECISIONS
# ============================================================
#
# These decisions were already produced by the existing
# walk-forward S2 regime router.
#
# DO NOT derive these from OOS performance here.
#
# Window 2:
#   60-80  -> B
#   80-100 -> B
#
# Window 4:
#   0-20   -> B
#
# ============================================================

VALID_S2_WINDOW_REGIMES = {
    2: {"60-80", "80-100"},
    4: {"0-20"},
}


# ============================================================
# HELPERS
# ============================================================


def profit_factor(r_values: pd.Series) -> float:
    """
    Calculate profit factor:

        gross profits / gross losses

    Losses are converted to positive magnitude.
    """

    profits = r_values[r_values > 0].sum()

    losses = -r_values[r_values < 0].sum()

    if losses == 0:
        if profits > 0:
            return np.inf
        return np.nan

    return float(profits / losses)


def max_drawdown(r_values: pd.Series) -> float:
    """
    Chronological maximum drawdown in R.
    """

    if r_values.empty:
        return np.nan

    equity = r_values.cumsum()

    running_peak = equity.cummax()

    drawdown = equity - running_peak

    return float(drawdown.min())


def longest_losing_streak(r_values: pd.Series) -> int:
    """
    Longest consecutive sequence of losing trades.
    """

    longest = 0
    current = 0

    for value in r_values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """
    Calculate descriptive performance metrics for a
    chronological set of S2 trades.
    """

    if trades.empty:
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
            "trading_days": 0,
            "profitable_days": 0,
            "losing_days": 0,
            "mean_daily_R": np.nan,
            "median_daily_R": np.nan,
            "best_day_R": np.nan,
            "worst_day_R": np.nan,
            "trades_per_day": np.nan,
            "mean_monthly_R": np.nan,
            "median_monthly_R": np.nan,
            "best_month_R": np.nan,
            "worst_month_R": np.nan,
        }

    x = trades.copy()

    x = x.sort_values("entry_timestamp").reset_index(drop=True)

    r = x["net_R"].astype(float)

    # --------------------------------------------------------
    # TRADE METRICS
    # --------------------------------------------------------

    wins = int((r > 0).sum())

    losses = int((r < 0).sum())

    trades_count = len(r)

    win_rate = wins / trades_count

    mean_R = float(r.mean())

    median_R = float(r.median())

    total_R = float(r.sum())

    pf = profit_factor(r)

    dd = max_drawdown(r)

    streak = longest_losing_streak(r)

    # --------------------------------------------------------
    # DAILY METRICS
    # --------------------------------------------------------

    x["date"] = x["entry_timestamp"].dt.date

    daily = x.groupby("date")["net_R"].sum()

    trading_days = len(daily)

    profitable_days = int((daily > 0).sum())

    losing_days = int((daily < 0).sum())

    mean_daily_R = float(daily.mean())

    median_daily_R = float(daily.median())

    best_day_R = float(daily.max())

    worst_day_R = float(daily.min())

    trades_per_day = trades_count / trading_days

    # --------------------------------------------------------
    # MONTHLY METRICS
    # --------------------------------------------------------

    x["month"] = x["entry_timestamp"].dt.to_period("M")

    monthly = x.groupby("month")["net_R"].sum()

    mean_monthly_R = float(monthly.mean())

    median_monthly_R = float(monthly.median())

    best_month_R = float(monthly.max())

    worst_month_R = float(monthly.min())

    return {
        "trades": trades_count,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "mean_R": mean_R,
        "median_R": median_R,
        "total_R": total_R,
        "profit_factor": pf,
        "max_drawdown_R": dd,
        "longest_losing_streak": streak,
        "trading_days": trading_days,
        "profitable_days": profitable_days,
        "losing_days": losing_days,
        "mean_daily_R": mean_daily_R,
        "median_daily_R": median_daily_R,
        "best_day_R": best_day_R,
        "worst_day_R": worst_day_R,
        "trades_per_day": trades_per_day,
        "mean_monthly_R": mean_monthly_R,
        "median_monthly_R": median_monthly_R,
        "best_month_R": best_month_R,
        "worst_month_R": worst_month_R,
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 110)
    print("S2 FINAL REGIME METRICS")
    print("=" * 110)

    print()
    print("FROZEN S2")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Quality >= {QUALITY_THRESHOLD}")
    print(f"RR = {RR}")
    print(f"{TAIL_PERCENT}% lower-tail")
    print(f"{STOP_POINTS:.0f}-point stop")
    print(f"{HORIZON}-bar horizon")

    print()
    print("FROZEN ROUTING DECISIONS:")

    for window, regimes in VALID_S2_WINDOW_REGIMES.items():
        for regime in sorted(regimes):
            print(f"Window {window}: {regime} -> B")

    print()
    print("Input:")
    print(INPUT_FILE)

    # ========================================================
    # LOAD
    # ========================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "entry_timestamp",
        "net_R",
        "vol_bucket",
        "window",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ========================================================
    # TIMESTAMP
    # ========================================================

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
    )

    df["net_R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["window"] = pd.to_numeric(
        df["window"],
        errors="coerce",
    ).astype("Int64")

    df = df.dropna(
        subset=[
            "entry_timestamp",
            "net_R",
            "window",
            "vol_bucket",
        ]
    )

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    # ========================================================
    # FILTER TO FROZEN S2 ROUTING
    # ========================================================

    masks = []

    for window, regimes in VALID_S2_WINDOW_REGIMES.items():
        mask = (df["window"] == window) & (df["vol_bucket"].isin(regimes))

        masks.append(mask)

    valid_mask = masks[0]

    for mask in masks[1:]:
        valid_mask = valid_mask | mask

    selected = df.loc[valid_mask].copy()

    selected = selected.sort_values("entry_timestamp").reset_index(drop=True)

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if selected.empty:
        raise ValueError("No trades matched the frozen S2 routing decisions.")

    # Verify every selected trade belongs to an allowed
    # window/regime combination.

    for _, row in selected.iterrows():
        window = int(row["window"])

        regime = row["vol_bucket"]

        if (
            window not in VALID_S2_WINDOW_REGIMES
            or regime not in VALID_S2_WINDOW_REGIMES[window]
        ):
            raise RuntimeError("Invalid trade passed the frozen routing filter.")

    # ========================================================
    # PRINT TRADE COVERAGE
    # ========================================================

    print()
    print("=" * 110)
    print("FILTERED OOS SAMPLE")
    print("=" * 110)

    print(f"Original B trades: {len(df)}")
    print(f"Trades in frozen S2 scope: {len(selected)}")

    print()
    print("Selected window/regime combinations:")

    coverage = (
        selected.groupby(["window", "vol_bucket"]).size().reset_index(name="trades")
    )

    print(coverage.to_string(index=False))

    # ========================================================
    # REGIME-LEVEL RESULTS
    # ========================================================

    regime_rows = []

    for (
        window,
        regime,
    ), group in selected.groupby(
        ["window", "vol_bucket"],
        sort=True,
    ):
        metrics = calculate_metrics(group)

        regime_rows.append(
            {
                "scope": "S2_REGIME",
                "window": int(window),
                "regime": regime,
                **metrics,
            }
        )

    by_regime_df = pd.DataFrame(regime_rows)

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    combined_metrics = calculate_metrics(selected)

    combined_row = {
        "scope": "S2_VALID_REGIMES",
        "window": "COMBINED",
        "regime": "COMBINED",
        **combined_metrics,
    }

    final_df = pd.DataFrame([combined_row])

    # ========================================================
    # PRINT REGIME RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("REGIME-LEVEL OOS RESULTS")
    print("=" * 110)

    display_columns = [
        "window",
        "regime",
        "trades",
        "win_rate",
        "mean_R",
        "median_R",
        "total_R",
        "profit_factor",
        "max_drawdown_R",
        "longest_losing_streak",
    ]

    print(by_regime_df[display_columns].to_string(index=False))

    # ========================================================
    # PRINT COMBINED RESULTS
    # ========================================================

    print()
    print("=" * 110)
    print("COMBINED VALID S2 REGIMES")
    print("=" * 110)

    for key, value in combined_metrics.items():
        if isinstance(value, float):
            if np.isfinite(value):
                print(f"{key:30s}: {value:.6f}")
            else:
                print(f"{key:30s}: {value}")

        else:
            print(f"{key:30s}: {value}")

    # ========================================================
    # SAVE
    # ========================================================

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    by_regime_df.to_csv(
        OUTPUT_BY_REGIME_FILE,
        index=False,
    )

    print()
    print("=" * 110)
    print("S2 FINAL REGIME METRICS COMPLETE")
    print("=" * 110)

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print(OUTPUT_BY_REGIME_FILE)

    print()
    print(
        "This report is descriptive OOS performance "
        "for the already-established S2 routing decisions."
    )

    print(
        "No regime discovery, parameter optimization, "
        "bootstrap, Monte Carlo, or funded-account simulation "
        "was performed."
    )


if __name__ == "__main__":
    main()
