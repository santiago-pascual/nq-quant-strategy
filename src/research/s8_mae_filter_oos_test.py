"""
S8 — MAE FILTER OOS TEST

Purpose
-------
Test whether exiting a trade when adverse excursion reaches a given
MAE threshold improves the frozen S2 benchmark.

This is a temporal OOS test.

The benchmark remains completely frozen.

For each MAE threshold:

    0.50R
    0.60R
    0.70R
    0.75R
    0.80R
    0.90R
    1.00R
    1.10R

the strategy does:

    - enter exactly like the frozen benchmark
    - monitor the intratrade path
    - if MAE reaches the threshold, exit immediately
    - otherwise retain the original benchmark outcome

The threshold is selected ONLY using development windows.

The selected threshold is then evaluated on untouched holdout windows.

No optimization is performed on holdout data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"

INPUT_FILE = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"

BENCHMARK_FILE = RESULTS_DIR / "s2_benchmark_trades.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "s8_mae_filter_oos_summary.csv"

OUTPUT_TRADES = RESULTS_DIR / "s8_mae_filter_oos_trades.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "s8_mae_filter_oos_by_window.csv"

# ============================================================
# FROZEN CONFIGURATION
# ============================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

TARGET_R = RR

# Candidate filters discovered in S7.
MAE_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
    1.10,
]

# Temporal split.
# Same 22-window structure used throughout S2.
DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# ============================================================
# UTILITIES
# ============================================================


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def find_close_columns(df):
    """
    Find close_1R ... close_20R path columns.
    """

    columns = {}

    for col in df.columns:
        if not col.startswith("close_"):
            continue

        try:
            bar = int(col.split("_")[1].replace("R", ""))
        except (
            ValueError,
            IndexError,
        ):
            continue

        columns[bar] = col

    return dict(sorted(columns.items()))


def find_mae_columns(df):
    """
    Find mae_1R ... mae_20R path columns.
    """

    columns = {}

    for col in df.columns:
        if not col.startswith("mae_"):
            continue

        try:
            bar = int(col.split("_")[1].replace("R", ""))
        except (
            ValueError,
            IndexError,
        ):
            continue

        columns[bar] = col

    return dict(sorted(columns.items()))


# ============================================================
# ENTRY / PATH
# ============================================================


def first_mae_crossing(
    row,
    mae_columns,
    threshold,
):
    """
    Return the first bar at which MAE reaches threshold.

    MAE is positive adverse excursion.

    Example:

        0.75R = price moved 0.75R against the short.
    """

    for bar, col in mae_columns.items():
        value = row[col]

        if pd.isna(value):
            continue

        if float(value) >= threshold:
            return bar

    return None


def close_at_bar(
    row,
    close_columns,
    bar,
):
    """
    Return close_R at a particular path bar.

    For a SHORT:

        positive close_R = adverse
        negative close_R = favorable
    """

    col = close_columns.get(bar)

    if col is None:
        return np.nan

    value = row[col]

    if pd.isna(value):
        return np.nan

    return float(value)


# ============================================================
# FILTERED TRADE SIMULATION
# ============================================================


def simulate_filter(
    row,
    close_columns,
    mae_columns,
    threshold,
):
    """
    Simulate an MAE exit using only information available
    up to the decision point.

    IMPORTANT:

    If MAE threshold is crossed at bar N:

        exit at the CLOSE of bar N.

    This avoids using future information.

    If threshold is never crossed:

        retain the original benchmark result.
    """

    crossing_bar = first_mae_crossing(
        row,
        mae_columns,
        threshold,
    )

    original_R = float(row["net_R"])

    original_reason = str(
        row.get(
            "exit_reason",
            row.get(
                "exit_reason_path",
                "unknown",
            ),
        )
    )

    # --------------------------------------------------------
    # NO FILTER TRIGGER
    # --------------------------------------------------------

    if crossing_bar is None:
        return {
            "strategy_R": original_R,
            "exit_bar": row.get(
                "holding_bars",
                np.nan,
            ),
            "exit_reason": original_reason,
            "filter_triggered": False,
            "mae_crossing_bar": np.nan,
        }

    # --------------------------------------------------------
    # FILTER TRIGGERED
    # --------------------------------------------------------

    close_R = close_at_bar(
        row,
        close_columns,
        crossing_bar,
    )

    if pd.isna(close_R):
        # Safety fallback:
        # never manufacture an exit price.
        return {
            "strategy_R": original_R,
            "exit_bar": row.get(
                "holding_bars",
                np.nan,
            ),
            "exit_reason": original_reason,
            "filter_triggered": False,
            "mae_crossing_bar": crossing_bar,
        }

    # close_R is measured from entry.
    #
    # For a short:
    #
    #   close_R = +0.50 means -0.50R PnL
    #   close_R = -0.50 means +0.50R PnL
    #
    # Therefore PnL R = -close_R.

    strategy_R = -close_R

    return {
        "strategy_R": strategy_R,
        "exit_bar": crossing_bar,
        "exit_reason": "mae_filter_exit",
        "filter_triggered": True,
        "mae_crossing_bar": crossing_bar,
    }


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades,
):
    """
    Calculate standard performance metrics.
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
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
            "longest_losing_streak": 0,
        }

    pnl = numeric(trades["strategy_R"]).dropna()

    if pnl.empty:
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
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
            "longest_losing_streak": 0,
        }

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    if gross_loss > 0:
        PF = gross_profit / gross_loss
    else:
        PF = np.inf

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    max_dd = drawdown.min()

    # --------------------------------------------------------
    # LOSING STREAK
    # --------------------------------------------------------

    longest = 0
    current = 0

    for value in pnl:
        if value < 0:
            current += 1

            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    # --------------------------------------------------------
    # WINDOW METRICS
    # --------------------------------------------------------

    if "window" in trades.columns:
        window_R = trades.groupby("window")["strategy_R"].sum()

        positive_pct = (window_R > 0).mean()

        worst_window = window_R.min()

        best_window = window_R.max()

    else:
        positive_pct = np.nan
        worst_window = np.nan
        best_window = np.nan

    return {
        "trades": int(len(pnl)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "median_R": float(pnl.median()),
        "total_R": float(pnl.sum()),
        "profit_factor": float(PF),
        "max_drawdown_R": float(max_dd),
        "positive_window_pct": float(positive_pct),
        "worst_window_R": float(worst_window),
        "best_window_R": float(best_window),
        "longest_losing_streak": int(longest),
    }


# ============================================================
# TEST ONE THRESHOLD
# ============================================================


def test_threshold(
    df,
    close_columns,
    mae_columns,
    threshold,
):
    """
    Apply one MAE threshold to every trade.
    """

    records = []

    for idx, row in df.iterrows():
        result = simulate_filter(
            row,
            close_columns,
            mae_columns,
            threshold,
        )

        record = row.to_dict()

        record.update(
            {
                "trade_index": idx,
                "mae_threshold": threshold,
                "strategy_R": result["strategy_R"],
                "strategy_exit_bar": result["exit_bar"],
                "strategy_exit_reason": result["exit_reason"],
                "filter_triggered": result["filter_triggered"],
                "mae_crossing_bar": result["mae_crossing_bar"],
            }
        )

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# WINDOW SUMMARY
# ============================================================


def window_summary(
    trades,
    threshold,
):
    """
    Produce window-level results.
    """

    rows = []

    for window, group in trades.groupby(
        "window",
        sort=True,
    ):
        benchmark_R = numeric(group["net_R"]).sum()

        strategy_R = numeric(group["strategy_R"]).sum()

        benchmark_wins = numeric(group["net_R"]) > 0

        strategy_wins = numeric(group["strategy_R"]) > 0

        benchmark_losses = numeric(group["net_R"]) < 0

        strategy_losses = numeric(group["strategy_R"]) < 0

        # Benchmark PF
        bp = numeric(group["net_R"])

        bp_profit = bp[bp > 0].sum()

        bp_loss = -bp[bp < 0].sum()

        benchmark_pf = bp_profit / bp_loss if bp_loss > 0 else np.inf

        # Strategy PF
        sp = numeric(group["strategy_R"])

        sp_profit = sp[sp > 0].sum()

        sp_loss = -sp[sp < 0].sum()

        strategy_pf = sp_profit / sp_loss if sp_loss > 0 else np.inf

        rows.append(
            {
                "window": window,
                "mae_threshold": threshold,
                "trades": len(group),
                "triggered_trades": int(group["filter_triggered"].sum()),
                "benchmark_R": float(benchmark_R),
                "strategy_R": float(strategy_R),
                "delta_R": float(strategy_R - benchmark_R),
                "benchmark_WR": float(benchmark_wins.mean()),
                "strategy_WR": float(strategy_wins.mean()),
                "benchmark_PF": float(benchmark_pf),
                "strategy_PF": float(strategy_pf),
                "benchmark_losses": int(benchmark_losses.sum()),
                "strategy_losses": int(strategy_losses.sum()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S8 MAE FILTER — TEMPORAL OOS TEST")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop            = {STOP_POINTS} points")
    print(f"  RR              = {RR}")
    print(f"  Horizon         = {HORIZON} bars")

    print()
    print("Candidate MAE filters:")
    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))

    print()
    print(f"Development windows : {DEVELOPMENT_WINDOWS}")

    print(f"Holdout windows     : {HOLDOUT_WINDOWS}")

    # ========================================================
    # LOAD
    # ========================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing enriched dataset:\n{INPUT_FILE}")

    print()
    print("Loading enriched benchmark...")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    # ========================================================
    # PATH COLUMNS
    # ========================================================

    close_columns = find_close_columns(df)

    mae_columns = find_mae_columns(df)

    if not close_columns:
        raise RuntimeError("No close_* path columns found.")

    if not mae_columns:
        raise RuntimeError("No mae_* path columns found.")

    print()
    print("Detected paths:")
    print(f"  close bars : {len(close_columns)}")
    print(f"  MAE bars   : {len(mae_columns)}")

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required = [
        "net_R",
        "window",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    # ========================================================
    # TEST ALL THRESHOLDS
    # ========================================================

    all_trades = []
    all_windows = []
    summaries = []

    print()
    print("=" * 110)
    print("RUNNING MAE FILTER TESTS")
    print("=" * 110)

    for threshold in MAE_THRESHOLDS:
        print(f"Testing MAE >= {threshold:.2f}R...")

        tested = test_threshold(
            df,
            close_columns,
            mae_columns,
            threshold,
        )

        windows = window_summary(
            tested,
            threshold,
        )

        metrics = calculate_metrics(tested)

        metrics.update(
            {
                "mae_threshold": threshold,
                "triggered_trades": int(tested["filter_triggered"].sum()),
            }
        )

        summaries.append(metrics)

        all_trades.append(tested)

        all_windows.append(windows)

        print(f"  Triggered: {metrics['triggered_trades']}")

    summary = pd.DataFrame(summaries)

    windows = pd.concat(
        all_windows,
        ignore_index=True,
    )

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    # ========================================================
    # DEVELOPMENT SEARCH
    # ========================================================

    development = summary[summary["mae_threshold"].notna()].copy()

    development_windows = windows[windows["window"].isin(DEVELOPMENT_WINDOWS)]

    development_summary = development_windows.groupby(
        "mae_threshold",
        as_index=False,
    ).agg(
        development_trades=(
            "trades",
            "sum",
        ),
        development_triggered=(
            "triggered_trades",
            "sum",
        ),
        development_benchmark_R=(
            "benchmark_R",
            "sum",
        ),
        development_strategy_R=(
            "strategy_R",
            "sum",
        ),
        development_delta_R=(
            "delta_R",
            "sum",
        ),
        development_mean_delta_R=(
            "delta_R",
            "mean",
        ),
        development_positive_windows=(
            "delta_R",
            lambda x: int((x > 0).sum()),
        ),
        development_windows=(
            "delta_R",
            "size",
        ),
    )

    development_summary["development_positive_window_pct"] = (
        development_summary["development_positive_windows"]
        / development_summary["development_windows"]
    )

    # --------------------------------------------------------
    # SELECT RULE
    # --------------------------------------------------------
    #
    # Primary criterion:
    #   maximum development total delta R.
    #
    # Secondary:
    #   positive-window percentage.
    #
    # This is deliberately simple.
    #

    development_summary = development_summary.sort_values(
        [
            "development_delta_R",
            "development_positive_window_pct",
        ],
        ascending=False,
    ).reset_index(drop=True)

    selected_threshold = float(development_summary.iloc[0]["mae_threshold"])

    # ========================================================
    # HOLDOUT
    # ========================================================

    holdout_windows = windows[windows["window"].isin(HOLDOUT_WINDOWS)]

    selected_holdout = holdout_windows[
        holdout_windows["mae_threshold"] == selected_threshold
    ].copy()

    selected_holdout_trades = trades[
        (trades["mae_threshold"] == selected_threshold)
        & (trades["window"].isin(HOLDOUT_WINDOWS))
    ].copy()

    benchmark_holdout = selected_holdout_trades.copy()

    # ========================================================
    # HOLDOUT METRICS
    # ========================================================

    def benchmark_metrics(
        data,
    ):

        pnl = numeric(data["net_R"]).dropna()

        wins = pnl[pnl > 0]

        losses = pnl[pnl < 0]

        gross_profit = wins.sum()

        gross_loss = -losses.sum()

        PF = gross_profit / gross_loss if gross_loss > 0 else np.inf

        equity = pnl.cumsum()

        dd = equity - equity.cummax()

        return {
            "trades": len(pnl),
            "win_rate": float((pnl > 0).mean()),
            "mean_R": float(pnl.mean()),
            "total_R": float(pnl.sum()),
            "profit_factor": float(PF),
            "max_drawdown_R": float(dd.min()),
        }

    benchmark_oos = benchmark_metrics(benchmark_holdout)

    strategy_oos = calculate_metrics(selected_holdout_trades)

    delta_R = strategy_oos["total_R"] - benchmark_oos["total_R"]

    delta_PF = strategy_oos["profit_factor"] - benchmark_oos["profit_factor"]

    delta_DD = strategy_oos["max_drawdown_R"] - benchmark_oos["max_drawdown_R"]

    # ========================================================
    # PRINT DEVELOPMENT
    # ========================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    print(
        development_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    selected_dev = development_summary.iloc[0]

    print(f"MAE threshold       : {selected_threshold:.2f}R")

    print(f"Development delta R : {selected_dev['development_delta_R']:.4f}")

    print(
        f"Positive windows    : {selected_dev['development_positive_window_pct']:.4f}"
    )

    # ========================================================
    # PRINT OOS
    # ========================================================

    print()
    print("=" * 110)
    print("HOLDOUT OOS RESULT")
    print("=" * 110)

    print()
    print("Frozen rule:")
    print(f"  MAE filter = {selected_threshold:.2f}R")

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades          : {benchmark_oos['trades']}")

    print(f"  Win rate        : {benchmark_oos['win_rate']:.4f}")

    print(f"  Mean R          : {benchmark_oos['mean_R']:.4f}")

    print(f"  Total R         : {benchmark_oos['total_R']:.4f}")

    print(f"  PF              : {benchmark_oos['profit_factor']:.4f}")

    print(f"  Max DD          : {benchmark_oos['max_drawdown_R']:.4f}")

    print()
    print("MAE FILTER HOLDOUT")

    print(f"  Trades          : {strategy_oos['trades']}")

    print(f"  Win rate        : {strategy_oos['win_rate']:.4f}")

    print(f"  Mean R          : {strategy_oos['mean_R']:.4f}")

    print(f"  Total R         : {strategy_oos['total_R']:.4f}")

    print(f"  PF              : {strategy_oos['profit_factor']:.4f}")

    print(f"  Max DD          : {strategy_oos['max_drawdown_R']:.4f}")

    print()
    print("IMPROVEMENT")

    print(f"  Delta R         : {delta_R:.4f}")

    print(f"  Delta PF        : {delta_PF:.4f}")

    print(f"  Delta Max DD    : {delta_DD:.4f}")

    # ========================================================
    # WINDOW-BY-WINDOW
    # ========================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW HOLDOUT")
    print("=" * 110)

    print(
        selected_holdout[
            [
                "window",
                "trades",
                "triggered_trades",
                "benchmark_R",
                "strategy_R",
                "delta_R",
                "benchmark_WR",
                "strategy_WR",
                "benchmark_PF",
                "strategy_PF",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Add OOS metadata to summary.
    summary["selected_by_development"] = summary["mae_threshold"] == selected_threshold

    summary["development_delta_R"] = np.nan

    summary["development_positive_window_pct"] = np.nan

    for _, row in development_summary.iterrows():
        mask = summary["mae_threshold"] == row["mae_threshold"]

        summary.loc[
            mask,
            "development_delta_R",
        ] = row["development_delta_R"]

        summary.loc[
            mask,
            "development_positive_window_pct",
        ] = row["development_positive_window_pct"]

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    trades.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(OUTPUT_SUMMARY)
    print(OUTPUT_TRADES)
    print(OUTPUT_WINDOWS)

    print()
    print("=" * 110)
    print("S8 MAE FILTER OOS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
