from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

OUTPUT_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"

MAE_THRESHOLD = 0.80

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# ============================================================
# HELPERS
# ============================================================


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return np.nan


def calculate_metrics(df):
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
        }

    pnl = pd.to_numeric(df["strategy_R"], errors="coerce").dropna()

    if pnl.empty:
        return {
            "trades": len(df),
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = -losses.sum()

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = np.inf
    else:
        pf = np.nan

    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min()

    if "window" in df.columns:
        window_R = df.groupby("window")["strategy_R"].sum()

        positive_window_pct = (window_R > 0).mean() if len(window_R) else np.nan

        worst_window_R = window_R.min() if len(window_R) else np.nan

        best_window_R = window_R.max() if len(window_R) else np.nan
    else:
        positive_window_pct = np.nan
        worst_window_R = np.nan
        best_window_R = np.nan

    return {
        "trades": len(pnl),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": (pnl > 0).mean(),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "profit_factor": pf,
        "max_drawdown_R": max_dd,
        "positive_window_pct": positive_window_pct,
        "worst_window_R": worst_window_R,
        "best_window_R": best_window_R,
    }


# ============================================================
# PATH DETECTION
# ============================================================


def detect_columns(df):
    mae_cols = {}

    close_cols = {}

    open_cols = {}

    for column in df.columns:
        name = str(column)

        if name.startswith("mae_") and name.endswith("R"):
            try:
                bar = int(name[4:-1])
                mae_cols[bar] = name
            except ValueError:
                pass

        elif name.startswith("close_") and name.endswith("R"):
            try:
                bar = int(name[6:-1])
                close_cols[bar] = name
            except ValueError:
                pass

        elif name.startswith("open_") and name.endswith("R"):
            try:
                bar = int(name[5:-1])
                open_cols[bar] = name
            except ValueError:
                pass

    return mae_cols, close_cols, open_cols


# ============================================================
# FIND FIRST MAE CROSSING
# ============================================================


def find_first_crossing(row, mae_cols, threshold):
    for bar in sorted(mae_cols):
        value = safe_float(row[mae_cols[bar]])

        if np.isfinite(value) and value >= threshold:
            return bar

    return np.nan


# ============================================================
# BUILD EXECUTION SCENARIOS
# ============================================================


def build_execution_scenarios(df, mae_cols, close_cols, open_cols):
    records = []

    for _, row in df.iterrows():
        crossing_bar = find_first_crossing(
            row,
            mae_cols,
            MAE_THRESHOLD,
        )

        base_R = safe_float(row["net_R"])

        if not np.isfinite(base_R):
            continue

        # ----------------------------------------------------
        # No threshold crossing
        # ----------------------------------------------------

        if pd.isna(crossing_bar):
            records.append(
                {
                    "trade_index": row.get("trade_index", np.nan),
                    "window": row.get("window", np.nan),
                    "entry_timestamp": row.get(
                        "entry_timestamp",
                        np.nan,
                    ),
                    "crossing_bar": np.nan,
                    "triggered": False,
                    "benchmark_R": base_R,
                    "threshold_exit_R": base_R,
                    "same_bar_close_R": base_R,
                    "next_bar_open_R": base_R,
                    "next_bar_close_R": base_R,
                }
            )

            continue

        crossing_bar = int(crossing_bar)

        # ----------------------------------------------------
        # Threshold exit
        #
        # Conservative assumption:
        # exit exactly when MAE reaches 0.80R.
        #
        # For a SHORT:
        # adverse movement = +0.80R.
        # Therefore PnL = -0.80R.
        #
        # This intentionally ignores the close.
        # ----------------------------------------------------

        threshold_exit_R = -MAE_THRESHOLD

        # ----------------------------------------------------
        # Same-bar close
        # ----------------------------------------------------

        if crossing_bar in close_cols:
            close_R = safe_float(row[close_cols[crossing_bar]])
        else:
            close_R = threshold_exit_R

        # ----------------------------------------------------
        # Next-bar open
        # ----------------------------------------------------

        next_bar = crossing_bar + 1

        if next_bar in open_cols:
            next_open_R = safe_float(row[open_cols[next_bar]])
        else:
            # If the next-bar open is unavailable,
            # conservatively use threshold execution.
            next_open_R = threshold_exit_R

        # ----------------------------------------------------
        # Next-bar close
        # ----------------------------------------------------

        if next_bar in close_cols:
            next_close_R = safe_float(row[close_cols[next_bar]])
        else:
            next_close_R = threshold_exit_R

        records.append(
            {
                "trade_index": row.get(
                    "trade_index",
                    np.nan,
                ),
                "window": row.get(
                    "window",
                    np.nan,
                ),
                "entry_timestamp": row.get(
                    "entry_timestamp",
                    np.nan,
                ),
                "crossing_bar": crossing_bar,
                "triggered": True,
                "benchmark_R": base_R,
                "threshold_exit_R": threshold_exit_R,
                "same_bar_close_R": close_R,
                "next_bar_open_R": next_open_R,
                "next_bar_close_R": next_close_R,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# CREATE SCENARIO DATAFRAME
# ============================================================


def scenario_dataframe(df, scenario):
    result = df.copy()

    result["strategy_R"] = pd.to_numeric(
        result[scenario],
        errors="coerce",
    )

    return result


# ============================================================
# WINDOW ANALYSIS
# ============================================================


def window_summary(df, scenario):
    temp = scenario_dataframe(df, scenario)

    rows = []

    for window, group in temp.groupby(
        "window",
        sort=True,
    ):
        metrics = calculate_metrics(group)

        rows.append(
            {
                "window": window,
                "scenario": scenario,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S9 MAE BOUNDARY EXECUTION AUDIT")
    print("=" * 110)

    print()
    print("Frozen hypothesis:")
    print(f"  MAE boundary = {MAE_THRESHOLD:.2f}R")
    print("  Direction    = SHORT")
    print("  Benchmark    = original frozen strategy")
    print()

    print("Execution scenarios:")
    print("  1. THRESHOLD_EXIT")
    print("     Exit exactly at -0.80R")
    print()
    print("  2. SAME_BAR_CLOSE")
    print("     Exit at the close of the bar that first crosses 0.80R")
    print()
    print("  3. NEXT_BAR_OPEN")
    print("     Detect the crossing, then exit at next bar open")
    print()
    print("  4. NEXT_BAR_CLOSE")
    print("     Detect the crossing, then exit at next bar close")
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    # --------------------------------------------------------
    # DETECT PATHS
    # --------------------------------------------------------

    mae_cols, close_cols, open_cols = detect_columns(df)

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_cols)}")
    print(f"  Close bars : {len(close_cols)}")
    print(f"  Open bars  : {len(open_cols)}")

    if not mae_cols:
        raise RuntimeError("No MAE path columns found.")

    if not close_cols:
        raise RuntimeError("No close path columns found.")

    # --------------------------------------------------------
    # BUILD SCENARIOS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("BUILDING EXECUTION SCENARIOS")
    print("=" * 110)

    scenarios = build_execution_scenarios(
        df,
        mae_cols,
        close_cols,
        open_cols,
    )

    triggered = int(scenarios["triggered"].sum())

    print()
    print(f"MAE >= {MAE_THRESHOLD:.2f}R crossings: {triggered}")
    print(f"Trades without crossing       : {len(scenarios) - triggered}")

    if triggered:
        print(
            "Mean crossing bar             : "
            f"{scenarios.loc[scenarios['triggered'], 'crossing_bar'].mean():.2f}"
        )

    # --------------------------------------------------------
    # OVERALL SCENARIOS
    # --------------------------------------------------------

    scenario_names = {
        "benchmark_R": "BENCHMARK",
        "threshold_exit_R": "THRESHOLD_EXIT",
        "same_bar_close_R": "SAME_BAR_CLOSE",
        "next_bar_open_R": "NEXT_BAR_OPEN",
        "next_bar_close_R": "NEXT_BAR_CLOSE",
    }

    summary_rows = []

    print()
    print("=" * 110)
    print("OVERALL EXECUTION COMPARISON")
    print("=" * 110)

    for column, label in scenario_names.items():
        temp = scenarios.copy()

        temp["strategy_R"] = pd.to_numeric(
            temp[column],
            errors="coerce",
        )

        metrics = calculate_metrics(temp)

        metrics["scenario"] = label
        metrics["scenario_column"] = column

        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)

    display_columns = [
        "scenario",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "mean_R",
        "total_R",
        "profit_factor",
        "max_drawdown_R",
        "positive_window_pct",
        "worst_window_R",
        "best_window_R",
    ]

    print(summary[display_columns].to_string(index=False))

    # --------------------------------------------------------
    # HOLDOUT OOS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("TEMPORAL OOS EXECUTION AUDIT")
    print("=" * 110)

    print()
    print(f"Development windows : {DEVELOPMENT_WINDOWS}")

    print(f"Holdout windows     : {HOLDOUT_WINDOWS}")

    # We are NOT selecting a threshold here.
    # 0.80R is frozen from S8.
    #
    # The purpose is purely to compare execution models.

    holdout = scenarios[scenarios["window"].isin(HOLDOUT_WINDOWS)].copy()

    print()
    print(f"Holdout trades: {len(holdout)}")

    oos_rows = []

    for column, label in scenario_names.items():
        temp = holdout.copy()

        temp["strategy_R"] = pd.to_numeric(
            temp[column],
            errors="coerce",
        )

        metrics = calculate_metrics(temp)

        metrics["scenario"] = label

        oos_rows.append(metrics)

    oos_summary = pd.DataFrame(oos_rows)

    print()
    print(oos_summary[display_columns].to_string(index=False))

    # --------------------------------------------------------
    # DELTA VS BENCHMARK
    # --------------------------------------------------------

    benchmark_oos = oos_summary[oos_summary["scenario"] == "BENCHMARK"].iloc[0]

    delta_rows = []

    for _, row in oos_summary.iterrows():
        if row["scenario"] == "BENCHMARK":
            continue

        delta_rows.append(
            {
                "scenario": row["scenario"],
                "delta_R": (row["total_R"] - benchmark_oos["total_R"]),
                "delta_mean_R": (row["mean_R"] - benchmark_oos["mean_R"]),
                "delta_win_rate": (row["win_rate"] - benchmark_oos["win_rate"]),
                "delta_max_DD_R": (
                    row["max_drawdown_R"] - benchmark_oos["max_drawdown_R"]
                ),
            }
        )

    delta_df = pd.DataFrame(delta_rows)

    print()
    print("=" * 110)
    print("OOS DELTA VS BENCHMARK")
    print("=" * 110)

    print(delta_df.to_string(index=False))

    # --------------------------------------------------------
    # WINDOW-BY-WINDOW
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    window_frames = []

    for column, label in scenario_names.items():
        temp = holdout.copy()

        temp["strategy_R"] = pd.to_numeric(
            temp[column],
            errors="coerce",
        )

        ws = (
            temp.groupby("window")
            .agg(
                trades=("strategy_R", "size"),
                total_R=("strategy_R", "sum"),
                win_rate=("strategy_R", lambda x: (x > 0).mean()),
            )
            .reset_index()
        )

        ws["scenario"] = label

        window_frames.append(ws)

    window_long = pd.concat(
        window_frames,
        ignore_index=True,
    )

    window_pivot = window_long.pivot(
        index="window",
        columns="scenario",
        values="total_R",
    ).reset_index()

    print(window_pivot.to_string(index=False))

    # --------------------------------------------------------
    # CROSSING-BAR DISTRIBUTION
    # --------------------------------------------------------

    crossing = scenarios[scenarios["triggered"]].copy()

    if not crossing.empty:
        crossing_summary = (
            crossing["crossing_bar"]
            .value_counts()
            .sort_index()
            .rename_axis("crossing_bar")
            .reset_index(name="trades")
        )

    else:
        crossing_summary = pd.DataFrame(
            columns=[
                "crossing_bar",
                "trades",
            ]
        )

    print()
    print("=" * 110)
    print("MAE CROSSING DISTRIBUTION")
    print("=" * 110)

    print(crossing_summary.to_string(index=False))

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = OUTPUT_DIR / "s9_mae_boundary_execution_summary.csv"

    oos_path = OUTPUT_DIR / "s9_mae_boundary_execution_oos.csv"

    delta_path = OUTPUT_DIR / "s9_mae_boundary_execution_delta.csv"

    window_path = OUTPUT_DIR / "s9_mae_boundary_execution_by_window.csv"

    crossing_path = OUTPUT_DIR / "s9_mae_boundary_crossing_distribution.csv"

    trades_path = OUTPUT_DIR / "s9_mae_boundary_execution_trades.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    oos_summary.to_csv(
        oos_path,
        index=False,
    )

    delta_df.to_csv(
        delta_path,
        index=False,
    )

    window_long.to_csv(
        window_path,
        index=False,
    )

    crossing_summary.to_csv(
        crossing_path,
        index=False,
    )

    scenarios.to_csv(
        trades_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(summary_path)
    print(oos_path)
    print(delta_path)
    print(window_path)
    print(crossing_path)
    print(trades_path)

    print()
    print("=" * 110)
    print("S9 MAE BOUNDARY EXECUTION AUDIT COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
