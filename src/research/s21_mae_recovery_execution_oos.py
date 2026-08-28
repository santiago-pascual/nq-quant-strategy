"""
S21 — MAE + RECOVERY EXECUTION OOS

Purpose
-------
Convert the S20 MAE/recovery evidence into an executable temporal OOS test.

Hypothesis
----------
After a significant adverse excursion (MAE), the important information is
whether price subsequently recovers.

We test:

1. BENCHMARK
   Frozen original strategy.

2. RECOVERY EXIT
   After MAE >= threshold, exit when price recovers to recovery_level.

3. FAILURE EXIT
   After MAE >= threshold, if price does not recover to recovery_level
   within the deadline, exit at the deadline.

The resulting rule is selected ONLY on development windows and then tested
unchanged on holdout windows.

No holdout information is used for rule selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

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


# Frozen benchmark
STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# Temporal split
DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# MAE trigger candidates
MAE_THRESHOLDS = [
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# Recovery levels
#
# Important:
# These are CLOSE_R levels after the MAE trigger.
#
# Example:
#   MAE >= 0.70R
#   recovery level = -0.20R
#
# means:
#   trade first becomes sufficiently adverse,
#   then we wait to see whether it recovers to -0.20R.
#
RECOVERY_LEVELS = [
    -0.50,
    -0.40,
    -0.30,
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
]


# Number of bars allowed for recovery after MAE trigger
RECOVERY_DEADLINES = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
]


# ============================================================================
# COLUMN DETECTION
# ============================================================================


def detect_columns(
    df: pd.DataFrame,
) -> Tuple[List[int], List[int]]:
    """
    Detect:

        close_1R ... close_NR
        mae_1R   ... mae_NR

    Returns
    -------
    close_bars
    mae_bars
    """

    close_bars: List[int] = []
    mae_bars: List[int] = []

    for column in df.columns:
        if column.startswith("close_") and column.endswith("R"):
            try:
                number = int(column.split("_")[1][:-1])
                close_bars.append(number)
            except ValueError:
                pass

        if column.startswith("mae_") and column.endswith("R"):
            try:
                number = int(column.split("_")[1][:-1])
                mae_bars.append(number)
            except ValueError:
                pass

    close_bars = sorted(set(close_bars))
    mae_bars = sorted(set(mae_bars))

    if not close_bars:
        raise RuntimeError(
            "No close path columns found. Expected close_1R, close_2R, ..."
        )

    if not mae_bars:
        raise RuntimeError("No MAE path columns found. Expected mae_1R, mae_2R, ...")

    return close_bars, mae_bars


# ============================================================================
# WINDOW NORMALIZATION
# ============================================================================


def normalize_window(value):
    """
    Normalize window labels.

    Supports:
        1
        1.0
        "1"
        "window_1"
        "Window 1"
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

    text = str(value).strip()

    digits = "".join(character for character in text if character.isdigit())

    if digits:
        return int(digits)

    return np.nan


# ============================================================================
# METRICS
# ============================================================================


def profit_factor(values: pd.Series) -> float:
    """
    Profit Factor = gross profits / gross losses.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    gross_profit = values[values > 0].sum()

    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return float(gross_profit / gross_loss)


def max_drawdown(values: pd.Series) -> float:
    """
    Maximum drawdown in R units.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:
        return 0.0

    equity = values.cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    return float(drawdown.min())


def calculate_metrics(
    df: pd.DataFrame,
    r_column: str,
) -> Dict[str, float]:
    """
    Calculate core strategy metrics.
    """

    if len(df) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }

    values = pd.to_numeric(
        df[r_column],
        errors="coerce",
    ).fillna(0.0)

    wins = int((values > 0).sum())

    losses = int((values <= 0).sum())

    return {
        "trades": int(len(values)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(values)),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
    }


# ============================================================================
# BENCHMARK
# ============================================================================


def build_benchmark(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    The benchmark is frozen.

    We do NOT reconstruct the benchmark.

    We simply use final_close_R from S4.
    """

    out = df.copy()

    if "final_close_R" not in out.columns:
        raise RuntimeError(
            "Missing final_close_R. The S4 enriched dataset is required."
        )

    out["_benchmark_R"] = pd.to_numeric(
        out["final_close_R"],
        errors="coerce",
    )

    if out["_benchmark_R"].isna().all():
        raise RuntimeError("final_close_R contains no valid values.")

    return out


# ============================================================================
# FIRST MAE CROSSING
# ============================================================================


def first_mae_crossing(
    row: pd.Series,
    mae_bars: List[int],
    threshold: float,
) -> int | None:
    """
    Return the first bar where MAE >= threshold.
    """

    for bar in mae_bars:
        value = row.get(
            f"mae_{bar}R",
            np.nan,
        )

        if pd.isna(value):
            continue

        value = float(value)

        if value >= threshold:
            return bar

    return None


# ============================================================================
# FIRST RECOVERY
# ============================================================================


def first_recovery_after_mae(
    row: pd.Series,
    close_bars: List[int],
    mae_bar: int,
    recovery_level: float,
    deadline: int,
) -> int | None:
    """
    Find first CLOSE_R >= recovery_level after the MAE trigger.

    Recovery begins on the bar AFTER the MAE trigger.

    Example:

        MAE trigger = bar 4
        deadline    = 3

    Search:

        bars 5, 6, 7
    """

    start_bar = mae_bar + 1

    end_bar = min(
        mae_bar + deadline,
        max(close_bars),
    )

    for bar in close_bars:
        if bar < start_bar:
            continue

        if bar > end_bar:
            break

        value = row.get(
            f"close_{bar}R",
            np.nan,
        )

        if pd.isna(value):
            continue

        value = float(value)

        if value >= recovery_level:
            return bar

    return None


# ============================================================================
# SINGLE TRADE EXECUTION
# ============================================================================


def evaluate_trade(
    row: pd.Series,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> Dict[str, object]:
    """
    Execute the S21 state machine.

    State machine
    -------------

    INITIAL
       |
       | MAE >= threshold
       v
    ADVERSE
       |
       | recovery >= recovery_level
       | before deadline
       v
    RECOVERED
       |
       | no recovery by deadline
       v
    FAILED_TO_RECOVER

    Execution price is CLOSE_R because S20's evidence was based on
    the close path.

    If MAE never triggers, the original benchmark outcome is preserved.
    """

    mae_bar = first_mae_crossing(
        row=row,
        mae_bars=mae_bars,
        threshold=mae_threshold,
    )

    # ------------------------------------------------------------------
    # No MAE trigger
    # ------------------------------------------------------------------

    if mae_bar is None:
        return {
            "_strategy_R": float(row["_benchmark_R"]),
            "state": "NO_MAE_TRIGGER",
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "exit_bar": np.nan,
            "exit_type": "BENCHMARK",
        }

    # ------------------------------------------------------------------
    # Search for recovery
    # ------------------------------------------------------------------

    recovery_bar = first_recovery_after_mae(
        row=row,
        close_bars=close_bars,
        mae_bar=mae_bar,
        recovery_level=recovery_level,
        deadline=deadline,
    )

    # ------------------------------------------------------------------
    # Recovery occurred
    # ------------------------------------------------------------------

    if recovery_bar is not None:
        recovery_value = row.get(
            f"close_{recovery_bar}R",
            np.nan,
        )

        if not pd.isna(recovery_value):
            return {
                "_strategy_R": float(recovery_value),
                "state": "RECOVERED",
                "mae_bar": mae_bar,
                "recovery_bar": recovery_bar,
                "exit_bar": recovery_bar,
                "exit_type": "RECOVERY_EXIT",
            }

    # ------------------------------------------------------------------
    # Recovery failed
    # ------------------------------------------------------------------

    failure_bar = min(
        mae_bar + deadline,
        max(close_bars),
    )

    failure_value = row.get(
        f"close_{failure_bar}R",
        np.nan,
    )

    # ------------------------------------------------------------------
    # No execution data
    # ------------------------------------------------------------------

    if pd.isna(failure_value):
        return {
            "_strategy_R": float(row["_benchmark_R"]),
            "state": "NO_EXECUTION_DATA",
            "mae_bar": mae_bar,
            "recovery_bar": np.nan,
            "exit_bar": np.nan,
            "exit_type": "BENCHMARK_FALLBACK",
        }

    # ------------------------------------------------------------------
    # Failed to recover
    # ------------------------------------------------------------------

    return {
        "_strategy_R": float(failure_value),
        "state": "FAILED_TO_RECOVER",
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "exit_bar": failure_bar,
        "exit_type": "FAILURE_EXIT",
    }


# ============================================================================
# EVALUATE ONE COMPLETE RULE
# ============================================================================


def evaluate_rule(
    df: pd.DataFrame,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:
    """
    Apply one frozen rule to every trade.
    """

    records = []

    for index, row in df.iterrows():
        result = evaluate_trade(
            row=row,
            mae_threshold=mae_threshold,
            recovery_level=recovery_level,
            deadline=deadline,
            close_bars=close_bars,
            mae_bars=mae_bars,
        )

        record = {
            "original_index": index,
            "mae_threshold": float(mae_threshold),
            "recovery_level": float(recovery_level),
            "deadline": int(deadline),
            "benchmark_R": float(row["_benchmark_R"]),
            "_strategy_R": float(result["_strategy_R"]),
            "state": result["state"],
            "mae_bar": result["mae_bar"],
            "recovery_bar": result["recovery_bar"],
            "exit_bar": result["exit_bar"],
            "exit_type": result["exit_type"],
        }

        if "window" in row.index:
            record["window"] = row["window"]

        records.append(record)

    result_df = pd.DataFrame(records)

    result_df["_window_numeric"] = (
        result_df["window"].apply(normalize_window)
        if "window" in result_df.columns
        else np.nan
    )

    return result_df


# ============================================================================
# DEVELOPMENT SEARCH
# ============================================================================


def search_development_rules(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:
    """
    Search all candidate rules ONLY on development windows.
    """

    development = df[df["_window_numeric"].isin(DEVELOPMENT_WINDOWS)].copy()

    if len(development) == 0:
        raise RuntimeError("No development trades found.")

    benchmark = development[["_benchmark_R"]].copy()

    benchmark = benchmark.rename(columns={"_benchmark_R": "_strategy_R"})

    benchmark_metrics = calculate_metrics(
        benchmark,
        "_strategy_R",
    )

    records = []

    total_rules = len(MAE_THRESHOLDS) * len(RECOVERY_LEVELS) * len(RECOVERY_DEADLINES)

    processed = 0

    print(f"Testing {total_rules} candidate execution rules...")

    for mae_threshold in MAE_THRESHOLDS:
        for recovery_level in RECOVERY_LEVELS:
            for deadline in RECOVERY_DEADLINES:
                processed += 1

                if processed == 1 or processed % 50 == 0 or processed == total_rules:
                    print(f"  Processing {processed}/{total_rules}...")

                trades = evaluate_rule(
                    development,
                    mae_threshold,
                    recovery_level,
                    deadline,
                    close_bars,
                    mae_bars,
                )

                strategy_metrics = calculate_metrics(
                    trades,
                    "_strategy_R",
                )

                delta_R = strategy_metrics["total_R"] - benchmark_metrics["total_R"]

                delta_mean_R = strategy_metrics["mean_R"] - benchmark_metrics["mean_R"]

                delta_wr = strategy_metrics["win_rate"] - benchmark_metrics["win_rate"]

                delta_dd = (
                    strategy_metrics["max_drawdown_R"]
                    - benchmark_metrics["max_drawdown_R"]
                )

                grouped = trades.groupby("_window_numeric")["_strategy_R"].sum()

                positive_window_pct = (
                    float((grouped > 0).mean()) if len(grouped) > 0 else np.nan
                )

                triggered = int((trades["state"] != "NO_MAE_TRIGGER").sum())

                recovered = int((trades["state"] == "RECOVERED").sum())

                failed = int((trades["state"] == "FAILED_TO_RECOVER").sum())

                records.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "deadline": deadline,
                        "development_trades": len(trades),
                        "triggered_trades": triggered,
                        "recovered_trades": recovered,
                        "failed_recovery_trades": failed,
                        "development_win_rate": strategy_metrics["win_rate"],
                        "development_mean_R": strategy_metrics["mean_R"],
                        "development_total_R": strategy_metrics["total_R"],
                        "development_delta_R": delta_R,
                        "development_delta_mean_R": delta_mean_R,
                        "development_delta_win_rate": delta_wr,
                        "development_delta_max_DD_R": delta_dd,
                        "development_profit_factor": strategy_metrics["profit_factor"],
                        "development_max_drawdown_R": strategy_metrics[
                            "max_drawdown_R"
                        ],
                        "development_positive_window_pct": positive_window_pct,
                    }
                )

    results = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT optimize solely for win rate.
    #
    # Primary:
    #   total R improvement
    #
    # Secondary:
    #   positive windows
    #   lower drawdown
    #   PF
    # ------------------------------------------------------------------

    results = results.sort_values(
        by=[
            "development_delta_R",
            "development_positive_window_pct",
            "development_profit_factor",
            "development_max_drawdown_R",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return results


# ============================================================================
# HOLDOUT EVALUATION
# ============================================================================


def evaluate_holdout(
    df: pd.DataFrame,
    selected_rule: pd.Series,
    close_bars: List[int],
    mae_bars: List[int],
) -> Tuple[
    pd.DataFrame,
    Dict[str, float],
    Dict[str, float],
]:
    """
    Evaluate the frozen development-selected rule on holdout windows.
    """

    holdout = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    if len(holdout) == 0:
        raise RuntimeError("No holdout trades found.")

    trades = evaluate_rule(
        holdout,
        float(selected_rule["mae_threshold"]),
        float(selected_rule["recovery_level"]),
        int(selected_rule["deadline"]),
        close_bars,
        mae_bars,
    )

    benchmark = holdout[["_benchmark_R"]].copy()

    benchmark = benchmark.rename(columns={"_benchmark_R": "_strategy_R"})

    benchmark_metrics = calculate_metrics(
        benchmark,
        "_strategy_R",
    )

    strategy_metrics = calculate_metrics(
        trades,
        "_strategy_R",
    )

    return (
        trades,
        benchmark_metrics,
        strategy_metrics,
    )


# ============================================================================
# STATE SUMMARY
# ============================================================================


def build_state_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare:

        NO_MAE_TRIGGER
        RECOVERED
        FAILED_TO_RECOVER
        NO_EXECUTION_DATA
    """

    records = []

    for state, subset in trades.groupby("state"):
        m = calculate_metrics(
            subset,
            "_strategy_R",
        )

        records.append(
            {
                "state": state,
                "trades": m["trades"],
                "wins": m["wins"],
                "losses": m["losses"],
                "win_rate": m["win_rate"],
                "mean_R": m["mean_R"],
                "total_R": m["total_R"],
                "profit_factor": m["profit_factor"],
                "max_drawdown_R": m["max_drawdown_R"],
            }
        )

    return pd.DataFrame(records).sort_values("state").reset_index(drop=True)


# ============================================================================
# WINDOW-BY-WINDOW ANALYSIS
# ============================================================================


def build_window_table(
    trades: pd.DataFrame,
    original_df: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """
    Compare strategy and benchmark for every holdout window.
    """

    records = []

    for window in windows:
        strategy = trades[trades["_window_numeric"] == window].copy()

        benchmark = original_df[original_df["_window_numeric"] == window].copy()

        if len(strategy) == 0:
            continue

        strategy_R = strategy["_strategy_R"]

        benchmark_R = benchmark["_benchmark_R"]

        records.append(
            {
                "window": window,
                "trades": len(strategy),
                "triggered_trades": int((strategy["state"] != "NO_MAE_TRIGGER").sum()),
                "recovered_trades": int((strategy["state"] == "RECOVERED").sum()),
                "failed_recovery_trades": int(
                    (strategy["state"] == "FAILED_TO_RECOVER").sum()
                ),
                "benchmark_R": float(benchmark_R.sum()),
                "strategy_R": float(strategy_R.sum()),
                "delta_R": float(strategy_R.sum() - benchmark_R.sum()),
                "benchmark_WR": float((benchmark_R > 0).mean()),
                "strategy_WR": float((strategy_R > 0).mean()),
                "benchmark_PF": profit_factor(benchmark_R),
                "strategy_PF": profit_factor(strategy_R),
                "benchmark_DD": max_drawdown(benchmark_R),
                "strategy_DD": max_drawdown(strategy_R),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 110)
    print("S21 MAE + RECOVERY EXECUTION OOS")
    print("=" * 110)

    print()
    print("Research objective:")
    print("Convert the S20 MAE/recovery separation into an executable rule.")

    print()
    print("Frozen benchmark:")
    print(f"  Stop       = {STOP_POINTS} points")
    print(f"  RR         = {RR}")
    print(f"  Horizon    = {HORIZON}")

    print()
    print("MAE thresholds:")
    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))

    print()
    print("Recovery levels:")
    print("  " + ", ".join(f"{x:+.2f}R" for x in RECOVERY_LEVELS))

    print()
    print("Recovery deadlines:")
    print(f"  {RECOVERY_DEADLINES}")

    print()
    print(f"Development windows: {DEVELOPMENT_WINDOWS}")

    print(f"Holdout windows    : {HOLDOUT_WINDOWS}")

    # =====================================================================
    # LOAD DATA
    # =====================================================================

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    # =====================================================================
    # DETECT PATHS
    # =====================================================================

    close_bars, mae_bars = detect_columns(df)

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_bars)}")
    print(f"  MAE range  : {min(mae_bars)} -> {max(mae_bars)}")
    print(f"  Close bars : {len(close_bars)}")
    print(f"  Close range: {min(close_bars)} -> {max(close_bars)}")

    # =====================================================================
    # NORMALIZE WINDOWS
    # =====================================================================

    if "window" not in df.columns:
        raise RuntimeError("Missing required column: window")

    # Avoid repeated dataframe inserts.
    # This also avoids pandas fragmentation warnings.
    normalized_window = df["window"].apply(normalize_window)

    df = df.copy()

    df["_window_numeric"] = normalized_window

    if df["_window_numeric"].isna().all():
        raise RuntimeError("Could not normalize any window values.")

    # =====================================================================
    # BUILD FROZEN BENCHMARK
    # =====================================================================

    df = build_benchmark(df)

    # =====================================================================
    # DEVELOPMENT SEARCH
    # =====================================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    development_results = search_development_rules(
        df,
        close_bars,
        mae_bars,
    )

    display_columns = [
        "mae_threshold",
        "recovery_level",
        "deadline",
        "triggered_trades",
        "recovered_trades",
        "failed_recovery_trades",
        "development_win_rate",
        "development_mean_R",
        "development_total_R",
        "development_delta_R",
        "development_profit_factor",
        "development_max_drawdown_R",
        "development_positive_window_pct",
    ]

    print()
    print(development_results[display_columns].head(30).to_string(index=False))

    # =====================================================================
    # SELECT RULE
    # =====================================================================

    selected_rule = development_results.iloc[0]

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(f"MAE threshold     : {selected_rule['mae_threshold']:.2f}R")

    print(f"Recovery level    : {selected_rule['recovery_level']:+.2f}R")

    print(f"Deadline          : {int(selected_rule['deadline'])} bars")

    print(f"Triggered trades  : {int(selected_rule['triggered_trades'])}")

    print(f"Recovered trades  : {int(selected_rule['recovered_trades'])}")

    print(f"Failed recovery   : {int(selected_rule['failed_recovery_trades'])}")

    print(f"Development delta : {selected_rule['development_delta_R']:.4f}R")

    print(f"Development PF    : {selected_rule['development_profit_factor']:.4f}")

    print(f"Development DD    : {selected_rule['development_max_drawdown_R']:.4f}R")

    # =====================================================================
    # HOLDOUT OOS
    # =====================================================================

    print()
    print("=" * 110)
    print("HOLDOUT OOS")
    print("=" * 110)

    print()
    print("Frozen rule:")

    print(f"  MAE threshold  = {selected_rule['mae_threshold']:.2f}R")

    print(f"  Recovery level = {selected_rule['recovery_level']:+.2f}R")

    print(f"  Deadline       = {int(selected_rule['deadline'])} bars")

    (
        holdout_trades,
        benchmark_metrics,
        strategy_metrics,
    ) = evaluate_holdout(
        df,
        selected_rule,
        close_bars,
        mae_bars,
    )

    # =====================================================================
    # BENCHMARK
    # =====================================================================

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades          : {benchmark_metrics['trades']}")

    print(f"  Win rate        : {benchmark_metrics['win_rate']:.4f}")

    print(f"  Mean R          : {benchmark_metrics['mean_R']:.4f}")

    print(f"  Total R         : {benchmark_metrics['total_R']:.4f}")

    print(f"  Profit Factor   : {benchmark_metrics['profit_factor']}")

    print(f"  Max DD          : {benchmark_metrics['max_drawdown_R']:.4f}")

    # =====================================================================
    # STRATEGY
    # =====================================================================

    print()
    print("RECOVERY EXECUTION HOLDOUT")

    print(f"  Trades          : {strategy_metrics['trades']}")

    print(f"  Win rate        : {strategy_metrics['win_rate']:.4f}")

    print(f"  Mean R          : {strategy_metrics['mean_R']:.4f}")

    print(f"  Total R         : {strategy_metrics['total_R']:.4f}")

    print(f"  Profit Factor   : {strategy_metrics['profit_factor']}")

    print(f"  Max DD          : {strategy_metrics['max_drawdown_R']:.4f}")

    # =====================================================================
    # IMPROVEMENT
    # =====================================================================

    delta_R = strategy_metrics["total_R"] - benchmark_metrics["total_R"]

    delta_mean_R = strategy_metrics["mean_R"] - benchmark_metrics["mean_R"]

    delta_wr = strategy_metrics["win_rate"] - benchmark_metrics["win_rate"]

    delta_dd = strategy_metrics["max_drawdown_R"] - benchmark_metrics["max_drawdown_R"]

    print()
    print("IMPROVEMENT")

    print(f"  Delta R         : {delta_R:.4f}")

    print(f"  Delta Mean R    : {delta_mean_R:.4f}")

    print(f"  Delta Win Rate  : {delta_wr:.4f}")

    print(f"  Delta Max DD    : {delta_dd:.4f}")

    # =====================================================================
    # STATE SUMMARY
    # =====================================================================

    print()
    print("=" * 110)
    print("HOLDOUT STATE SUMMARY")
    print("=" * 110)

    state_df = build_state_summary(holdout_trades)

    if len(state_df) > 0:
        print(state_df.to_string(index=False))

    # =====================================================================
    # WINDOW ANALYSIS
    # =====================================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW HOLDOUT")
    print("=" * 110)

    window_df = build_window_table(
        holdout_trades,
        df,
        HOLDOUT_WINDOWS,
    )

    if len(window_df) > 0:
        print(window_df.to_string(index=False))

    # =====================================================================
    # SAVE FILES
    # =====================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = OUTPUT_DIR / "s21_mae_recovery_execution_development.csv"

    holdout_path = OUTPUT_DIR / "s21_mae_recovery_execution_holdout.csv"

    window_path = OUTPUT_DIR / "s21_mae_recovery_execution_by_window.csv"

    state_path = OUTPUT_DIR / "s21_mae_recovery_execution_states.csv"

    selected_path = OUTPUT_DIR / "s21_selected_recovery_execution_rule.csv"

    development_results.to_csv(
        development_path,
        index=False,
    )

    holdout_trades.to_csv(
        holdout_path,
        index=False,
    )

    window_df.to_csv(
        window_path,
        index=False,
    )

    state_df.to_csv(
        state_path,
        index=False,
    )

    pd.DataFrame([selected_rule]).to_csv(
        selected_path,
        index=False,
    )

    # =====================================================================
    # FINAL OUTPUT
    # =====================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(development_path)

    print(holdout_path)

    print(window_path)

    print(state_path)

    print(selected_path)

    print()
    print("=" * 110)
    print("S21 MAE + RECOVERY EXECUTION OOS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
