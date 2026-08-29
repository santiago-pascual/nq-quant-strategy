"""
S23 — RECOVERY EXIT LEVEL TEMPORAL OOS

Objective
---------
Convert the MAE -> recovery relationship discovered in S20/S21/S22
into an executable exit rule.

Core hypothesis
---------------
After a trade reaches a significant adverse excursion (MAE), the future
path contains information about whether the trade is likely to recover.

Instead of asking only:

    "Did the trade recover?"

we now ask:

    "If we decide to exit based on the post-MAE recovery path,
     which recovery level produces the best executable outcome?"

Recovery levels:
    -0.50R ... +0.40R in 0.10R increments.

MAE thresholds:
    0.60R ... 0.90R.

Deadlines:
    3, 5, 8, 10, 12 bars.

Temporal split:
    Development = windows 1..11
    Holdout     = windows 12..22

IMPORTANT
---------
This is the final major optimization of the MAE/recovery branch.

The selected rule is frozen before holdout evaluation.

No holdout data is used for parameter selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

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


# =============================================================================
# FROZEN BENCHMARK
# =============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# =============================================================================
# SEARCH GRID
# =============================================================================

MAE_THRESHOLDS = [
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


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


DEADLINES = [
    3,
    5,
    8,
    10,
    12,
]


DEVELOPMENT_WINDOWS = list(range(1, 12))

HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# HELPERS
# =============================================================================


def normalize_window(value):
    """
    Convert window labels into integers.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

    text = str(value).strip()

    digits = "".join(c for c in text if c.isdigit())

    if digits:
        return int(digits)

    return np.nan


def detect_path_columns(
    df: pd.DataFrame,
) -> Tuple[List[int], List[int]]:
    """
    Detect close_N_R and mae_N_R columns.
    """

    close_bars = []
    mae_bars = []

    for column in df.columns:
        if column.startswith("close_") and column.endswith("R"):
            try:
                bar = int(column.split("_")[1][:-1])
                close_bars.append(bar)
            except ValueError:
                pass

        if column.startswith("mae_") and column.endswith("R"):
            try:
                bar = int(column.split("_")[1][:-1])
                mae_bars.append(bar)
            except ValueError:
                pass

    close_bars = sorted(set(close_bars))
    mae_bars = sorted(set(mae_bars))

    if not close_bars:
        raise RuntimeError("No close_N_R path columns found.")

    if not mae_bars:
        raise RuntimeError("No mae_N_R path columns found.")

    return close_bars, mae_bars


def max_drawdown(
    values: pd.Series,
) -> float:
    """
    Maximum cumulative drawdown.
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


def profit_factor(
    values: pd.Series,
) -> float:
    """
    Gross profit / gross loss.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return 0.0

    gross_profit = values[values > 0].sum()

    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return float(gross_profit / gross_loss)


def metrics(
    values: pd.Series,
) -> Dict[str, float]:
    """
    Standard R metrics.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:
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


# =============================================================================
# MAE CROSSING
# =============================================================================


def find_mae_crossing(
    row: pd.Series,
    mae_bars: List[int],
    threshold: float,
) -> Optional[int]:
    """
    Return first bar where MAE >= threshold.
    """

    for bar in mae_bars:
        value = row.get(
            f"mae_{bar}R",
            np.nan,
        )

        if pd.isna(value):
            continue

        if float(value) >= threshold:
            return bar

    return None


# =============================================================================
# RECOVERY CROSSING
# =============================================================================


def find_recovery_crossing(
    row: pd.Series,
    close_bars: List[int],
    mae_bar: int,
    recovery_level: float,
    deadline: int,
) -> Optional[int]:
    """
    Find first recovery crossing AFTER the MAE event.

    The MAE bar itself is excluded.

    Example:

        MAE at bar 4
        deadline = 5

    Search:

        bars 5..9
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

        if float(value) >= recovery_level:
            return bar

    return None


# =============================================================================
# EXECUTION MODEL
# =============================================================================


def execute_rule(
    row: pd.Series,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> Dict[str, object]:
    """
    Execute the recovery rule on one trade.

    Logic
    -----
    1. Wait until MAE reaches threshold.
    2. Once adverse state is activated:
       - if price recovers to recovery_level before deadline,
         EXIT at that recovery level.
       - if it does not recover before deadline,
         EXIT at the close at the deadline.
    3. If MAE never reaches threshold:
         keep original benchmark result.

    This makes the rule fully executable using the available path data.
    """

    benchmark_R = float(row["_benchmark_R"])

    mae_bar = find_mae_crossing(
        row,
        mae_bars,
        mae_threshold,
    )

    # -------------------------------------------------------------------------
    # No adverse state
    # -------------------------------------------------------------------------

    if mae_bar is None:
        return {
            "strategy_R": benchmark_R,
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "recovery_R": np.nan,
            "exit_type": "BENCHMARK",
        }

    # -------------------------------------------------------------------------
    # Recovery crossing
    # -------------------------------------------------------------------------

    recovery_bar = find_recovery_crossing(
        row,
        close_bars,
        mae_bar,
        recovery_level,
        deadline,
    )

    if recovery_bar is not None:
        recovery_R = float(row[f"close_{recovery_bar}R"])

        return {
            "strategy_R": recovery_R,
            "mae_bar": mae_bar,
            "recovery_bar": recovery_bar,
            "recovery_R": recovery_R,
            "exit_type": "RECOVERY_EXIT",
        }

    # -------------------------------------------------------------------------
    # Did not recover before deadline
    # -------------------------------------------------------------------------

    deadline_bar = min(
        mae_bar + deadline,
        max(close_bars),
    )

    deadline_value = row.get(
        f"close_{deadline_bar}R",
        np.nan,
    )

    if pd.isna(deadline_value):
        strategy_R = benchmark_R

    else:
        strategy_R = float(deadline_value)

    return {
        "strategy_R": strategy_R,
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "recovery_R": np.nan,
        "exit_type": "DEADLINE_EXIT",
    }


# =============================================================================
# BUILD EXECUTION DATASET
# =============================================================================


def build_execution_dataset(
    df: pd.DataFrame,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:
    """
    Apply one executable rule to every trade.
    """

    records = []

    for index, row in df.iterrows():
        result = execute_rule(
            row,
            mae_threshold,
            recovery_level,
            deadline,
            close_bars,
            mae_bars,
        )

        records.append(
            {
                "original_index": index,
                "window": row["_window_numeric"],
                "mae_threshold": mae_threshold,
                "recovery_level": recovery_level,
                "deadline": deadline,
                "benchmark_R": float(row["_benchmark_R"]),
                "strategy_R": float(result["strategy_R"]),
                "mae_bar": result["mae_bar"],
                "recovery_bar": result["recovery_bar"],
                "recovery_R": result["recovery_R"],
                "exit_type": result["exit_type"],
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# RULE METRICS
# =============================================================================


def evaluate_rule(
    trades: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate one executable rule.
    """

    strategy = metrics(trades["strategy_R"])

    benchmark = metrics(trades["benchmark_R"])

    mae_triggered = trades[trades["exit_type"] != "BENCHMARK"]

    recovery_exits = trades[trades["exit_type"] == "RECOVERY_EXIT"]

    deadline_exits = trades[trades["exit_type"] == "DEADLINE_EXIT"]

    return {
        "trades": strategy["trades"],
        "wins": strategy["wins"],
        "losses": strategy["losses"],
        "win_rate": strategy["win_rate"],
        "mean_R": strategy["mean_R"],
        "total_R": strategy["total_R"],
        "profit_factor": strategy["profit_factor"],
        "max_drawdown_R": strategy["max_drawdown_R"],
        "benchmark_total_R": benchmark["total_R"],
        "benchmark_mean_R": benchmark["mean_R"],
        "benchmark_win_rate": benchmark["win_rate"],
        "delta_R": strategy["total_R"] - benchmark["total_R"],
        "delta_mean_R": strategy["mean_R"] - benchmark["mean_R"],
        "delta_win_rate": strategy["win_rate"] - benchmark["win_rate"],
        "delta_max_DD": strategy["max_drawdown_R"] - benchmark["max_drawdown_R"],
        "mae_triggered": len(mae_triggered),
        "recovery_exits": len(recovery_exits),
        "deadline_exits": len(deadline_exits),
        "recovery_exit_pct": (
            len(recovery_exits) / len(mae_triggered)
            if len(mae_triggered) > 0
            else np.nan
        ),
    }


# =============================================================================
# DEVELOPMENT SEARCH
# =============================================================================


def development_search(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
):
    """
    Search executable recovery rules on development only.
    """

    development = df[df["_window_numeric"].isin(DEVELOPMENT_WINDOWS)].copy()

    if len(development) == 0:
        raise RuntimeError("No development trades.")

    records = []

    total_rules = len(MAE_THRESHOLDS) * len(RECOVERY_LEVELS) * len(DEADLINES)

    processed = 0

    print(f"Testing {total_rules} executable recovery rules...")

    for mae_threshold in MAE_THRESHOLDS:
        for recovery_level in RECOVERY_LEVELS:
            for deadline in DEADLINES:
                processed += 1

                if processed == 1 or processed % 50 == 0 or processed == total_rules:
                    print(f"  Processing {processed}/{total_rules}...")

                trades = build_execution_dataset(
                    development,
                    mae_threshold,
                    recovery_level,
                    deadline,
                    close_bars,
                    mae_bars,
                )

                m = evaluate_rule(trades)

                records.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "deadline": deadline,
                        **{f"development_{k}": v for k, v in m.items()},
                    }
                )

    results = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Selection criterion
    #
    # We care primarily about actual executable performance:
    #
    #   1. Positive delta R
    #   2. Positive mean R
    #   3. Lower drawdown
    #
    # A modest penalty is applied to rules that trigger on almost every trade,
    # because they effectively replace the benchmark with an entirely
    # different strategy.
    # -------------------------------------------------------------------------

    results["_score"] = (
        results["development_delta_R"].fillna(-999.0)
        + 0.50 * results["development_delta_mean_R"].fillna(-999.0)
        + 0.10 * results["development_delta_max_DD"].fillna(-999.0)
    )

    # Prefer rules with positive development delta.
    results.loc[results["development_delta_R"] <= 0, "_score"] -= 100.0

    results = results.sort_values(
        by=[
            "_score",
            "development_delta_R",
            "development_mean_R",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return results


# =============================================================================
# HOLDOUT
# =============================================================================


def run_holdout(
    df: pd.DataFrame,
    selected_rule: pd.Series,
    close_bars: List[int],
    mae_bars: List[int],
):
    """
    Execute frozen development rule on holdout.
    """

    holdout = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    if len(holdout) == 0:
        raise RuntimeError("No holdout trades.")

    trades = build_execution_dataset(
        holdout,
        float(selected_rule["mae_threshold"]),
        float(selected_rule["recovery_level"]),
        int(selected_rule["deadline"]),
        close_bars,
        mae_bars,
    )

    summary = evaluate_rule(trades)

    return trades, summary


# =============================================================================
# WINDOW ANALYSIS
# =============================================================================


def window_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare executable strategy vs benchmark window by window.
    """

    records = []

    for window in sorted(trades["window"].dropna().unique()):
        subset = trades[trades["window"] == window]

        strategy = metrics(subset["strategy_R"])

        benchmark = metrics(subset["benchmark_R"])

        records.append(
            {
                "window": int(window),
                "trades": len(subset),
                "benchmark_R": benchmark["total_R"],
                "strategy_R": strategy["total_R"],
                "delta_R": strategy["total_R"] - benchmark["total_R"],
                "benchmark_WR": benchmark["win_rate"],
                "strategy_WR": strategy["win_rate"],
                "benchmark_PF": benchmark["profit_factor"],
                "strategy_PF": strategy["profit_factor"],
                "benchmark_DD": benchmark["max_drawdown_R"],
                "strategy_DD": strategy["max_drawdown_R"],
                "recovery_exits": int((subset["exit_type"] == "RECOVERY_EXIT").sum()),
                "deadline_exits": int((subset["exit_type"] == "DEADLINE_EXIT").sum()),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# LEVEL SENSITIVITY
# =============================================================================


def level_sensitivity(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
    mae_threshold: float,
    deadline: int,
) -> pd.DataFrame:
    """
    Hold MAE threshold and deadline fixed.

    Test every recovery level on holdout.

    This is the critical sensitivity table.

    If the result is only good at one exact level,
    the rule may be overfit.

    If several neighboring levels remain useful,
    the result is much more robust.
    """

    records = []

    for level in RECOVERY_LEVELS:
        trades = build_execution_dataset(
            df,
            mae_threshold,
            level,
            deadline,
            close_bars,
            mae_bars,
        )

        m = evaluate_rule(trades)

        records.append(
            {
                "recovery_level": level,
                **m,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# DEADLINE SENSITIVITY
# =============================================================================


def deadline_sensitivity(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
    mae_threshold: float,
    recovery_level: float,
) -> pd.DataFrame:
    """
    Hold MAE and recovery level fixed.

    Test deadline sensitivity on holdout.
    """

    records = []

    for deadline in DEADLINES:
        trades = build_execution_dataset(
            df,
            mae_threshold,
            recovery_level,
            deadline,
            close_bars,
            mae_bars,
        )

        m = evaluate_rule(trades)

        records.append(
            {
                "deadline": deadline,
                **m,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# MAE SENSITIVITY
# =============================================================================


def mae_sensitivity(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
    recovery_level: float,
    deadline: int,
) -> pd.DataFrame:
    """
    Hold recovery and deadline fixed.

    Test MAE threshold sensitivity on holdout.
    """

    records = []

    for threshold in MAE_THRESHOLDS:
        trades = build_execution_dataset(
            df,
            threshold,
            recovery_level,
            deadline,
            close_bars,
            mae_bars,
        )

        m = evaluate_rule(trades)

        records.append(
            {
                "mae_threshold": threshold,
                **m,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S23 RECOVERY EXIT LEVEL TEMPORAL OOS")
    print("=" * 110)

    print()
    print("Research objective:")
    print("Convert the MAE -> recovery relationship into an executable exit rule.")

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
    print("Deadlines:")
    print(f"  {DEADLINES}")

    print()
    print(f"Development windows: {DEVELOPMENT_WINDOWS}")

    print(f"Holdout windows    : {HOLDOUT_WINDOWS}")

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    if "final_close_R" not in df.columns:
        raise RuntimeError("Missing final_close_R.")

    if "window" not in df.columns:
        raise RuntimeError("Missing window.")

    # =========================================================================
    # PATHS
    # =========================================================================

    close_bars, mae_bars = detect_path_columns(df)

    print()
    print("Detected paths:")

    print(f"  MAE bars   : {len(mae_bars)}")

    print(f"  MAE range  : {min(mae_bars)} -> {max(mae_bars)}")

    print(f"  Close bars : {len(close_bars)}")

    print(f"  Close range: {min(close_bars)} -> {max(close_bars)}")

    # =========================================================================
    # PREP
    # =========================================================================

    df = df.copy()

    df["_window_numeric"] = df["window"].apply(normalize_window)

    df["_benchmark_R"] = pd.to_numeric(
        df["final_close_R"],
        errors="coerce",
    )

    if df["_benchmark_R"].isna().all():
        raise RuntimeError("No valid benchmark R values.")

    # =========================================================================
    # DEVELOPMENT
    # =========================================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    development_results = development_search(
        df,
        close_bars,
        mae_bars,
    )

    development_display = [
        "mae_threshold",
        "recovery_level",
        "deadline",
        "development_trades",
        "development_mae_triggered",
        "development_recovery_exits",
        "development_deadline_exits",
        "development_win_rate",
        "development_mean_R",
        "development_total_R",
        "development_profit_factor",
        "development_max_drawdown_R",
        "development_delta_R",
        "development_delta_mean_R",
        "development_delta_win_rate",
    ]

    print()

    print(development_results[development_display].head(30).to_string(index=False))

    # =========================================================================
    # FREEZE RULE
    # =========================================================================

    selected_rule = development_results.iloc[0]

    selected_mae = float(selected_rule["mae_threshold"])

    selected_recovery = float(selected_rule["recovery_level"])

    selected_deadline = int(selected_rule["deadline"])

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE — FROZEN")
    print("=" * 110)

    print(f"MAE threshold : {selected_mae:.2f}R")

    print(f"Recovery level: {selected_recovery:+.2f}R")

    print(f"Deadline      : {selected_deadline} bars")

    print(f"Development R : {selected_rule['development_total_R']:.4f}")

    print(f"Development ΔR: {selected_rule['development_delta_R']:.4f}")

    print(f"Development WR: {selected_rule['development_win_rate']:.4f}")

    # =========================================================================
    # HOLDOUT
    # =========================================================================

    print()
    print("=" * 110)
    print("HOLDOUT OOS")
    print("=" * 110)

    print()
    print("Frozen rule:")

    print(f"  MAE threshold = {selected_mae:.2f}R")

    print(f"  Recovery      = {selected_recovery:+.2f}R")

    print(f"  Deadline      = {selected_deadline} bars")

    holdout_trades, holdout_summary = run_holdout(
        df,
        selected_rule,
        close_bars,
        mae_bars,
    )

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades   : {holdout_summary['trades']}")

    print(f"  WR       : {holdout_summary['benchmark_win_rate']:.4f}")

    print(f"  Mean R   : {holdout_summary['benchmark_mean_R']:.4f}")

    print(f"  Total R  : {holdout_summary['benchmark_total_R']:.4f}")

    print()
    print("RECOVERY EXIT HOLDOUT")

    print(f"  Trades   : {holdout_summary['trades']}")

    print(f"  WR       : {holdout_summary['win_rate']:.4f}")

    print(f"  Mean R   : {holdout_summary['mean_R']:.4f}")

    print(f"  Total R  : {holdout_summary['total_R']:.4f}")

    print(f"  PF       : {holdout_summary['profit_factor']:.4f}")

    print(f"  Max DD   : {holdout_summary['max_drawdown_R']:.4f}")

    print()
    print("EXECUTION IMPROVEMENT")

    print(f"  Delta R       : {holdout_summary['delta_R']:.4f}")

    print(f"  Delta mean R  : {holdout_summary['delta_mean_R']:.4f}")

    print(f"  Delta WR      : {holdout_summary['delta_win_rate']:.4f}")

    print(f"  Delta Max DD  : {holdout_summary['delta_max_DD']:.4f}")

    print(f"  MAE triggered : {holdout_summary['mae_triggered']}")

    print(f"  Recovery exits: {holdout_summary['recovery_exits']}")

    print(f"  Deadline exits: {holdout_summary['deadline_exits']}")

    # =========================================================================
    # WINDOW ANALYSIS
    # =========================================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW HOLDOUT")
    print("=" * 110)

    window_df = window_analysis(holdout_trades)

    if len(window_df) > 0:
        print(window_df.to_string(index=False))

    # =========================================================================
    # LEVEL SENSITIVITY
    # =========================================================================

    print()
    print("=" * 110)
    print("RECOVERY LEVEL SENSITIVITY — HOLDOUT")
    print("=" * 110)

    holdout_only = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    level_df = level_sensitivity(
        holdout_only,
        close_bars,
        mae_bars,
        selected_mae,
        selected_deadline,
    )

    print(
        level_df[
            [
                "recovery_level",
                "mae_triggered",
                "recovery_exits",
                "deadline_exits",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "delta_R",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # DEADLINE SENSITIVITY
    # =========================================================================

    print()
    print("=" * 110)
    print("DEADLINE SENSITIVITY — HOLDOUT")
    print("=" * 110)

    deadline_df = deadline_sensitivity(
        holdout_only,
        close_bars,
        mae_bars,
        selected_mae,
        selected_recovery,
    )

    print(
        deadline_df[
            [
                "deadline",
                "mae_triggered",
                "recovery_exits",
                "deadline_exits",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "delta_R",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # MAE SENSITIVITY
    # =========================================================================

    print()
    print("=" * 110)
    print("MAE THRESHOLD SENSITIVITY — HOLDOUT")
    print("=" * 110)

    mae_df = mae_sensitivity(
        holdout_only,
        close_bars,
        mae_bars,
        selected_recovery,
        selected_deadline,
    )

    print(
        mae_df[
            [
                "mae_threshold",
                "mae_triggered",
                "recovery_exits",
                "deadline_exits",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "delta_R",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = OUTPUT_DIR / "s23_recovery_exit_development.csv"

    holdout_path = OUTPUT_DIR / "s23_recovery_exit_holdout.csv"

    window_path = OUTPUT_DIR / "s23_recovery_exit_by_window.csv"

    level_path = OUTPUT_DIR / "s23_recovery_exit_level_sensitivity.csv"

    deadline_path = OUTPUT_DIR / "s23_recovery_exit_deadline_sensitivity.csv"

    mae_path = OUTPUT_DIR / "s23_recovery_exit_mae_sensitivity.csv"

    selected_path = OUTPUT_DIR / "s23_selected_recovery_exit_rule.csv"

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

    level_df.to_csv(
        level_path,
        index=False,
    )

    deadline_df.to_csv(
        deadline_path,
        index=False,
    )

    mae_df.to_csv(
        mae_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "mae_threshold": selected_mae,
                "recovery_level": selected_recovery,
                "deadline": selected_deadline,
            }
        ]
    ).to_csv(
        selected_path,
        index=False,
    )

    # =========================================================================
    # FINAL DIAGNOSTIC
    # =========================================================================

    print()
    print("=" * 110)
    print("FINAL DIAGNOSTIC")
    print("=" * 110)

    if holdout_summary["delta_R"] > 0:
        print("RESULT: EXECUTABLE RULE IMPROVED TOTAL R IN HOLDOUT.")

    else:
        print("RESULT: EXECUTABLE RULE DID NOT IMPROVE TOTAL R IN HOLDOUT.")

    if holdout_summary["delta_mean_R"] > 0:
        print("Mean R improved.")

    else:
        print("Mean R did not improve.")

    if holdout_summary["delta_max_DD"] > 0:
        print("Maximum drawdown improved.")

    else:
        print("Maximum drawdown did not improve.")

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(development_path)
    print(holdout_path)
    print(window_path)
    print(level_path)
    print(deadline_path)
    print(mae_path)
    print(selected_path)

    print()
    print("=" * 110)
    print("S23 RECOVERY EXIT LEVEL TEMPORAL OOS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
