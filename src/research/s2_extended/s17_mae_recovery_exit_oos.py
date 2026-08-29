"""
S17 MAE -> RECOVERY EXIT OOS TEST

Purpose
-------
Convert the discovered MAE failure phenomenon into a conditional
recovery-management rule.

Hypothesis
----------
Once a trade reaches a sufficiently adverse MAE level, its probability
of eventual failure becomes very high. However, some of these trades
subsequently recover.

We therefore test:

    MAE >= X
        |
        +--> RECOVERY to Y -> exit at Y
        |
        +--> NO RECOVERY -> keep original benchmark outcome

The rule is selected on development windows only and then frozen
for temporal OOS validation.

IMPORTANT
---------
This is a research test. It does NOT modify the frozen benchmark.
No future information is used to trigger the decision.

"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

# Frozen benchmark
STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

# MAE boundary candidates
MAE_THRESHOLDS = [
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

# Recovery levels.
# Negative = partial recovery.
# 0.00 = break-even.
# Positive = recovery into profit.
RECOVERY_LEVELS = [
    -0.20,
    0.00,
    0.25,
    0.50,
]

# How long after the MAE trigger we allow recovery.
RECOVERY_HORIZONS = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
]

# Development / holdout split
DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))

# Minimum number of triggered trades required for a candidate to be
# considered during development.
MIN_TRIGGERED = 15


# =============================================================================
# UTILITIES
# =============================================================================


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise RuntimeError(f"Missing required column: {column}")
    return pd.to_numeric(df[column], errors="coerce")


def normalise_window(value):
    """
    Converts common window formats into integer window IDs.

    Examples:
        1       -> 1
        "1"     -> 1
        "W01"   -> 1
        "window_12" -> 12
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return int(value)
        return np.nan

    text = str(value).strip()

    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))

    return np.nan


def detect_path_columns(df: pd.DataFrame, prefix: str):
    """
    Detect columns such as:

        close_1R
        close_2R
        ...
        mae_1R
        mae_2R
        ...

    Returns sorted list of (bar_number, column_name).
    """
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)R$",
        re.IGNORECASE,
    )

    found = []

    for col in df.columns:
        match = pattern.match(str(col))
        if match:
            bar = int(match.group(1))
            found.append((bar, col))

    found.sort(key=lambda x: x[0])

    return found


def get_path_value(row, path_columns, bar):
    """
    Get a path value at a specific bar.
    Returns NaN if that bar is unavailable.
    """
    for path_bar, column in path_columns:
        if path_bar == bar:
            return safe_float(row[column])
    return np.nan


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================


def profit_factor(returns):
    returns = pd.Series(returns, dtype=float)

    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return gross_profit / gross_loss


def max_drawdown(returns):
    returns = pd.Series(returns, dtype=float).fillna(0.0)

    equity = returns.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_metrics(
    returns,
    windows=None,
):
    returns = pd.Series(returns, dtype=float)

    trades = len(returns)

    if trades == 0:
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

    wins = int((returns > 0).sum())
    losses = int((returns <= 0).sum())

    result = {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades,
        "mean_R": float(returns.mean()),
        "total_R": float(returns.sum()),
        "profit_factor": profit_factor(returns),
        "max_drawdown_R": max_drawdown(returns),
        "positive_window_pct": np.nan,
        "worst_window_R": np.nan,
        "best_window_R": np.nan,
    }

    if windows is not None:
        temp = pd.DataFrame(
            {
                "window": windows,
                "R": returns.values,
            }
        )

        window_results = temp.groupby("window")["R"].sum()

        if len(window_results) > 0:
            result["positive_window_pct"] = float((window_results > 0).mean())
            result["worst_window_R"] = float(window_results.min())
            result["best_window_R"] = float(window_results.max())

    return result


# =============================================================================
# DATA PREPARATION
# =============================================================================


def load_data():
    print("=" * 110)
    print("S17 MAE -> RECOVERY EXIT OOS TEST")
    print("=" * 110)
    print()

    print("Frozen benchmark:")
    print(f"  Stop       = {STOP_POINTS:.1f} points")
    print(f"  RR         = {RR:.2f}")
    print(f"  Horizon    = {HORIZON} bars")
    print()

    print("MAE boundaries:")
    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))
    print()

    print("Recovery levels:")
    print("  " + ", ".join(f"{x:+.2f}R" for x in RECOVERY_LEVELS))
    print()

    print("Recovery horizons:")
    print(f"  {RECOVERY_HORIZONS}")
    print()

    print("Development windows:")
    print(f"  {DEVELOPMENT_WINDOWS}")
    print()

    print("Holdout windows:")
    print(f"  {HOLDOUT_WINDOWS}")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found:\n{INPUT_FILE}")

    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")
    print()

    required = [
        "net_R",
        "final_close_R",
        "window",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    close_columns = detect_path_columns(df, "close")
    mae_columns = detect_path_columns(df, "mae")

    if not close_columns:
        raise RuntimeError(
            "No close path columns found. Expected close_1R, close_2R, ..."
        )

    if not mae_columns:
        raise RuntimeError("No MAE path columns found. Expected mae_1R, mae_2R, ...")

    print("Detected paths:")
    print(f"  Close bars : {len(close_columns)}")
    print(f"  Close range: {close_columns[0][0]} -> {close_columns[-1][0]}")
    print(f"  MAE bars   : {len(mae_columns)}")
    print(f"  MAE range  : {mae_columns[0][0]} -> {mae_columns[-1][0]}")
    print()

    df["_window_numeric"] = df["window"].apply(normalise_window)

    df["final_R_numeric"] = pd.to_numeric(
        df["final_close_R"],
        errors="coerce",
    )

    # If final_close_R is missing for any reason, fall back to net_R.
    fallback = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["final_R_numeric"] = df["final_R_numeric"].fillna(fallback)

    return df, close_columns, mae_columns


# =============================================================================
# MAE TRIGGER DETECTION
# =============================================================================


def find_mae_trigger(
    row,
    mae_columns,
    threshold,
):
    """
    Find the FIRST bar where MAE reaches the threshold.

    This is the causal trigger.

    Returns:
        trigger_bar or None
    """

    for bar, column in mae_columns:
        mae = safe_float(row[column])

        if np.isfinite(mae) and mae >= threshold:
            return bar

    return None


# =============================================================================
# RECOVERY DETECTION
# =============================================================================


def find_recovery_after_trigger(
    row,
    close_columns,
    trigger_bar,
    recovery_level,
    recovery_horizon,
):
    """
    Search for recovery AFTER the MAE trigger.

    The first bar after/equal to the trigger where:

        close_R >= recovery_level

    is considered the recovery event.

    Returns:
        recovery_bar, recovery_R

    or:
        None, None
    """

    last_bar = trigger_bar + recovery_horizon

    for bar, column in close_columns:
        if bar < trigger_bar:
            continue

        if bar > last_bar:
            break

        close_R = safe_float(row[column])

        if not np.isfinite(close_R):
            continue

        if close_R >= recovery_level:
            return bar, recovery_level

    return None, None


# =============================================================================
# BUILD CANDIDATE STRATEGY
# =============================================================================


def apply_recovery_rule(
    df,
    mae_columns,
    close_columns,
    mae_threshold,
    recovery_level,
    recovery_horizon,
):
    """
    Apply one recovery-management rule.

    Logic:

        1. Find MAE trigger.
        2. If no trigger:
               benchmark result unchanged.
        3. If trigger:
               search for recovery.
        4. If recovery:
               exit at recovery level.
        5. If no recovery:
               retain benchmark final result.

    This allows us to measure whether conditional recovery management
    adds value without pretending that every adverse trade must be
    immediately closed.
    """

    strategy_R = []
    trigger_bars = []
    recovery_bars = []
    actions = []

    for _, row in df.iterrows():
        benchmark_R = safe_float(row["final_R_numeric"])

        trigger_bar = find_mae_trigger(
            row,
            mae_columns,
            mae_threshold,
        )

        if trigger_bar is None:
            strategy_R.append(benchmark_R)
            trigger_bars.append(np.nan)
            recovery_bars.append(np.nan)
            actions.append("NO_TRIGGER")
            continue

        recovery_bar, recovery_R = find_recovery_after_trigger(
            row,
            close_columns,
            trigger_bar,
            recovery_level,
            recovery_horizon,
        )

        if recovery_bar is not None:
            strategy_R.append(recovery_R)
            trigger_bars.append(trigger_bar)
            recovery_bars.append(recovery_bar)
            actions.append("RECOVERY_EXIT")
        else:
            strategy_R.append(benchmark_R)
            trigger_bars.append(trigger_bar)
            recovery_bars.append(np.nan)
            actions.append("NO_RECOVERY_BENCHMARK")

    result = df.copy()

    result["strategy_R"] = strategy_R
    result["trigger_bar"] = trigger_bars
    result["recovery_bar"] = recovery_bars
    result["action"] = actions

    return result


# =============================================================================
# IMMEDIATE EXIT REFERENCE
# =============================================================================


def apply_immediate_mae_exit(
    df,
    mae_columns,
    threshold,
):
    """
    Reference strategy:

        MAE >= threshold -> immediately exit at -threshold.

    This is NOT the proposed final strategy.

    It exists only to compare the recovery approach against the
    naive immediate-stop interpretation of the MAE phenomenon.
    """

    strategy_R = []
    triggered = []

    for _, row in df.iterrows():
        benchmark_R = safe_float(row["final_R_numeric"])

        trigger_bar = find_mae_trigger(
            row,
            mae_columns,
            threshold,
        )

        if trigger_bar is None:
            strategy_R.append(benchmark_R)
            triggered.append(False)
        else:
            strategy_R.append(-threshold)
            triggered.append(True)

    result = df.copy()
    result["strategy_R"] = strategy_R
    result["triggered"] = triggered

    return result


# =============================================================================
# DEVELOPMENT SEARCH
# =============================================================================


def development_search(
    df,
    mae_columns,
    close_columns,
):
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)
    print()

    development_df = df[df["_window_numeric"].isin(DEVELOPMENT_WINDOWS)].copy()

    print(f"Development trades: {len(development_df)}")
    print()

    benchmark_metrics = calculate_metrics(
        development_df["final_R_numeric"],
        development_df["_window_numeric"],
    )

    rows = []

    candidates = []

    for mae_threshold in MAE_THRESHOLDS:
        for recovery_level in RECOVERY_LEVELS:
            for recovery_horizon in RECOVERY_HORIZONS:
                candidates.append(
                    (
                        mae_threshold,
                        recovery_level,
                        recovery_horizon,
                    )
                )

    print(f"Testing {len(candidates)} recovery rules...")
    print()

    for index, (
        mae_threshold,
        recovery_level,
        recovery_horizon,
    ) in enumerate(candidates, start=1):
        candidate = apply_recovery_rule(
            development_df,
            mae_columns,
            close_columns,
            mae_threshold,
            recovery_level,
            recovery_horizon,
        )

        triggered = candidate["trigger_bar"].notna()

        triggered_count = int(triggered.sum())

        recovery_exits = int((candidate["action"] == "RECOVERY_EXIT").sum())

        metrics = calculate_metrics(
            candidate["strategy_R"],
            candidate["_window_numeric"],
        )

        delta_R = metrics["total_R"] - benchmark_metrics["total_R"]

        rows.append(
            {
                "mae_threshold": mae_threshold,
                "recovery_level": recovery_level,
                "recovery_horizon": recovery_horizon,
                "triggered_trades": triggered_count,
                "recovery_exits": recovery_exits,
                "trades": metrics["trades"],
                "wins": metrics["wins"],
                "losses": metrics["losses"],
                "win_rate": metrics["win_rate"],
                "mean_R": metrics["mean_R"],
                "total_R": metrics["total_R"],
                "delta_R": delta_R,
                "profit_factor": metrics["profit_factor"],
                "max_drawdown_R": metrics["max_drawdown_R"],
                "positive_window_pct": metrics["positive_window_pct"],
                "worst_window_R": metrics["worst_window_R"],
                "best_window_R": metrics["best_window_R"],
            }
        )

        if index % 50 == 0 or index == len(candidates):
            print(f"Processing {index}/{len(candidates)}...")

    result = pd.DataFrame(rows)

    # Require a meaningful number of triggers.
    eligible = result[result["triggered_trades"] >= MIN_TRIGGERED].copy()

    if eligible.empty:
        raise RuntimeError(
            "No recovery rule has enough triggered trades for development selection."
        )

    # Selection priority:
    # 1. Positive delta R
    # 2. Positive window percentage
    # 3. Mean R
    #
    # We intentionally do NOT optimize solely for PF.
    eligible = eligible.sort_values(
        [
            "delta_R",
            "positive_window_pct",
            "mean_R",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    print()
    print("=" * 110)
    print("TOP RECOVERY RULES — DEVELOPMENT")
    print("=" * 110)

    display_columns = [
        "mae_threshold",
        "recovery_level",
        "recovery_horizon",
        "triggered_trades",
        "recovery_exits",
        "win_rate",
        "mean_R",
        "total_R",
        "delta_R",
        "profit_factor",
        "max_drawdown_R",
        "positive_window_pct",
    ]

    print(eligible[display_columns].head(25).to_string(index=False))

    best = eligible.iloc[0]

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(f"MAE threshold    : {best['mae_threshold']:.2f}R")

    print(f"Recovery level   : {best['recovery_level']:+.2f}R")

    print(f"Recovery horizon : {int(best['recovery_horizon'])} bars")

    print(f"Triggered trades : {int(best['triggered_trades'])}")

    print(f"Recovery exits   : {int(best['recovery_exits'])}")

    print(f"Development WR   : {best['win_rate']:.4f}")

    print(f"Development R    : {best['total_R']:.4f}")

    print(f"Development ΔR   : {best['delta_R']:.4f}")

    print(f"Development PF   : {best['profit_factor']:.4f}")

    print(f"Development DD   : {best['max_drawdown_R']:.4f}")

    return result, best


# =============================================================================
# OOS TEST
# =============================================================================


def oos_test(
    df,
    mae_columns,
    close_columns,
    best_rule,
):
    print()
    print("=" * 110)
    print("HOLDOUT OOS TEST")
    print("=" * 110)
    print()

    mae_threshold = float(best_rule["mae_threshold"])

    recovery_level = float(best_rule["recovery_level"])

    recovery_horizon = int(best_rule["recovery_horizon"])

    holdout_df = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    print(f"Holdout trades: {len(holdout_df)}")
    print()

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    benchmark_metrics = calculate_metrics(
        holdout_df["final_R_numeric"],
        holdout_df["_window_numeric"],
    )

    # ---------------------------------------------------------
    # Recovery strategy
    # ---------------------------------------------------------

    strategy_df = apply_recovery_rule(
        holdout_df,
        mae_columns,
        close_columns,
        mae_threshold,
        recovery_level,
        recovery_horizon,
    )

    strategy_metrics = calculate_metrics(
        strategy_df["strategy_R"],
        strategy_df["_window_numeric"],
    )

    # ---------------------------------------------------------
    # Immediate MAE exit reference
    # ---------------------------------------------------------

    immediate_df = apply_immediate_mae_exit(
        holdout_df,
        mae_columns,
        mae_threshold,
    )

    immediate_metrics = calculate_metrics(
        immediate_df["strategy_R"],
        immediate_df["_window_numeric"],
    )

    print("=" * 110)
    print("FROZEN RULE")
    print("=" * 110)

    print(f"MAE threshold    = {mae_threshold:.2f}R")

    print(f"Recovery level   = {recovery_level:+.2f}R")

    print(f"Recovery horizon = {recovery_horizon} bars")

    print()

    print("=" * 110)
    print("BENCHMARK HOLDOUT")
    print("=" * 110)

    print(f"Trades       : {benchmark_metrics['trades']}")
    print(f"Win rate     : {benchmark_metrics['win_rate']:.4f}")
    print(f"Mean R       : {benchmark_metrics['mean_R']:.4f}")
    print(f"Total R      : {benchmark_metrics['total_R']:.4f}")
    print(f"PF           : {benchmark_metrics['profit_factor']}")
    print(f"Max DD       : {benchmark_metrics['max_drawdown_R']:.4f}")

    print()

    print("=" * 110)
    print("RECOVERY EXIT HOLDOUT")
    print("=" * 110)

    print(f"Trades       : {strategy_metrics['trades']}")
    print(f"Win rate     : {strategy_metrics['win_rate']:.4f}")
    print(f"Mean R       : {strategy_metrics['mean_R']:.4f}")
    print(f"Total R      : {strategy_metrics['total_R']:.4f}")
    print(f"PF           : {strategy_metrics['profit_factor']}")
    print(f"Max DD       : {strategy_metrics['max_drawdown_R']:.4f}")

    recovery_count = int((strategy_df["action"] == "RECOVERY_EXIT").sum())

    triggered_count = int(strategy_df["trigger_bar"].notna().sum())

    print(f"MAE triggers : {triggered_count}")

    print(f"Recovery exits: {recovery_count}")

    print()

    print("=" * 110)
    print("IMMEDIATE MAE EXIT REFERENCE")
    print("=" * 110)

    print(f"Win rate     : {immediate_metrics['win_rate']:.4f}")
    print(f"Mean R       : {immediate_metrics['mean_R']:.4f}")
    print(f"Total R      : {immediate_metrics['total_R']:.4f}")
    print(f"PF           : {immediate_metrics['profit_factor']}")
    print(f"Max DD       : {immediate_metrics['max_drawdown_R']:.4f}")

    # ---------------------------------------------------------
    # Improvement
    # ---------------------------------------------------------

    print()
    print("=" * 110)
    print("OOS IMPROVEMENT VS BENCHMARK")
    print("=" * 110)

    delta_R = strategy_metrics["total_R"] - benchmark_metrics["total_R"]

    delta_mean_R = strategy_metrics["mean_R"] - benchmark_metrics["mean_R"]

    delta_wr = strategy_metrics["win_rate"] - benchmark_metrics["win_rate"]

    delta_dd = strategy_metrics["max_drawdown_R"] - benchmark_metrics["max_drawdown_R"]

    print(f"Delta R          : {delta_R:.4f}")

    print(f"Delta mean R     : {delta_mean_R:.4f}")

    print(f"Delta win rate   : {delta_wr:+.4f}")

    print(f"Delta max DD     : {delta_dd:+.4f}")

    # ---------------------------------------------------------
    # Window-by-window
    # ---------------------------------------------------------

    rows = []

    for window in HOLDOUT_WINDOWS:
        benchmark_window = holdout_df[holdout_df["_window_numeric"] == window]

        strategy_window = strategy_df[strategy_df["_window_numeric"] == window]

        if len(benchmark_window) == 0:
            continue

        b_metrics = calculate_metrics(benchmark_window["final_R_numeric"])

        s_metrics = calculate_metrics(strategy_window["strategy_R"])

        rows.append(
            {
                "window": window,
                "trades": len(benchmark_window),
                "triggered_trades": int(strategy_window["trigger_bar"].notna().sum()),
                "recovery_exits": int(
                    (strategy_window["action"] == "RECOVERY_EXIT").sum()
                ),
                "benchmark_R": b_metrics["total_R"],
                "strategy_R": s_metrics["total_R"],
                "delta_R": (s_metrics["total_R"] - b_metrics["total_R"]),
                "benchmark_WR": b_metrics["win_rate"],
                "strategy_WR": s_metrics["win_rate"],
                "benchmark_PF": b_metrics["profit_factor"],
                "strategy_PF": s_metrics["profit_factor"],
                "benchmark_DD": b_metrics["max_drawdown_R"],
                "strategy_DD": s_metrics["max_drawdown_R"],
            }
        )

    window_df = pd.DataFrame(rows)

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    if not window_df.empty:
        print(window_df.to_string(index=False))

    return (
        strategy_df,
        immediate_df,
        window_df,
    )


# =============================================================================
# RECOVERY DIAGNOSTIC
# =============================================================================


def recovery_diagnostic(
    strategy_df,
):
    """
    Additional diagnostic specifically answering:

        Of the trades that trigger the MAE condition,
        how many actually recover?

    This helps distinguish:
        "MAE is predictive"
    from:
        "MAE can actually be monetized through recovery exits."
    """

    print()
    print("=" * 110)
    print("RECOVERY DIAGNOSTIC")
    print("=" * 110)

    triggered = strategy_df[strategy_df["trigger_bar"].notna()].copy()

    if triggered.empty:
        print("No MAE-triggered trades.")
        return pd.DataFrame()

    total = len(triggered)

    recovery = triggered[triggered["action"] == "RECOVERY_EXIT"]

    no_recovery = triggered[triggered["action"] == "NO_RECOVERY_BENCHMARK"]

    print()
    print(f"Triggered trades       : {total}")

    print(f"Recovered trades       : {len(recovery)}")

    print(f"No recovery            : {len(no_recovery)}")

    print(f"Recovery percentage    : {len(recovery) / total:.4f}")

    if len(recovery) > 0:
        print()
        print("Recovery exits:")

        print(
            recovery[
                [
                    "trigger_bar",
                    "recovery_bar",
                    "strategy_R",
                ]
            ]
            .describe()
            .to_string()
        )

    return triggered


# =============================================================================
# SAVE RESULTS
# =============================================================================


def save_results(
    development_df,
    strategy_df,
    immediate_df,
    window_df,
    best_rule,
    diagnostic_df,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_file = OUTPUT_DIR / "s17_mae_recovery_exit_development.csv"

    oos_file = OUTPUT_DIR / "s17_mae_recovery_exit_oos_trades.csv"

    window_file = OUTPUT_DIR / "s17_mae_recovery_exit_oos_by_window.csv"

    diagnostic_file = OUTPUT_DIR / "s17_mae_recovery_exit_diagnostic.csv"

    summary_file = OUTPUT_DIR / "s17_mae_recovery_exit_summary.csv"

    development_df.to_csv(
        development_file,
        index=False,
    )

    combined = strategy_df.copy()

    combined["immediate_mae_exit_R"] = immediate_df["strategy_R"].values

    combined.to_csv(
        oos_file,
        index=False,
    )

    window_df.to_csv(
        window_file,
        index=False,
    )

    if diagnostic_df is not None:
        diagnostic_df.to_csv(
            diagnostic_file,
            index=False,
        )

    summary = pd.DataFrame(
        [
            {
                "mae_threshold": float(best_rule["mae_threshold"]),
                "recovery_level": float(best_rule["recovery_level"]),
                "recovery_horizon": int(best_rule["recovery_horizon"]),
                "development_triggered_trades": int(best_rule["triggered_trades"]),
                "development_recovery_exits": int(best_rule["recovery_exits"]),
                "development_win_rate": float(best_rule["win_rate"]),
                "development_mean_R": float(best_rule["mean_R"]),
                "development_total_R": float(best_rule["total_R"]),
                "development_delta_R": float(best_rule["delta_R"]),
                "development_profit_factor": float(best_rule["profit_factor"]),
                "development_max_drawdown_R": float(best_rule["max_drawdown_R"]),
            }
        ]
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(development_file)
    print(oos_file)
    print(window_file)
    print(diagnostic_file)
    print(summary_file)


# =============================================================================
# MAIN
# =============================================================================


def main():

    (
        df,
        close_columns,
        mae_columns,
    ) = load_data()

    # -------------------------------------------------------------------------
    # Basic sanity check
    # -------------------------------------------------------------------------

    print("=" * 110)
    print("DATA SANITY CHECK")
    print("=" * 110)

    print(f"Total trades : {len(df)}")

    print(
        f"Development  : {int(df['_window_numeric'].isin(DEVELOPMENT_WINDOWS).sum())}"
    )

    print(f"Holdout      : {int(df['_window_numeric'].isin(HOLDOUT_WINDOWS).sum())}")

    print()

    # -------------------------------------------------------------------------
    # Development search
    # -------------------------------------------------------------------------

    (
        development_results,
        best_rule,
    ) = development_search(
        df,
        mae_columns,
        close_columns,
    )

    # -------------------------------------------------------------------------
    # OOS
    # -------------------------------------------------------------------------

    (
        strategy_df,
        immediate_df,
        window_df,
    ) = oos_test(
        df,
        mae_columns,
        close_columns,
        best_rule,
    )

    # -------------------------------------------------------------------------
    # Diagnostic
    # -------------------------------------------------------------------------

    diagnostic_df = recovery_diagnostic(strategy_df)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        development_results,
        strategy_df,
        immediate_df,
        window_df,
        best_rule,
        diagnostic_df,
    )

    print()
    print("=" * 110)
    print("S17 MAE -> RECOVERY EXIT OOS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
