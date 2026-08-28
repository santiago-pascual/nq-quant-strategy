from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# S18 MAE RECOVERY GRID — TEMPORAL OOS TEST
# =============================================================================
#
# Research question:
#
#   Once a trade reaches a sufficiently adverse MAE level, can we exploit
#   subsequent recovery by exiting at a smaller loss / breakeven / small gain?
#
# This test deliberately evaluates the FULL recovery range instead of assuming
# +0.50R is the correct recovery level.
#
# MAE trigger:
#   0.60R -> 1.00R
#
# Recovery exit:
#   -0.50R -> +0.40R in 0.10R increments
#
# Recovery horizon:
#   2, 3, 4, 5, 6, 8, 10, 12 bars after MAE trigger
#
# Rule selection:
#   Development windows only
#
# Final evaluation:
#   Frozen rule on holdout windows
#
# IMPORTANT:
#   This is a research test.
#   It does NOT modify the frozen benchmark.
#   It does NOT claim that a rule is tradable until OOS execution confirms it.
# =============================================================================


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


# -----------------------------------------------------------------------------
# Frozen benchmark
# -----------------------------------------------------------------------------

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# -----------------------------------------------------------------------------
# MAE thresholds
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Recovery levels
#
# IMPORTANT:
# This is the key expansion versus the previous S17 test.
#
# We are testing:
#
# -0.50
# -0.40
# -0.30
# -0.20
# -0.10
#  0.00
# +0.10
# +0.20
# +0.30
# +0.40
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Recovery horizons
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Temporal split
# -----------------------------------------------------------------------------

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# -----------------------------------------------------------------------------
# Minimum triggered observations required for a candidate to be considered.
#
# This prevents extremely rare rules from winning simply because of a tiny
# sample.
# -----------------------------------------------------------------------------

MIN_TRIGGERED = 15


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def safe_float(value) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return np.nan


def normalise_window(value):
    """
    Convert window labels into numeric values.

    Supports:
        1
        "1"
        "window_1"
        "Window 1"
        "W1"
        etc.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return int(value)

    text = str(value).strip()

    match = re.search(r"(\d+)", text)

    if match:
        return int(match.group(1))

    return np.nan


def first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    for col in candidates:
        if col in df.columns:
            return col

    return None


# =============================================================================
# PATH COLUMN DETECTION
# =============================================================================


def detect_numbered_columns(
    df: pd.DataFrame,
    prefix: str,
) -> Dict[int, str]:
    """
    Detect columns such as:

        close_1R
        close_2R
        ...
        close_20R

    Also supports lowercase/uppercase variants.
    """

    found = {}

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)R$",
        re.IGNORECASE,
    )

    for col in df.columns:
        match = pattern.match(str(col))

        if match:
            bar = int(match.group(1))
            found[bar] = col

    return dict(sorted(found.items()))


def detect_paths(df: pd.DataFrame):
    """
    Detect MAE and close paths.

    Expected:
        mae_1R ... mae_20R
        close_1R ... close_20R
    """

    mae_cols = detect_numbered_columns(df, "mae")
    close_cols = detect_numbered_columns(df, "close")

    if not mae_cols:
        raise RuntimeError(
            "No MAE path columns found.\nExpected columns such as mae_1R, mae_2R, ..."
        )

    if not close_cols:
        raise RuntimeError(
            "No close path columns found.\n"
            "Expected columns such as close_1R, close_2R, ..."
        )

    common_bars = sorted(set(mae_cols.keys()) & set(close_cols.keys()))

    if not common_bars:
        raise RuntimeError("No common MAE/close path bars found.")

    return mae_cols, close_cols, common_bars


# =============================================================================
# WINDOW COLUMN
# =============================================================================


def detect_window_column(df: pd.DataFrame) -> str:

    candidates = [
        "window",
        "window_id",
        "validation_window",
        "walk_forward_window",
    ]

    col = first_existing_column(df, candidates)

    if col is None:
        raise RuntimeError(
            f"Could not find a window column. Expected one of: {candidates}"
        )

    return col


# =============================================================================
# FINAL R COLUMN
# =============================================================================


def detect_final_r_column(df: pd.DataFrame) -> Optional[str]:

    candidates = [
        "final_close_R",
        "net_R_path",
        "net_R",
    ]

    return first_existing_column(df, candidates)


# =============================================================================
# BENCHMARK METRICS
# =============================================================================


def profit_factor(values: pd.Series) -> float:

    values = pd.to_numeric(values, errors="coerce").dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()

    if gross_loss <= 0:
        if gross_profit > 0:
            return float("inf")
        return 0.0

    return float(gross_profit / gross_loss)


def max_drawdown(values: pd.Series) -> float:

    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return 0.0

    equity = values.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    return float(drawdown.min())


def positive_window_pct(
    df: pd.DataFrame,
    r_col: str,
    window_col: str,
) -> float:

    if df.empty:
        return 0.0

    grouped = df.groupby(window_col)[r_col].sum()

    if grouped.empty:
        return 0.0

    return float((grouped > 0).mean())


def strategy_metrics(
    df: pd.DataFrame,
    r_col: str,
    window_col: str,
) -> Dict[str, float]:

    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "mean_R": 0.0,
            "total_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
            "positive_window_pct": 0.0,
            "worst_window_R": 0.0,
            "best_window_R": 0.0,
        }

    values = pd.to_numeric(
        df[r_col],
        errors="coerce",
    ).dropna()

    wins = int((values > 0).sum())
    losses = int((values <= 0).sum())

    grouped = (
        df.assign(_metric_r=pd.to_numeric(df[r_col], errors="coerce"))
        .groupby(window_col)["_metric_r"]
        .sum()
    )

    return {
        "trades": int(len(values)),
        "wins": wins,
        "losses": losses,
        "win_rate": float((values > 0).mean()),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
        "positive_window_pct": (
            float((grouped > 0).mean()) if not grouped.empty else 0.0
        ),
        "worst_window_R": (float(grouped.min()) if not grouped.empty else 0.0),
        "best_window_R": (float(grouped.max()) if not grouped.empty else 0.0),
    }


# =============================================================================
# TRADE-LEVEL PATH LOGIC
# =============================================================================


def get_path_values(
    row: pd.Series,
    columns: Dict[int, str],
) -> List[Tuple[int, float]]:

    values = []

    for bar, col in sorted(columns.items()):
        value = safe_float(row.get(col))

        if np.isfinite(value):
            values.append((bar, value))

    return values


def find_mae_crossing(
    row: pd.Series,
    mae_cols: Dict[int, str],
    threshold: float,
) -> Optional[int]:
    """
    Find the FIRST bar where MAE >= threshold.

    MAE is stored as positive adverse excursion in R.
    """

    for bar, col in sorted(mae_cols.items()):
        value = safe_float(row.get(col))

        if not np.isfinite(value):
            continue

        if value >= threshold:
            return bar

    return None


def find_recovery_exit(
    row: pd.Series,
    close_cols: Dict[int, str],
    trigger_bar: int,
    recovery_level: float,
    recovery_horizon: int,
) -> Optional[Tuple[int, float]]:
    """
    After the MAE trigger, find the first close that recovers to the chosen
    recovery level.

    Example:

        MAE trigger = 0.80R
        recovery level = -0.20R

    Once the trade reaches 0.80R MAE, if a subsequent close is >= -0.20R,
    the recovery exit is executed at that close.

    The trigger bar itself is included.

    This is intentional:
    if the bar that first crosses the MAE boundary closes back above the
    recovery level, that information is available at that bar's close.
    """

    max_bar = trigger_bar + recovery_horizon

    for bar in sorted(close_cols):
        if bar < trigger_bar:
            continue

        if bar > max_bar:
            break

        col = close_cols[bar]

        close_r = safe_float(row.get(col))

        if not np.isfinite(close_r):
            continue

        if close_r >= recovery_level:
            return bar, close_r

    return None


# =============================================================================
# BUILD TRADE-LEVEL CANDIDATE RESULT
# =============================================================================


def evaluate_rule(
    df: pd.DataFrame,
    mae_cols: Dict[int, str],
    close_cols: Dict[int, str],
    mae_threshold: float,
    recovery_level: float,
    recovery_horizon: int,
) -> pd.DataFrame:

    rows = []

    for index, row in df.iterrows():
        original_r = safe_float(row["_final_R"])

        if not np.isfinite(original_r):
            continue

        trigger_bar = find_mae_crossing(
            row,
            mae_cols,
            mae_threshold,
        )

        # ---------------------------------------------------------------------
        # No MAE trigger:
        # preserve benchmark trade.
        # ---------------------------------------------------------------------

        if trigger_bar is None:
            rows.append(
                {
                    "_source_index": index,
                    "original_R": original_r,
                    "strategy_R": original_r,
                    "triggered": False,
                    "recovery_triggered": False,
                    "mae_trigger_bar": np.nan,
                    "recovery_exit_bar": np.nan,
                    "recovery_exit_R": np.nan,
                    "exit_type": "BENCHMARK",
                    "loss_rescued": False,
                    "winner_damaged": False,
                }
            )

            continue

        # ---------------------------------------------------------------------
        # MAE trigger exists.
        # ---------------------------------------------------------------------

        recovery = find_recovery_exit(
            row,
            close_cols,
            trigger_bar,
            recovery_level,
            recovery_horizon,
        )

        # ---------------------------------------------------------------------
        # Recovery achieved:
        # replace original outcome with recovery close.
        # ---------------------------------------------------------------------

        if recovery is not None:
            recovery_bar, recovery_r = recovery

            loss_rescued = original_r < 0 and recovery_r > original_r

            winner_damaged = original_r > 0 and recovery_r < original_r

            rows.append(
                {
                    "_source_index": index,
                    "original_R": original_r,
                    "strategy_R": recovery_r,
                    "triggered": True,
                    "recovery_triggered": True,
                    "mae_trigger_bar": trigger_bar,
                    "recovery_exit_bar": recovery_bar,
                    "recovery_exit_R": recovery_r,
                    "exit_type": "RECOVERY_EXIT",
                    "loss_rescued": bool(loss_rescued),
                    "winner_damaged": bool(winner_damaged),
                }
            )

        # ---------------------------------------------------------------------
        # MAE trigger but NO recovery:
        # preserve original benchmark outcome.
        # ---------------------------------------------------------------------

        else:
            rows.append(
                {
                    "_source_index": index,
                    "original_R": original_r,
                    "strategy_R": original_r,
                    "triggered": True,
                    "recovery_triggered": False,
                    "mae_trigger_bar": trigger_bar,
                    "recovery_exit_bar": np.nan,
                    "recovery_exit_R": np.nan,
                    "exit_type": "BENCHMARK_AFTER_NO_RECOVERY",
                    "loss_rescued": False,
                    "winner_damaged": False,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# CANDIDATE METRICS
# =============================================================================


def candidate_metrics(
    result_df: pd.DataFrame,
    source_df: pd.DataFrame,
    window_col: str,
) -> Dict[str, float]:

    if result_df.empty:
        return {}

    result = result_df.copy()

    source_windows = source_df[window_col].to_dict()

    result["_window"] = result["_source_index"].map(source_windows)

    result = result.dropna(subset=["_window"])

    values = pd.to_numeric(
        result["strategy_R"],
        errors="coerce",
    ).dropna()

    benchmark_values = pd.to_numeric(
        result["original_R"],
        errors="coerce",
    ).dropna()

    wins = int((values > 0).sum())
    losses = int((values <= 0).sum())

    triggered = int(result["triggered"].sum())
    recovered = int(result["recovery_triggered"].sum())

    losses_rescued = int(result["loss_rescued"].sum())
    winners_damaged = int(result["winner_damaged"].sum())

    grouped = result.groupby("_window")["strategy_R"].sum()

    benchmark_grouped = result.groupby("_window")["original_R"].sum()

    delta_values = result["strategy_R"] - result["original_R"]

    return {
        "trades": int(len(values)),
        "wins": wins,
        "losses": losses,
        "win_rate": float((values > 0).mean()),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
        "positive_window_pct": (
            float((grouped > 0).mean()) if not grouped.empty else 0.0
        ),
        "worst_window_R": (float(grouped.min()) if not grouped.empty else 0.0),
        "best_window_R": (float(grouped.max()) if not grouped.empty else 0.0),
        "triggered_trades": triggered,
        "recovery_exits": recovered,
        "losses_rescued": losses_rescued,
        "winners_damaged": winners_damaged,
        "delta_R": float(delta_values.sum()),
        "mean_delta_R": float(delta_values.mean()),
        "benchmark_R": float(benchmark_values.sum()),
        "benchmark_win_rate": float((benchmark_values > 0).mean()),
        "benchmark_profit_factor": profit_factor(benchmark_values),
        "benchmark_max_drawdown_R": max_drawdown(benchmark_values),
        "development_window_positive_pct": (
            float((grouped > benchmark_grouped * 0).mean())
            if not grouped.empty
            else 0.0
        ),
    }


# =============================================================================
# DISCOVERY / DEVELOPMENT SEARCH
# =============================================================================


def run_development_search(
    df: pd.DataFrame,
    mae_cols: Dict[int, str],
    close_cols: Dict[int, str],
    window_col: str,
) -> Tuple[pd.DataFrame, Dict]:

    development_df = df[df[window_col].isin(DEVELOPMENT_WINDOWS)].copy()

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    print(f"Development trades: {len(development_df)}")

    candidate_rows = []

    total_candidates = (
        len(MAE_THRESHOLDS) * len(RECOVERY_LEVELS) * len(RECOVERY_HORIZONS)
    )

    processed = 0

    for mae_threshold in MAE_THRESHOLDS:
        print()
        print(f"MAE threshold = {mae_threshold:.2f}R")

        for recovery_level in RECOVERY_LEVELS:
            for recovery_horizon in RECOVERY_HORIZONS:
                processed += 1

                result = evaluate_rule(
                    development_df,
                    mae_cols,
                    close_cols,
                    mae_threshold,
                    recovery_level,
                    recovery_horizon,
                )

                metrics = candidate_metrics(
                    result,
                    development_df,
                    window_col,
                )

                if not metrics:
                    continue

                candidate_rows.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "recovery_horizon": recovery_horizon,
                        **metrics,
                    }
                )

                if processed % 100 == 0 or processed == total_candidates:
                    print(f"  Processing {processed}/{total_candidates}...")

    candidates = pd.DataFrame(candidate_rows)

    if candidates.empty:
        raise RuntimeError("No candidate rules were generated.")

    # -------------------------------------------------------------------------
    # Filter rare candidates.
    # -------------------------------------------------------------------------

    eligible = candidates[candidates["triggered_trades"] >= MIN_TRIGGERED].copy()

    if eligible.empty:
        print()
        print(
            "WARNING: No candidate reached the minimum "
            f"trigger count of {MIN_TRIGGERED}."
        )

        eligible = candidates.copy()

    # -------------------------------------------------------------------------
    # Primary ranking:
    #
    # We care first about delta R.
    #
    # Then positive windows.
    #
    # Then lower drawdown.
    #
    # Then fewer damaged winners.
    #
    # This avoids selecting a rule solely because it has a high WR.
    # -------------------------------------------------------------------------

    eligible = eligible.sort_values(
        by=[
            "delta_R",
            "positive_window_pct",
            "max_drawdown_R",
            "winners_damaged",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    best = eligible.iloc[0].to_dict()

    return candidates, best


# =============================================================================
# HOLDOUT TEST
# =============================================================================


def run_holdout_test(
    df: pd.DataFrame,
    mae_cols: Dict[int, str],
    close_cols: Dict[int, str],
    window_col: str,
    best_rule: Dict,
):

    mae_threshold = float(best_rule["mae_threshold"])

    recovery_level = float(best_rule["recovery_level"])

    recovery_horizon = int(best_rule["recovery_horizon"])

    holdout_df = df[df[window_col].isin(HOLDOUT_WINDOWS)].copy()

    print()
    print("=" * 110)
    print("HOLDOUT OOS TEST")
    print("=" * 110)

    print()
    print("Frozen rule:")
    print(f"  MAE threshold    = {mae_threshold:.2f}R")
    print(f"  Recovery level   = {recovery_level:+.2f}R")
    print(f"  Recovery horizon = {recovery_horizon} bars")

    print()
    print(f"Holdout trades: {len(holdout_df)}")

    result = evaluate_rule(
        holdout_df,
        mae_cols,
        close_cols,
        mae_threshold,
        recovery_level,
        recovery_horizon,
    )

    metrics = candidate_metrics(
        result,
        holdout_df,
        window_col,
    )

    return holdout_df, result, metrics


# =============================================================================
# WINDOW-BY-WINDOW OOS
# =============================================================================


def build_window_oos(
    holdout_df: pd.DataFrame,
    result_df: pd.DataFrame,
    window_col: str,
) -> pd.DataFrame:

    source_windows = holdout_df[window_col].to_dict()

    result = result_df.copy()

    result["_window"] = result["_source_index"].map(source_windows)

    rows = []

    for window in sorted(result["_window"].dropna().unique()):
        subset = result[result["_window"] == window]

        benchmark_r = subset["original_R"].sum()

        strategy_r = subset["strategy_R"].sum()

        benchmark_values = subset["original_R"]

        strategy_values = subset["strategy_R"]

        rows.append(
            {
                "window": int(window),
                "trades": int(len(subset)),
                "triggered_trades": int(subset["triggered"].sum()),
                "recovery_exits": int(subset["recovery_triggered"].sum()),
                "losses_rescued": int(subset["loss_rescued"].sum()),
                "winners_damaged": int(subset["winner_damaged"].sum()),
                "benchmark_R": float(benchmark_r),
                "strategy_R": float(strategy_r),
                "delta_R": float(strategy_r - benchmark_r),
                "benchmark_WR": float((benchmark_values > 0).mean()),
                "strategy_WR": float((strategy_values > 0).mean()),
                "benchmark_PF": profit_factor(benchmark_values),
                "strategy_PF": profit_factor(strategy_values),
                "benchmark_DD": max_drawdown(benchmark_values),
                "strategy_DD": max_drawdown(strategy_values),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# RECOVERY DISTRIBUTION
# =============================================================================


def build_recovery_distribution(
    result_df: pd.DataFrame,
) -> pd.DataFrame:

    recovered = result_df[result_df["recovery_triggered"]].copy()

    if recovered.empty:
        return pd.DataFrame()

    return (
        recovered[
            [
                "mae_trigger_bar",
                "recovery_exit_bar",
                "recovery_exit_R",
                "original_R",
                "strategy_R",
                "loss_rescued",
                "winner_damaged",
            ]
        ]
        .copy()
        .sort_values(
            by=[
                "mae_trigger_bar",
                "recovery_exit_bar",
            ]
        )
    )


# =============================================================================
# FULL GRID OOS DIAGNOSTIC
# =============================================================================


def run_full_holdout_grid(
    df: pd.DataFrame,
    mae_cols: Dict[int, str],
    close_cols: Dict[int, str],
    window_col: str,
) -> pd.DataFrame:

    holdout_df = df[df[window_col].isin(HOLDOUT_WINDOWS)].copy()

    rows = []

    total = len(MAE_THRESHOLDS) * len(RECOVERY_LEVELS) * len(RECOVERY_HORIZONS)

    processed = 0

    print()
    print("=" * 110)
    print("FULL HOLDOUT GRID — DIAGNOSTIC ONLY")
    print("=" * 110)

    for mae_threshold in MAE_THRESHOLDS:
        for recovery_level in RECOVERY_LEVELS:
            for recovery_horizon in RECOVERY_HORIZONS:
                processed += 1

                result = evaluate_rule(
                    holdout_df,
                    mae_cols,
                    close_cols,
                    mae_threshold,
                    recovery_level,
                    recovery_horizon,
                )

                metrics = candidate_metrics(
                    result,
                    holdout_df,
                    window_col,
                )

                rows.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "recovery_horizon": recovery_horizon,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S18 MAE RECOVERY GRID — TEMPORAL OOS TEST")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop       = {STOP_POINTS} points")
    print(f"  RR         = {RR}")
    print(f"  Horizon    = {HORIZON} bars")

    print()
    print("MAE thresholds:")
    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))

    print()
    print("Recovery levels:")
    print("  " + ", ".join(f"{x:+.2f}R" for x in RECOVERY_LEVELS))

    print()
    print("Recovery horizons:")
    print(f"  {RECOVERY_HORIZONS}")

    print()
    print(f"Development windows: {DEVELOPMENT_WINDOWS}")

    print(f"Holdout windows    : {HOLDOUT_WINDOWS}")

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    # -------------------------------------------------------------------------
    # Detect columns
    # -------------------------------------------------------------------------

    mae_cols, close_cols, common_bars = detect_paths(df)

    window_col = detect_window_column(df)

    final_r_col = detect_final_r_column(df)

    if final_r_col is None:
        raise RuntimeError(
            "Could not detect final R column.\n"
            "Expected one of: "
            "final_close_R, net_R_path, net_R"
        )

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_cols)}")
    print(f"  MAE range  : {min(mae_cols)} -> {max(mae_cols)}")
    print(f"  Close bars : {len(close_cols)}")
    print(f"  Close range: {min(close_cols)} -> {max(close_cols)}")
    print(f"  Window col : {window_col}")
    print(f"  Final R    : {final_r_col}")

    # -------------------------------------------------------------------------
    # Normalize window
    # -------------------------------------------------------------------------

    df["_window_numeric"] = df[window_col].map(normalise_window)

    df["_final_R"] = pd.to_numeric(
        df[final_r_col],
        errors="coerce",
    )

    df = df[df["_window_numeric"].notna() & df["_final_R"].notna()].copy()

    df["_window_numeric"] = df["_window_numeric"].astype(int)

    # Use normalized window from here onward.
    window_col_internal = "_window_numeric"

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BASELINE")
    print("=" * 110)

    baseline_holdout = df[df[window_col_internal].isin(HOLDOUT_WINDOWS)].copy()

    baseline_metrics = strategy_metrics(
        baseline_holdout,
        "_final_R",
        window_col_internal,
    )

    for key, value in baseline_metrics.items():
        print(f"{key:30s}: {value}")

    # -------------------------------------------------------------------------
    # Development search
    # -------------------------------------------------------------------------

    candidates, best_rule = run_development_search(
        df,
        mae_cols,
        close_cols,
        window_col_internal,
    )

    # -------------------------------------------------------------------------
    # Top development rules
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("TOP DEVELOPMENT RULES")
    print("=" * 110)

    display_cols = [
        "mae_threshold",
        "recovery_level",
        "recovery_horizon",
        "triggered_trades",
        "recovery_exits",
        "losses_rescued",
        "winners_damaged",
        "win_rate",
        "mean_R",
        "total_R",
        "delta_R",
        "profit_factor",
        "max_drawdown_R",
        "positive_window_pct",
        "worst_window_R",
        "best_window_R",
    ]

    top = candidates[candidates["triggered_trades"] >= MIN_TRIGGERED].copy()

    if top.empty:
        top = candidates.copy()

    top = top.sort_values(
        by="delta_R",
        ascending=False,
    ).head(30)

    print(top[display_cols].to_string(index=False))

    # -------------------------------------------------------------------------
    # Selected development rule
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(f"MAE threshold    : {best_rule['mae_threshold']:.2f}R")

    print(f"Recovery level   : {best_rule['recovery_level']:+.2f}R")

    print(f"Recovery horizon : {int(best_rule['recovery_horizon'])} bars")

    print(f"Triggered trades : {int(best_rule['triggered_trades'])}")

    print(f"Recovery exits   : {int(best_rule['recovery_exits'])}")

    print(f"Losses rescued   : {int(best_rule['losses_rescued'])}")

    print(f"Winners damaged  : {int(best_rule['winners_damaged'])}")

    print(f"Development WR   : {best_rule['win_rate']:.4f}")

    print(f"Development R    : {best_rule['total_R']:.4f}")

    print(f"Development ΔR   : {best_rule['delta_R']:.4f}")

    print(f"Development PF   : {best_rule['profit_factor']:.4f}")

    print(f"Development DD   : {best_rule['max_drawdown_R']:.4f}")

    # -------------------------------------------------------------------------
    # Holdout OOS
    # -------------------------------------------------------------------------

    holdout_df, holdout_result, holdout_metrics = run_holdout_test(
        df,
        mae_cols,
        close_cols,
        window_col_internal,
        best_rule,
    )

    # -------------------------------------------------------------------------
    # Benchmark holdout metrics
    # -------------------------------------------------------------------------

    benchmark_values = holdout_df["_final_R"]

    benchmark_holdout_metrics = {
        "trades": int(len(benchmark_values)),
        "wins": int((benchmark_values > 0).sum()),
        "win_rate": float((benchmark_values > 0).mean()),
        "mean_R": float(benchmark_values.mean()),
        "total_R": float(benchmark_values.sum()),
        "profit_factor": profit_factor(benchmark_values),
        "max_drawdown_R": max_drawdown(benchmark_values),
    }

    # -------------------------------------------------------------------------
    # OOS result
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("HOLDOUT OOS RESULT")
    print("=" * 110)

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades      : {benchmark_holdout_metrics['trades']}")

    print(f"  Win rate    : {benchmark_holdout_metrics['win_rate']:.4f}")

    print(f"  Mean R      : {benchmark_holdout_metrics['mean_R']:.4f}")

    print(f"  Total R     : {benchmark_holdout_metrics['total_R']:.4f}")

    print(f"  PF          : {benchmark_holdout_metrics['profit_factor']:.4f}")

    print(f"  Max DD      : {benchmark_holdout_metrics['max_drawdown_R']:.4f}")

    print()
    print("RECOVERY EXIT HOLDOUT")

    print(f"  Trades             : {holdout_metrics['trades']}")

    print(f"  Win rate           : {holdout_metrics['win_rate']:.4f}")

    print(f"  Mean R             : {holdout_metrics['mean_R']:.4f}")

    print(f"  Total R            : {holdout_metrics['total_R']:.4f}")

    print(f"  PF                 : {holdout_metrics['profit_factor']:.4f}")

    print(f"  Max DD             : {holdout_metrics['max_drawdown_R']:.4f}")

    print(f"  Triggered trades   : {holdout_metrics['triggered_trades']}")

    print(f"  Recovery exits     : {holdout_metrics['recovery_exits']}")

    print(f"  Losses rescued     : {holdout_metrics['losses_rescued']}")

    print(f"  Winners damaged    : {holdout_metrics['winners_damaged']}")

    # -------------------------------------------------------------------------
    # Improvement
    # -------------------------------------------------------------------------

    delta_r = holdout_metrics["total_R"] - benchmark_holdout_metrics["total_R"]

    delta_mean_r = holdout_metrics["mean_R"] - benchmark_holdout_metrics["mean_R"]

    delta_wr = holdout_metrics["win_rate"] - benchmark_holdout_metrics["win_rate"]

    benchmark_pf = benchmark_holdout_metrics["profit_factor"]

    strategy_pf = holdout_metrics["profit_factor"]

    if math.isinf(strategy_pf):
        delta_pf_text = "inf"
    elif math.isinf(benchmark_pf):
        delta_pf_text = "-inf"
    else:
        delta_pf_text = f"{strategy_pf - benchmark_pf:.4f}"

    delta_dd = (
        holdout_metrics["max_drawdown_R"] - benchmark_holdout_metrics["max_drawdown_R"]
    )

    print()
    print("=" * 110)
    print("OOS IMPROVEMENT")
    print("=" * 110)

    print(f"Delta R         : {delta_r:.4f}")

    print(f"Delta mean R    : {delta_mean_r:.4f}")

    print(f"Delta win rate  : {delta_wr:.4f}")

    print(f"Delta PF        : {delta_pf_text}")

    print(f"Delta Max DD    : {delta_dd:.4f}")

    # -------------------------------------------------------------------------
    # Window-by-window
    # -------------------------------------------------------------------------

    window_oos = build_window_oos(
        holdout_df,
        holdout_result,
        window_col_internal,
    )

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    if window_oos.empty:
        print("No window-level results available.")
    else:
        print(window_oos.to_string(index=False))

    # -------------------------------------------------------------------------
    # Recovery distribution
    # -------------------------------------------------------------------------

    recovery_distribution = build_recovery_distribution(holdout_result)

    print()
    print("=" * 110)
    print("OOS RECOVERY EXIT DISTRIBUTION")
    print("=" * 110)

    if recovery_distribution.empty:
        print("No recovery exits occurred.")
    else:
        print(f"Recovery exits: {len(recovery_distribution)}")

        print(
            f"Mean recovery exit R: "
            f"{recovery_distribution['recovery_exit_R'].mean():.4f}"
        )

        print(
            f"Median recovery exit R: "
            f"{recovery_distribution['recovery_exit_R'].median():.4f}"
        )

        print(
            f"Mean MAE trigger bar: "
            f"{recovery_distribution['mae_trigger_bar'].mean():.2f}"
        )

        print(
            f"Mean recovery exit bar: "
            f"{recovery_distribution['recovery_exit_bar'].mean():.2f}"
        )

    # -------------------------------------------------------------------------
    # Full holdout grid diagnostic
    #
    # IMPORTANT:
    # This is NOT used to select the final rule.
    #
    # The final rule was selected using development only.
    #
    # This grid is saved so we can inspect whether the OOS result is robust
    # around neighboring thresholds / recovery levels.
    # -------------------------------------------------------------------------

    full_oos_grid = run_full_holdout_grid(
        df,
        mae_cols,
        close_cols,
        window_col_internal,
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = OUTPUT_DIR / "s18_mae_recovery_grid_oos_summary.csv"

    trades_path = OUTPUT_DIR / "s18_mae_recovery_grid_oos_trades.csv"

    windows_path = OUTPUT_DIR / "s18_mae_recovery_grid_oos_by_window.csv"

    development_path = OUTPUT_DIR / "s18_mae_recovery_grid_development.csv"

    full_grid_path = OUTPUT_DIR / "s18_mae_recovery_grid_full_oos.csv"

    recovery_distribution_path = (
        OUTPUT_DIR / "s18_mae_recovery_grid_recovery_distribution.csv"
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "result": "BENCHMARK_HOLDOUT",
                "mae_threshold": np.nan,
                "recovery_level": np.nan,
                "recovery_horizon": np.nan,
                **benchmark_holdout_metrics,
            },
            {
                "result": "RECOVERY_EXIT_HOLDOUT",
                "mae_threshold": best_rule["mae_threshold"],
                "recovery_level": best_rule["recovery_level"],
                "recovery_horizon": best_rule["recovery_horizon"],
                **holdout_metrics,
            },
        ]
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Trade-level output
    # -------------------------------------------------------------------------

    trade_output = holdout_result.copy()

    source_columns = [
        col
        for col in [
            "entry_timestamp",
            "exit_timestamp",
            "session_id",
            "quality",
            "vol_percentile",
            "stop_points",
            "rr",
            "horizon",
            "raw_points",
            "net_points",
            "net_R",
            "exit_reason",
            "holding_bars",
            "window",
            "validation_start",
            "validation_end",
            "trade_index",
            "entry_price",
        ]
        if col in holdout_df.columns
    ]

    source_subset = holdout_df[source_columns].copy()

    source_subset["_source_index"] = source_subset.index

    trade_output = trade_output.merge(
        source_subset,
        on="_source_index",
        how="left",
        suffixes=("", "_source"),
    )

    trade_output.to_csv(
        trades_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Window output
    # -------------------------------------------------------------------------

    window_oos.to_csv(
        windows_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Development grid
    # -------------------------------------------------------------------------

    candidates.to_csv(
        development_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Full OOS diagnostic grid
    # -------------------------------------------------------------------------

    full_oos_grid.to_csv(
        full_grid_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Recovery distribution
    # -------------------------------------------------------------------------

    recovery_distribution.to_csv(
        recovery_distribution_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final paths
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(summary_path)
    print(trades_path)
    print(windows_path)
    print(development_path)
    print(full_grid_path)
    print(recovery_distribution_path)

    print()
    print("=" * 110)
    print("S18 MAE RECOVERY GRID OOS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
