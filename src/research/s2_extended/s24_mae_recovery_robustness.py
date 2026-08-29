"""
S24 — MAE + RECOVERY ROBUSTNESS TEST

FINAL ROBUSTNESS TEST FOR THE MAE/RECOVERY BRANCH

Objective
---------
Validate whether the MAE -> recovery exit relationship remains useful
across nearby parameter values and temporal sub-periods.

This is NOT another broad optimization.

We deliberately restrict the search to the robust region discovered
in S23:

    MAE:
        0.65R, 0.70R, 0.75R, 0.80R, 0.85R

    Recovery:
        -0.10R, 0.00R, +0.10R, +0.20R, +0.30R

    Deadline:
        4, 5, 6 bars

The purpose is to determine whether there is a ROBUST REGION rather
than a single optimized parameter combination.

Temporal evaluation
-------------------
Development:
    windows 1..11

Holdout:
    windows 12..22

Additionally, the holdout is divided into temporal blocks so that
we can inspect consistency across different periods.

IMPORTANT
---------
No parameter is selected using holdout performance.

The final recommendation is based on:

    1. Development performance
    2. Robustness across neighboring parameters
    3. Holdout confirmation
    4. Number of positive holdout windows
    5. Drawdown behavior
    6. Stability of the parameter neighborhood

If the evidence is sufficiently stable, this branch should be
FROZEN and no further MAE/recovery optimization should be performed.
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
# ROBUSTNESS GRID
# =============================================================================

MAE_THRESHOLDS = [
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]

RECOVERY_LEVELS = [
    -0.10,
    0.00,
    0.10,
    0.20,
    0.30,
]

DEADLINES = [
    4,
    5,
    6,
]


# =============================================================================
# TEMPORAL WINDOWS
# =============================================================================

DEVELOPMENT_WINDOWS = list(range(1, 12))

HOLDOUT_WINDOWS = list(range(12, 23))


# Holdout temporal blocks.
#
# These are NOT used to optimize.
# They are only used to evaluate stability.

HOLDOUT_BLOCKS = {
    "EARLY_OOS": [12, 13, 14, 15],
    "MID_OOS": [16, 18, 19],
    "LATE_OOS": [20, 21, 22],
}


# =============================================================================
# HELPERS
# =============================================================================


def normalize_window(value):
    """
    Normalize window labels to integers.
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
    Detect close_N_R and mae_N_R path columns.
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
        raise RuntimeError("No close_N_R columns found.")

    if not mae_bars:
        raise RuntimeError("No mae_N_R columns found.")

    return close_bars, mae_bars


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
        "trades": len(values),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(values),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
    }


# =============================================================================
# MAE EVENT
# =============================================================================


def find_mae_crossing(
    row: pd.Series,
    mae_bars: List[int],
    threshold: float,
) -> Optional[int]:
    """
    Find first bar where MAE >= threshold.
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
# RECOVERY EVENT
# =============================================================================


def find_recovery_crossing(
    row: pd.Series,
    close_bars: List[int],
    mae_bar: int,
    recovery_level: float,
    deadline: int,
) -> Optional[int]:
    """
    Find the first close reaching recovery_level after MAE.

    Search begins AFTER the MAE bar.
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
# EXECUTION
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
    Execute one recovery rule.

    Logic:

        No MAE event
            -> benchmark

        MAE event
            -> wait deadline

        Recovery reached
            -> exit at observed recovery close

        Recovery not reached
            -> exit at deadline close
    """

    benchmark_R = float(row["_benchmark_R"])

    mae_bar = find_mae_crossing(
        row,
        mae_bars,
        mae_threshold,
    )

    # -------------------------------------------------------------------------
    # No MAE event
    # -------------------------------------------------------------------------

    if mae_bar is None:
        return {
            "strategy_R": benchmark_R,
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "exit_type": "BENCHMARK",
        }

    # -------------------------------------------------------------------------
    # Recovery event
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
            "exit_type": "RECOVERY_EXIT",
        }

    # -------------------------------------------------------------------------
    # Deadline event
    # -------------------------------------------------------------------------

    deadline_bar = min(
        mae_bar + deadline,
        max(close_bars),
    )

    deadline_R = row.get(
        f"close_{deadline_bar}R",
        np.nan,
    )

    if pd.isna(deadline_R):
        strategy_R = benchmark_R

    else:
        strategy_R = float(deadline_R)

    return {
        "strategy_R": strategy_R,
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "exit_type": "DEADLINE_EXIT",
    }


# =============================================================================
# APPLY RULE
# =============================================================================


def apply_rule(
    df: pd.DataFrame,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:
    """
    Apply one rule to all trades.
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
                "exit_type": result["exit_type"],
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# RULE EVALUATION
# =============================================================================


def evaluate(
    trades: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate strategy versus benchmark.
    """

    strategy = metrics(trades["strategy_R"])

    benchmark = metrics(trades["benchmark_R"])

    triggered = trades[trades["exit_type"] != "BENCHMARK"]

    recovery = trades[trades["exit_type"] == "RECOVERY_EXIT"]

    deadline = trades[trades["exit_type"] == "DEADLINE_EXIT"]

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
        "benchmark_max_drawdown_R": benchmark["max_drawdown_R"],
        "delta_R": strategy["total_R"] - benchmark["total_R"],
        "delta_mean_R": strategy["mean_R"] - benchmark["mean_R"],
        "delta_win_rate": strategy["win_rate"] - benchmark["win_rate"],
        "delta_DD": strategy["max_drawdown_R"] - benchmark["max_drawdown_R"],
        "mae_triggered": len(triggered),
        "recovery_exits": len(recovery),
        "deadline_exits": len(deadline),
        "recovery_exit_pct": (
            len(recovery) / len(triggered) if len(triggered) > 0 else np.nan
        ),
    }


# =============================================================================
# FULL PARAMETER GRID
# =============================================================================


def run_grid(
    df: pd.DataFrame,
    windows: List[int],
    close_bars: List[int],
    mae_bars: List[int],
) -> Tuple[pd.DataFrame, Dict[Tuple[float, float, int], pd.DataFrame]]:
    """
    Evaluate all robustness combinations on a specified temporal set.
    """

    subset = df[df["_window_numeric"].isin(windows)].copy()

    records = {}

    rows = []

    total = len(MAE_THRESHOLDS) * len(RECOVERY_LEVELS) * len(DEADLINES)

    counter = 0

    for mae in MAE_THRESHOLDS:
        for recovery in RECOVERY_LEVELS:
            for deadline in DEADLINES:
                counter += 1

                trades = apply_rule(
                    subset,
                    mae,
                    recovery,
                    deadline,
                    close_bars,
                    mae_bars,
                )

                m = evaluate(trades)

                key = (
                    mae,
                    recovery,
                    deadline,
                )

                records[key] = trades

                rows.append(
                    {
                        "mae_threshold": mae,
                        "recovery_level": recovery,
                        "deadline": deadline,
                        **m,
                    }
                )

    result = pd.DataFrame(rows)

    return result, records


# =============================================================================
# ROBUSTNESS SCORING
# =============================================================================


def calculate_robustness_score(
    development: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score parameter combinations using DEVELOPMENT only.

    We explicitly reward:

        + delta R
        + positive mean R
        + lower drawdown

    We do NOT use holdout here.
    """

    result = development.copy()

    result["_score"] = (
        result["delta_R"].fillna(-999)
        + 20.0 * result["delta_mean_R"].fillna(-999)
        + 0.25 * result["delta_DD"].fillna(-999)
    )

    return result.sort_values(
        "_score",
        ascending=False,
    ).reset_index(drop=True)


# =============================================================================
# TEMPORAL BLOCK ANALYSIS
# =============================================================================


def block_analysis(
    df: pd.DataFrame,
    rule: Tuple[float, float, int],
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:
    """
    Evaluate frozen rule independently across temporal blocks.
    """

    mae, recovery, deadline = rule

    rows = []

    for block_name, windows in HOLDOUT_BLOCKS.items():
        subset = df[df["_window_numeric"].isin(windows)].copy()

        if len(subset) == 0:
            continue

        trades = apply_rule(
            subset,
            mae,
            recovery,
            deadline,
            close_bars,
            mae_bars,
        )

        m = evaluate(trades)

        rows.append(
            {
                "block": block_name,
                "windows": ",".join(str(x) for x in windows),
                **m,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# NEIGHBORHOOD ANALYSIS
# =============================================================================


def neighborhood_analysis(
    holdout_grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify whether nearby parameter combinations also work.

    This is intentionally descriptive.

    We are looking for a REGION, not a single point.
    """

    result = holdout_grid.copy()

    result["positive_delta"] = result["delta_R"] > 0

    result["positive_mean"] = result["mean_R"] > 0

    result["improved_DD"] = result["delta_DD"] > 0

    result["all_positive"] = result["positive_delta"] & result["improved_DD"]

    return result


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S24 MAE + RECOVERY ROBUSTNESS TEST")
    print("=" * 110)

    print()
    print("Purpose:")

    print("Determine whether the MAE/recovery edge is a robust parameter region.")

    print()
    print("Frozen benchmark:")

    print(f"  Stop       = {STOP_POINTS} points")

    print(f"  RR         = {RR}")

    print(f"  Horizon    = {HORIZON}")

    print()
    print("Robustness MAE grid:")

    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))

    print()
    print("Robustness recovery grid:")

    print("  " + ", ".join(f"{x:+.2f}R" for x in RECOVERY_LEVELS))

    print()
    print("Robustness deadlines:")

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

    required = [
        "final_close_R",
        "window",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

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

    df = df[df["_benchmark_R"].notna()].copy()

    # =========================================================================
    # DEVELOPMENT GRID
    # =========================================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT ROBUSTNESS GRID")
    print("=" * 110)

    development_grid, development_trades = run_grid(
        df,
        DEVELOPMENT_WINDOWS,
        close_bars,
        mae_bars,
    )

    development_scored = calculate_robustness_score(development_grid)

    print()
    print(
        development_scored[
            [
                "mae_threshold",
                "recovery_level",
                "deadline",
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "delta_R",
                "delta_mean_R",
                "delta_win_rate",
                "delta_DD",
                "recovery_exits",
                "deadline_exits",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )

    # =========================================================================
    # SELECT DEVELOPMENT CENTER
    # =========================================================================

    selected = development_scored.iloc[0]

    selected_rule = (
        float(selected["mae_threshold"]),
        float(selected["recovery_level"]),
        int(selected["deadline"]),
    )

    print()
    print("=" * 110)
    print("DEVELOPMENT CENTER")
    print("=" * 110)

    print(f"MAE threshold : {selected_rule[0]:.2f}R")

    print(f"Recovery      : {selected_rule[1]:+.2f}R")

    print(f"Deadline      : {selected_rule[2]} bars")

    print(f"Development ΔR: {selected['delta_R']:.4f}")

    print(f"Development mean ΔR: {selected['delta_mean_R']:.4f}")

    print(f"Development DD Δ: {selected['delta_DD']:.4f}")

    # =========================================================================
    # HOLDOUT GRID
    # =========================================================================

    print()
    print("=" * 110)
    print("HOLDOUT ROBUSTNESS GRID")
    print("=" * 110)

    holdout_grid, holdout_trades = run_grid(
        df,
        HOLDOUT_WINDOWS,
        close_bars,
        mae_bars,
    )

    print()

    print(
        holdout_grid[
            [
                "mae_threshold",
                "recovery_level",
                "deadline",
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "delta_R",
                "delta_mean_R",
                "delta_win_rate",
                "delta_DD",
                "recovery_exits",
                "deadline_exits",
            ]
        ]
        .sort_values(
            "delta_R",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )

    # =========================================================================
    # FROZEN DEVELOPMENT RULE OOS
    # =========================================================================

    frozen_key = selected_rule

    frozen_holdout = holdout_trades[frozen_key]

    frozen_summary = evaluate(frozen_holdout)

    print()
    print("=" * 110)
    print("FROZEN DEVELOPMENT RULE — OOS")
    print("=" * 110)

    print(f"MAE threshold : {selected_rule[0]:.2f}R")

    print(f"Recovery      : {selected_rule[1]:+.2f}R")

    print(f"Deadline      : {selected_rule[2]} bars")

    print()
    print("BENCHMARK OOS")

    print(f"  WR      : {frozen_summary['benchmark_win_rate']:.4f}")

    print(f"  Mean R  : {frozen_summary['benchmark_mean_R']:.4f}")

    print(f"  Total R : {frozen_summary['benchmark_total_R']:.4f}")

    print(f"  DD      : {frozen_summary['benchmark_max_drawdown_R']:.4f}")

    print()
    print("RECOVERY RULE OOS")

    print(f"  WR      : {frozen_summary['win_rate']:.4f}")

    print(f"  Mean R  : {frozen_summary['mean_R']:.4f}")

    print(f"  Total R : {frozen_summary['total_R']:.4f}")

    print(f"  PF      : {frozen_summary['profit_factor']:.4f}")

    print(f"  DD      : {frozen_summary['max_drawdown_R']:.4f}")

    print()
    print("DELTA")

    print(f"  ΔR      : {frozen_summary['delta_R']:.4f}")

    print(f"  ΔMean R : {frozen_summary['delta_mean_R']:.4f}")

    print(f"  ΔWR     : {frozen_summary['delta_win_rate']:.4f}")

    print(f"  ΔDD     : {frozen_summary['delta_DD']:.4f}")

    # =========================================================================
    # TEMPORAL BLOCKS
    # =========================================================================

    print()
    print("=" * 110)
    print("TEMPORAL BLOCK STABILITY — FROZEN RULE")
    print("=" * 110)

    block_df = block_analysis(
        df,
        selected_rule,
        close_bars,
        mae_bars,
    )

    print(
        block_df[
            [
                "block",
                "windows",
                "trades",
                "benchmark_win_rate",
                "win_rate",
                "benchmark_mean_R",
                "mean_R",
                "benchmark_total_R",
                "total_R",
                "delta_R",
                "benchmark_max_drawdown_R",
                "max_drawdown_R",
                "delta_DD",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # NEIGHBORHOOD
    # =========================================================================

    print()
    print("=" * 110)
    print("PARAMETER NEIGHBORHOOD — HOLDOUT")
    print("=" * 110)

    neighborhood = neighborhood_analysis(holdout_grid)

    positive_count = int(neighborhood["positive_delta"].sum())

    total_count = len(neighborhood)

    positive_dd_count = int(neighborhood["improved_DD"].sum())

    print(f"Rules with positive ΔR: {positive_count}/{total_count}")

    print(f"Rules with improved DD: {positive_dd_count}/{total_count}")

    print()

    print(
        neighborhood[
            [
                "mae_threshold",
                "recovery_level",
                "deadline",
                "delta_R",
                "delta_mean_R",
                "delta_win_rate",
                "delta_DD",
                "positive_delta",
                "improved_DD",
                "all_positive",
            ]
        ]
        .sort_values(
            "delta_R",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )

    # =========================================================================
    # ROBUSTNESS CONCLUSION
    # =========================================================================

    print()
    print("=" * 110)
    print("S24 ROBUSTNESS CONCLUSION")
    print("=" * 110)

    block_positive = int((block_df["delta_R"] > 0).sum())

    block_total = len(block_df)

    if (
        frozen_summary["delta_R"] > 0
        and positive_count / total_count >= 0.50
        and block_positive / max(block_total, 1) >= 0.50
    ):
        print("PASS: MAE/RECOVERY EDGE SHOWS ROBUST OOS SUPPORT.")

        print()
        print("RECOMMENDATION:")

        print("Freeze the MAE/recovery branch and move to final strategy validation.")

    else:
        print(
            "FAIL / INCONCLUSIVE: "
            "MAE/recovery edge is not sufficiently "
            "robust to freeze."
        )

        print()
        print("RECOMMENDATION:")

        print("Do not incorporate this branch into the final strategy yet.")

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = OUTPUT_DIR / "s24_mae_recovery_development_grid.csv"

    holdout_path = OUTPUT_DIR / "s24_mae_recovery_holdout_grid.csv"

    frozen_path = OUTPUT_DIR / "s24_mae_recovery_frozen_oos.csv"

    block_path = OUTPUT_DIR / "s24_mae_recovery_temporal_blocks.csv"

    neighborhood_path = OUTPUT_DIR / "s24_mae_recovery_neighborhood.csv"

    selected_path = OUTPUT_DIR / "s24_mae_recovery_selected_rule.csv"

    development_scored.to_csv(
        development_path,
        index=False,
    )

    holdout_grid.to_csv(
        holdout_path,
        index=False,
    )

    frozen_holdout.to_csv(
        frozen_path,
        index=False,
    )

    block_df.to_csv(
        block_path,
        index=False,
    )

    neighborhood.to_csv(
        neighborhood_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "mae_threshold": selected_rule[0],
                "recovery_level": selected_rule[1],
                "deadline": selected_rule[2],
                "oos_delta_R": frozen_summary["delta_R"],
                "oos_delta_mean_R": frozen_summary["delta_mean_R"],
                "oos_delta_win_rate": frozen_summary["delta_win_rate"],
                "oos_delta_DD": frozen_summary["delta_DD"],
                "positive_holdout_rules": positive_count,
                "total_holdout_rules": total_count,
                "positive_block_count": block_positive,
                "total_block_count": block_total,
            }
        ]
    ).to_csv(
        selected_path,
        index=False,
    )

    # =========================================================================
    # FILES
    # =========================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(development_path)
    print(holdout_path)
    print(frozen_path)
    print(block_path)
    print(neighborhood_path)
    print(selected_path)

    print()
    print("=" * 110)
    print("S24 MAE + RECOVERY ROBUSTNESS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
