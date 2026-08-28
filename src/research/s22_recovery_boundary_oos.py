"""
S22 — RECOVERY BOUNDARY TEMPORAL OOS

Research question
-----------------
After a significant MAE event, what recovery level best separates
trades that eventually survive from trades that fail?

This test is deliberately narrower than S21.

We are NOT searching hundreds of unrelated features.

We are mapping the recovery boundary:

    MAE trigger
        |
        v
    adverse state
        |
        +---- recovers to X R ----> RECOVERED
        |
        +---- does not recover ---> FAILED

Recovery levels:
    -0.50R ... +0.40R

The purpose is to determine whether there is a robust recovery boundary
that can later become an executable filter.

Temporal OOS:
    Development windows = 1..11
    Holdout windows     = 12..22

IMPORTANT
---------
The selected rule is frozen after development.

No holdout information is used for selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
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


# ============================================================================
# FROZEN BENCHMARK
# ============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# ============================================================================
# RESEARCH GRID
# ============================================================================

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


# ============================================================================
# HELPERS
# ============================================================================


def normalize_window(value):
    """
    Normalize window labels into integers.
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
    Detect close_N_R and mae_N_R style columns.

    Actual expected names:

        close_1R
        close_2R
        ...
        mae_1R
        mae_2R
        ...
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
        raise RuntimeError("No close path columns found.")

    if not mae_bars:
        raise RuntimeError("No MAE path columns found.")

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
    Maximum cumulative drawdown in R.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:
        return 0.0

    equity = values.cumsum()

    peak = equity.cummax()

    dd = equity - peak

    return float(dd.min())


def metrics(
    df: pd.DataFrame,
    r_column: str,
) -> Dict[str, float]:
    """
    Standard strategy metrics.
    """

    values = pd.to_numeric(
        df[r_column],
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


# ============================================================================
# FIRST MAE CROSSING
# ============================================================================


def find_mae_bar(
    row: pd.Series,
    mae_bars: List[int],
    threshold: float,
) -> Optional[int]:
    """
    Find the first bar where MAE reaches the threshold.
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


# ============================================================================
# RECOVERY AFTER MAE
# ============================================================================


def find_recovery(
    row: pd.Series,
    close_bars: List[int],
    mae_bar: int,
    recovery_level: float,
    deadline: int,
) -> Optional[int]:
    """
    Find first close >= recovery level after MAE trigger.

    Search begins on the NEXT bar.

    Example:

        MAE trigger = bar 4
        deadline    = 5

    Search:

        5, 6, 7, 8, 9
    """

    start = mae_bar + 1

    end = min(
        mae_bar + deadline,
        max(close_bars),
    )

    for bar in close_bars:
        if bar < start:
            continue

        if bar > end:
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


# ============================================================================
# SINGLE TRADE STATE
# ============================================================================


def classify_trade(
    row: pd.Series,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> Dict[str, object]:
    """
    Classify one trade.

    States:

        NO_MAE
        RECOVERED
        FAILED

    We retain the original benchmark R as the eventual outcome.

    This test is primarily about classification/separation.

    The executable R calculation is also included so that the result
    can be tested later.
    """

    final_R = float(row["_benchmark_R"])

    mae_bar = find_mae_bar(
        row,
        mae_bars,
        mae_threshold,
    )

    # ---------------------------------------------------------------
    # Never reached the MAE threshold
    # ---------------------------------------------------------------

    if mae_bar is None:
        return {
            "state": "NO_MAE",
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "recovery_R": np.nan,
            "strategy_R": final_R,
        }

    # ---------------------------------------------------------------
    # Search recovery
    # ---------------------------------------------------------------

    recovery_bar = find_recovery(
        row,
        close_bars,
        mae_bar,
        recovery_level,
        deadline,
    )

    # ---------------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------------

    if recovery_bar is not None:
        recovery_R = float(row[f"close_{recovery_bar}R"])

        return {
            "state": "RECOVERED",
            "mae_bar": mae_bar,
            "recovery_bar": recovery_bar,
            "recovery_R": recovery_R,
            "strategy_R": final_R,
        }

    # ---------------------------------------------------------------
    # Failure to recover
    # ---------------------------------------------------------------

    return {
        "state": "FAILED",
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "recovery_R": np.nan,
        "strategy_R": final_R,
    }


# ============================================================================
# BUILD CLASSIFICATION DATASET
# ============================================================================


def build_classification_dataset(
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
        result = classify_trade(
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
                "mae_threshold": mae_threshold,
                "recovery_level": recovery_level,
                "deadline": deadline,
                "benchmark_R": final_value(row["_benchmark_R"]),
                "_strategy_R": final_value(result["strategy_R"]),
                "state": result["state"],
                "mae_bar": result["mae_bar"],
                "recovery_bar": result["recovery_bar"],
                "recovery_R": result["recovery_R"],
                "window": row["_window_numeric"],
            }
        )

    return pd.DataFrame(records)


def final_value(
    value,
) -> float:
    """
    Safe float conversion.
    """

    if pd.isna(value):
        return 0.0

    return float(value)


# ============================================================================
# RULE ANALYSIS
# ============================================================================


def analyze_rule(
    trades: pd.DataFrame,
) -> Dict[str, float]:
    """
    Analyze one recovery rule.

    Critical quantities:

        recovered WR
        failed WR
        separation
        recovered mean R
        failed mean R
    """

    recovered = trades[trades["state"] == "RECOVERED"]

    failed = trades[trades["state"] == "FAILED"]

    no_mae = trades[trades["state"] == "NO_MAE"]

    recovered_metrics = metrics(
        recovered,
        "_strategy_R",
    )

    failed_metrics = metrics(
        failed,
        "_strategy_R",
    )

    no_mae_metrics = metrics(
        no_mae,
        "_strategy_R",
    )

    # Difference in win rate between recovered and failed.
    if len(recovered) > 0 and len(failed) > 0:
        separation = recovered_metrics["win_rate"] - failed_metrics["win_rate"]

    else:
        separation = np.nan

    # Outcome difference.
    if len(recovered) > 0 and len(failed) > 0:
        mean_R_separation = recovered_metrics["mean_R"] - failed_metrics["mean_R"]

    else:
        mean_R_separation = np.nan

    return {
        "trades": len(trades),
        "mae_triggered": len(recovered) + len(failed),
        "recovered": len(recovered),
        "failed": len(failed),
        "no_mae": len(no_mae),
        "recovery_rate": (
            len(recovered) / (len(recovered) + len(failed))
            if (len(recovered) + len(failed)) > 0
            else np.nan
        ),
        "recovered_win_rate": recovered_metrics["win_rate"],
        "failed_win_rate": failed_metrics["win_rate"],
        "win_rate_separation": separation,
        "recovered_mean_R": recovered_metrics["mean_R"],
        "failed_mean_R": failed_metrics["mean_R"],
        "mean_R_separation": mean_R_separation,
        "recovered_total_R": recovered_metrics["total_R"],
        "failed_total_R": failed_metrics["total_R"],
        "recovered_PF": recovered_metrics["profit_factor"],
        "failed_PF": failed_metrics["profit_factor"],
    }


# ============================================================================
# DEVELOPMENT SEARCH
# ============================================================================


def development_search(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
) -> Tuple[
    pd.DataFrame,
    Dict[Tuple[float, float, int], pd.DataFrame],
]:
    """
    Search recovery boundary candidates on development only.
    """

    development = df[df["_window_numeric"].isin(DEVELOPMENT_WINDOWS)].copy()

    if len(development) == 0:
        raise RuntimeError("No development trades.")

    records = []

    datasets = {}

    total_rules = len(MAE_THRESHOLDS) * len(RECOVERY_LEVELS) * len(DEADLINES)

    processed = 0

    print(f"Testing {total_rules} recovery boundary candidates...")

    for mae_threshold in MAE_THRESHOLDS:
        for recovery_level in RECOVERY_LEVELS:
            for deadline in DEADLINES:
                processed += 1

                if processed == 1 or processed % 50 == 0 or processed == total_rules:
                    print(f"  Processing {processed}/{total_rules}...")

                trades = build_classification_dataset(
                    development,
                    mae_threshold,
                    recovery_level,
                    deadline,
                    close_bars,
                    mae_bars,
                )

                datasets[
                    (
                        mae_threshold,
                        recovery_level,
                        deadline,
                    )
                ] = trades

                a = analyze_rule(trades)

                records.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "deadline": deadline,
                        **{f"development_{key}": value for key, value in a.items()},
                    }
                )

    results = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Selection philosophy
    #
    # We want:
    #
    # 1. Large separation between recovered and failed
    # 2. Recovered cohort genuinely positive
    # 3. Failed cohort clearly negative
    # 4. Enough recovered observations
    #
    # We deliberately do NOT select solely on total R.
    # This stage is boundary discovery.
    # ------------------------------------------------------------------

    results["_selection_score"] = results["development_win_rate_separation"].fillna(
        -999.0
    ) + 0.50 * results["development_mean_R_separation"].fillna(-999.0)

    # Penalize extremely tiny recovered samples.
    results.loc[results["development_recovered"] < 5, "_selection_score"] -= 10.0

    results = results.sort_values(
        by=[
            "_selection_score",
            "development_recovered_win_rate",
            "development_failed_win_rate",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return results, datasets


# ============================================================================
# HOLDOUT
# ============================================================================


def holdout_test(
    df: pd.DataFrame,
    selected_rule: pd.Series,
    close_bars: List[int],
    mae_bars: List[int],
) -> Tuple[
    pd.DataFrame,
    Dict[str, float],
]:
    """
    Test frozen rule on holdout.
    """

    holdout = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    if len(holdout) == 0:
        raise RuntimeError("No holdout trades.")

    trades = build_classification_dataset(
        holdout,
        float(selected_rule["mae_threshold"]),
        float(selected_rule["recovery_level"]),
        int(selected_rule["deadline"]),
        close_bars,
        mae_bars,
    )

    summary = analyze_rule(trades)

    return trades, summary


# ============================================================================
# HOLDOUT WINDOW TABLE
# ============================================================================


def window_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show recovery separation window by window.
    """

    records = []

    for window in sorted(trades["window"].dropna().unique()):
        subset = trades[trades["window"] == window]

        recovered = subset[subset["state"] == "RECOVERED"]

        failed = subset[subset["state"] == "FAILED"]

        recovered_wr = (
            float((recovered["_strategy_R"] > 0).mean())
            if len(recovered) > 0
            else np.nan
        )

        failed_wr = (
            float((failed["_strategy_R"] > 0).mean()) if len(failed) > 0 else np.nan
        )

        records.append(
            {
                "window": int(window),
                "trades": len(subset),
                "recovered": len(recovered),
                "failed": len(failed),
                "recovery_rate": (
                    len(recovered) / (len(recovered) + len(failed))
                    if (len(recovered) + len(failed)) > 0
                    else np.nan
                ),
                "recovered_WR": recovered_wr,
                "failed_WR": failed_wr,
                "WR_separation": (
                    recovered_wr - failed_wr
                    if (not pd.isna(recovered_wr) and not pd.isna(failed_wr))
                    else np.nan
                ),
                "recovered_R": float(recovered["_strategy_R"].sum()),
                "failed_R": float(failed["_strategy_R"].sum()),
                "total_R": float(subset["_strategy_R"].sum()),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# RECOVERY LEVEL MAP
# ============================================================================


def recovery_level_map(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
    mae_threshold: float,
    deadline: int,
) -> pd.DataFrame:
    """
    Hold the MAE threshold and deadline fixed and map every recovery level.

    This is one of the most important outputs of S22.

    It answers:

        At this MAE state and this time horizon,
        how does the winner/loser separation change
        as the required recovery level moves from
        -0.50R to +0.40R?
    """

    records = []

    for recovery_level in RECOVERY_LEVELS:
        trades = build_classification_dataset(
            df,
            mae_threshold,
            recovery_level,
            deadline,
            close_bars,
            mae_bars,
        )

        a = analyze_rule(trades)

        records.append(
            {
                "mae_threshold": mae_threshold,
                "deadline": deadline,
                "recovery_level": recovery_level,
                **a,
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 110)
    print("S22 RECOVERY BOUNDARY TEMPORAL OOS")
    print("=" * 110)

    print()
    print("Research question:")
    print(
        "After significant MAE, what recovery level best separates winners from losers?"
    )

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

    # =====================================================================
    # LOAD
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
    # REQUIRED COLUMNS
    # =====================================================================

    if "final_close_R" not in df.columns:
        raise RuntimeError("Missing final_close_R.")

    if "window" not in df.columns:
        raise RuntimeError("Missing window.")

    # =====================================================================
    # PATHS
    # =====================================================================

    close_bars, mae_bars = detect_path_columns(df)

    print()
    print("Detected paths:")

    print(f"  MAE bars   : {len(mae_bars)}")

    print(f"  MAE range  : {min(mae_bars)} -> {max(mae_bars)}")

    print(f"  Close bars : {len(close_bars)}")

    print(f"  Close range: {min(close_bars)} -> {max(close_bars)}")

    # =====================================================================
    # PREPARE DATA
    # =====================================================================

    df = df.copy()

    df["_window_numeric"] = df["window"].apply(normalize_window)

    df["_benchmark_R"] = pd.to_numeric(
        df["final_close_R"],
        errors="coerce",
    )

    if df["_benchmark_R"].isna().all():
        raise RuntimeError("No valid benchmark R values.")

    # =====================================================================
    # DEVELOPMENT
    # =====================================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    development_results, _ = development_search(
        df,
        close_bars,
        mae_bars,
    )

    display_columns = [
        "mae_threshold",
        "recovery_level",
        "deadline",
        "development_mae_triggered",
        "development_recovered",
        "development_failed",
        "development_recovery_rate",
        "development_recovered_win_rate",
        "development_failed_win_rate",
        "development_win_rate_separation",
        "development_recovered_mean_R",
        "development_failed_mean_R",
        "development_mean_R_separation",
        "development_recovered_total_R",
        "development_failed_total_R",
    ]

    print()

    print(development_results[display_columns].head(30).to_string(index=False))

    # =====================================================================
    # SELECTED RULE
    # =====================================================================

    selected_rule = development_results.iloc[0]

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT BOUNDARY")
    print("=" * 110)

    print(f"MAE threshold       : {selected_rule['mae_threshold']:.2f}R")

    print(f"Recovery level      : {selected_rule['recovery_level']:+.2f}R")

    print(f"Deadline            : {int(selected_rule['deadline'])} bars")

    print(f"Recovered trades    : {int(selected_rule['development_recovered'])}")

    print(f"Failed trades       : {int(selected_rule['development_failed'])}")

    print(
        f"Recovered WR        : {selected_rule['development_recovered_win_rate']:.4f}"
    )

    print(f"Failed WR           : {selected_rule['development_failed_win_rate']:.4f}")

    print(
        f"WR separation       : {selected_rule['development_win_rate_separation']:.4f}"
    )

    print(f"Recovered mean R    : {selected_rule['development_recovered_mean_R']:.4f}")

    print(f"Failed mean R       : {selected_rule['development_failed_mean_R']:.4f}")

    print(f"Mean R separation   : {selected_rule['development_mean_R_separation']:.4f}")

    # =====================================================================
    # HOLDOUT
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

    holdout_trades, holdout_summary = holdout_test(
        df,
        selected_rule,
        close_bars,
        mae_bars,
    )

    print()
    print("HOLDOUT RECOVERY BOUNDARY")

    print(f"  MAE triggered     : {holdout_summary['mae_triggered']}")

    print(f"  Recovered         : {holdout_summary['recovered']}")

    print(f"  Failed            : {holdout_summary['failed']}")

    print(f"  Recovery rate     : {holdout_summary['recovery_rate']:.4f}")

    print(f"  Recovered WR     : {holdout_summary['recovered_win_rate']:.4f}")

    print(f"  Failed WR        : {holdout_summary['failed_win_rate']:.4f}")

    print(f"  WR separation     : {holdout_summary['win_rate_separation']:.4f}")

    print(f"  Recovered mean R  : {holdout_summary['recovered_mean_R']:.4f}")

    print(f"  Failed mean R     : {holdout_summary['failed_mean_R']:.4f}")

    print(f"  Mean R separation : {holdout_summary['mean_R_separation']:.4f}")

    print(f"  Recovered Total R : {holdout_summary['recovered_total_R']:.4f}")

    print(f"  Failed Total R    : {holdout_summary['failed_total_R']:.4f}")

    # =====================================================================
    # WINDOW BY WINDOW
    # =====================================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW HOLDOUT")
    print("=" * 110)

    window_df = window_analysis(holdout_trades)

    if len(window_df) > 0:
        print(window_df.to_string(index=False))

    # =====================================================================
    # RECOVERY LEVEL MAP
    # =====================================================================

    print()
    print("=" * 110)
    print("HOLDOUT RECOVERY LEVEL MAP")
    print("=" * 110)

    selected_mae = float(selected_rule["mae_threshold"])

    selected_deadline = int(selected_rule["deadline"])

    level_map = recovery_level_map(
        df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy(),
        close_bars,
        mae_bars,
        selected_mae,
        selected_deadline,
    )

    print(
        level_map[
            [
                "recovery_level",
                "mae_triggered",
                "recovered",
                "failed",
                "recovery_rate",
                "recovered_win_rate",
                "failed_win_rate",
                "win_rate_separation",
                "recovered_mean_R",
                "failed_mean_R",
                "mean_R_separation",
            ]
        ].to_string(index=False)
    )

    # =====================================================================
    # SAVE
    # =====================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = OUTPUT_DIR / "s22_recovery_boundary_development.csv"

    holdout_path = OUTPUT_DIR / "s22_recovery_boundary_holdout.csv"

    window_path = OUTPUT_DIR / "s22_recovery_boundary_by_window.csv"

    level_map_path = OUTPUT_DIR / "s22_recovery_boundary_level_map.csv"

    selected_path = OUTPUT_DIR / "s22_selected_recovery_boundary.csv"

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

    level_map.to_csv(
        level_map_path,
        index=False,
    )

    pd.DataFrame([selected_rule]).to_csv(
        selected_path,
        index=False,
    )

    # =====================================================================
    # FINAL
    # =====================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(development_path)

    print(holdout_path)

    print(window_path)

    print(level_map_path)

    print(selected_path)

    print()
    print("=" * 110)
    print("S22 RECOVERY BOUNDARY TEMPORAL OOS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
