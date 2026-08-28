from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# S19 POST-MAE STATE CLASSIFICATION
# =============================================================================
#
# PURPOSE
# -------
# Determine whether the path AFTER an adverse MAE event can distinguish:
#
#   1. trades that eventually recover
#   2. trades that eventually fail
#
# This is a DISCOVERY test.
#
# It does NOT yet implement a trading filter.
#
# The objective is to identify robust post-MAE state variables that can later
# be converted into a simple OOS trading rule.
#
# IMPORTANT:
# We deliberately avoid optimizing the final strategy here.
# We first study the mechanism.
#
#
# CORE QUESTION
# -------------
#
# After:
#
#       MAE >= X R
#
# what happens during the next N bars?
#
# Examples:
#
#   - Does price recover to -0.10R?
#   - Does it recover to 0R?
#   - Does it recover to +0.10R?
#   - How quickly?
#   - Does it make a new adverse low?
#   - Does it generate positive MFE?
#   - Does the close improve or deteriorate?
#
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
# MAE boundaries
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
# Post-MAE observation horizons
# -----------------------------------------------------------------------------

POST_MAE_HORIZONS = [
    1,
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
# Recovery levels
#
# These are NOT trading exits yet.
#
# They are simply state markers.
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
# Development / holdout split
# -----------------------------------------------------------------------------

DEVELOPMENT_WINDOWS = list(range(1, 12))

HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# HELPERS
# =============================================================================


def safe_float(value) -> float:

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return np.nan


def normalise_window(value):

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


def detect_numbered_columns(
    df: pd.DataFrame,
    prefix: str,
) -> Dict[int, str]:

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)R$",
        re.IGNORECASE,
    )

    result = {}

    for column in df.columns:
        match = pattern.match(str(column))

        if match:
            bar = int(match.group(1))

            result[bar] = column

    return dict(sorted(result.items()))


def detect_final_r_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = [
        "final_close_R",
        "net_R_path",
        "net_R",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def detect_window_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = [
        "window",
        "window_id",
        "validation_window",
        "walk_forward_window",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


# =============================================================================
# METRICS
# =============================================================================


def profit_factor(
    values: pd.Series,
) -> float:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    gross_profit = values[values > 0].sum()

    gross_loss = -values[values < 0].sum()

    if gross_loss <= 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return float(gross_profit / gross_loss)


def max_drawdown(
    values: pd.Series,
) -> float:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    equity = values.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    return float(drawdown.min())


# =============================================================================
# PATH EXTRACTION
# =============================================================================


def get_path(
    row: pd.Series,
    columns: Dict[int, str],
) -> Dict[int, float]:

    result = {}

    for bar, column in columns.items():
        value = safe_float(row.get(column))

        if np.isfinite(value):
            result[bar] = value

    return result


# =============================================================================
# MAE CROSSING
# =============================================================================


def find_mae_crossing(
    row: pd.Series,
    mae_cols: Dict[int, str],
    threshold: float,
) -> Optional[int]:

    for bar, column in sorted(mae_cols.items()):
        mae = safe_float(row.get(column))

        if not np.isfinite(mae):
            continue

        if mae >= threshold:
            return bar

    return None


# =============================================================================
# POST-MAE STATE ANALYSIS
# =============================================================================


def analyse_trade_state(
    row: pd.Series,
    mae_cols: Dict[int, str],
    close_cols: Dict[int, str],
    mae_threshold: float,
) -> Optional[Dict]:

    trigger_bar = find_mae_crossing(
        row,
        mae_cols,
        mae_threshold,
    )

    if trigger_bar is None:
        return None

    final_r = safe_float(row["_final_R"])

    if not np.isfinite(final_r):
        return None

    close_path = get_path(
        row,
        close_cols,
    )

    mae_path = get_path(
        row,
        mae_cols,
    )

    # -------------------------------------------------------------------------
    # State observations are made AFTER the MAE trigger.
    #
    # We start at trigger_bar and then observe subsequent bars.
    # -------------------------------------------------------------------------

    available_bars = sorted(set(close_path.keys()) & set(mae_path.keys()))

    post_bars = [bar for bar in available_bars if bar >= trigger_bar]

    if not post_bars:
        return None

    # -------------------------------------------------------------------------
    # Initial post-MAE state
    # -------------------------------------------------------------------------

    trigger_close = close_path.get(
        trigger_bar,
        np.nan,
    )

    trigger_mae = mae_path.get(
        trigger_bar,
        np.nan,
    )

    result = {
        "trigger_bar": trigger_bar,
        "trigger_mae_R": trigger_mae,
        "trigger_close_R": trigger_close,
        "final_R": final_r,
    }

    # -------------------------------------------------------------------------
    # For every horizon, measure the state after the trigger.
    # -------------------------------------------------------------------------

    for horizon in POST_MAE_HORIZONS:
        max_bar = trigger_bar + horizon

        horizon_bars = [bar for bar in post_bars if bar <= max_bar]

        if not horizon_bars:
            continue

        closes = [
            close_path[bar]
            for bar in horizon_bars
            if np.isfinite(
                close_path.get(
                    bar,
                    np.nan,
                )
            )
        ]

        maes = [
            mae_path[bar]
            for bar in horizon_bars
            if np.isfinite(
                mae_path.get(
                    bar,
                    np.nan,
                )
            )
        ]

        if not closes:
            continue

        # ---------------------------------------------------------------------
        # Close at horizon
        # ---------------------------------------------------------------------

        close_at_horizon = closes[-1]

        # ---------------------------------------------------------------------
        # Best recovery after MAE trigger
        # ---------------------------------------------------------------------

        max_close = max(closes)

        # ---------------------------------------------------------------------
        # Worst additional adverse excursion
        # ---------------------------------------------------------------------

        max_mae_after = max(maes)

        # ---------------------------------------------------------------------
        # Improvement from trigger close
        # ---------------------------------------------------------------------

        close_improvement = close_at_horizon - trigger_close

        best_recovery_from_trigger = max_close - trigger_close

        # ---------------------------------------------------------------------
        # New adverse movement AFTER trigger.
        #
        # This is especially important.
        #
        # A trade that touches 0.70R MAE and immediately recovers is very
        # different from a trade that touches 0.70R and then continues to
        # 0.90R / 1.00R.
        # ---------------------------------------------------------------------

        additional_mae = max_mae_after - trigger_mae

        # ---------------------------------------------------------------------
        # Recovery level flags
        # ---------------------------------------------------------------------

        for recovery_level in RECOVERY_LEVELS:
            key = f"h{horizon}_recovered_{recovery_level:+.2f}R"

            result[key] = bool(max_close >= recovery_level)

        # ---------------------------------------------------------------------
        # Store numerical state
        # ---------------------------------------------------------------------

        result[f"h{horizon}_close_R"] = close_at_horizon

        result[f"h{horizon}_max_close_R"] = max_close

        result[f"h{horizon}_max_MAE_R"] = max_mae_after

        result[f"h{horizon}_close_improvement_R"] = close_improvement

        result[f"h{horizon}_best_recovery_R"] = best_recovery_from_trigger

        result[f"h{horizon}_additional_MAE_R"] = additional_mae

        # ---------------------------------------------------------------------
        # Did price improve?
        # ---------------------------------------------------------------------

        result[f"h{horizon}_improved"] = bool(close_at_horizon > trigger_close)

        result[f"h{horizon}_positive"] = bool(close_at_horizon > 0)

        result[f"h{horizon}_breakeven"] = bool(max_close >= 0)

    return result


# =============================================================================
# BUILD STATE DATASET
# =============================================================================


def build_state_dataset(
    df: pd.DataFrame,
    mae_cols: Dict[int, str],
    close_cols: Dict[int, str],
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 110)
    print("BUILDING POST-MAE STATE DATASET")
    print("=" * 110)

    total = len(df) * len(MAE_THRESHOLDS)

    processed = 0

    for index, row in df.iterrows():
        for mae_threshold in MAE_THRESHOLDS:
            processed += 1

            state = analyse_trade_state(
                row,
                mae_cols,
                close_cols,
                mae_threshold,
            )

            if state is None:
                continue

            state["_source_index"] = index

            state["mae_threshold"] = mae_threshold

            state["_window"] = row["_window"]

            rows.append(state)

            if processed % 1000 == 0 or processed == total:
                print(f"Processing {processed}/{total}...")

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No post-MAE state observations were generated.")

    return result


# =============================================================================
# COHORT CLASSIFICATION
# =============================================================================


def classify_outcome(
    final_r: float,
) -> str:

    if final_r > 0:
        return "WIN"

    return "LOSS"


# =============================================================================
# RECOVERY PROBABILITY ANALYSIS
# =============================================================================


def recovery_probability_analysis(
    state_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 110)
    print("RECOVERY PROBABILITY BY POST-MAE STATE")
    print("=" * 110)

    for mae_threshold in MAE_THRESHOLDS:
        subset = state_df[state_df["mae_threshold"] == mae_threshold].copy()

        if subset.empty:
            continue

        subset["outcome"] = subset["final_R"].apply(classify_outcome)

        total = len(subset)

        wins = int((subset["outcome"] == "WIN").sum())

        losses = total - wins

        print()
        print(f"MAE >= {mae_threshold:.2f}R")

        print(f"  Observations: {total}")

        print(f"  Winners     : {wins}")

        print(f"  Losers      : {losses}")

        for horizon in POST_MAE_HORIZONS:
            for recovery_level in RECOVERY_LEVELS:
                column = f"h{horizon}_recovered_{recovery_level:+.2f}R"

                if column not in subset.columns:
                    continue

                condition = subset[column].fillna(False).astype(bool)

                reached = subset[condition]

                not_reached = subset[~condition]

                if reached.empty:
                    continue

                reached_win_rate = (reached["final_R"] > 0).mean()

                not_reached_win_rate = (
                    (not_reached["final_R"] > 0).mean() if not_reached.empty else np.nan
                )

                rows.append(
                    {
                        "mae_threshold": mae_threshold,
                        "horizon": horizon,
                        "recovery_level": recovery_level,
                        "observations": total,
                        "reached_trades": len(reached),
                        "not_reached_trades": len(not_reached),
                        "reached_pct": (len(reached) / total),
                        "reached_win_rate": (reached_win_rate),
                        "not_reached_win_rate": (not_reached_win_rate),
                        "win_rate_difference": (
                            reached_win_rate - not_reached_win_rate
                            if np.isfinite(not_reached_win_rate)
                            else np.nan
                        ),
                        "reached_mean_final_R": (reached["final_R"].mean()),
                        "not_reached_mean_final_R": (
                            not_reached["final_R"].mean()
                            if not_reached.empty
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# WIN VS LOSS STATE COMPARISON
# =============================================================================


def win_loss_state_comparison(
    state_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 110)
    print("WIN VS LOSS POST-MAE STATE COMPARISON")
    print("=" * 110)

    for mae_threshold in MAE_THRESHOLDS:
        subset = state_df[state_df["mae_threshold"] == mae_threshold].copy()

        if subset.empty:
            continue

        winners = subset[subset["final_R"] > 0]

        losers = subset[subset["final_R"] <= 0]

        if winners.empty or losers.empty:
            continue

        for horizon in POST_MAE_HORIZONS:
            numeric_columns = [
                f"h{horizon}_close_R",
                f"h{horizon}_max_close_R",
                f"h{horizon}_max_MAE_R",
                f"h{horizon}_close_improvement_R",
                f"h{horizon}_best_recovery_R",
                f"h{horizon}_additional_MAE_R",
            ]

            for column in numeric_columns:
                if column not in subset.columns:
                    continue

                winner_mean = (
                    pd.to_numeric(
                        winners[column],
                        errors="coerce",
                    )
                    .dropna()
                    .mean()
                )

                loser_mean = (
                    pd.to_numeric(
                        losers[column],
                        errors="coerce",
                    )
                    .dropna()
                    .mean()
                )

                if not (np.isfinite(winner_mean) and np.isfinite(loser_mean)):
                    continue

                rows.append(
                    {
                        "mae_threshold": (mae_threshold),
                        "horizon": horizon,
                        "feature": column,
                        "winner_mean": winner_mean,
                        "loser_mean": loser_mean,
                        "winner_minus_loser": (winner_mean - loser_mean),
                        "winner_n": len(winners),
                        "loser_n": len(losers),
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        by="winner_minus_loser",
        ascending=False,
    )


# =============================================================================
# CONTINUATION VS RECOVERY STATE
# =============================================================================


def continuation_recovery_analysis(
    state_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 110)
    print("CONTINUATION VS RECOVERY STATE")
    print("=" * 110)

    for mae_threshold in MAE_THRESHOLDS:
        subset = state_df[state_df["mae_threshold"] == mae_threshold].copy()

        if subset.empty:
            continue

        for horizon in POST_MAE_HORIZONS:
            close_col = f"h{horizon}_close_R"

            improvement_col = f"h{horizon}_close_improvement_R"

            additional_mae_col = f"h{horizon}_additional_MAE_R"

            best_recovery_col = f"h{horizon}_best_recovery_R"

            if close_col not in subset.columns or improvement_col not in subset.columns:
                continue

            # -----------------------------------------------------------------
            # Define simple state buckets.
            #
            # RECOVERY:
            # close improved versus trigger.
            #
            # CONTINUATION:
            # close deteriorated versus trigger.
            #
            # FLAT:
            # essentially unchanged.
            # -----------------------------------------------------------------

            improvement = pd.to_numeric(
                subset[improvement_col],
                errors="coerce",
            )

            subset["state"] = np.select(
                [
                    improvement > 0.05,
                    improvement < -0.05,
                ],
                [
                    "RECOVERY",
                    "CONTINUATION",
                ],
                default="FLAT",
            )

            for state in [
                "RECOVERY",
                "CONTINUATION",
                "FLAT",
            ]:
                state_subset = subset[subset["state"] == state]

                if state_subset.empty:
                    continue

                rows.append(
                    {
                        "mae_threshold": (mae_threshold),
                        "horizon": horizon,
                        "state": state,
                        "trades": len(state_subset),
                        "pct_of_cohort": (len(state_subset) / len(subset)),
                        "win_rate": (state_subset["final_R"] > 0).mean(),
                        "mean_final_R": (state_subset["final_R"].mean()),
                        "median_final_R": (state_subset["final_R"].median()),
                        "mean_close_R": (
                            pd.to_numeric(
                                state_subset[close_col],
                                errors="coerce",
                            ).mean()
                        ),
                        "mean_additional_MAE_R": (
                            pd.to_numeric(
                                state_subset[additional_mae_col],
                                errors="coerce",
                            ).mean()
                            if additional_mae_col in state_subset.columns
                            else np.nan
                        ),
                        "mean_best_recovery_R": (
                            pd.to_numeric(
                                state_subset[best_recovery_col],
                                errors="coerce",
                            ).mean()
                            if best_recovery_col in state_subset.columns
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# TEMPORAL OOS STATE CHECK
# =============================================================================


def temporal_oos_state_check(
    state_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 110)
    print("TEMPORAL OOS STATE CHECK")
    print("=" * 110)

    for dataset_name, windows in [
        (
            "DEVELOPMENT",
            DEVELOPMENT_WINDOWS,
        ),
        (
            "HOLDOUT",
            HOLDOUT_WINDOWS,
        ),
    ]:
        subset_dataset = state_df[state_df["_window"].isin(windows)].copy()

        if subset_dataset.empty:
            continue

        for mae_threshold in MAE_THRESHOLDS:
            subset = subset_dataset[subset_dataset["mae_threshold"] == mae_threshold]

            if subset.empty:
                continue

            for horizon in POST_MAE_HORIZONS:
                for recovery_level in RECOVERY_LEVELS:
                    column = f"h{horizon}_recovered_{recovery_level:+.2f}R"

                    if column not in subset.columns:
                        continue

                    condition = subset[column].fillna(False).astype(bool)

                    reached = subset[condition]

                    not_reached = subset[~condition]

                    if reached.empty:
                        continue

                    reached_win_rate = (reached["final_R"] > 0).mean()

                    not_reached_win_rate = (
                        (not_reached["final_R"] > 0).mean()
                        if not_reached.empty
                        else np.nan
                    )

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "mae_threshold": (mae_threshold),
                            "horizon": horizon,
                            "recovery_level": (recovery_level),
                            "trades": len(subset),
                            "reached_trades": len(reached),
                            "reached_pct": (len(reached) / len(subset)),
                            "reached_win_rate": (reached_win_rate),
                            "not_reached_win_rate": (not_reached_win_rate),
                            "win_rate_difference": (
                                reached_win_rate - not_reached_win_rate
                                if np.isfinite(not_reached_win_rate)
                                else np.nan
                            ),
                            "reached_mean_final_R": (reached["final_R"].mean()),
                            "not_reached_mean_final_R": (
                                not_reached["final_R"].mean()
                                if not_reached.empty
                                else np.nan
                            ),
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# BEST STATE SEPARATORS
# =============================================================================


def identify_best_state_separators(
    temporal_df: pd.DataFrame,
) -> pd.DataFrame:

    if temporal_df.empty:
        return pd.DataFrame()

    development = temporal_df[temporal_df["dataset"] == "DEVELOPMENT"].copy()

    holdout = temporal_df[temporal_df["dataset"] == "HOLDOUT"].copy()

    if development.empty:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # We want states that:
    #
    # 1. Have enough observations
    # 2. Strongly separate win rate
    # 3. Do not completely collapse OOS
    #
    # This is still discovery, NOT final optimization.
    # -------------------------------------------------------------------------

    development = development[development["reached_trades"] >= 10].copy()

    if development.empty:
        return pd.DataFrame()

    development = development.sort_values(
        by="win_rate_difference",
        ascending=False,
    )

    development = development.head(50)

    rows = []

    for _, dev_row in development.iterrows():
        mask = (
            (holdout["mae_threshold"] == dev_row["mae_threshold"])
            & (holdout["horizon"] == dev_row["horizon"])
            & (holdout["recovery_level"] == dev_row["recovery_level"])
        )

        matching = holdout[mask]

        if matching.empty:
            oos_difference = np.nan
            oos_reached_wr = np.nan
            oos_reached_pct = np.nan

        else:
            row = matching.iloc[0]

            oos_difference = row["win_rate_difference"]

            oos_reached_wr = row["reached_win_rate"]

            oos_reached_pct = row["reached_pct"]

        rows.append(
            {
                "mae_threshold": (dev_row["mae_threshold"]),
                "horizon": (dev_row["horizon"]),
                "recovery_level": (dev_row["recovery_level"]),
                "development_difference": (dev_row["win_rate_difference"]),
                "development_reached_pct": (dev_row["reached_pct"]),
                "development_reached_wr": (dev_row["reached_win_rate"]),
                "holdout_difference": (oos_difference),
                "holdout_reached_pct": (oos_reached_pct),
                "holdout_reached_wr": (oos_reached_wr),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        by=[
            "holdout_difference",
            "development_difference",
        ],
        ascending=False,
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S19 POST-MAE STATE CLASSIFICATION")
    print("=" * 110)

    print()
    print("Research question:")
    print(
        "After a trade reaches a significant MAE, "
        "can its subsequent path distinguish recovery "
        "from failure?"
    )

    print()
    print("MAE thresholds:")
    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))

    print()
    print("Recovery state levels:")
    print("  " + ", ".join(f"{x:+.2f}R" for x in RECOVERY_LEVELS))

    print()
    print("Post-MAE horizons:")
    print(f"  {POST_MAE_HORIZONS}")

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
    # Detect paths
    # -------------------------------------------------------------------------

    mae_cols = detect_numbered_columns(
        df,
        "mae",
    )

    close_cols = detect_numbered_columns(
        df,
        "close",
    )

    if not mae_cols:
        raise RuntimeError("No MAE path columns found. Expected mae_1R, mae_2R, ...")

    if not close_cols:
        raise RuntimeError(
            "No close path columns found. Expected close_1R, close_2R, ..."
        )

    final_r_col = detect_final_r_column(df)

    if final_r_col is None:
        raise RuntimeError("No final R column found.")

    window_col = detect_window_column(df)

    if window_col is None:
        raise RuntimeError("No window column found.")

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_cols)}")
    print(f"  MAE range  : {min(mae_cols)} -> {max(mae_cols)}")
    print(f"  Close bars : {len(close_cols)}")
    print(f"  Close range: {min(close_cols)} -> {max(close_cols)}")
    print(f"  Final R    : {final_r_col}")
    print(f"  Window     : {window_col}")

    # -------------------------------------------------------------------------
    # Normalize
    # -------------------------------------------------------------------------

    df["_window"] = df[window_col].map(normalise_window)

    df["_final_R"] = pd.to_numeric(
        df[final_r_col],
        errors="coerce",
    )

    df = df[df["_window"].notna() & df["_final_R"].notna()].copy()

    df["_window"] = df["_window"].astype(int)

    # -------------------------------------------------------------------------
    # Build state dataset
    # -------------------------------------------------------------------------

    state_df = build_state_dataset(
        df,
        mae_cols,
        close_cols,
    )

    print()
    print(f"Post-MAE observations: {len(state_df)}")

    # -------------------------------------------------------------------------
    # Basic cohort summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("MAE COHORT SIZES")
    print("=" * 110)

    cohort_rows = []

    for threshold in MAE_THRESHOLDS:
        subset = state_df[state_df["mae_threshold"] == threshold]

        winners = subset[subset["final_R"] > 0]

        losers = subset[subset["final_R"] <= 0]

        row = {
            "mae_threshold": threshold,
            "trades": len(subset),
            "wins": len(winners),
            "losses": len(losers),
            "win_rate": (
                (subset["final_R"] > 0).mean() if not subset.empty else np.nan
            ),
            "mean_final_R": (subset["final_R"].mean() if not subset.empty else np.nan),
        }

        cohort_rows.append(row)

    cohort_df = pd.DataFrame(cohort_rows)

    print(cohort_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Recovery probability
    # -------------------------------------------------------------------------

    recovery_df = recovery_probability_analysis(state_df)

    # -------------------------------------------------------------------------
    # Winner vs loser
    # -------------------------------------------------------------------------

    comparison_df = win_loss_state_comparison(state_df)

    # -------------------------------------------------------------------------
    # Recovery vs continuation
    # -------------------------------------------------------------------------

    state_comparison_df = continuation_recovery_analysis(state_df)

    # -------------------------------------------------------------------------
    # Temporal OOS
    # -------------------------------------------------------------------------

    temporal_df = temporal_oos_state_check(state_df)

    # -------------------------------------------------------------------------
    # Best separators
    # -------------------------------------------------------------------------

    separators_df = identify_best_state_separators(temporal_df)

    # -------------------------------------------------------------------------
    # Print key findings
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("TOP DEVELOPMENT STATE SEPARATORS")
    print("=" * 110)

    if separators_df.empty:
        print("No robust state separators found.")

    else:
        print(separators_df.head(20).to_string(index=False))

    print()
    print("=" * 110)
    print("TOP WIN VS LOSS FEATURES")
    print("=" * 110)

    if comparison_df.empty:
        print("No comparison features available.")

    else:
        print(comparison_df.head(30).to_string(index=False))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_path = OUTPUT_DIR / "s19_post_mae_state_observations.csv"

    cohort_path = OUTPUT_DIR / "s19_mae_cohort_summary.csv"

    recovery_path = OUTPUT_DIR / "s19_mae_recovery_probability.csv"

    comparison_path = OUTPUT_DIR / "s19_win_loss_state_comparison.csv"

    state_comparison_path = OUTPUT_DIR / "s19_recovery_continuation_states.csv"

    temporal_path = OUTPUT_DIR / "s19_temporal_oos_state_check.csv"

    separators_path = OUTPUT_DIR / "s19_best_state_separators.csv"

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    state_df.to_csv(
        state_path,
        index=False,
    )

    cohort_df.to_csv(
        cohort_path,
        index=False,
    )

    recovery_df.to_csv(
        recovery_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    state_comparison_df.to_csv(
        state_comparison_path,
        index=False,
    )

    temporal_df.to_csv(
        temporal_path,
        index=False,
    )

    separators_df.to_csv(
        separators_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(state_path)
    print(cohort_path)
    print(recovery_path)
    print(comparison_path)
    print(state_comparison_path)
    print(temporal_path)
    print(separators_path)

    print()
    print("=" * 110)
    print("S19 POST-MAE STATE CLASSIFICATION COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
