from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# S20 — RECOVERY LEVEL CONDITIONAL OOS
# =============================================================================
#
# Research question:
#
# After a significant adverse excursion (MAE), does recovery to a specific
# R-level distinguish winners from losers?
#
# We are NOT optimizing an execution rule yet.
#
# We are trying to discover the actual state transition:
#
#       MAE >= X R
#              |
#              v
#       recovery to Y R?
#          /       \
#        YES       NO
#         |         |
#      outcome    outcome
#
# The key idea is that recovery can happen at:
#
#   -0.50R
#   -0.40R
#   ...
#    0.00R
#   +0.10R
#   ...
#   +0.40R
#
# This tells us whether "partial recovery" itself contains information.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

RESULTS_DIR = ROOT / "src" / "research" / "results" / "s2_extended"


# =============================================================================
# FROZEN BENCHMARK
# =============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# =============================================================================
# MAE THRESHOLDS
# =============================================================================

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


# =============================================================================
# RECOVERY LEVELS
# =============================================================================

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


# =============================================================================
# POST-MAE HORIZONS
# =============================================================================

REQUESTED_RECOVERY_HORIZONS = [
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


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# HELPERS
# =============================================================================


def normalise_window(value) -> float:

    if pd.isna(value):
        return np.nan

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    digits = "".join(ch for ch in text if ch.isdigit())

    if digits:
        try:
            return float(digits)
        except ValueError:
            pass

    return np.nan


def detect_path_columns(
    df: pd.DataFrame,
    prefix: str,
) -> dict[int, str]:

    result = {}

    prefix_lower = prefix.lower()

    for column in df.columns:
        text = str(column)
        lower = text.lower()

        if not lower.startswith(prefix_lower):
            continue

        suffix = lower[len(prefix_lower) :]

        if suffix.endswith("r"):
            suffix = suffix[:-1]

        try:
            bar = int(suffix)
        except ValueError:
            continue

        result[bar] = column

    return dict(sorted(result.items()))


def numeric(value) -> float:

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return np.nan


def profit_factor(
    values: Iterable[float],
) -> float:

    arr = np.asarray(
        list(values),
        dtype=float,
    )

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    gross_profit = arr[arr > 0].sum()
    gross_loss = -arr[arr < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf

        return np.nan

    return float(gross_profit / gross_loss)


def max_drawdown(
    values: Iterable[float],
) -> float:

    arr = np.asarray(
        list(values),
        dtype=float,
    )

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    equity = np.cumsum(arr)

    running_max = np.maximum.accumulate(
        np.concatenate(
            [
                [0.0],
                equity,
            ]
        )
    )[1:]

    drawdown = equity - running_max

    return float(drawdown.min())


def metrics(
    df: pd.DataFrame,
) -> dict:

    # IMPORTANT:
    #
    # The observation dataset uses "final_R".
    # The previous version incorrectly looked for "_final_R".
    #

    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
        }

    r = pd.to_numeric(
        df["final_R"],
        errors="coerce",
    ).dropna()

    if r.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
        }

    wins = int((r > 0).sum())

    losses = int((r <= 0).sum())

    return {
        "trades": int(len(r)),
        "wins": wins,
        "losses": losses,
        "win_rate": float((r > 0).mean()),
        "mean_R": float(r.mean()),
        "total_R": float(r.sum()),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": max_drawdown(r),
    }


# =============================================================================
# CONDITIONAL METRICS
# =============================================================================


def conditional_metrics(
    df: pd.DataFrame,
    condition_column: str,
) -> dict:

    condition = df[condition_column].fillna(False).astype(bool)

    reached = df[condition]
    not_reached = df[~condition]

    reached_m = metrics(reached)
    not_reached_m = metrics(not_reached)

    if pd.notna(reached_m["win_rate"]) and pd.notna(not_reached_m["win_rate"]):
        wr_difference = reached_m["win_rate"] - not_reached_m["win_rate"]

    else:
        wr_difference = np.nan

    if pd.notna(reached_m["mean_R"]) and pd.notna(not_reached_m["mean_R"]):
        mean_difference = reached_m["mean_R"] - not_reached_m["mean_R"]

    else:
        mean_difference = np.nan

    return {
        "reached_trades": reached_m["trades"],
        "reached_wins": reached_m["wins"],
        "reached_losses": reached_m["losses"],
        "reached_pct": (len(reached) / len(df) if len(df) else np.nan),
        "reached_win_rate": reached_m["win_rate"],
        "reached_mean_R": reached_m["mean_R"],
        "reached_total_R": reached_m["total_R"],
        "reached_PF": reached_m["profit_factor"],
        "reached_DD": reached_m["max_drawdown_R"],
        "not_reached_trades": not_reached_m["trades"],
        "not_reached_wins": not_reached_m["wins"],
        "not_reached_losses": not_reached_m["losses"],
        "not_reached_pct": (len(not_reached) / len(df) if len(df) else np.nan),
        "not_reached_win_rate": not_reached_m["win_rate"],
        "not_reached_mean_R": not_reached_m["mean_R"],
        "not_reached_total_R": not_reached_m["total_R"],
        "not_reached_PF": not_reached_m["profit_factor"],
        "not_reached_DD": not_reached_m["max_drawdown_R"],
        "win_rate_difference": wr_difference,
        "mean_R_difference": mean_difference,
    }


# =============================================================================
# BUILD POST-MAE DATASET
# =============================================================================


def build_post_mae_dataset(
    df: pd.DataFrame,
    mae_cols: dict[int, str],
    close_cols: dict[int, str],
) -> pd.DataFrame:

    rows = []

    common_bars = sorted(set(mae_cols.keys()) & set(close_cols.keys()))

    if not common_bars:
        raise RuntimeError("No common MAE/Close path bars.")

    max_bar = max(common_bars)

    effective_horizons = [h for h in REQUESTED_RECOVERY_HORIZONS if h <= max_bar]

    print()
    print(f"Available path bars: {common_bars}")

    print(f"Requested horizons: {REQUESTED_RECOVERY_HORIZONS}")

    print(f"Effective horizons: {effective_horizons}")

    # -------------------------------------------------------------------------
    # Each source trade can generate multiple:
    #
    #   MAE threshold
    #       x
    #   crossing bar
    #       x
    #   post-MAE horizon
    #
    # observations.
    #
    # That is intentional.
    # -------------------------------------------------------------------------

    for source_index, row in df.iterrows():
        final_R = numeric(row["_final_R"])

        if not np.isfinite(final_R):
            continue

        window = row["_window"]

        # ---------------------------------------------------------------------
        # MAE path
        # ---------------------------------------------------------------------

        mae_path = {}

        for bar, column in mae_cols.items():
            value = numeric(row[column])

            if np.isfinite(value):
                mae_path[bar] = value

        # ---------------------------------------------------------------------
        # Close path
        # ---------------------------------------------------------------------

        close_path = {}

        for bar, column in close_cols.items():
            value = numeric(row[column])

            if np.isfinite(value):
                close_path[bar] = value

        if not mae_path or not close_path:
            continue

        # ---------------------------------------------------------------------
        # Every MAE threshold
        # ---------------------------------------------------------------------

        for mae_threshold in MAE_THRESHOLDS:
            crossing_bar = None

            for bar in sorted(mae_path):
                if mae_path[bar] >= mae_threshold:
                    crossing_bar = bar
                    break

            if crossing_bar is None:
                continue

            # -----------------------------------------------------------------
            # Need bars AFTER crossing.
            # -----------------------------------------------------------------

            for horizon in effective_horizons:
                target_bar = crossing_bar + horizon

                if target_bar > max_bar:
                    continue

                if target_bar not in close_path:
                    continue

                post_bars = [
                    bar for bar in common_bars if (crossing_bar < bar <= target_bar)
                ]

                if not post_bars:
                    continue

                post_closes = [
                    close_path[bar] for bar in post_bars if np.isfinite(close_path[bar])
                ]

                if not post_closes:
                    continue

                crossing_close = close_path.get(
                    crossing_bar,
                    np.nan,
                )

                close_at_horizon = close_path[target_bar]

                best_recovery = max(post_closes)

                worst_post_close = min(post_closes)

                if np.isfinite(crossing_close):
                    improvement = close_at_horizon - crossing_close

                else:
                    improvement = np.nan

                record = {
                    "source_index": source_index,
                    "window": window,
                    "mae_threshold": mae_threshold,
                    "crossing_bar": crossing_bar,
                    "horizon": horizon,
                    "target_bar": target_bar,
                    "crossing_mae_R": mae_path[crossing_bar],
                    "crossing_close_R": crossing_close,
                    "close_at_horizon_R": close_at_horizon,
                    "best_recovery_R": best_recovery,
                    "worst_post_close_R": worst_post_close,
                    "improvement_R": improvement,
                    # IMPORTANT:
                    # This is deliberately called final_R in the generated
                    # dataset. metrics() uses the same name.
                    "final_R": final_R,
                    "outcome": ("WIN" if final_R > 0 else "LOSS"),
                }

                # -------------------------------------------------------------
                # Recovery flags
                # -------------------------------------------------------------

                for level in RECOVERY_LEVELS:
                    record[f"reached_{level:+.2f}R"] = bool(best_recovery >= level)

                rows.append(record)

    return pd.DataFrame(rows)


# =============================================================================
# COHORT SUMMARY
# =============================================================================


def build_cohort_summary(
    recovery_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for threshold in MAE_THRESHOLDS:
        subset = recovery_df[recovery_df["mae_threshold"] == threshold]

        cohort = subset.sort_values(
            [
                "source_index",
                "horizon",
            ]
        ).drop_duplicates("source_index")

        m = metrics(cohort)

        rows.append(
            {
                "mae_threshold": threshold,
                **m,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CONDITIONAL RECOVERY ANALYSIS
# =============================================================================


def build_conditional_analysis(
    recovery_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    horizons = sorted(recovery_df["horizon"].dropna().unique())

    for threshold in MAE_THRESHOLDS:
        for horizon in horizons:
            subset = recovery_df[
                (recovery_df["mae_threshold"] == threshold)
                & (recovery_df["horizon"] == horizon)
            ].copy()

            if subset.empty:
                continue

            subset = subset.sort_values("source_index").drop_duplicates("source_index")

            for level in RECOVERY_LEVELS:
                condition_column = f"reached_{level:+.2f}R"

                result = conditional_metrics(
                    subset,
                    condition_column,
                )

                rows.append(
                    {
                        "mae_threshold": threshold,
                        "horizon": horizon,
                        "recovery_level": level,
                        **result,
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# TEMPORAL OOS
# =============================================================================


def build_temporal_oos(
    recovery_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, candidate in conditional_df.iterrows():
        threshold = candidate["mae_threshold"]

        horizon = candidate["horizon"]

        level = candidate["recovery_level"]

        subset = recovery_df[
            (recovery_df["mae_threshold"] == threshold)
            & (recovery_df["horizon"] == horizon)
        ].copy()

        subset = subset.sort_values("source_index").drop_duplicates("source_index")

        condition_column = f"reached_{level:+.2f}R"

        development = subset[subset["window"].isin(DEVELOPMENT_WINDOWS)]

        holdout = subset[subset["window"].isin(HOLDOUT_WINDOWS)]

        if development.empty:
            continue

        if holdout.empty:
            continue

        dev = conditional_metrics(
            development,
            condition_column,
        )

        oos = conditional_metrics(
            holdout,
            condition_column,
        )

        rows.append(
            {
                "mae_threshold": threshold,
                "horizon": horizon,
                "recovery_level": level,
                # Development
                "development_reached_trades": dev["reached_trades"],
                "development_reached_pct": dev["reached_pct"],
                "development_reached_WR": dev["reached_win_rate"],
                "development_not_reached_trades": dev["not_reached_trades"],
                "development_not_reached_WR": dev["not_reached_win_rate"],
                "development_WR_difference": dev["win_rate_difference"],
                "development_reached_mean_R": dev["reached_mean_R"],
                "development_not_reached_mean_R": dev["not_reached_mean_R"],
                "development_mean_R_difference": dev["mean_R_difference"],
                # Holdout
                "holdout_reached_trades": oos["reached_trades"],
                "holdout_reached_pct": oos["reached_pct"],
                "holdout_reached_WR": oos["reached_win_rate"],
                "holdout_not_reached_trades": oos["not_reached_trades"],
                "holdout_not_reached_WR": oos["not_reached_win_rate"],
                "holdout_WR_difference": oos["win_rate_difference"],
                "holdout_reached_mean_R": oos["reached_mean_R"],
                "holdout_not_reached_mean_R": oos["not_reached_mean_R"],
                "holdout_mean_R_difference": oos["mean_R_difference"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# RANK
# =============================================================================


def rank_candidates(
    oos_df: pd.DataFrame,
) -> pd.DataFrame:

    if oos_df.empty:
        return oos_df.copy()

    result = oos_df.copy()

    result["same_direction"] = (
        result["development_WR_difference"] * result["holdout_WR_difference"]
    ) > 0

    result["min_branch_pct"] = result[
        [
            "development_reached_pct",
            "holdout_reached_pct",
        ]
    ].min(axis=1)

    result["min_branch_trades"] = result[
        [
            "development_reached_trades",
            "development_not_reached_trades",
            "holdout_reached_trades",
            "holdout_not_reached_trades",
        ]
    ].min(axis=1)

    result["holdout_abs_WR_difference"] = result["holdout_WR_difference"].abs()

    result["robust_score"] = (
        result["holdout_abs_WR_difference"]
        * result["development_WR_difference"].abs()
        * result["min_branch_pct"]
        * result["same_direction"].astype(float)
    )

    return result.sort_values(
        [
            "same_direction",
            "robust_score",
            "holdout_abs_WR_difference",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )


# =============================================================================
# PRINT
# =============================================================================


def print_results(
    cohort_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
    oos_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 110)
    print("MAE COHORT SUMMARY")
    print("=" * 110)

    print(cohort_df.to_string(index=False))

    print()
    print("=" * 110)
    print("TOP CONDITIONAL RECOVERY STATES")
    print("=" * 110)

    if oos_df.empty:
        print("No temporal OOS observations available.")

        return

    ranked = rank_candidates(oos_df)

    display_columns = [
        "mae_threshold",
        "horizon",
        "recovery_level",
        "development_reached_trades",
        "development_reached_pct",
        "development_reached_WR",
        "development_not_reached_WR",
        "development_WR_difference",
        "holdout_reached_trades",
        "holdout_reached_pct",
        "holdout_reached_WR",
        "holdout_not_reached_WR",
        "holdout_WR_difference",
        "holdout_reached_mean_R",
        "holdout_not_reached_mean_R",
        "holdout_mean_R_difference",
    ]

    print(ranked[display_columns].head(40).to_string(index=False))

    # -------------------------------------------------------------------------
    # Strict sample-size filter
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("ROBUST OOS RECOVERY SEPARATORS")
    print("=" * 110)

    robust = ranked[
        (ranked["holdout_reached_trades"] >= 10)
        & (ranked["holdout_not_reached_trades"] >= 10)
    ].copy()

    if robust.empty:
        print("No candidate has >=10 trades in both holdout branches.")

    else:
        print(robust[display_columns].head(25).to_string(index=False))

    # -------------------------------------------------------------------------
    # Specifically show the 0R neighborhood.
    # This is important for the question:
    #
    # "If the trade comes back near breakeven, should we exit?"
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BREAKEVEN / NEAR-BREAKEVEN RECOVERY")
    print("=" * 110)

    near_zero = ranked[
        ranked["recovery_level"].isin(
            [
                -0.20,
                -0.10,
                0.00,
                0.10,
                0.20,
            ]
        )
    ].copy()

    print(near_zero[display_columns].head(30).to_string(index=False))


# =============================================================================
# SAVE
# =============================================================================


def save_results(
    recovery_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
    oos_df: pd.DataFrame,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "observations": RESULTS_DIR / "s20_recovery_level_observations.csv",
        "cohort": RESULTS_DIR / "s20_mae_cohort_summary.csv",
        "conditional": RESULTS_DIR / "s20_recovery_level_conditional.csv",
        "oos": RESULTS_DIR / "s20_recovery_level_temporal_oos.csv",
        "ranked": RESULTS_DIR / "s20_best_recovery_separators.csv",
    }

    recovery_df.to_csv(
        paths["observations"],
        index=False,
    )

    cohort_df.to_csv(
        paths["cohort"],
        index=False,
    )

    conditional_df.to_csv(
        paths["conditional"],
        index=False,
    )

    oos_df.to_csv(
        paths["oos"],
        index=False,
    )

    rank_candidates(oos_df).to_csv(
        paths["ranked"],
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    for path in paths.values():
        print(path)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S20 RECOVERY LEVEL CONDITIONAL OOS")
    print("=" * 110)

    print()
    print("Research question:")
    print(
        "After significant MAE, does recovery "
        "to a specific R-level distinguish "
        "winners from losers?"
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
    print("Requested recovery horizons:")
    print(f"  {REQUESTED_RECOVERY_HORIZONS}")

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

    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Trades loaded: {len(df)}")

    # =========================================================================
    # DETECT PATHS
    # =========================================================================

    mae_cols = detect_path_columns(
        df,
        "mae_",
    )

    close_cols = detect_path_columns(
        df,
        "close_",
    )

    if not mae_cols:
        raise RuntimeError("No MAE path columns found.")

    if not close_cols:
        raise RuntimeError("No CLOSE path columns found.")

    required = [
        "final_close_R",
        "window",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    print()
    print("Detected paths:")

    print(f"  MAE bars   : {len(mae_cols)}")

    print(f"  MAE range  : {min(mae_cols)} -> {max(mae_cols)}")

    print(f"  Close bars : {len(close_cols)}")

    print(f"  Close range: {min(close_cols)} -> {max(close_cols)}")

    print("  Final R    : final_close_R")

    print("  Window     : window")

    # =========================================================================
    # NORMALIZE
    # =========================================================================

    df = df.copy()

    df["_window"] = df["window"].map(normalise_window)

    df["_final_R"] = pd.to_numeric(
        df["final_close_R"],
        errors="coerce",
    )

    df = df[df["_window"].notna() & df["_final_R"].notna()].copy()

    # =========================================================================
    # BUILD
    # =========================================================================

    print()
    print("=" * 110)
    print("BUILDING POST-MAE RECOVERY DATASET")
    print("=" * 110)

    recovery_df = build_post_mae_dataset(
        df,
        mae_cols,
        close_cols,
    )

    print()
    print(f"Post-MAE observations: {len(recovery_df)}")

    if recovery_df.empty:
        raise RuntimeError("No post-MAE recovery observations were created.")

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    cohort_df = build_cohort_summary(recovery_df)

    print()
    print("=" * 110)
    print("RUNNING CONDITIONAL RECOVERY ANALYSIS")
    print("=" * 110)

    conditional_df = build_conditional_analysis(recovery_df)

    print()
    print("=" * 110)
    print("RUNNING TEMPORAL OOS ANALYSIS")
    print("=" * 110)

    oos_df = build_temporal_oos(
        recovery_df,
        conditional_df,
    )

    # =========================================================================
    # RESULTS
    # =========================================================================

    print_results(
        cohort_df,
        conditional_df,
        oos_df,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_results(
        recovery_df,
        cohort_df,
        conditional_df,
        oos_df,
    )

    print()
    print("=" * 110)
    print("S20 RECOVERY LEVEL CONDITIONAL OOS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
