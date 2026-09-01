from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# MEAN REVERSION — RESEARCH 02
# ============================================================
#
# PURPOSE
# -------
# Analyze the results produced by Research 01.
#
# Research 01 measured future price behavior conditional on
# current market characteristics.
#
# This script asks:
#
#   1. Is the effect statistically visible?
#   2. Is it consistent across OOS windows?
#   3. At which horizons does it appear?
#   4. Are positive and negative deviations symmetric?
#   5. Does the effect survive aggregation across windows?
#
# IMPORTANT
# ---------
# This script does NOT:
#
#   - create a trading strategy
#   - optimize entry thresholds
#   - optimize exits
#   - train ML models
#   - select parameters
#
# It is an analysis layer only.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CATEGORICAL_PATH = RESULTS_DIR / "research_01_categorical.csv"

CONTINUOUS_PATH = RESULTS_DIR / "research_01_continuous.csv"


# ============================================================
# HELPERS
# ============================================================


def safe_mean(series: pd.Series) -> float:
    """
    Return the mean of finite observations.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def consistency_ratio(
    series: pd.Series,
    positive: bool = True,
) -> float:
    """
    Measure the fraction of OOS windows in which the statistic
    has the expected sign.

    For positive=True:

        fraction(statistic > 0)

    For positive=False:

        fraction(statistic < 0)

    This is a descriptive consistency metric.

    It is NOT a statistical significance test.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty:
        return np.nan

    if positive:
        return float((values > 0).mean())

    return float((values < 0).mean())


# ============================================================
# LOAD RESULTS
# ============================================================


def load_results():
    """
    Load Research 01 outputs.
    """

    if not CATEGORICAL_PATH.exists():
        raise FileNotFoundError(
            f"Categorical research file not found:\n{CATEGORICAL_PATH}"
        )

    if not CONTINUOUS_PATH.exists():
        raise FileNotFoundError(
            f"Continuous research file not found:\n{CONTINUOUS_PATH}"
        )

    categorical = pd.read_csv(CATEGORICAL_PATH)

    continuous = pd.read_csv(CONTINUOUS_PATH)

    return categorical, continuous


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================


def analyze_categorical(
    categorical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze categorical conditions such as Z-score buckets.

    Produces:

        1. pooled summary
        2. OOS consistency summary
    """

    pooled_rows = []
    consistency_rows = []

    group_columns = [
        "condition",
        "condition_value",
        "horizon_bars",
    ]

    grouped = categorical.groupby(
        group_columns,
        dropna=True,
    )

    for (
        condition,
        condition_value,
        horizon,
    ), group in grouped:
        observations = group["observations"].sum()

        if observations <= 0:
            continue

        # ----------------------------------------------------
        # Pooled mean
        # ----------------------------------------------------
        #
        # Weight each OOS window by its number of observations.
        #

        mean_return = np.average(
            group["mean_return"],
            weights=group["observations"],
        )

        mean_displacement = np.average(
            group["mean_displacement_points"],
            weights=group["observations"],
        )

        mean_win_rate = np.average(
            group["win_rate_positive"],
            weights=group["observations"],
        )

        # ----------------------------------------------------
        # Median across windows
        # ----------------------------------------------------

        median_window_return = group["mean_return"].median()

        # ----------------------------------------------------
        # Window consistency
        # ----------------------------------------------------

        positive_windows = (group["mean_return"] > 0).mean()

        negative_windows = (group["mean_return"] < 0).mean()

        pooled_rows.append(
            {
                "condition": condition,
                "condition_value": condition_value,
                "horizon_bars": horizon,
                "total_observations": int(observations),
                "pooled_mean_return": (mean_return),
                "pooled_mean_displacement": (mean_displacement),
                "pooled_win_rate_positive": (mean_win_rate),
                "median_window_mean_return": (median_window_return),
            }
        )

        consistency_rows.append(
            {
                "condition": condition,
                "condition_value": condition_value,
                "horizon_bars": horizon,
                "windows_observed": len(group),
                "positive_return_windows": (int((group["mean_return"] > 0).sum())),
                "negative_return_windows": (int((group["mean_return"] < 0).sum())),
                "positive_window_ratio": (positive_windows),
                "negative_window_ratio": (negative_windows),
            }
        )

    return (
        pd.DataFrame(pooled_rows),
        pd.DataFrame(consistency_rows),
    )


# ============================================================
# EXTREME Z-SCORE ANALYSIS
# ============================================================


def analyze_zscore_symmetry(
    categorical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare extreme negative and positive Z-score buckets.

    For mean reversion we expect, if the phenomenon exists:

        negative extreme
            -> positive future return

        positive extreme
            -> negative future return

    This is a diagnostic symmetry test.
    """

    zscore_30 = categorical.loc[categorical["condition"] == "zscore_30"].copy()

    rows = []

    for horizon in sorted(zscore_30["horizon_bars"].unique()):
        subset = zscore_30.loc[zscore_30["horizon_bars"] == horizon]

        negative = subset.loc[
            subset["condition_value"].isin(
                [
                    "<=-2.5",
                    "-2.5:-2.0",
                ]
            )
        ]

        positive = subset.loc[
            subset["condition_value"].isin(
                [
                    "2.0:2.5",
                    ">=2.5",
                ]
            )
        ]

        if negative.empty or positive.empty:
            continue

        negative_return = np.average(
            negative["mean_return"],
            weights=negative["observations"],
        )

        positive_return = np.average(
            positive["mean_return"],
            weights=positive["observations"],
        )

        rows.append(
            {
                "horizon_bars": horizon,
                "negative_extreme_mean_return": (negative_return),
                "positive_extreme_mean_return": (positive_return),
                "negative_extreme_reverts": (negative_return > 0),
                "positive_extreme_reverts": (positive_return < 0),
                "symmetric_mean_reversion": (
                    negative_return > 0 and positive_return < 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CONTINUOUS FEATURE ANALYSIS
# ============================================================


def analyze_continuous(
    continuous: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze bottom/top 10% behavior for continuous features.

    For a mean-reverting variable we look for directional
    asymmetry:

        bottom 10%
            -> positive future return

        top 10%
            -> negative future return
    """

    rows = []

    grouped = continuous.groupby(
        [
            "feature",
            "horizon_bars",
        ],
        dropna=True,
    )

    for (
        feature,
        horizon,
    ), group in grouped:
        bottom = group.loc[group["condition"] == "bottom_10pct"]

        top = group.loc[group["condition"] == "top_10pct"]

        if bottom.empty or top.empty:
            continue

        bottom_return = safe_mean(bottom["mean_return"])

        top_return = safe_mean(top["mean_return"])

        rows.append(
            {
                "feature": feature,
                "horizon_bars": horizon,
                "bottom_10_mean_return": (bottom_return),
                "top_10_mean_return": (top_return),
                "bottom_10_reversion": (bottom_return > 0),
                "top_10_reversion": (top_return < 0),
                "symmetric_reversion": (bottom_return > 0 and top_return < 0),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# HORIZON ANALYSIS
# ============================================================


def analyze_horizons(
    categorical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine how the sign and magnitude of the effect evolve
    across future horizons.

    This is useful because a real short-term mean-reversion
    effect should not necessarily be equally strong at every
    horizon.
    """

    rows = []

    grouped = categorical.groupby(
        [
            "condition",
            "condition_value",
            "horizon_bars",
        ]
    )

    for (
        condition,
        condition_value,
        horizon,
    ), group in grouped:
        if group.empty:
            continue

        values = group["mean_return"].dropna()

        if values.empty:
            continue

        rows.append(
            {
                "condition": condition,
                "condition_value": condition_value,
                "horizon_bars": horizon,
                "mean_window_return": (values.mean()),
                "median_window_return": (values.median()),
                "min_window_return": (values.min()),
                "max_window_return": (values.max()),
                "positive_window_ratio": ((values > 0).mean()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — RESEARCH 02")
    print("=" * 100)

    print()
    print("ANALYZING RESEARCH 01")

    print("No strategy construction.")

    print("No optimization.")

    print("No machine learning.")

    print()

    # ========================================================
    # LOAD
    # ========================================================

    print("Loading Research 01 results...")

    categorical, continuous = load_results()

    print(f"Categorical rows: {len(categorical):,}")

    print(f"Continuous rows: {len(continuous):,}")

    # ========================================================
    # CATEGORICAL
    # ========================================================

    print()
    print("Analyzing categorical conditions...")

    (
        categorical_pooled,
        categorical_consistency,
    ) = analyze_categorical(categorical)

    # ========================================================
    # Z-SCORE
    # ========================================================

    print("Analyzing Z-score symmetry...")

    zscore_symmetry = analyze_zscore_symmetry(categorical)

    # ========================================================
    # CONTINUOUS
    # ========================================================

    print("Analyzing continuous features...")

    continuous_analysis = analyze_continuous(continuous)

    # ========================================================
    # HORIZONS
    # ========================================================

    print("Analyzing horizons...")

    horizon_analysis = analyze_horizons(categorical)

    # ========================================================
    # SAVE
    # ========================================================

    output_files = {
        "research_02_categorical_pooled.csv": categorical_pooled,
        "research_02_categorical_consistency.csv": categorical_consistency,
        "research_02_zscore_symmetry.csv": zscore_symmetry,
        "research_02_continuous.csv": continuous_analysis,
        "research_02_horizons.csv": horizon_analysis,
    }

    print()
    print("Saving results...")

    for filename, dataframe in output_files.items():
        path = RESULTS_DIR / filename

        dataframe.to_csv(
            path,
            index=False,
        )

        print(f"  {path}")

    # ========================================================
    # QUICK TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("RESEARCH 02 COMPLETE")
    print("=" * 100)

    print()

    if not zscore_symmetry.empty:
        print("Z-SCORE SYMMETRY")

        print(zscore_symmetry.to_string(index=False))

    print()

    if not continuous_analysis.empty:
        print("CONTINUOUS FEATURE SUMMARY")

        print(continuous_analysis.to_string(index=False))

    print()
    print("IMPORTANT:")

    print("These results are descriptive.")

    print("No strategy has been selected.")

    print("No threshold has been optimized.")

    print("XGBoost has not been used.")

    print("=" * 100)


if __name__ == "__main__":
    main()
