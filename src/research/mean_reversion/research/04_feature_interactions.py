from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# ============================================================
# MEAN REVERSION — RESEARCH 04
# ============================================================
#
# PURPOSE
# -------
# Investigate whether the mean-reversion effect depends on
# combinations of existing features.
#
# Research 03 showed an important asymmetry:
#
#   Negative Z-score extreme
#       -> meaningful LONG reversion
#
#   Positive Z-score extreme
#       -> much weaker / inconsistent SHORT reversion
#
# Therefore this research focuses primarily on LONG
# mean-reversion events.
#
# We investigate combinations such as:
#
#   Z-score
#       +
#   VWAP distance
#
#   Z-score
#       +
#   volatility regime
#
#   Z-score
#       +
#   OU residual
#
#   Z-score
#       +
#   half-life
#
# IMPORTANT
# ---------
# This is still exploratory research.
#
# NO:
#
#   - strategy construction
#   - stop/target optimization
#   - parameter optimization
#   - machine learning
#
# All measurements are performed independently inside the
# existing 22 OOS windows.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RESEARCH HORIZONS
# ============================================================

HORIZONS = (
    1,
    3,
    5,
    10,
    20,
    30,
    60,
)


# ============================================================
# BASE EVENT
# ============================================================
#
# Research 03 used +/- 1.5 only as a descriptive extreme
# region.
#
# We preserve that same definition here.
#
# It is NOT a proposed strategy threshold.
#
# ============================================================

Z_THRESHOLD = 1.5


# ============================================================
# COST MODEL
# ============================================================

MNQ_POINT_VALUE = 2.00

TOPSTEP_MNQ_RT_FEE_USD = 1.22

SLIPPAGE_POINTS_PER_SIDE = 0.25

TOTAL_SLIPPAGE_POINTS = 2.0 * SLIPPAGE_POINTS_PER_SIDE

TOPSTEP_FEE_POINTS = TOPSTEP_MNQ_RT_FEE_USD / MNQ_POINT_VALUE

TOTAL_COST_POINTS = TOTAL_SLIPPAGE_POINTS + TOPSTEP_FEE_POINTS


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    df: pd.DataFrame,
):
    """
    Reproduce the project's standard:

        2 years training
        3 months OOS

    structure.

    No model is trained here.
    """

    start = df.index.min()
    end = df.index.max()

    validation_start = start + pd.DateOffset(years=2)

    windows = []

    while validation_start < end:
        validation_end = min(
            validation_start + pd.DateOffset(months=3),
            end,
        )

        train_start = validation_start - pd.DateOffset(years=2)

        windows.append(
            (
                train_start,
                validation_start,
                validation_end,
            )
        )

        validation_start += pd.DateOffset(months=3)

    return windows


# ============================================================
# PREPARE RTH
# ============================================================


def prepare_rth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
        utc=True,
    )

    timestamps = timestamps.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = timestamps

    df = df.loc[df["market_period"] == "RTH"].copy()

    df = df.sort_values("_timestamp_et")

    df = df.set_index("_timestamp_et")

    df.index.name = "timestamp_et"

    if "session_date" in df.columns:
        df["_session_id"] = df["session_date"].astype(str)

    else:
        df["_session_id"] = df.index.date

    return df


# ============================================================
# FUTURE RETURN
# ============================================================


def add_future_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add future close-to-close returns for every research
    horizon.

    These are descriptive outcomes, not trading returns.
    """

    df = df.copy()

    for horizon in HORIZONS:
        future_close = df["close"].shift(-horizon)

        df[f"future_return_{horizon}"] = future_close / df["close"] - 1.0

        df[f"future_points_{horizon}"] = future_close - df["close"]

    return df


# ============================================================
# FEATURE BUCKETS
# ============================================================


def add_research_buckets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create coarse descriptive buckets.

    IMPORTANT:
    These are deliberately broad.

    We are trying to discover whether interactions exist,
    not optimize exact thresholds.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Z-score magnitude
    # --------------------------------------------------------

    df["z30_bucket"] = pd.cut(
        df["zscore_30"],
        bins=[
            -np.inf,
            -2.5,
            -2.0,
            -1.5,
            0.0,
            np.inf,
        ],
        labels=[
            "<=-2.5",
            "-2.5:-2.0",
            "-2.0:-1.5",
            "-1.5:0",
            ">0",
        ],
    )

    # --------------------------------------------------------
    # VWAP distance
    # --------------------------------------------------------
    #
    # We intentionally use sign and broad magnitude rather
    # than optimizing a specific cutoff.
    #

    df["vwap_distance_bucket"] = pd.cut(
        df["normalized_vwap_distance"],
        bins=[
            -np.inf,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            np.inf,
        ],
        labels=[
            "<=-2",
            "-2:-1",
            "-1:0",
            "0:1",
            "1:2",
            ">=2",
        ],
    )

    # --------------------------------------------------------
    # OU residual
    # --------------------------------------------------------

    df["ou_residual_bucket"] = pd.cut(
        df["normalized_ou_residual_30"],
        bins=[
            -np.inf,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            np.inf,
        ],
        labels=[
            "<=-2",
            "-2:-1",
            "-1:0",
            "0:1",
            "1:2",
            ">=2",
        ],
    )

    # --------------------------------------------------------
    # Half-life
    # --------------------------------------------------------

    df["half_life_bucket"] = pd.cut(
        df["half_life_30"],
        bins=[
            -np.inf,
            5.0,
            10.0,
            20.0,
            40.0,
            np.inf,
        ],
        labels=[
            "<=5",
            "5:10",
            "10:20",
            "20:40",
            ">40",
        ],
    )

    # --------------------------------------------------------
    # Short/long volatility relationship
    # --------------------------------------------------------

    df["vol_ratio_bucket"] = pd.cut(
        df["vol_ratio_5_30"],
        bins=[
            -np.inf,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            np.inf,
        ],
        labels=[
            "<=0.5",
            "0.5:0.75",
            "0.75:1.0",
            "1.0:1.25",
            "1.25:1.5",
            ">1.5",
        ],
    )

    return df


# ============================================================
# INTERACTION ANALYSIS
# ============================================================


def analyze_interaction(
    validation: pd.DataFrame,
    window_number: int,
    feature_a: str,
    feature_b: str,
    bucket_a: str,
    bucket_b: str,
) -> pd.DataFrame:
    """
    Measure future performance for combinations of two
    conditions.

    We only analyze negative Z-score extremes because Research
    03 showed the strongest economic evidence on that side.

    The base event is:

        zscore_30 <= -1.5

    Then we condition on the second feature.
    """

    base = validation.loc[validation["zscore_30"] <= -Z_THRESHOLD].copy()

    if base.empty:
        return pd.DataFrame()

    rows = []

    grouped = base.groupby(
        [
            feature_a,
            feature_b,
        ],
        observed=True,
        dropna=True,
    )

    for (
        value_a,
        value_b,
    ), group in grouped:
        if len(group) < 20:
            continue

        for horizon in HORIZONS:
            future_points = (
                group[f"future_points_{horizon}"]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            if future_points.empty:
                continue

            mean_points = future_points.mean()

            median_points = future_points.median()

            positive_probability = (future_points > 0).mean()

            cost_covered_probability = (future_points > TOTAL_COST_POINTS).mean()

            rows.append(
                {
                    "window": window_number,
                    "feature_a": feature_a,
                    "feature_a_value": str(value_a),
                    "feature_b": feature_b,
                    "feature_b_value": str(value_b),
                    "horizon_bars": horizon,
                    "observations": len(future_points),
                    "mean_future_points": (mean_points),
                    "median_future_points": (median_points),
                    "positive_probability": (positive_probability),
                    "cost_covered_probability": (cost_covered_probability),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SINGLE-FEATURE CONDITIONAL ANALYSIS
# ============================================================


def analyze_single_condition(
    validation: pd.DataFrame,
    window_number: int,
    feature: str,
) -> pd.DataFrame:
    """
    Analyze one additional feature conditioned on the negative
    Z-score extreme.

    This gives us a baseline before studying feature pairs.
    """

    base = validation.loc[validation["zscore_30"] <= -Z_THRESHOLD].copy()

    if base.empty:
        return pd.DataFrame()

    rows = []

    grouped = base.groupby(
        feature,
        observed=True,
        dropna=True,
    )

    for value, group in grouped:
        if len(group) < 20:
            continue

        for horizon in HORIZONS:
            future_points = (
                group[f"future_points_{horizon}"]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            if future_points.empty:
                continue

            rows.append(
                {
                    "window": window_number,
                    "feature": feature,
                    "feature_value": str(value),
                    "horizon_bars": horizon,
                    "observations": len(future_points),
                    "mean_future_points": (future_points.mean()),
                    "median_future_points": (future_points.median()),
                    "positive_probability": ((future_points > 0).mean()),
                    "cost_covered_probability": (
                        (future_points > TOTAL_COST_POINTS).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — RESEARCH 04")
    print("=" * 100)

    print()
    print("FEATURE INTERACTION ANALYSIS")
    print("-" * 100)

    print("No strategy construction.")

    print("No parameter optimization.")

    print("No XGBoost.")

    print("22 walk-forward OOS windows.")

    print()
    print(f"Base event: zscore_30 <= -{Z_THRESHOLD}")

    print(f"Estimated total cost: {TOTAL_COST_POINTS:.2f} points")

    # ========================================================
    # LOAD
    # ========================================================

    print()
    print("Loading MNQ data...")

    df = load_data()

    print(f"Rows loaded: {len(df):,}")

    # ========================================================
    # RTH
    # ========================================================

    print()
    print("Preparing RTH...")

    df = prepare_rth(df)

    print(f"RTH rows: {len(df):,}")

    # ========================================================
    # FEATURES
    # ========================================================

    print()
    print("Building complete feature set...")

    df = build_mean_reversion_features(df)

    print(f"Feature columns: {len(df.columns)}")

    # ========================================================
    # FUTURE OUTCOMES
    # ========================================================

    print()
    print("Building future outcome measurements...")

    df = add_future_returns(df)

    # ========================================================
    # BUCKETS
    # ========================================================

    print()
    print("Building broad research buckets...")

    df = add_research_buckets(df)

    # ========================================================
    # WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print()
    print(f"Walk-forward windows: {len(windows)}")

    # ========================================================
    # FEATURE DEFINITIONS
    # ========================================================

    single_features = [
        "vwap_distance_bucket",
        "ou_residual_bucket",
        "half_life_bucket",
        "vol_ratio_bucket",
    ]

    interactions = [
        (
            "vwap_distance_bucket",
            "ou_residual_bucket",
        ),
        (
            "vwap_distance_bucket",
            "half_life_bucket",
        ),
        (
            "vwap_distance_bucket",
            "vol_ratio_bucket",
        ),
        (
            "ou_residual_bucket",
            "half_life_bucket",
        ),
        (
            "ou_residual_bucket",
            "vol_ratio_bucket",
        ),
        (
            "half_life_bucket",
            "vol_ratio_bucket",
        ),
    ]

    single_results = []

    interaction_results = []

    # ========================================================
    # OOS LOOP
    # ========================================================

    for number, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        print()
        print(f"Processing OOS window {number}/{len(windows)}...")

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        print(f"  OOS rows: {len(validation):,}")

        if validation.empty:
            continue

        # ----------------------------------------------------
        # Single feature conditions
        # ----------------------------------------------------

        for feature in single_features:
            result = analyze_single_condition(
                validation,
                number,
                feature,
            )

            if not result.empty:
                single_results.append(result)

        # ----------------------------------------------------
        # Feature interactions
        # ----------------------------------------------------

        for feature_a, feature_b in interactions:
            result = analyze_interaction(
                validation,
                number,
                feature_a,
                feature_b,
                feature_a,
                feature_b,
            )

            if not result.empty:
                interaction_results.append(result)

    # ========================================================
    # COMBINE
    # ========================================================

    if single_results:
        single_results = pd.concat(
            single_results,
            ignore_index=True,
        )

    else:
        single_results = pd.DataFrame()

    if interaction_results:
        interaction_results = pd.concat(
            interaction_results,
            ignore_index=True,
        )

    else:
        interaction_results = pd.DataFrame()

    # ========================================================
    # AGGREGATED INTERACTION SUMMARY
    # ========================================================

    if not interaction_results.empty:
        grouped = interaction_results.groupby(
            [
                "feature_a",
                "feature_a_value",
                "feature_b",
                "feature_b_value",
                "horizon_bars",
            ],
            dropna=True,
        )

        summary_rows = []

        for keys, group in grouped:
            (
                feature_a,
                value_a,
                feature_b,
                value_b,
                horizon,
            ) = keys

            observations = group["observations"].sum()

            if observations <= 0:
                continue

            mean_points = np.average(
                group["mean_future_points"],
                weights=group["observations"],
            )

            positive_probability = np.average(
                group["positive_probability"],
                weights=group["observations"],
            )

            cost_probability = np.average(
                group["cost_covered_probability"],
                weights=group["observations"],
            )

            window_means = (
                group["mean_future_points"]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            summary_rows.append(
                {
                    "feature_a": feature_a,
                    "feature_a_value": value_a,
                    "feature_b": feature_b,
                    "feature_b_value": value_b,
                    "horizon_bars": horizon,
                    "total_observations": int(observations),
                    "pooled_mean_future_points": (mean_points),
                    "pooled_positive_probability": (positive_probability),
                    "pooled_cost_covered_probability": (cost_probability),
                    "median_window_mean_points": (window_means.median()),
                    "positive_window_ratio": ((window_means > 0).mean()),
                    "cost_positive_window_ratio": (
                        (group["mean_future_points"] > TOTAL_COST_POINTS).mean()
                    ),
                    "windows_observed": (len(group)),
                }
            )

        interaction_summary = pd.DataFrame(summary_rows)

    else:
        interaction_summary = pd.DataFrame()

    # ========================================================
    # SAVE
    # ========================================================

    single_path = RESULTS_DIR / "research_04_single_conditions.csv"

    interaction_path = RESULTS_DIR / "research_04_interactions_oos.csv"

    summary_path = RESULTS_DIR / "research_04_interactions_summary.csv"

    single_results.to_csv(
        single_path,
        index=False,
    )

    interaction_results.to_csv(
        interaction_path,
        index=False,
    )

    interaction_summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print("RESEARCH 04 COMPLETE")
    print("=" * 100)

    print()

    print(f"Single-condition rows: {len(single_results):,}")

    print(f"Interaction rows: {len(interaction_results):,}")

    print()

    if not interaction_summary.empty:
        print("TOP INTERACTIONS BY POOLED FUTURE POINTS")

        display = interaction_summary.sort_values(
            [
                "horizon_bars",
                "pooled_mean_future_points",
            ],
            ascending=[
                True,
                False,
            ],
        ).head(40)

        print(display.to_string(index=False))

    print()
    print("FILES SAVED")

    print(single_path)

    print(interaction_path)

    print(summary_path)

    print()
    print("IMPORTANT:")

    print("Interactions are descriptive.")

    print("No strategy was constructed.")

    print("No thresholds were optimized.")

    print("XGBoost was not used.")

    print("=" * 100)


if __name__ == "__main__":
    main()
