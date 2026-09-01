from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# ============================================================
# MEAN REVERSION — RESEARCH 01
# ============================================================
#
# PURPOSE
# -------
# This is the first statistical investigation of the Mean
# Reversion hypothesis on real MNQ data.
#
# We are NOT building a trading strategy yet.
#
# We are asking:
#
#   "Given the information available at time t, what tends
#    to happen to price over the following H bars?"
#
# The research is performed using the existing 22-window
# walk-forward structure.
#
# IMPORTANT
# ---------
# This script does NOT:
#
#   - optimize entry parameters
#   - optimize stop/target
#   - generate trades
#   - train XGBoost
#   - select a strategy
#
# It only measures future behavior conditional on current
# market state.
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
# RESEARCH CONFIGURATION
# ============================================================

# Future horizons in bars.
#
# Since the source data is 1-minute MNQ data:
#
#     1  = 1 minute
#     3  = 3 minutes
#     5  = 5 minutes
#     10 = 10 minutes
#     20 = 20 minutes
#     30 = 30 minutes
#     60 = 60 minutes
#
FUTURE_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
    30,
    60,
)


# Z-score buckets.
#
# These are descriptive buckets.
# They are NOT entry thresholds.
#
ZSCORE_BUCKETS = (
    (-np.inf, -2.5, "<=-2.5"),
    (-2.5, -2.0, "-2.5:-2.0"),
    (-2.0, -1.5, "-2.0:-1.5"),
    (-1.5, -1.0, "-1.5:-1.0"),
    (-1.0, 0.0, "-1.0:0"),
    (0.0, 1.0, "0:1.0"),
    (1.0, 1.5, "1.0:1.5"),
    (1.5, 2.0, "1.5:2.0"),
    (2.0, 2.5, "2.0:2.5"),
    (2.5, np.inf, ">=2.5"),
)


# Volatility percentile buckets.
#
# These are deliberately broad descriptive regimes.
#
VOL_BUCKETS = (
    (0.0, 0.20, "0-20"),
    (0.20, 0.40, "20-40"),
    (0.40, 0.60, "40-60"),
    (0.60, 0.80, "60-80"),
    (0.80, 1.01, "80-100"),
)


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    df: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate the same 22-style rolling walk-forward structure
    used elsewhere in the project.

    Structure:

        TRAIN = previous 2 years
        OOS   = following 3 months

    The first OOS period begins two years after the beginning
    of the available dataset.

    IMPORTANT:

    This function defines the temporal boundaries only.
    No fitting occurs here.
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
    """
    Restrict the research dataset to RTH.

    We preserve chronological order and create a session
    identifier where necessary.

    No observations are forward-filled or fabricated.
    """

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
# FUTURE RETURNS
# ============================================================


def add_future_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate future close-to-close returns.

    For horizon H:

        future_return_H(t) =
            close[t+H] / close[t] - 1

    This is the TARGET VARIABLE of this research.

    IMPORTANT:

    These columns are intentionally future-looking.

    They MUST NEVER be used as model features or inputs to
    a live strategy.

    They exist only for ex-post statistical measurement.
    """

    if "close" not in df.columns:
        raise KeyError("Missing required column: 'close'")

    df = df.copy()

    for horizon in FUTURE_HORIZONS:
        df[f"future_return_{horizon}"] = df["close"].shift(-horizon) / df["close"] - 1.0

    return df


# ============================================================
# FUTURE PRICE DISPLACEMENT
# ============================================================


def add_future_displacement(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate future price displacement in NQ points.

        future_displacement_H =
            close[t+H] - close[t]

    This complements percentage return because NQ is a
    point-based instrument.
    """

    df = df.copy()

    for horizon in FUTURE_HORIZONS:
        df[f"future_displacement_{horizon}"] = df["close"].shift(-horizon) - df["close"]

    return df


# ============================================================
# Z-SCORE BUCKET
# ============================================================


def classify_zscore(
    value: float,
) -> str | None:
    """
    Assign a descriptive Z-score bucket.
    """

    if pd.isna(value):
        return None

    for lower, upper, label in ZSCORE_BUCKETS:
        if value > lower and value <= upper:
            return label

    return None


# ============================================================
# VOLATILITY BUCKET
# ============================================================


def classify_volatility(
    value: float,
) -> str | None:
    """
    Assign a descriptive volatility percentile bucket.
    """

    if pd.isna(value):
        return None

    for lower, upper, label in VOL_BUCKETS:
        if value >= lower and value < upper:
            return label

    return None


# ============================================================
# CONDITIONAL STATISTICS
# ============================================================


def calculate_conditional_statistics(
    df: pd.DataFrame,
    condition_column: str,
    condition_name: str,
) -> pd.DataFrame:
    """
    Calculate future-return statistics conditional on a
    categorical market condition.

    For each condition and future horizon we calculate:

        observations
        mean return
        median return
        standard deviation
        win rate
        mean displacement
        median displacement

    No strategy assumptions are made.
    """

    rows = []

    grouped = df.groupby(
        condition_column,
        dropna=True,
    )

    for condition_value, group in grouped:
        for horizon in FUTURE_HORIZONS:
            return_column = f"future_return_{horizon}"

            displacement_column = f"future_displacement_{horizon}"

            values = (
                group[return_column]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            displacement = (
                group[displacement_column]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            if values.empty:
                continue

            rows.append(
                {
                    "condition": condition_name,
                    "condition_value": condition_value,
                    "horizon_bars": horizon,
                    "observations": len(values),
                    "mean_return": values.mean(),
                    "median_return": values.median(),
                    "std_return": values.std(),
                    "win_rate_positive": (values > 0).mean(),
                    "mean_displacement_points": (displacement.mean()),
                    "median_displacement_points": (displacement.median()),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CONTINUOUS CONDITION STATISTICS
# ============================================================


def calculate_continuous_statistics(
    df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """
    Calculate simple conditional statistics for extreme
    values of a continuous feature.

    This is intentionally descriptive.

    We examine:

        feature <= 10th percentile
        feature >= 90th percentile

    separately.

    The percentile boundaries are calculated independently
    inside each OOS window.

    This prevents information from later periods entering
    the classification.
    """

    rows = []

    valid = (
        df[feature]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if valid.empty:
        return pd.DataFrame()

    low_threshold = valid.quantile(0.10)

    high_threshold = valid.quantile(0.90)

    conditions = {
        "bottom_10pct": df[feature] <= low_threshold,
        "top_10pct": df[feature] >= high_threshold,
    }

    for condition_name, mask in conditions.items():
        subset = df.loc[mask]

        for horizon in FUTURE_HORIZONS:
            column = f"future_return_{horizon}"

            values = (
                subset[column]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            if values.empty:
                continue

            rows.append(
                {
                    "feature": feature,
                    "condition": condition_name,
                    "threshold": (
                        low_threshold
                        if condition_name == "bottom_10pct"
                        else high_threshold
                    ),
                    "horizon_bars": horizon,
                    "observations": len(values),
                    "mean_return": values.mean(),
                    "median_return": values.median(),
                    "win_rate_positive": (values > 0).mean(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SINGLE OOS WINDOW
# ============================================================


def analyze_window(
    validation: pd.DataFrame,
    window_number: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Analyze one OOS window.

    Returns:

        categorical_statistics
        continuous_statistics

    The window is already strictly OOS when passed here.
    """

    data = validation.copy()

    # --------------------------------------------------------
    # Future targets
    # --------------------------------------------------------

    data = add_future_returns(data)

    data = add_future_displacement(data)

    # --------------------------------------------------------
    # Descriptive buckets
    # --------------------------------------------------------

    data["zscore_30_bucket"] = data["zscore_30"].apply(classify_zscore)

    data["zscore_60_bucket"] = data["zscore_60"].apply(classify_zscore)

    # --------------------------------------------------------
    # Categorical research
    # --------------------------------------------------------

    categorical_results = []

    for column, name in [
        (
            "zscore_30_bucket",
            "zscore_30",
        ),
        (
            "zscore_60_bucket",
            "zscore_60",
        ),
    ]:
        result = calculate_conditional_statistics(
            data,
            column,
            name,
        )

        if not result.empty:
            result["window"] = window_number

            categorical_results.append(result)

    if categorical_results:
        categorical = pd.concat(
            categorical_results,
            ignore_index=True,
        )

    else:
        categorical = pd.DataFrame()

    # --------------------------------------------------------
    # Continuous research
    # --------------------------------------------------------

    continuous_results = []

    for feature in [
        "normalized_vwap_distance",
        "half_life_30",
        "half_life_60",
        "normalized_ou_residual_30",
        "realized_vol_30",
        "realized_vol_60",
        "vol_ratio_5_30",
        "vol_ratio_5_60",
        "vol_ratio_30_60",
    ]:
        if feature not in data.columns:
            continue

        result = calculate_continuous_statistics(
            data,
            feature,
        )

        if not result.empty:
            result["window"] = window_number

            continuous_results.append(result)

    if continuous_results:
        continuous = pd.concat(
            continuous_results,
            ignore_index=True,
        )

    else:
        continuous = pd.DataFrame()

    return (
        categorical,
        continuous,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — RESEARCH 01")
    print("=" * 100)

    print()
    print("STATISTICAL EDGE EXPLORATION")
    print("-" * 100)

    print("No strategy construction.")

    print("No XGBoost.")

    print("No parameter optimization.")

    print("22 walk-forward OOS windows.")

    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("Loading MNQ data...")

    df = load_data()

    print(f"Rows loaded: {len(df):,}")

    # ========================================================
    # PREPARE RTH
    # ========================================================

    print()
    print("Preparing RTH...")

    df = prepare_rth(df)

    print(f"RTH rows: {len(df):,}")

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    print()
    print("Building complete feature set...")

    df = build_mean_reversion_features(df)

    print(f"Feature columns: {len(df.columns)}")

    # ========================================================
    # GENERATE WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print()
    print(f"Walk-forward windows: {len(windows)}")

    # ========================================================
    # PROCESS WINDOWS
    # ========================================================

    all_categorical = []
    all_continuous = []

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
            print("  WARNING: empty OOS window")
            continue

        categorical, continuous = analyze_window(
            validation,
            number,
        )

        if not categorical.empty:
            categorical["validation_start"] = validation_start

            categorical["validation_end"] = validation_end

            all_categorical.append(categorical)

        if not continuous.empty:
            continuous["validation_start"] = validation_start

            continuous["validation_end"] = validation_end

            all_continuous.append(continuous)

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    print()
    print("=" * 100)

    if all_categorical:
        categorical_results = pd.concat(
            all_categorical,
            ignore_index=True,
        )

    else:
        categorical_results = pd.DataFrame()

    if all_continuous:
        continuous_results = pd.concat(
            all_continuous,
            ignore_index=True,
        )

    else:
        continuous_results = pd.DataFrame()

    # ========================================================
    # SAVE
    # ========================================================

    categorical_path = RESULTS_DIR / "research_01_categorical.csv"

    continuous_path = RESULTS_DIR / "research_01_continuous.csv"

    categorical_results.to_csv(
        categorical_path,
        index=False,
    )

    continuous_results.to_csv(
        continuous_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("MEAN REVERSION RESEARCH COMPLETE")

    print("=" * 100)

    print()

    print(f"Categorical rows: {len(categorical_results):,}")

    print(f"Continuous rows: {len(continuous_results):,}")

    print()

    print("FILES SAVED")

    print(categorical_path)

    print(continuous_path)

    print()

    print("No strategy was constructed.")

    print("Results represent descriptive OOS statistics only.")

    print("=" * 100)


if __name__ == "__main__":
    main()
