from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# ============================================================
# MEAN REVERSION — RESEARCH 03
# ============================================================
#
# PURPOSE
# -------
# Determine whether the statistical mean-reversion effects
# discovered in Research 01/02 have meaningful economic
# magnitude on MNQ.
#
# We are NOT building a strategy.
#
# We are measuring:
#
#   - future returns
#   - future displacement in MNQ points
#   - MFE
#   - MAE
#   - probability of reaching positive/negative excursions
#   - economic magnitude after estimated transaction costs
#
# Everything is evaluated through the existing 22 OOS
# walk-forward windows.
#
# NO:
#
#   - parameter optimization
#   - strategy construction
#   - stop/target optimization
#   - XGBoost
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
# CONFIGURATION
# ============================================================

# Horizons already used in Research 01.
FUTURE_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
    30,
    60,
)


# Z-score extreme regions.
#
# These are descriptive research regions.
# They are NOT proposed strategy entry thresholds.
#
EXTREME_ZSCORE_THRESHOLD = 1.5


# MNQ contract specifications.
MNQ_POINT_VALUE = 2.00

# Existing project cost assumptions.
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
) -> list[
    tuple[
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:
    """
    Generate the existing rolling walk-forward structure.

        Training = previous 2 years
        OOS      = following 3 months

    The training period is used only to define the temporal
    structure here. No model is fitted in Research 03.
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
    Prepare the MNQ dataset for RTH research.
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
# FUTURE PATH
# ============================================================


def calculate_future_path(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Calculate the future path relative to the current close.

    For every observation t:

        future_high =
            maximum high from t+1 through t+H

        future_low =
            minimum low from t+1 through t+H

    Therefore:

        future_MFE_long =
            future_high - current_close

        future_MAE_long =
            future_low - current_close

    For a SHORT:

        future_MFE_short =
            current_close - future_low

        future_MAE_short =
            current_close - future_high

    These are ex-post measurements only.
    """

    result = df.copy()

    future_highs = []

    future_lows = []

    for shift in range(
        1,
        horizon + 1,
    ):
        future_highs.append(df["high"].shift(-shift))

        future_lows.append(df["low"].shift(-shift))

    future_high = pd.concat(
        future_highs,
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        future_lows,
        axis=1,
    ).min(axis=1)

    result["future_high"] = future_high

    result["future_low"] = future_low

    result["mfe_long"] = future_high - df["close"]

    result["mae_long"] = future_low - df["close"]

    result["mfe_short"] = df["close"] - future_low

    result["mae_short"] = df["close"] - future_high

    return result


# ============================================================
# EXTREME Z-SCORE ANALYSIS
# ============================================================


def analyze_extreme_zscore(
    validation: pd.DataFrame,
    window_number: int,
) -> pd.DataFrame:
    """
    Analyze the economic behavior of extreme Z-score states.

    Long-side research:

        zscore_30 <= -threshold

    Short-side research:

        zscore_30 >= +threshold

    We measure future path statistics without imposing an
    entry, stop, target, or execution rule.
    """

    rows = []

    long_mask = validation["zscore_30"] <= -EXTREME_ZSCORE_THRESHOLD

    short_mask = validation["zscore_30"] >= EXTREME_ZSCORE_THRESHOLD

    for direction, mask in [
        ("long_reversion", long_mask),
        ("short_reversion", short_mask),
    ]:
        subset = validation.loc[mask].copy()

        if subset.empty:
            continue

        for horizon in FUTURE_HORIZONS:
            path = calculate_future_path(
                subset,
                horizon,
            )

            if direction == "long_reversion":
                mfe = (
                    path["mfe_long"]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .dropna()
                )

                mae = (
                    path["mae_long"]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .dropna()
                )

                final_displacement = subset["close"].shift(-horizon) - subset["close"]

            else:
                mfe = (
                    path["mfe_short"]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .dropna()
                )

                mae = (
                    path["mae_short"]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .dropna()
                )

                final_displacement = subset["close"] - subset["close"].shift(-horizon)

            final_displacement = final_displacement.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            ).dropna()

            if final_displacement.empty:
                continue

            # ------------------------------------------------
            # Economic return
            # ------------------------------------------------

            mean_points = final_displacement.mean()

            median_points = final_displacement.median()

            net_mean_points = mean_points - TOTAL_COST_POINTS

            # ------------------------------------------------
            # Probability of positive movement
            # ------------------------------------------------

            positive_probability = (final_displacement > 0).mean()

            # ------------------------------------------------
            # Probability of covering costs
            # ------------------------------------------------

            cost_covered_probability = (final_displacement > TOTAL_COST_POINTS).mean()

            rows.append(
                {
                    "window": window_number,
                    "direction": direction,
                    "horizon_bars": horizon,
                    "observations": len(final_displacement),
                    "mean_future_points": (mean_points),
                    "median_future_points": (median_points),
                    "mean_future_points_after_cost": (net_mean_points),
                    "mean_mfe_points": (mfe.mean()),
                    "median_mfe_points": (mfe.median()),
                    "mean_mae_points": (mae.mean()),
                    "median_mae_points": (mae.median()),
                    "positive_probability": (positive_probability),
                    "cost_covered_probability": (cost_covered_probability),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MFE / MAE DISTRIBUTION
# ============================================================


def calculate_excursion_quantiles(
    validation: pd.DataFrame,
    window_number: int,
) -> pd.DataFrame:
    """
    Calculate MFE/MAE quantiles for extreme Z-score states.

    Quantiles are useful because means can hide the shape of
    the distribution.
    """

    rows = []

    conditions = {
        "long_reversion": (validation["zscore_30"] <= -EXTREME_ZSCORE_THRESHOLD),
        "short_reversion": (validation["zscore_30"] >= EXTREME_ZSCORE_THRESHOLD),
    }

    for direction, mask in conditions.items():
        subset = validation.loc[mask].copy()

        if subset.empty:
            continue

        for horizon in FUTURE_HORIZONS:
            path = calculate_future_path(
                subset,
                horizon,
            )

            if direction == "long_reversion":
                mfe = path["mfe_long"]

                mae = path["mae_long"]

            else:
                mfe = path["mfe_short"]

                mae = path["mae_short"]

            mfe = mfe.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            ).dropna()

            mae = mae.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            ).dropna()

            if mfe.empty or mae.empty:
                continue

            for quantile in (
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ):
                rows.append(
                    {
                        "window": window_number,
                        "direction": direction,
                        "horizon_bars": horizon,
                        "quantile": quantile,
                        "mfe_points": mfe.quantile(quantile),
                        "mae_points": mae.quantile(quantile),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# WINDOW SUMMARY
# ============================================================


def summarize_windows(
    economic_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the economic results across OOS windows.

    This lets us distinguish pooled behavior from
    cross-window consistency.
    """

    if economic_results.empty:
        return pd.DataFrame()

    rows = []

    grouped = economic_results.groupby(
        [
            "direction",
            "horizon_bars",
        ]
    )

    for (
        direction,
        horizon,
    ), group in grouped:
        values = (
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

        if values.empty:
            continue

        rows.append(
            {
                "direction": direction,
                "horizon_bars": horizon,
                "windows": len(values),
                "mean_of_window_means": (values.mean()),
                "median_of_window_means": (values.median()),
                "positive_window_ratio": ((values > 0).mean()),
                "cost_positive_window_ratio": (
                    (group["mean_future_points_after_cost"] > 0).mean()
                ),
                "min_window_mean": (values.min()),
                "max_window_mean": (values.max()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — RESEARCH 03")
    print("=" * 100)

    print()
    print("ECONOMIC SIGNIFICANCE ANALYSIS")
    print("-" * 100)

    print("No strategy construction.")

    print("No optimization.")

    print("No XGBoost.")

    print("22 walk-forward OOS windows.")

    print()
    print(f"MNQ point value: ${MNQ_POINT_VALUE:.2f}")

    print(f"Estimated total cost: {TOTAL_COST_POINTS:.2f} points")

    # ========================================================
    # LOAD
    # ========================================================

    print()
    print("Loading MNQ data...")

    df = load_data()

    print(f"Rows loaded: {len(df):,}")

    # ========================================================
    # PREPARE
    # ========================================================

    print()
    print("Preparing RTH...")

    df = prepare_rth(df)

    print(f"RTH rows: {len(df):,}")

    # ========================================================
    # FEATURES
    # ========================================================

    print()
    print("Building Mean Reversion features...")

    df = build_mean_reversion_features(df)

    print(f"Feature columns: {len(df.columns)}")

    # ========================================================
    # WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print()
    print(f"Walk-forward windows: {len(windows)}")

    # ========================================================
    # ANALYSIS
    # ========================================================

    economic_results = []
    excursion_results = []

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

        economic = analyze_extreme_zscore(
            validation,
            number,
        )

        excursions = calculate_excursion_quantiles(
            validation,
            number,
        )

        if not economic.empty:
            economic["validation_start"] = validation_start

            economic["validation_end"] = validation_end

            economic_results.append(economic)

        if not excursions.empty:
            excursions["validation_start"] = validation_start

            excursions["validation_end"] = validation_end

            excursion_results.append(excursions)

    # ========================================================
    # COMBINE
    # ========================================================

    if economic_results:
        economic_results = pd.concat(
            economic_results,
            ignore_index=True,
        )

    else:
        economic_results = pd.DataFrame()

    if excursion_results:
        excursion_results = pd.concat(
            excursion_results,
            ignore_index=True,
        )

    else:
        excursion_results = pd.DataFrame()

    # ========================================================
    # SUMMARY
    # ========================================================

    window_summary = summarize_windows(economic_results)

    # ========================================================
    # SAVE
    # ========================================================

    economic_path = RESULTS_DIR / "research_03_economic.csv"

    excursion_path = RESULTS_DIR / "research_03_excursions.csv"

    summary_path = RESULTS_DIR / "research_03_window_summary.csv"

    economic_results.to_csv(
        economic_path,
        index=False,
    )

    excursion_results.to_csv(
        excursion_path,
        index=False,
    )

    window_summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("RESEARCH 03 COMPLETE")
    print("=" * 100)

    print()

    print(f"Economic observations: {len(economic_results):,}")

    print(f"Excursion observations: {len(excursion_results):,}")

    print()

    print("WINDOW SUMMARY")

    if not window_summary.empty:
        print(window_summary.to_string(index=False))

    print()
    print("FILES SAVED")

    print(economic_path)

    print(excursion_path)

    print(summary_path)

    print()
    print("No strategy was constructed.")

    print("No parameters were optimized.")

    print("XGBoost was not used.")

    print("=" * 100)


if __name__ == "__main__":
    main()
