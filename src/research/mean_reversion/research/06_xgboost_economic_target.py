from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


# ============================================================
# MEAN REVERSION — RESEARCH 06
# ============================================================
#
# XGBOOST ECONOMIC TARGET
# ============================================================
#
# RESEARCH QUESTION
# -----------------
#
# Research 05 asked:
#
#     "Can XGBoost predict whether the next 10 bars
#      finish higher?"
#
# The answer was:
#
#     Some predictive structure exists, but directional
#     accuracy was not sufficiently robust across all OOS
#     windows.
#
# Research 05 also showed something more interesting:
#
#     Higher model probabilities were associated with
#     larger future price movements.
#
# Therefore Research 06 changes the target.
#
# Instead of predicting:
#
#     future_return > 0
#
# we ask:
#
#     "Can the feature set identify situations where the
#      favorable future movement reaches a meaningful
#      number of MNQ points?"
#
# ------------------------------------------------------------
#
# IMPORTANT
# ---------
#
# We do NOT optimize the target threshold.
#
# We evaluate several fixed economic targets:
#
#     +1 point
#     +2 points
#     +3 points
#     +5 points
#
# across fixed horizons:
#
#     5
#     10
#     20
#     30 bars
#
# Each target/horizon combination is a separate research
# experiment.
#
# The model is trained independently inside every walk-forward
# window.
#
# NO information from the OOS period is used for training.
#
# NO final strategy is constructed.
#
# ============================================================


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


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
# ECONOMIC TARGETS
# ============================================================
#
# Fixed research grid.
#
# These are not optimized trading parameters.
#
# They represent progressively larger favorable MNQ
# movements.
#
# ============================================================

TARGET_POINTS = (
    1.0,
    2.0,
    3.0,
    5.0,
)

TARGET_HORIZONS = (
    5,
    10,
    20,
    30,
)


# ============================================================
# BASE EVENT
# ============================================================
#
# Research 03/04 established the strongest preliminary
# phenomenon around negative Z-score extremes.
#
# Therefore we investigate:
#
#     zscore_30 <= -1.5
#
# This remains an exploratory event definition.
#
# It is NOT the final entry rule.
#
# ============================================================

BASE_ZSCORE_THRESHOLD = -1.5


# ============================================================
# FIXED MODEL CONFIGURATION
# ============================================================
#
# Same conservative XGBoost architecture used in Research 05.
#
# No hyperparameter search.
#
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

MIN_CHILD_WEIGHT = 10
REG_ALPHA = 0.1
REG_LAMBDA = 1.0


# ============================================================
# FEATURES
# ============================================================
#
# These are all pre-existing features.
#
# NO future_* variables are allowed.
#
# ============================================================

MODEL_FEATURES = [
    # Returns / momentum
    "past_return_1",
    "past_return_3",
    "past_return_5",
    "past_return_10",
    "past_return_15",
    "past_return_30",
    # Volatility
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "vol_ratio_5_30",
    "vol_ratio_5_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
    # Range / ATR
    "rolling_range_5",
    "rolling_range_15",
    "rolling_range_30",
    "rolling_range_60",
    "atr_5",
    "atr_15",
    "atr_30",
    "atr_60",
    "normalized_range_5",
    "normalized_range_15",
    "normalized_range_30",
    "normalized_range_60",
    # VWAP
    "vwap_distance",
    "vwap_distance_pct",
    "normalized_vwap_distance",
    # Mean distance
    "mean_distance_5",
    "mean_distance_15",
    "mean_distance_30",
    "mean_distance_60",
    # Z-score
    "zscore_5",
    "zscore_15",
    "zscore_30",
    "zscore_60",
    "abs_zscore_5",
    "abs_zscore_15",
    "abs_zscore_30",
    "abs_zscore_60",
    "zscore_direction_5",
    "zscore_direction_15",
    "zscore_direction_30",
    "zscore_direction_60",
    # Log-price deviation
    "log_zscore_5",
    "log_zscore_30",
    "log_zscore_60",
    # OU / mean reversion
    "autocorrelation_30",
    "autocorrelation_60",
    "ar1_coefficient_30",
    "ar1_coefficient_60",
    "mean_reversion_speed_30",
    "mean_reversion_speed_60",
    "half_life_30",
    "half_life_60",
    "ou_residual_30",
    "ou_residual_60",
    "normalized_ou_residual_30",
    "normalized_ou_residual_60",
]


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(
    df: pd.DataFrame,
):
    """
    Generate the standard project walk-forward structure:

        2 years TRAIN
        3 months OOS

    No random train/test splitting is used.
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
# FUTURE PATH
# ============================================================


def add_future_paths(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build future high / low paths.

    These columns are ONLY used to construct future targets.

    They must NEVER be included in MODEL_FEATURES.
    """

    df = df.copy()

    maximum_horizon = max(TARGET_HORIZONS)

    future_highs = []
    future_lows = []

    for shift in range(
        1,
        maximum_horizon + 1,
    ):
        future_highs.append(df["high"].shift(-shift))

        future_lows.append(df["low"].shift(-shift))

    future_high_matrix = pd.concat(
        future_highs,
        axis=1,
    )

    future_low_matrix = pd.concat(
        future_lows,
        axis=1,
    )

    for horizon in TARGET_HORIZONS:
        high_window = future_high_matrix.iloc[
            :,
            :horizon,
        ]

        low_window = future_low_matrix.iloc[
            :,
            :horizon,
        ]

        df[f"future_high_{horizon}"] = high_window.max(axis=1)

        df[f"future_low_{horizon}"] = low_window.min(axis=1)

        # Favorable movement for LONG mean reversion.
        df[f"future_mfe_long_{horizon}"] = df[f"future_high_{horizon}"] - df["close"]

    return df


# ============================================================
# TARGET CONSTRUCTION
# ============================================================


def build_economic_target(
    df: pd.DataFrame,
    horizon: int,
    target_points: float,
) -> pd.DataFrame:
    """
    Construct one binary economic target.

    target = 1

        if the future high reaches at least
        `target_points` above the current close
        within `horizon` bars.

    target = 0

        otherwise.

    This represents whether a favorable LONG movement of a
    specified magnitude occurred.

    IMPORTANT:

    The target uses future information only because it is the
    supervised-learning label.

    The target itself is never provided to the model as a
    feature.
    """

    df = df.copy()

    mfe = df[f"future_mfe_long_{horizon}"]

    df["economic_target"] = (mfe >= target_points).astype(float)

    return df


# ============================================================
# MODEL
# ============================================================


def create_model():
    """
    Create the fixed XGBoost classifier.

    No hyperparameter optimization.
    """

    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_weight=MIN_CHILD_WEIGHT,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


# ============================================================
# BASELINE
# ============================================================


def calculate_baseline(
    validation: pd.DataFrame,
    horizon: int,
    target_points: float,
) -> dict:
    """
    Calculate the unconditional baseline probability.

    This is critical.

    XGBoost must beat the simple probability of the economic
    event occurring within the base mean-reversion population.

    """

    mfe = validation[f"future_mfe_long_{horizon}"]

    mfe = mfe.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if mfe.empty:
        return {
            "baseline_probability": np.nan,
            "observations": 0,
        }

    probability = (mfe >= target_points).mean()

    return {
        "baseline_probability": float(probability),
        "observations": int(len(mfe)),
    }


# ============================================================
# PROBABILITY DECILES
# ============================================================


def analyze_probability_deciles(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Divide OOS predictions into probability deciles.

    If the model has useful economic information, the realized
    probability of reaching the target should increase as the
    predicted probability increases.

    The analysis is performed independently per:

        target
        horizon
        OOS window
    """

    rows = []

    grouping = predictions.groupby(
        [
            "target_points",
            "horizon_bars",
            "window",
        ]
    )

    for (
        target_points,
        horizon,
        window,
    ), group in grouping:
        if len(group) < 50:
            continue

        group = group.copy()

        try:
            group["probability_decile"] = pd.qcut(
                group["probability"],
                q=10,
                labels=False,
                duplicates="drop",
            )

        except ValueError:
            continue

        for decile, subset in group.groupby(
            "probability_decile",
            observed=True,
        ):
            if subset.empty:
                continue

            rows.append(
                {
                    "target_points": (target_points),
                    "horizon_bars": (horizon),
                    "window": (window),
                    "probability_decile": (int(decile)),
                    "observations": len(subset),
                    "mean_probability": (subset["probability"].mean()),
                    "realized_probability": (subset["target"].mean()),
                    "mean_future_mfe": (subset["future_mfe"].mean()),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def extract_feature_importance(
    model: XGBClassifier,
    target_points: float,
    horizon: int,
    window_number: int,
) -> pd.DataFrame:
    """
    Extract gain-based feature importance.
    """

    booster = model.get_booster()

    gain = booster.get_score(importance_type="gain")

    rows = []

    for feature in MODEL_FEATURES:
        rows.append(
            {
                "target_points": (target_points),
                "horizon_bars": (horizon),
                "window": (window_number),
                "feature": feature,
                "gain": float(
                    gain.get(
                        feature,
                        0.0,
                    )
                ),
            }
        )

    result = pd.DataFrame(rows)

    total_gain = result["gain"].sum()

    if total_gain > 0:
        result["gain_normalized"] = result["gain"] / total_gain

    else:
        result["gain_normalized"] = 0.0

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — RESEARCH 06")
    print("=" * 100)

    print()
    print("XGBOOST ECONOMIC TARGET")
    print("-" * 100)

    print("No final strategy.")

    print("No hyperparameter optimization.")

    print("No stop / target optimization.")

    print("22 walk-forward OOS windows.")

    print()

    print("Target points:")

    print(TARGET_POINTS)

    print("Target horizons:")

    print(TARGET_HORIZONS)

    print(f"Base event: zscore_30 <= {BASE_ZSCORE_THRESHOLD}")

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
    # VALIDATE FEATURES
    # ========================================================

    missing = [feature for feature in MODEL_FEATURES if feature not in df.columns]

    if missing:
        raise KeyError("Missing model features:\n" + "\n".join(missing))

    # ========================================================
    # FUTURE PATH
    # ========================================================

    print()
    print("Building future price paths...")

    df = add_future_paths(df)

    # ========================================================
    # WALK FORWARD
    # ========================================================

    windows = generate_windows(df)

    print()
    print(f"Walk-forward windows: {len(windows)}")

    prediction_results = []
    importance_results = []
    baseline_results = []

    # ========================================================
    # EXPERIMENT LOOP
    # ========================================================

    for target_points in TARGET_POINTS:
        for horizon in TARGET_HORIZONS:
            print()
            print("=" * 100)

            print(f"TARGET = +{target_points:.1f} POINTS")

            print(f"HORIZON = {horizon} BARS")

            print("=" * 100)

            # ------------------------------------------------
            # Build target
            # ------------------------------------------------

            experiment_df = build_economic_target(
                df,
                horizon,
                target_points,
            )

            # ------------------------------------------------
            # OOS windows
            # ------------------------------------------------

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

                train = experiment_df.loc[
                    (experiment_df.index >= train_start)
                    & (experiment_df.index < validation_start)
                ].copy()

                validation = experiment_df.loc[
                    (experiment_df.index >= validation_start)
                    & (experiment_df.index < validation_end)
                ].copy()

                # ------------------------------------------------
                # Base mean-reversion event
                # ------------------------------------------------

                train = train.loc[train["zscore_30"] <= BASE_ZSCORE_THRESHOLD].copy()

                validation = validation.loc[
                    validation["zscore_30"] <= BASE_ZSCORE_THRESHOLD
                ].copy()

                print(f"  Train events: {len(train):,}")

                print(f"  OOS events: {len(validation):,}")

                # ------------------------------------------------
                # Baseline
                # ------------------------------------------------

                baseline = calculate_baseline(
                    validation,
                    horizon,
                    target_points,
                )

                baseline_results.append(
                    {
                        "target_points": (target_points),
                        "horizon_bars": (horizon),
                        "window": (number),
                        **baseline,
                    }
                )

                if len(train) < 100:
                    print("  Skipping: insufficient training.")
                    continue

                if validation.empty:
                    continue

                # ------------------------------------------------
                # Remove missing observations
                # ------------------------------------------------

                required_columns = MODEL_FEATURES + [
                    "economic_target",
                    f"future_mfe_long_{horizon}",
                ]

                train = train.dropna(subset=required_columns)

                validation = validation.dropna(subset=required_columns)

                if len(train) < 100:
                    print("  Skipping after NaN removal.")
                    continue

                if validation.empty:
                    continue

                # ------------------------------------------------
                # TRAIN
                # ------------------------------------------------

                X_train = train[MODEL_FEATURES]

                y_train = train["economic_target"].astype(int)

                X_validation = validation[MODEL_FEATURES]

                y_validation = validation["economic_target"].astype(int)

                # ------------------------------------------------
                # Class balance check
                # ------------------------------------------------

                if y_train.nunique() < 2:
                    print("  Skipping: training target contains only one class.")

                    continue

                model = create_model()

                model.fit(
                    X_train,
                    y_train,
                    verbose=False,
                )

                # ------------------------------------------------
                # OOS probability
                # ------------------------------------------------

                probability = model.predict_proba(X_validation)[:, 1]

                # ------------------------------------------------
                # Store predictions
                # ------------------------------------------------

                result = pd.DataFrame(
                    {
                        "timestamp": (validation.index),
                        "target_points": (target_points),
                        "horizon_bars": (horizon),
                        "window": (number),
                        "probability": (probability),
                        "target": (y_validation.to_numpy()),
                        "future_mfe": (
                            validation[f"future_mfe_long_{horizon}"].to_numpy()
                        ),
                        "zscore_30": (validation["zscore_30"].to_numpy()),
                        "normalized_vwap_distance": (
                            validation["normalized_vwap_distance"].to_numpy()
                        ),
                        "normalized_ou_residual_30": (
                            validation["normalized_ou_residual_30"].to_numpy()
                        ),
                        "half_life_30": (validation["half_life_30"].to_numpy()),
                        "realized_vol_30": (validation["realized_vol_30"].to_numpy()),
                    }
                )

                prediction_results.append(result)

                # ------------------------------------------------
                # Feature importance
                # ------------------------------------------------

                importance = extract_feature_importance(
                    model,
                    target_points,
                    horizon,
                    number,
                )

                importance_results.append(importance)

                print(f"  Baseline P(target): {baseline['baseline_probability']:.4f}")

                print(f"  Mean OOS probability: {probability.mean():.4f}")

    # ========================================================
    # COMBINE
    # ========================================================

    if prediction_results:
        predictions = pd.concat(
            prediction_results,
            ignore_index=True,
        )

    else:
        predictions = pd.DataFrame()

    if importance_results:
        feature_importance = pd.concat(
            importance_results,
            ignore_index=True,
        )

    else:
        feature_importance = pd.DataFrame()

    baseline = pd.DataFrame(baseline_results)

    # ========================================================
    # DECILES
    # ========================================================

    if not predictions.empty:
        deciles = analyze_probability_deciles(predictions)

    else:
        deciles = pd.DataFrame()

    # ========================================================
    # AGGREGATED EXPERIMENT SUMMARY
    # ========================================================

    summary_rows = []

    if not predictions.empty:
        grouping = predictions.groupby(
            [
                "target_points",
                "horizon_bars",
                "window",
            ]
        )

        for (
            target_points,
            horizon,
            window,
        ), group in grouping:
            if group.empty:
                continue

            baseline_subset = baseline.loc[
                (baseline["target_points"] == target_points)
                & (baseline["horizon_bars"] == horizon)
                & (baseline["window"] == window)
            ]

            if baseline_subset.empty:
                baseline_probability = np.nan
            else:
                baseline_probability = baseline_subset.iloc[0]["baseline_probability"]

            probability = group["probability"]

            realized = group["target"]

            # Top probability decile
            # is used only descriptively.
            #
            # It is NOT a proposed strategy threshold.
            try:
                decile_labels = pd.qcut(
                    probability,
                    q=10,
                    labels=False,
                    duplicates="drop",
                )

                top_decile = decile_labels == decile_labels.max()

                top_decile_realized = realized.loc[top_decile].mean()

            except Exception:
                top_decile_realized = np.nan

            summary_rows.append(
                {
                    "target_points": (target_points),
                    "horizon_bars": (horizon),
                    "window": (window),
                    "observations": (len(group)),
                    "baseline_probability": (baseline_probability),
                    "mean_model_probability": (probability.mean()),
                    "realized_probability": (realized.mean()),
                    "probability_lift_vs_baseline": (
                        realized.mean() - baseline_probability
                    ),
                    "top_decile_realized_probability": (top_decile_realized),
                    "top_decile_lift_vs_baseline": (
                        top_decile_realized - baseline_probability
                    ),
                    "mean_future_mfe": (group["future_mfe"].mean()),
                }
            )

    summary = pd.DataFrame(summary_rows)

    # ========================================================
    # CROSS-WINDOW SUMMARY
    # ========================================================

    cross_window_rows = []

    if not summary.empty:
        grouped = summary.groupby(
            [
                "target_points",
                "horizon_bars",
            ]
        )

        for (
            target_points,
            horizon,
        ), group in grouped:
            valid = group.dropna(subset=["probability_lift_vs_baseline"])

            if valid.empty:
                continue

            cross_window_rows.append(
                {
                    "target_points": (target_points),
                    "horizon_bars": (horizon),
                    "windows": len(valid),
                    "mean_realized_probability": (valid["realized_probability"].mean()),
                    "mean_baseline_probability": (valid["baseline_probability"].mean()),
                    "mean_probability_lift": (
                        valid["probability_lift_vs_baseline"].mean()
                    ),
                    "median_probability_lift": (
                        valid["probability_lift_vs_baseline"].median()
                    ),
                    "positive_lift_window_ratio": (
                        (valid["probability_lift_vs_baseline"] > 0).mean()
                    ),
                    "mean_top_decile_lift": (
                        valid["top_decile_lift_vs_baseline"].mean()
                    ),
                    "positive_top_decile_lift_ratio": (
                        (valid["top_decile_lift_vs_baseline"] > 0).mean()
                    ),
                    "mean_future_mfe": (valid["mean_future_mfe"].mean()),
                }
            )

    cross_window_summary = pd.DataFrame(cross_window_rows)

    # ========================================================
    # SAVE
    # ========================================================

    predictions_path = RESULTS_DIR / "research_06_xgboost_economic_predictions.csv"

    importance_path = (
        RESULTS_DIR / "research_06_xgboost_economic_feature_importance.csv"
    )

    deciles_path = RESULTS_DIR / "research_06_xgboost_economic_deciles.csv"

    baseline_path = RESULTS_DIR / "research_06_xgboost_economic_baseline.csv"

    summary_path = RESULTS_DIR / "research_06_xgboost_economic_summary.csv"

    cross_window_path = RESULTS_DIR / "research_06_xgboost_economic_cross_window.csv"

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    feature_importance.to_csv(
        importance_path,
        index=False,
    )

    deciles.to_csv(
        deciles_path,
        index=False,
    )

    baseline.to_csv(
        baseline_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    cross_window_summary.to_csv(
        cross_window_path,
        index=False,
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("RESEARCH 06 COMPLETE")
    print("=" * 100)

    print()

    print(f"OOS predictions: {len(predictions):,}")

    print(f"Feature importance rows: {len(feature_importance):,}")

    print()

    if not cross_window_summary.empty:
        print("CROSS-WINDOW ECONOMIC RESULTS")

        print(cross_window_summary.to_string(index=False))

    print()

    if not deciles.empty:
        print("PROBABILITY DECILES — POOLED")

        pooled = deciles.groupby(
            [
                "target_points",
                "horizon_bars",
                "probability_decile",
            ]
        )[
            [
                "mean_probability",
                "realized_probability",
                "mean_future_mfe",
            ]
        ].mean()

        print(pooled.to_string())

    print()

    if not feature_importance.empty:
        print("TOP FEATURES BY EXPERIMENT")

        top_features = (
            feature_importance.groupby(
                [
                    "target_points",
                    "horizon_bars",
                    "feature",
                ]
            )["gain_normalized"]
            .mean()
            .reset_index()
            .sort_values(
                [
                    "target_points",
                    "horizon_bars",
                    "gain_normalized",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
        )

        for (
            target_points,
            horizon,
        ), group in top_features.groupby(
            [
                "target_points",
                "horizon_bars",
            ]
        ):
            print()
            print(f"+{target_points:.1f} points / {horizon} bars")

            print(group.head(10).to_string(index=False))

    print()
    print("FILES SAVED")

    print(predictions_path)

    print(importance_path)

    print(deciles_path)

    print(baseline_path)

    print(summary_path)

    print(cross_window_path)

    print()
    print("IMPORTANT:")

    print("XGBoost was used only for discovery.")

    print("Economic targets were fixed before OOS evaluation.")

    print("No target threshold was optimized.")

    print("No final strategy was constructed.")

    print("=" * 100)


if __name__ == "__main__":
    main()
