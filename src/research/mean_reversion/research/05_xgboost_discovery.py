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
# MEAN REVERSION — RESEARCH 05
# ============================================================
#
# XGBOOST DISCOVERY
# ============================================================
#
# PURPOSE
# -------
# Determine whether the existing Mean Reversion feature set
# contains nonlinear predictive information that was not
# captured by the previous descriptive research.
#
# XGBoost is used ONLY as a discovery model.
#
# We are NOT:
#
#   - constructing the final strategy
#   - optimizing trading parameters
#   - selecting a final threshold
#   - optimizing stop / target
#   - tuning XGBoost against the OOS data
#
# The temporal structure is strictly preserved:
#
#       TRAIN: previous 2 years
#       OOS:   following 3 months
#
# repeated across the existing 22 windows.
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
# MODEL CONFIGURATION
# ============================================================
#
# These are intentionally conservative fixed model settings.
#
# They are NOT optimized.
#
# The purpose is to test whether predictive structure exists,
# not to squeeze maximum performance out of the model.
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
# TARGET
# ============================================================
#
# We use a directional future-return target.
#
# For each observation:
#
#       target = 1
#           if future 10-bar return > 0
#
#       target = 0
#           otherwise
#
# This is deliberately simple.
#
# We are NOT yet asking the model to predict whether a trade
# reaches a specific target.
#
# First question:
#
#       "Can the feature set predict the direction of the
#        subsequent short-term move?"
#
# Later research can test economically meaningful targets.
#
# ============================================================

TARGET_HORIZON = 10


# ============================================================
# BASE EVENT
# ============================================================
#
# XGBoost will focus on observations where there is a meaningful
# mean-reversion setup:
#
#       zscore_30 <= -1.5
#
# This follows the strongest directional phenomenon discovered
# in Research 03/04.
#
# IMPORTANT:
#
# The threshold is used as an exploratory event definition.
# It is NOT a final strategy parameter.
#
# ============================================================

BASE_ZSCORE_THRESHOLD = -1.5


# ============================================================
# FEATURES
# ============================================================
#
# ONLY features that exist in the current Feature Engine.
#
# No future_* columns are allowed.
#
# No target-derived information is allowed.
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
    # Price range / ATR
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
    # Rolling mean / deviation
    "mean_distance_5",
    "mean_distance_15",
    "mean_distance_30",
    "mean_distance_60",
    "zscore_5",
    "zscore_15",
    "zscore_30",
    "zscore_60",
    "abs_zscore_5",
    "abs_zscore_15",
    "abs_zscore_30",
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
    Generate the project's standard temporal structure:

        2 years training
        3 months OOS

    No random train/test split is permitted.
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
    Convert timestamps to New York time and retain RTH data.
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
# TARGET CONSTRUCTION
# ============================================================


def build_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the future directional target.

    target = 1
        future close > current close

    target = 0
        future close <= current close

    The future information is used ONLY for the target.

    It must NEVER enter MODEL_FEATURES.
    """

    df = df.copy()

    future_close = df["close"].shift(-TARGET_HORIZON)

    future_return = future_close / df["close"] - 1.0

    df["target_future_return"] = future_return

    df["target"] = (future_return > 0).astype(float)

    return df


# ============================================================
# DATA VALIDATION
# ============================================================


def validate_features(
    df: pd.DataFrame,
):
    """
    Verify that every requested model feature exists.
    """

    missing = [feature for feature in MODEL_FEATURES if feature not in df.columns]

    if missing:
        raise KeyError("Missing model features:\n" + "\n".join(missing))


# ============================================================
# MODEL
# ============================================================


def create_model():
    """
    Create a fixed XGBoost classifier.

    No hyperparameter search is performed.
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
# METRICS
# ============================================================


def calculate_classification_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict:
    """
    Calculate basic OOS classification metrics.

    The default decision threshold of 0.50 is used ONLY as a
    descriptive reference.

    The probability itself is retained because later research
    will examine whether higher predicted probabilities
    correspond to stronger economic outcomes.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    mask = np.isfinite(y_true) & np.isfinite(probability)

    y_true = y_true[mask]
    probability = probability[mask]

    if len(y_true) == 0:
        return {
            "observations": 0,
            "accuracy": np.nan,
            "brier_score": np.nan,
            "mean_probability": np.nan,
        }

    prediction = (probability >= 0.50).astype(float)

    accuracy = (prediction == y_true).mean()

    brier_score = np.mean((probability - y_true) ** 2)

    return {
        "observations": int(len(y_true)),
        "accuracy": float(accuracy),
        "brier_score": float(brier_score),
        "mean_probability": float(probability.mean()),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def extract_feature_importance(
    model: XGBClassifier,
    window_number: int,
) -> pd.DataFrame:
    """
    Extract XGBoost gain-based feature importance.

    Importance is saved per OOS window so that we can later
    determine whether the same features repeatedly matter.
    """

    booster = model.get_booster()

    gain = booster.get_score(importance_type="gain")

    rows = []

    for feature in MODEL_FEATURES:
        rows.append(
            {
                "window": window_number,
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

    return result.sort_values(
        "gain_normalized",
        ascending=False,
    )


# ============================================================
# PROBABILITY DECILES
# ============================================================


def probability_deciles(
    oos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze realized outcomes by model probability decile.

    This is one of the most important discovery tests.

    If XGBoost is finding useful structure, higher predicted
    probabilities should correspond to higher realized
    future returns / hit rates.

    Deciles are calculated separately inside each OOS window.
    """

    rows = []

    for window, group in oos.groupby("window"):
        group = group.copy()

        if len(group) < 20:
            continue

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
                    "window": window,
                    "probability_decile": (int(decile)),
                    "observations": len(subset),
                    "mean_probability": (subset["probability"].mean()),
                    "realized_positive_rate": (subset["target"].mean()),
                    "mean_future_points": (subset["future_points"].mean()),
                    "median_future_points": (subset["future_points"].median()),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 100)
    print("MEAN REVERSION — RESEARCH 05")
    print("=" * 100)

    print()
    print("XGBOOST DISCOVERY")
    print("-" * 100)

    print("No final strategy.")

    print("No parameter optimization.")

    print("No stop / target optimization.")

    print("22 walk-forward OOS windows.")

    print()
    print(f"Target horizon: {TARGET_HORIZON} bars")

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

    validate_features(df)

    # ========================================================
    # TARGET
    # ========================================================

    print()
    print("Building future target...")

    df = build_target(df)

    # ========================================================
    # FUTURE POINTS
    # ========================================================

    df["future_points"] = df["close"].shift(-TARGET_HORIZON) - df["close"]

    # ========================================================
    # WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print()
    print(f"Walk-forward windows: {len(windows)}")

    # ========================================================
    # OUTPUT STORAGE
    # ========================================================

    prediction_results = []

    importance_results = []

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

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()

        # ----------------------------------------------------
        # OOS
        # ----------------------------------------------------

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        # ----------------------------------------------------
        # Mean-reversion event
        # ----------------------------------------------------

        train = train.loc[train["zscore_30"] <= BASE_ZSCORE_THRESHOLD].copy()

        validation = validation.loc[
            validation["zscore_30"] <= BASE_ZSCORE_THRESHOLD
        ].copy()

        print(f"  Train events: {len(train):,}")

        print(f"  OOS events: {len(validation):,}")

        if len(train) < 100:
            print("  Skipping: insufficient training observations.")
            continue

        if len(validation) == 0:
            continue

        # ----------------------------------------------------
        # Remove rows with missing model data
        # ----------------------------------------------------

        train = train.dropna(
            subset=MODEL_FEATURES
            + [
                "target",
                "target_future_return",
            ]
        )

        validation = validation.dropna(
            subset=MODEL_FEATURES
            + [
                "target",
                "target_future_return",
            ]
        )

        if len(train) < 100:
            print("  Skipping: insufficient clean training observations.")
            continue

        if validation.empty:
            continue

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        X_train = train[MODEL_FEATURES]

        y_train = train["target"].astype(int)

        X_validation = validation[MODEL_FEATURES]

        y_validation = validation["target"].astype(int)

        model = create_model()

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        # ----------------------------------------------------
        # OOS PREDICTIONS
        # ----------------------------------------------------

        probability = model.predict_proba(X_validation)[:, 1]

        metrics = calculate_classification_metrics(
            y_validation,
            probability,
        )

        print(f"  OOS accuracy: {metrics['accuracy']:.4f}")

        print(f"  OOS Brier: {metrics['brier_score']:.4f}")

        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        result = pd.DataFrame(
            {
                "timestamp": validation.index,
                "window": number,
                "probability": probability,
                "target": y_validation.to_numpy(),
                "future_return": (validation["target_future_return"].to_numpy()),
                "future_points": (validation["future_points"].to_numpy()),
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

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        importance = extract_feature_importance(
            model,
            number,
        )

        importance_results.append(importance)

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

    # ========================================================
    # DECILES
    # ========================================================

    if not predictions.empty:
        deciles = probability_deciles(predictions)

    else:
        deciles = pd.DataFrame()

    # ========================================================
    # WINDOW METRICS
    # ========================================================

    window_metrics = []

    if not predictions.empty:
        for window, group in predictions.groupby("window"):
            metrics = calculate_classification_metrics(
                group["target"],
                group["probability"],
            )

            window_metrics.append(
                {
                    "window": window,
                    **metrics,
                    "mean_future_points": (group["future_points"].mean()),
                    "median_future_points": (group["future_points"].median()),
                }
            )

    window_metrics = pd.DataFrame(window_metrics)

    # ========================================================
    # SAVE
    # ========================================================

    predictions_path = RESULTS_DIR / "research_05_xgboost_predictions.csv"

    importance_path = RESULTS_DIR / "research_05_xgboost_feature_importance.csv"

    deciles_path = RESULTS_DIR / "research_05_xgboost_probability_deciles.csv"

    metrics_path = RESULTS_DIR / "research_05_xgboost_window_metrics.csv"

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

    window_metrics.to_csv(
        metrics_path,
        index=False,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print("RESEARCH 05 COMPLETE")
    print("=" * 100)

    print()

    print(f"OOS predictions: {len(predictions):,}")

    print(f"Feature importance rows: {len(feature_importance):,}")

    print()

    if not window_metrics.empty:
        print("OOS WINDOW METRICS")

        print(window_metrics.to_string(index=False))

    print()

    if not deciles.empty:
        print("PROBABILITY DECILES")

        print(
            deciles.groupby("probability_decile")[
                [
                    "mean_probability",
                    "realized_positive_rate",
                    "mean_future_points",
                ]
            ]
            .mean()
            .to_string()
        )

    print()

    if not feature_importance.empty:
        print("TOP FEATURES BY MEAN NORMALIZED GAIN")

        top_features = (
            feature_importance.groupby("feature")["gain_normalized"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )

        print(top_features.to_string())

    print()
    print("FILES SAVED")

    print(predictions_path)

    print(importance_path)

    print(deciles_path)

    print(metrics_path)

    print()
    print("IMPORTANT:")

    print("XGBoost was used only for discovery.")

    print("No final strategy was constructed.")

    print("No hyperparameter optimization was performed.")

    print("=" * 100)


if __name__ == "__main__":
    main()
