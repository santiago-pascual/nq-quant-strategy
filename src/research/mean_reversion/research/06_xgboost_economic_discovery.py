"""
MEAN REVERSION — RESEARCH 06
=============================

XGBOOST ECONOMIC-MAGNITUDE DISCOVERY

Purpose
-------
Determine whether the information available at the moment of a
mean-reversion event contains predictive information about the
SIZE of the subsequent favorable price excursion.

This is a RESEARCH experiment.

It is NOT a trading strategy.

IMPORTANT
---------
This experiment intentionally does NOT model:

    - stop losses
    - take profits
    - trade management
    - execution
    - slippage
    - commissions
    - intrabar target/stop ordering

Those belong to the subsequent path-dependent research stage.

The present experiment answers a simpler question:

    "Given a mean-reversion event, can the model distinguish
     situations that subsequently produce small versus large
     favorable excursions?"

Targets
-------
5 / 10 / 15 / 20 / 25 / 35 / 50 / 75 / 100 points

Horizons
--------
5 / 10 / 20 / 30 / 60 / 120 bars

Validation
----------
22 walk-forward OOS windows.

No:
    - hyperparameter optimization
    - target optimization
    - entry optimization
    - strategy construction
    - parameter selection based on OOS results

XGBoost is used ONLY as a discovery tool.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_POINTS = [
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    35.0,
    50.0,
    75.0,
    100.0,
]

HORIZONS = [
    5,
    10,
    20,
    30,
    60,
    120,
]

# Base event.
#
# We are deliberately keeping the event fixed.
# We are NOT searching for the best entry threshold here.
BASE_ZSCORE_THRESHOLD = -1.5

N_WINDOWS = 22

# Minimum number of training observations required.
MIN_TRAIN_EVENTS = 500

RANDOM_STATE = 42


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OUTPUT FILES
# =============================================================================

PREDICTIONS_FILE = (
    RESULTS_DIR / "research_06_xgboost_economic_magnitude_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR / "research_06_xgboost_economic_magnitude_feature_importance.csv"
)

DECILES_FILE = RESULTS_DIR / "research_06_xgboost_economic_magnitude_deciles.csv"

SUMMARY_FILE = RESULTS_DIR / "research_06_xgboost_economic_magnitude_summary.csv"

CROSS_WINDOW_FILE = (
    RESULTS_DIR / "research_06_xgboost_economic_magnitude_cross_window.csv"
)


# =============================================================================
# PRINT HELPERS
# =============================================================================


def separator(char: str = "=", length: int = 100) -> None:
    print(char * length)


def section(title: str) -> None:
    print()
    separator()
    print(title)
    separator()


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_rth(data: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the research universe to RTH.

    The mean-reversion research is designed around the regular
    trading session. ETH data is therefore excluded from the
    event universe.
    """

    if "market_period" not in data.columns:
        raise KeyError("Required column 'market_period' is missing.")

    result = data.loc[data["market_period"].eq("RTH")].copy()

    result = result.sort_values("timestamp ET")
    result = result.reset_index(drop=True)

    return result


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete frozen mean-reversion feature set.

    The feature engine contains the previously validated features:

        - returns
        - volatility
        - VWAP
        - rolling mean/std
        - z-scores
        - OU / AR features
        - etc.

    No future target information is supplied to this function.
    """

    result = build_mean_reversion_features(data.copy())

    return result


# =============================================================================
# EVENT DEFINITION
# =============================================================================


def build_events(data: pd.DataFrame) -> pd.DataFrame:
    """
    Define the fixed research event.

    Long-side mean-reversion event:

        zscore_30 <= -1.5

    This means price is sufficiently below its 30-bar rolling mean.

    IMPORTANT:
        This threshold is frozen for this experiment.

    We are NOT optimizing it.
    """

    if "zscore_30" not in data.columns:
        raise KeyError("Required feature 'zscore_30' is missing.")

    result = data.copy()

    result["research_event"] = result["zscore_30"] <= BASE_ZSCORE_THRESHOLD

    return result


# =============================================================================
# TARGET CONSTRUCTION
# =============================================================================


def build_target(
    data: pd.DataFrame,
    target_points: float,
    horizon: int,
) -> pd.Series:
    """
    Construct a future favorable-excursion target.

    For a LONG mean-reversion event:

        target = 1

    if the future HIGH reaches:

        current close + target_points

    at any point within the next `horizon` bars.

    Otherwise:

        target = 0

    IMPORTANT
    ---------
    This is a MAGNITUDE discovery label.

    It does NOT ask whether a real trade would survive a stop.

    Therefore:

        future HIGH

    is intentionally used here.

    Stop-loss / target ordering will be handled in the
    path-dependent experiment later.
    """

    close = data["close"]
    high = data["high"]

    future_high = pd.concat(
        [high.shift(-i) for i in range(1, horizon + 1)],
        axis=1,
    ).max(axis=1)

    target_price = close + target_points

    target = (future_high >= target_price).astype(float)

    # The final `horizon` observations do not have a complete
    # future window and therefore cannot be labeled.
    target.iloc[-horizon:] = np.nan

    return target


# =============================================================================
# FEATURE SELECTION
# =============================================================================


def select_features(data: pd.DataFrame) -> list[str]:
    """
    Select model features.

    Only information known at the current bar may enter XGBoost.

    Explicitly excluded:

        - future_* columns
        - target columns
        - research_event
        - timestamp
        - symbol
        - session metadata
    """

    excluded_prefixes = ("future_",)

    excluded_exact = {
        "timestamp ET",
        "symbol",
        "session_date",
        "market_period",
        "research_event",
    }

    features = []

    for column in data.columns:
        if column in excluded_exact:
            continue

        if column.startswith(excluded_prefixes):
            continue

        if column.startswith("target_"):
            continue

        if pd.api.types.is_numeric_dtype(data[column]):
            features.append(column)

    return features


# =============================================================================
# WALK-FORWARD WINDOWS
# =============================================================================


def build_walk_forward_windows(
    data: pd.DataFrame,
    n_windows: int = N_WINDOWS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Construct chronological walk-forward windows.

    Each window uses:

        all previous observations -> training
        current block             -> OOS

    No future information is used to train a given OOS window.
    """

    n = len(data)

    if n_windows < 2:
        raise ValueError("n_windows must be >= 2.")

    block_size = n // n_windows

    windows = []

    for i in range(n_windows):
        oos_start = i * block_size

        if i == n_windows - 1:
            oos_end = n
        else:
            oos_end = (i + 1) * block_size

        if oos_start == 0:
            # There is no historical training data for
            # the first block.
            continue

        train_idx = np.arange(0, oos_start)
        oos_idx = np.arange(oos_start, oos_end)

        windows.append((train_idx, oos_idx))

    return windows


# =============================================================================
# MODEL
# =============================================================================


def create_model() -> XGBClassifier:
    """
    Create the frozen XGBoost discovery model.

    These are intentionally fixed research parameters.

    They are NOT optimized against the OOS results.
    """

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================


def run_experiment(
    data: pd.DataFrame,
    features: list[str],
    target_points: float,
    horizon: int,
    windows: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one complete target/horizon experiment.

    Example:

        +35 points / 60 bars

    is treated as a completely separate discovery experiment.

    Every OOS window gets its own model trained only on
    information available before that OOS block.
    """

    target = build_target(
        data,
        target_points=target_points,
        horizon=horizon,
    )

    event_mask = data["research_event"]

    model_rows = []
    importance_rows = []

    for window_number, (train_idx, oos_idx) in enumerate(
        windows,
        start=1,
    ):
        train_mask = event_mask.iloc[train_idx] & target.iloc[train_idx].notna()

        oos_mask = event_mask.iloc[oos_idx] & target.iloc[oos_idx].notna()

        train_indices = train_idx[train_mask.to_numpy()]
        oos_indices = oos_idx[oos_mask.to_numpy()]

        print(
            f"    Window {window_number:02d}/{len(windows):02d}"
            f" | train={len(train_indices):,}"
            f" | OOS={len(oos_indices):,}"
        )

        if len(train_indices) < MIN_TRAIN_EVENTS:
            print("      skipped: insufficient training events")
            continue

        if len(oos_indices) == 0:
            print("      skipped: no OOS events")
            continue

        X_train = data.loc[
            train_indices,
            features,
        ].replace([np.inf, -np.inf], np.nan)

        X_oos = data.loc[
            oos_indices,
            features,
        ].replace([np.inf, -np.inf], np.nan)

        y_train = target.loc[train_indices]
        y_oos = target.loc[oos_indices]

        valid_train = X_train.notna().all(axis=1)
        valid_oos = X_oos.notna().all(axis=1)

        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        X_oos = X_oos.loc[valid_oos]
        y_oos = y_oos.loc[valid_oos]

        if len(X_train) < MIN_TRAIN_EVENTS:
            print("      skipped: insufficient valid training data")
            continue

        if len(X_oos) == 0:
            print("      skipped: no valid OOS data")
            continue

        # -------------------------------------------------------------
        # BASELINE
        # -------------------------------------------------------------

        baseline_probability = float(y_train.mean())

        # -------------------------------------------------------------
        # MODEL
        # -------------------------------------------------------------

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(X_oos)[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        actual = y_oos.to_numpy(dtype=int)

        accuracy = float((predictions == actual).mean())

        brier = float(np.mean((probabilities - actual) ** 2))

        probability_lift = probabilities.mean() - baseline_probability

        # -------------------------------------------------------------
        # FUTURE MFE
        # -------------------------------------------------------------

        future_high = pd.concat(
            [data["high"].shift(-i) for i in range(1, horizon + 1)],
            axis=1,
        ).max(axis=1)

        future_mfe = future_high.loc[X_oos.index] - data.loc[X_oos.index, "close"]

        # -------------------------------------------------------------
        # SAVE OOS PREDICTIONS
        # -------------------------------------------------------------

        for idx, probability, actual_value, mfe in zip(
            X_oos.index,
            probabilities,
            actual,
            future_mfe,
        ):
            model_rows.append(
                {
                    "window": window_number,
                    "target_points": target_points,
                    "horizon_bars": horizon,
                    "timestamp": data.loc[
                        idx,
                        "timestamp ET",
                    ],
                    "probability": float(probability),
                    "actual_target": int(actual_value),
                    "baseline_probability": (baseline_probability),
                    "future_mfe": float(mfe),
                }
            )

        # -------------------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------------------

        importances = model.feature_importances_

        total_gain = importances.sum()

        if total_gain > 0:
            normalized = importances / total_gain
        else:
            normalized = importances

        for feature, importance in zip(
            features,
            normalized,
        ):
            importance_rows.append(
                {
                    "window": window_number,
                    "target_points": target_points,
                    "horizon_bars": horizon,
                    "feature": feature,
                    "gain_normalized": float(importance),
                }
            )

        print(
            f"      baseline={baseline_probability:.4f}"
            f" | mean_p={probabilities.mean():.4f}"
            f" | lift={probability_lift:+.4f}"
            f" | accuracy={accuracy:.4f}"
            f" | brier={brier:.4f}"
        )

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(importance_rows),
    )


# =============================================================================
# DECILES
# =============================================================================


def calculate_deciles(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Divide predictions into probability deciles.

    This asks:

        Do higher model probabilities correspond to
        higher realized target frequencies?

    This is particularly important because raw accuracy
    can hide useful probability ranking.
    """

    if predictions.empty:
        return pd.DataFrame()

    frames = []

    grouped = predictions.groupby(
        [
            "target_points",
            "horizon_bars",
        ],
    )

    for (
        target_points,
        horizon,
    ), group in grouped:
        if len(group) < 20:
            continue

        group = group.copy()

        group["probability_decile"] = pd.qcut(
            group["probability"],
            q=10,
            labels=False,
            duplicates="drop",
        )

        summary = (
            group.groupby("probability_decile")
            .agg(
                mean_probability=(
                    "probability",
                    "mean",
                ),
                realized_probability=(
                    "actual_target",
                    "mean",
                ),
                mean_future_mfe=(
                    "future_mfe",
                    "mean",
                ),
                observations=(
                    "actual_target",
                    "size",
                ),
            )
            .reset_index()
        )

        summary["target_points"] = target_points
        summary["horizon_bars"] = horizon

        frames.append(summary)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =============================================================================
# CROSS-WINDOW SUMMARY
# =============================================================================


def calculate_cross_window_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate model performance across the 22 OOS windows.

    The unit of analysis is the WINDOW.

    This prevents the largest windows from completely dominating
    the summary.
    """

    if predictions.empty:
        return pd.DataFrame()

    rows = []

    for (
        target_points,
        horizon,
        window,
    ), group in predictions.groupby(
        [
            "target_points",
            "horizon_bars",
            "window",
        ],
    ):
        baseline = group["baseline_probability"].iloc[0]

        realized = group["actual_target"].mean()

        mean_probability = group["probability"].mean()

        probability_lift = mean_probability - baseline

        # Top probability decile.
        if len(group) >= 10:
            threshold = group["probability"].quantile(0.9)

            top = group[group["probability"] >= threshold]

            top_lift = top["actual_target"].mean() - baseline

        else:
            top_lift = np.nan

        rows.append(
            {
                "target_points": target_points,
                "horizon_bars": horizon,
                "window": window,
                "observations": len(group),
                "mean_realized_probability": realized,
                "mean_model_probability": mean_probability,
                "baseline_probability": baseline,
                "probability_lift": probability_lift,
                "top_decile_lift": top_lift,
                "mean_future_mfe": group["future_mfe"].mean(),
                "median_future_mfe": group["future_mfe"].median(),
            }
        )

    result = pd.DataFrame(rows)

    return result


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    section("MEAN REVERSION — RESEARCH 06")

    print("XGBOOST ECONOMIC-MAGNITUDE DISCOVERY")

    print()
    print("No final strategy.")
    print("No parameter optimization.")
    print("No stop / target optimization.")
    print("22 walk-forward OOS windows.")

    print()
    print(f"Base event: zscore_30 <= {BASE_ZSCORE_THRESHOLD}")

    print("Targets: " + ", ".join(f"{x:.0f}" for x in TARGET_POINTS) + " points")

    print("Horizons: " + ", ".join(str(x) for x in HORIZONS) + " bars")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    section("LOADING MNQ DATA")

    print("Loading MNQ data...")

    data = load_data()

    print(f"Rows loaded: {len(data):,}")

    # -------------------------------------------------------------------------
    # RTH
    # -------------------------------------------------------------------------

    section("PREPARING RTH")

    data = prepare_rth(data)

    print(f"RTH rows: {len(data):,}")

    # -------------------------------------------------------------------------
    # FEATURES
    # -------------------------------------------------------------------------

    section("BUILDING COMPLETE FEATURE SET")

    data = build_features(data)

    print(f"Feature columns: {len(data.columns):,}")

    # -------------------------------------------------------------------------
    # EVENT
    # -------------------------------------------------------------------------

    data = build_events(data)

    event_count = int(data["research_event"].sum())

    print(f"Research events: {event_count:,}")

    # -------------------------------------------------------------------------
    # FEATURES
    # -------------------------------------------------------------------------

    features = select_features(data)

    print(f"Model features: {len(features):,}")

    # -------------------------------------------------------------------------
    # WALK FORWARD
    # -------------------------------------------------------------------------

    windows = build_walk_forward_windows(
        data,
        n_windows=N_WINDOWS,
    )

    print(f"Walk-forward windows: {len(windows)}")

    # -------------------------------------------------------------------------
    # EXPERIMENT LOOP
    # -------------------------------------------------------------------------

    all_predictions = []
    all_importances = []

    experiment_count = len(TARGET_POINTS) * len(HORIZONS)

    experiment_number = 0

    for target_points in TARGET_POINTS:
        for horizon in HORIZONS:
            experiment_number += 1

            section(f"EXPERIMENT {experiment_number}/{experiment_count}")

            print(f"TARGET = +{target_points:.0f} POINTS")

            print(f"HORIZON = {horizon} BARS")

            predictions, importances = run_experiment(
                data=data,
                features=features,
                target_points=target_points,
                horizon=horizon,
                windows=windows,
            )

            if not predictions.empty:
                all_predictions.append(predictions)

            if not importances.empty:
                all_importances.append(importances)

    # -------------------------------------------------------------------------
    # COMBINE
    # -------------------------------------------------------------------------

    section("RESEARCH 06 COMPLETE")

    if all_predictions:
        predictions = pd.concat(
            all_predictions,
            ignore_index=True,
        )
    else:
        predictions = pd.DataFrame()

    if all_importances:
        importances = pd.concat(
            all_importances,
            ignore_index=True,
        )
    else:
        importances = pd.DataFrame()

    print(f"OOS predictions: {len(predictions):,}")

    print(f"Feature importance rows: {len(importances):,}")

    # -------------------------------------------------------------------------
    # DECILES
    # -------------------------------------------------------------------------

    deciles = calculate_deciles(predictions)

    # -------------------------------------------------------------------------
    # CROSS WINDOW
    # -------------------------------------------------------------------------

    cross_window = calculate_cross_window_summary(predictions)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    if not cross_window.empty:
        summary = (
            cross_window.groupby(
                [
                    "target_points",
                    "horizon_bars",
                ]
            )
            .agg(
                windows=(
                    "window",
                    "nunique",
                ),
                mean_realized_probability=(
                    "mean_realized_probability",
                    "mean",
                ),
                mean_baseline_probability=(
                    "baseline_probability",
                    "mean",
                ),
                mean_probability_lift=(
                    "probability_lift",
                    "mean",
                ),
                median_probability_lift=(
                    "probability_lift",
                    "median",
                ),
                positive_lift_window_ratio=(
                    "probability_lift",
                    lambda x: (x > 0).mean(),
                ),
                mean_top_decile_lift=(
                    "top_decile_lift",
                    "mean",
                ),
                positive_top_decile_lift_ratio=(
                    "top_decile_lift",
                    lambda x: (x > 0).mean(),
                ),
                mean_future_mfe=(
                    "mean_future_mfe",
                    "mean",
                ),
                median_future_mfe=(
                    "median_future_mfe",
                    "median",
                ),
            )
            .reset_index()
        )

    else:
        summary = pd.DataFrame()

    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE SUMMARY
    # -------------------------------------------------------------------------

    if not importances.empty:
        feature_summary = (
            importances.groupby(
                [
                    "target_points",
                    "horizon_bars",
                    "feature",
                ]
            )["gain_normalized"]
            .mean()
            .reset_index()
        )

        feature_summary = feature_summary.sort_values(
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

    else:
        feature_summary = pd.DataFrame()

    # -------------------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------------------

    section("CROSS-WINDOW ECONOMIC RESULTS")

    if not summary.empty:
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    section("TOP FEATURES BY EXPERIMENT")

    if not feature_summary.empty:
        for (
            target_points,
            horizon,
        ), group in feature_summary.groupby(
            [
                "target_points",
                "horizon_bars",
            ]
        ):
            print()
            print(f"+{target_points:.0f} points / {horizon} bars")

            print(
                group.head(10).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}",
                )
            )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    feature_summary.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    deciles.to_csv(
        DECILES_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    cross_window.to_csv(
        CROSS_WINDOW_FILE,
        index=False,
    )

    section("FILES SAVED")

    print(PREDICTIONS_FILE)
    print(FEATURE_IMPORTANCE_FILE)
    print(DECILES_FILE)
    print(SUMMARY_FILE)
    print(CROSS_WINDOW_FILE)

    print()
    print("IMPORTANT:")
    print("XGBoost was used only for discovery.")
    print("Economic targets were fixed before OOS evaluation.")
    print("No target threshold was optimized.")
    print("No stop / target optimization was performed.")
    print("No final strategy was constructed.")
    print("Path-dependent stop/target ordering belongs to the next research stage.")

    separator()


if __name__ == "__main__":
    main()
