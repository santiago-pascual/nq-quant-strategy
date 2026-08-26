from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_END = pd.Timestamp(
    "2024-12-31 23:59:00",
    tz="America/New_York",
)

RANDOM_STATE = 42

N_STATES = 3

# We deliberately start with a relatively small model.
# The goal is to test whether nonlinear information exists,
# NOT to maximize in-sample fit.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# FEATURE SET
# ============================================================

BASE_FEATURES = [
    "past_return_5",
    "past_return_10",
    "past_return_15",
    "past_return_30",
    "directional_pressure_5",
    "directional_pressure_10",
    "directional_pressure_15",
    "directional_pressure_30",
    "direction_streak",
    "up_streak",
    "down_streak",
    "close_location_5",
    "close_location_10",
    "close_location_15",
    "close_location_30",
    "normalized_momentum_10",
    "normalized_momentum_15",
    "normalized_momentum_30",
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "vol_ratio_5_30",
    "vol_ratio_5_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
]


# ============================================================
# DATA PREPARATION
# ============================================================


def get_timestamp_series(df: pd.DataFrame) -> pd.Series:

    if "timestamp ET" not in df.columns:
        raise KeyError("Expected 'timestamp ET' column.")

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    return timestamps


def prepare_rth_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_timestamp_et"] = get_timestamp_series(df)

    if "market_period" not in df.columns:
        raise KeyError("Missing market_period column.")

    rth = df.loc[df["market_period"] == "RTH"].copy()

    rth = rth.sort_values("_timestamp_et")

    rth = rth.set_index("_timestamp_et")

    rth.index.name = "timestamp_et"

    return rth


# ============================================================
# TARGET CONSTRUCTION
# ============================================================


def add_direction_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary directional targets.

    LONG target:
        1 if future_return_15 > 0
        0 otherwise

    SHORT target:
        1 if future_return_15 < 0
        0 otherwise

    We start with the 15-bar horizon because the previous
    research showed the strongest recurring directional
    relationships around that horizon.

    This is classification rather than regression because
    we care about identifying repeatable direction, not
    predicting gigantic percentage moves.
    """

    df = df.copy()

    if "future_return_15" not in df.columns:
        raise KeyError("Missing future_return_15.")

    future_return = df["future_return_15"]

    df["target_long"] = (future_return > 0).astype(int)

    df["target_short"] = (future_return < 0).astype(int)

    return df


# ============================================================
# HMM REGIME
# ============================================================


def add_hmm_states(
    train: pd.DataFrame,
    oos: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    VolatilityRegimeModel,
]:
    """
    Fit HMM exclusively on TRAIN.

    The fitted model is then applied to OOS.

    This prevents OOS information from entering the
    regime model during training.
    """

    hmm = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    hmm.fit(train)

    train = train.copy()
    oos = oos.copy()

    train_states = hmm.predict_states(train)

    oos_states = hmm.predict_states(oos)

    train["hmm_state"] = train_states

    oos["hmm_state"] = oos_states

    return (
        train,
        oos,
        hmm,
    )


# ============================================================
# FEATURES
# ============================================================


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    missing = [feature for feature in BASE_FEATURES if feature not in df.columns]

    if missing:
        raise KeyError("Missing directional/XGBoost features:\n" + "\n".join(missing))

    return BASE_FEATURES + [
        "hmm_state",
    ]


def prepare_model_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Prepare model matrix.

    No forward-looking feature is constructed here.

    Future return exists only as the target.
    """

    required = feature_columns + [target_column]

    data = df[required].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna()

    X = data[feature_columns].copy()

    y = data[target_column].astype(int)

    return X, y


# ============================================================
# MODEL
# ============================================================


def build_model() -> XGBClassifier:

    return XGBClassifier(**XGB_PARAMS)


def evaluate_probability_signal(
    probabilities: np.ndarray,
    actual_returns: pd.Series,
    threshold: float,
    direction: str,
) -> dict[str, float]:
    """
    Evaluate a simple probability threshold.

    This is NOT the final trading strategy.

    It is only a first diagnostic of whether high-confidence
    model predictions contain directional expectancy.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    returns = actual_returns.to_numpy(dtype=float)

    if direction == "long":
        mask = probabilities >= threshold

        trade_returns = returns[mask]

    elif direction == "short":
        mask = probabilities >= threshold

        trade_returns = -returns[mask]

    else:
        raise ValueError("direction must be 'long' or 'short'.")

    trade_returns = trade_returns[np.isfinite(trade_returns)]

    if len(trade_returns) == 0:
        return {
            "trades": 0,
            "mean_return": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    positive = trade_returns[trade_returns > 0]

    negative = trade_returns[trade_returns < 0]

    gross_profit = positive.sum()

    gross_loss = -negative.sum()

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    return {
        "trades": len(trade_returns),
        "mean_return": trade_returns.mean(),
        "win_rate": (trade_returns > 0).mean(),
        "profit_factor": profit_factor,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)

    print("XGBOOST DIRECTIONAL MODEL — OOS VALIDATION")

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    print("\n=== ADDING DIRECTIONAL FEATURES ===")

    df = add_directional_features(df)

    rth = prepare_rth_data(df)

    rth = add_direction_targets(rth)

    print(f"\nRTH observations: {len(rth)}")

    print(f"Start: {rth.index.min()}")

    print(f"End: {rth.index.max()}")

    # --------------------------------------------------------
    # TRAIN / OOS SPLIT
    # --------------------------------------------------------

    train = rth.loc[rth.index <= TRAIN_END].copy()

    oos = rth.loc[rth.index > TRAIN_END].copy()

    print("\n=== DATA SPLIT ===")

    print(f"Train observations: {len(train)}")

    print(f"Train start: {train.index.min()}")

    print(f"Train end: {train.index.max()}")

    print(f"OOS observations: {len(oos)}")

    print(f"OOS start: {oos.index.min()}")

    print(f"OOS end: {oos.index.max()}")

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    print("\n=== FITTING HMM ON TRAIN ===")

    (
        train,
        oos,
        hmm,
    ) = add_hmm_states(
        train,
        oos,
    )

    print(f"Converged: {hmm.model.monitor_.converged}")

    print(f"Iterations: {hmm.model.monitor_.iter}")

    print("\nTrain regime proportions:")

    print(train["hmm_state"].value_counts(normalize=True).sort_index())

    print("\nOOS regime proportions:")

    print(oos["hmm_state"].value_counts(normalize=True).sort_index())

    # --------------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------------

    feature_columns = get_feature_columns(train)

    print("\n=== MODEL FEATURES ===")

    for feature in feature_columns:
        print(f"  {feature}")

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    X_train_long, y_train_long = prepare_model_data(
        train,
        feature_columns,
        "target_long",
    )

    X_oos_long, y_oos_long = prepare_model_data(
        oos,
        feature_columns,
        "target_long",
    )

    X_train_short, y_train_short = prepare_model_data(
        train,
        feature_columns,
        "target_short",
    )

    X_oos_short, y_oos_short = prepare_model_data(
        oos,
        feature_columns,
        "target_short",
    )

    print("\n=== MODEL DATA ===")

    print(f"LONG train: {len(X_train_long)}")

    print(f"LONG OOS: {len(X_oos_long)}")

    print(f"SHORT train: {len(X_train_short)}")

    print(f"SHORT OOS: {len(X_oos_short)}")

    # --------------------------------------------------------
    # LONG MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("TRAINING LONG MODEL")

    print("=" * 70)

    long_model = build_model()

    long_model.fit(
        X_train_long,
        y_train_long,
        verbose=False,
    )

    long_probability = long_model.predict_proba(X_oos_long)[:, 1]

    print("\nLONG probability statistics:")

    print(pd.Series(long_probability).describe())

    # --------------------------------------------------------
    # SHORT MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("TRAINING SHORT MODEL")

    print("=" * 70)

    short_model = build_model()

    short_model.fit(
        X_train_short,
        y_train_short,
        verbose=False,
    )

    short_probability = short_model.predict_proba(X_oos_short)[:, 1]

    print("\nSHORT probability statistics:")

    print(pd.Series(short_probability).describe())

    # --------------------------------------------------------
    # OOS DIAGNOSTICS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("OOS PROBABILITY DIAGNOSTICS")

    print("=" * 70)

    thresholds = [
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    # Actual future returns corresponding to the
    # valid OOS observations.

    long_returns = oos.loc[
        X_oos_long.index,
        "future_return_15",
    ]

    short_returns = oos.loc[
        X_oos_short.index,
        "future_return_15",
    ]

    print("\nLONG MODEL")

    for threshold in thresholds:
        metrics = evaluate_probability_signal(
            long_probability,
            long_returns,
            threshold,
            "long",
        )

        print(f"\nThreshold: {threshold:.2f}")

        print(f"Trades: {metrics['trades']}")

        print(f"Mean return: {metrics['mean_return']:.10f}")

        print(f"Win rate: {metrics['win_rate']:.4%}")

        print(f"Profit factor: {metrics['profit_factor']:.4f}")

    print("\nSHORT MODEL")

    for threshold in thresholds:
        metrics = evaluate_probability_signal(
            short_probability,
            short_returns,
            threshold,
            "short",
        )

        print(f"\nThreshold: {threshold:.2f}")

        print(f"Trades: {metrics['trades']}")

        print(f"Mean return: {metrics['mean_return']:.10f}")

        print(f"Win rate: {metrics['win_rate']:.4%}")

        print(f"Profit factor: {metrics['profit_factor']:.4f}")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("LONG FEATURE IMPORTANCE")

    print("=" * 70)

    long_importance = pd.Series(
        long_model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    print(long_importance)

    print("\n" + "=" * 70)

    print("SHORT FEATURE IMPORTANCE")

    print("=" * 70)

    short_importance = pd.Series(
        short_model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    print(short_importance)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("XGBOOST OOS VALIDATION COMPLETE")

    print("=" * 70)

    print("\nThis experiment is diagnostic only.")

    print("No optimized trading threshold has been selected.")

    print("No stop-loss, take-profit, position sizing, or")

    print("funded-account execution rules have been applied.")


if __name__ == "__main__":
    main()
