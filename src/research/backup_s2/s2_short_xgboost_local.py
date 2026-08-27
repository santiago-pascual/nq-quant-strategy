from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 SHORT — SPECIALIZED XGBOOST META MODEL
# ============================================================
#
# IMPORTANT:
#
# This is NOT a generic directional XGBoost model.
#
# The base strategy is FIXED:
#
#   HMM State 2
#   AND past_return_30 in lower tail
#   AND directional_pressure_30 in lower tail
#   AND close_location_30 in lower tail
#   AND normalized_momentum_30 in lower tail
#
# XGBoost is allowed to operate ONLY inside this population.
#
# Its job:
#
#   1. Separate better trades from worse trades.
#   2. Identify characteristics of losing trades.
#   3. Produce a quality probability.
#   4. Test whether filtering by that probability improves
#      WR / expectancy / PF / drawdown OOS.
#
# PAYOFF REGION:
#
#   RR: 1.20 / 1.25 / 1.30 / 1.35
#   Horizon: 10 / 12 / 15 bars
#
# We are NOT reopening the previously rejected RR region.
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

N_STATES = 3
STATE = 2

STOP_POINTS = 20.0

TAIL_QUANTILES = [
    0.90,
    0.875,
    0.85,
    0.825,
    0.80,
]

RR_VALUES = [
    1.20,
    1.25,
    1.30,
    1.35,
]

HORIZONS = [
    10,
    12,
    15,
]

# Probability filters applied to OOS predictions.
#
# We deliberately don't optimize an absurdly fine grid.
#
PROBABILITY_THRESHOLDS = [
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60,
]

# Minimum number of OOS trades for a candidate to be considered.
MIN_TOTAL_TRADES = 100


# ============================================================
# FIXED BASE CONDITIONS
# ============================================================

BASE_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


# ============================================================
# XGBOOST FEATURES
# ============================================================
#
# These are NOT used to discover a new strategy.
#
# They refine the already-valid S2-short population.
#
# The 30-bar base conditions remain mandatory.
#
# We allow XGBoost to see:
#
#   - exact magnitude of the existing conditions
#   - shorter-term versions of the same concepts
#   - HMM probability
#   - volatility context
#   - volatility ratios
#   - streak/context information
#
# No generic long/short target is used.
# ============================================================

XGB_FEATURES = [
    # --------------------------------------------------------
    # Exact magnitude of proven 30-bar conditions
    # --------------------------------------------------------
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
    # --------------------------------------------------------
    # Shorter-term confirmation
    # --------------------------------------------------------
    "past_return_5",
    "past_return_10",
    "past_return_15",
    "directional_pressure_5",
    "directional_pressure_10",
    "directional_pressure_15",
    "close_location_5",
    "close_location_10",
    "close_location_15",
    "normalized_momentum_10",
    "normalized_momentum_15",
    # --------------------------------------------------------
    # Volatility context
    # --------------------------------------------------------
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "vol_ratio_5_30",
    "vol_ratio_5_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
    # --------------------------------------------------------
    # Directional persistence
    # --------------------------------------------------------
    "direction_streak",
    "up_streak",
    "down_streak",
]


# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_rth(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    required = [
        "timestamp ET",
        "market_period",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError("Missing required columns:\n" + "\n".join(missing))

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.isna().all():
        raise ValueError("Could not parse timestamp ET.")

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
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
# WALK-FORWARD WINDOWS
# ============================================================


def generate_windows(df):

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
# HMM
# ============================================================


def fit_hmm(
    train,
    validation,
):

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    # TRAIN ONLY
    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    # --------------------------------------------------------
    # HMM probabilities.
    #
    # These are useful because:
    #
    # State == 2
    #
    # is discrete, while probability tells us how confident
    # the model is about the state.
    # --------------------------------------------------------

    train_probs = model.predict_probabilities(train)

    validation_probs = model.predict_probabilities(validation)

    for column in train_probs.columns:
        train[column] = train_probs[column]

        validation[column] = validation_probs[column]

    return train, validation


# ============================================================
# TRAIN-ONLY BASE THRESHOLDS
# ============================================================


def calculate_base_thresholds(
    train,
    quantile,
):

    state_train = train.loc[train["hmm_state"] == STATE]

    if state_train.empty:
        raise ValueError(f"No State {STATE} observations in training data.")

    lower_quantile = 1.0 - quantile

    thresholds = {}

    for feature in BASE_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if values.empty:
            raise ValueError(f"No valid values for {feature}.")

        thresholds[feature] = float(values.quantile(lower_quantile))

    return thresholds


# ============================================================
# BASE SETUP GATE
# ============================================================


def base_setup_mask(
    df,
    thresholds,
):

    mask = df["hmm_state"] == STATE

    for feature in BASE_FEATURES:
        mask &= df[feature] <= thresholds[feature]

    return mask


# ============================================================
# BUILD TRADE OUTCOME
# ============================================================


def resolve_trade(
    session,
    entry_position,
    target_points,
    horizon,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = close[entry_position]

    target_price = entry_price - target_points

    stop_price = entry_price + STOP_POINTS

    last_position = min(
        entry_position + horizon,
        len(session) - 1,
    )

    for i in range(
        entry_position + 1,
        last_position + 1,
    ):
        target_hit = low[i] <= target_price

        stop_hit = high[i] >= stop_price

        if target_hit and stop_hit:
            return {
                "pnl_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        if target_hit:
            return {
                "pnl_points": target_points,
                "reason": "target",
                "exit_position": i,
            }

        if stop_hit:
            return {
                "pnl_points": -STOP_POINTS,
                "reason": "stop",
                "exit_position": i,
            }

    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    return {
        "pnl_points": pnl_points,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# BUILD DATASET OF VALID SETUPS
# ============================================================
#
# For every valid base setup we create a trade observation.
#
# The XGBoost target is:
#
#     1 = trade reaches target before stop
#     0 = it doesn't
#
# This means XGBoost is learning:
#
# "Which already-valid setups are likely to win?"
#
# ============================================================


def build_trade_dataset(
    df,
    thresholds,
    rr,
    horizon,
    window_number,
):

    target_points = STOP_POINTS * rr

    records = []

    for session_id, session in df.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= horizon:
            continue

        mask = base_setup_mask(
            session,
            thresholds,
        )

        positions = np.flatnonzero(mask.to_numpy())

        # ----------------------------------------------------
        # Non-overlapping setup population.
        # ----------------------------------------------------

        i = 0

        while i < len(positions):
            position = positions[i]

            if position >= len(session) - horizon:
                break

            row = session.iloc[position]

            result = resolve_trade(
                session=session,
                entry_position=position,
                target_points=target_points,
                horizon=horizon,
            )

            # ------------------------------------------------
            # Features
            # ------------------------------------------------

            record = {}

            for feature in XGB_FEATURES:
                if feature not in session.columns:
                    continue

                record[feature] = row[feature]

            # HMM probabilities
            for state in range(N_STATES):
                column = f"state_probability_{state}"

                if column in row.index:
                    record[column] = row[column]

            # ------------------------------------------------
            # Useful derived features.
            #
            # These measure how far inside the proven region
            # the setup sits.
            # ------------------------------------------------

            for feature in BASE_FEATURES:
                value = row[feature]

                threshold = thresholds[feature]

                record[f"{feature}_distance"] = value - threshold

            # ------------------------------------------------
            # Outcome
            # ------------------------------------------------

            record["target"] = int(result["reason"] == "target")

            record["pnl_R"] = result["pnl_points"] / STOP_POINTS

            record["pnl_points"] = result["pnl_points"]

            record["entry_timestamp"] = session.index[position]

            record["exit_timestamp"] = session.index[result["exit_position"]]

            record["holding_bars"] = result["exit_position"] - position

            record["exit_reason"] = result["reason"]

            record["window"] = window_number

            record["rr"] = rr

            record["horizon"] = horizon

            records.append(record)

            # ------------------------------------------------
            # IMPORTANT:
            #
            # No overlapping trades.
            # Move after the trade exits.
            # ------------------------------------------------

            exit_position = result["exit_position"]

            next_positions = positions[positions > exit_position]

            if len(next_positions) == 0:
                break

            next_position = next_positions[0]

            i = np.searchsorted(
                positions,
                next_position,
            )

    return pd.DataFrame(records)


# ============================================================
# CLEAN XGBOOST DATA
# ============================================================


def prepare_xgb_data(
    train_df,
    validation_df,
):

    excluded = {
        "target",
        "pnl_R",
        "pnl_points",
        "entry_timestamp",
        "exit_timestamp",
        "exit_reason",
        "window",
        "rr",
        "horizon",
    }

    feature_columns = [c for c in train_df.columns if c not in excluded]

    # Keep only numerical columns.
    feature_columns = [
        c for c in feature_columns if pd.api.types.is_numeric_dtype(train_df[c])
    ]

    X_train = train_df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_validation = validation_df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median imputation using TRAIN only.
    medians = X_train.median()

    X_train = X_train.fillna(medians)

    X_validation = X_validation.fillna(medians)

    y_train = train_df["target"].astype(int)

    y_validation = validation_df["target"].astype(int)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


# ============================================================
# XGBOOST
# ============================================================


def train_xgboost(
    X_train,
    y_train,
):

    model = XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=8,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades,
):

    if trades.empty:
        return None

    pnl = trades["pnl_R"].astype(float)

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    if gross_loss > 0:
        pf = gross_profit / gross_loss

    else:
        pf = np.inf

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    # --------------------------------------------------------
    # Losing streak
    # --------------------------------------------------------

    longest_streak = 0
    current_streak = 0

    for value in pnl:
        if value < 0:
            current_streak += 1

            longest_streak = max(
                longest_streak,
                current_streak,
            )

        else:
            current_streak = 0

    # --------------------------------------------------------
    # Daily
    # --------------------------------------------------------

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    if len(daily) > 1 and daily.std(ddof=1) > 0:
        daily_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)

    else:
        daily_sharpe = np.nan

    return {
        "trades": len(trades),
        "win_rate": (pnl > 0).mean(),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "profit_factor": pf,
        "max_drawdown_R": drawdown.min(),
        "longest_losing_streak": longest_streak,
        "average_holding_bars": trades["holding_bars"].mean(),
        "profitable_days": int((daily > 0).sum()),
        "losing_days": int((daily < 0).sum()),
        "daily_sharpe": daily_sharpe,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 SHORT — SPECIALIZED XGBOOST META MODEL")
    print("=" * 110)

    print("\nFIXED BASE STRATEGY:")

    print("HMM State 2")

    print("AND all four 30-bar bearish conditions")

    print("\nXGBoost is NOT searching the whole market.")

    print("It is learning inside the already-proven setup.")

    print("\nLOCAL PAYOFF REGION:")

    print("RR: 1.20 / 1.25 / 1.30 / 1.35")

    print("Horizon: 10 / 12 / 15 bars")

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()

    df = prepare_rth(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    # ========================================================
    # FEATURES
    # ========================================================

    print("\n=== ADDING DIRECTIONAL FEATURES ===")

    df = add_directional_features(df)

    # ========================================================
    # WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_results = []
    all_trades = []

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for (
        window_number,
        (
            train_start,
            validation_start,
            validation_end,
        ),
    ) in enumerate(
        windows,
        start=1,
    ):
        print("\n" + "#" * 110)

        print(f"WINDOW {window_number}")

        print("#" * 110)

        train = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        print(f"Train: {train_start.date()} → {validation_start.date()}")

        print(f"Validation: {validation_start.date()} → {validation_end.date()}")

        # ----------------------------------------------------
        # HMM TRAIN ONLY
        # ----------------------------------------------------

        (
            train,
            validation,
        ) = fit_hmm(
            train,
            validation,
        )

        # ----------------------------------------------------
        # LOOP THROUGH NARROW PAYOFF REGION
        # ----------------------------------------------------

        for quantile in TAIL_QUANTILES:
            thresholds = calculate_base_thresholds(
                train,
                quantile,
            )

            print(f"\nTAIL = {(1 - quantile) * 100:.1f}%")

            # ------------------------------------------------
            # We train one model for each payoff configuration.
            #
            # This prevents the model from being asked to
            # predict multiple different definitions of "win"
            # at once.
            # ------------------------------------------------

            for rr in RR_VALUES:
                for horizon in HORIZONS:
                    print(f"\n  RR={rr:.2f} H={horizon}")

                    # ========================================
                    # BUILD TRAIN TRADE POPULATION
                    # ========================================

                    train_trades = build_trade_dataset(
                        train,
                        thresholds,
                        rr,
                        horizon,
                        window_number,
                    )

                    # ========================================
                    # BUILD OOS TRADE POPULATION
                    # ========================================

                    validation_trades = build_trade_dataset(
                        validation,
                        thresholds,
                        rr,
                        horizon,
                        window_number,
                    )

                    if train_trades.empty or validation_trades.empty:
                        print("    No valid setups.")

                        continue

                    # ------------------------------------------------
                    # Need both classes.
                    # ------------------------------------------------

                    if train_trades["target"].nunique() < 2:
                        print("    Training population contains only one class.")

                        continue

                    # ========================================
                    # XGBOOST DATA
                    # ========================================

                    (
                        X_train,
                        y_train,
                        X_validation,
                        y_validation,
                        feature_columns,
                    ) = prepare_xgb_data(
                        train_trades,
                        validation_trades,
                    )

                    # ========================================
                    # TRAIN MODEL
                    # ========================================

                    model = train_xgboost(
                        X_train,
                        y_train,
                    )

                    # ========================================
                    # OOS PREDICTIONS
                    # ========================================

                    probabilities = model.predict_proba(X_validation)[:, 1]

                    validation_trades = validation_trades.copy()

                    validation_trades["xgb_probability"] = probabilities

                    # ========================================
                    # AUC
                    # ========================================

                    if y_validation.nunique() >= 2:
                        auc = roc_auc_score(
                            y_validation,
                            probabilities,
                        )

                    else:
                        auc = np.nan

                    print(f"    Train setups: {len(train_trades)}")

                    print(f"    OOS setups: {len(validation_trades)}")

                    print(f"    OOS AUC: {auc:.4f}")

                    # ========================================
                    # BASELINE
                    # ========================================

                    baseline_metrics = calculate_metrics(validation_trades)

                    if baseline_metrics:
                        baseline_metrics.update(
                            {
                                "window": window_number,
                                "tail_percent": (1 - quantile) * 100,
                                "rr": rr,
                                "horizon": horizon,
                                "probability_threshold": 0.0,
                                "model": "baseline",
                                "auc": auc,
                            }
                        )

                        all_results.append(baseline_metrics)

                    # ========================================
                    # XGBOOST FILTERS
                    # ========================================

                    for threshold in PROBABILITY_THRESHOLDS:
                        selected = validation_trades.loc[
                            validation_trades["xgb_probability"] >= threshold
                        ].copy()

                        if len(selected) == 0:
                            continue

                        metrics = calculate_metrics(selected)

                        if metrics is None:
                            continue

                        metrics.update(
                            {
                                "window": window_number,
                                "tail_percent": (1 - quantile) * 100,
                                "rr": rr,
                                "horizon": horizon,
                                "probability_threshold": threshold,
                                "model": "xgboost",
                                "auc": auc,
                            }
                        )

                        all_results.append(metrics)

                        selected = selected.copy()

                        selected["window"] = window_number

                        selected["tail_percent"] = (1 - quantile) * 100

                        selected["rr"] = rr

                        selected["horizon"] = horizon

                        selected["probability_threshold"] = threshold

                        selected["model"] = "xgboost"

                        all_trades.append(selected)

    # ========================================================
    # RESULTS
    # ========================================================

    results = pd.DataFrame(all_results)

    if results.empty:
        print("\nNo results generated.")

        return

    trades = (
        pd.concat(
            all_trades,
            ignore_index=True,
        )
        if all_trades
        else pd.DataFrame()
    )

    # ========================================================
    # OOS SUMMARY
    # ========================================================

    print("\n" + "=" * 110)

    print("XGBOOST OOS SUMMARY")

    print("=" * 110)

    summary = (
        results.groupby(
            [
                "model",
                "tail_percent",
                "rr",
                "horizon",
                "probability_threshold",
            ]
        )
        .agg(
            windows=(
                "window",
                "count",
            ),
            total_trades=(
                "trades",
                "sum",
            ),
            median_WR=(
                "win_rate",
                "median",
            ),
            mean_WR=(
                "win_rate",
                "mean",
            ),
            median_R=(
                "mean_R",
                "median",
            ),
            mean_R=(
                "mean_R",
                "mean",
            ),
            median_PF=(
                "profit_factor",
                "median",
            ),
            mean_PF=(
                "profit_factor",
                "mean",
            ),
            total_R=(
                "total_R",
                "sum",
            ),
            worst_DD=(
                "max_drawdown_R",
                "min",
            ),
            worst_streak=(
                "longest_losing_streak",
                "max",
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "median_R",
            "median_PF",
        ],
        ascending=False,
    )

    print(summary.to_string(index=False))

    # ========================================================
    # POSITIVE IN EVERY WINDOW
    # ========================================================

    print("\n" + "=" * 110)

    print("XGBOOST CONFIGURATIONS POSITIVE IN ALL OOS WINDOWS")

    print("=" * 110)

    consistency = (
        results.groupby(
            [
                "model",
                "tail_percent",
                "rr",
                "horizon",
                "probability_threshold",
            ]
        )
        .agg(
            windows=(
                "window",
                "count",
            ),
            positive_windows=(
                "mean_R",
                lambda x: int((x > 0).sum()),
            ),
            median_R=(
                "mean_R",
                "median",
            ),
            median_PF=(
                "profit_factor",
                "median",
            ),
            total_trades=(
                "trades",
                "sum",
            ),
            median_WR=(
                "win_rate",
                "median",
            ),
            worst_DD=(
                "max_drawdown_R",
                "min",
            ),
            worst_streak=(
                "longest_losing_streak",
                "max",
            ),
        )
        .reset_index()
    )

    consistency["all_positive"] = (
        consistency["positive_windows"] == consistency["windows"]
    )

    robust = consistency.loc[
        (consistency["all_positive"])
        & (consistency["total_trades"] >= MIN_TOTAL_TRADES)
    ].sort_values(
        [
            "median_R",
            "median_PF",
        ],
        ascending=False,
    )

    if robust.empty:
        print("No configuration was positive in every OOS window.")

    else:
        print(robust.to_string(index=False))

    # ========================================================
    # WIN RATE IMPROVEMENT
    # ========================================================

    print("\n" + "=" * 110)

    print("WIN RATE IMPROVEMENT VS BASELINE")

    print("=" * 110)

    baseline = results.loc[results["model"] == "baseline"].copy()

    xgb = results.loc[results["model"] == "xgboost"].copy()

    if not baseline.empty and not xgb.empty:
        baseline_group = (
            baseline.groupby(
                [
                    "window",
                    "tail_percent",
                    "rr",
                    "horizon",
                ]
            )["win_rate"]
            .mean()
            .rename("baseline_WR")
        )

        xgb_group = (
            xgb.groupby(
                [
                    "window",
                    "tail_percent",
                    "rr",
                    "horizon",
                    "probability_threshold",
                ]
            )["win_rate"]
            .mean()
            .reset_index()
        )

        comparison = xgb_group.merge(
            baseline_group.reset_index(),
            on=[
                "window",
                "tail_percent",
                "rr",
                "horizon",
            ],
            how="left",
        )

        comparison["WR_improvement"] = (
            comparison["win_rate"] - comparison["baseline_WR"]
        )

        comparison = (
            comparison.groupby(
                [
                    "tail_percent",
                    "rr",
                    "horizon",
                    "probability_threshold",
                ]
            )
            .agg(
                windows=(
                    "window",
                    "count",
                ),
                median_WR_improvement=(
                    "WR_improvement",
                    "median",
                ),
                mean_WR_improvement=(
                    "WR_improvement",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                "median_WR_improvement",
                ascending=False,
            )
        )

        print(comparison.head(30).to_string(index=False))

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 110)

    print("FEATURE IMPORTANCE — FINAL TRAINED MODELS")

    print("=" * 110)

    # Train one representative model on the last window's
    # narrowest/highest-quality population for diagnostics.
    #
    # This is NOT used to select the final strategy.
    # ========================================================

    try:
        final_train = df.loc[
            (df.index >= windows[-1][0]) & (df.index < windows[-1][1])
        ].copy()

        final_train, _ = fit_hmm(
            final_train,
            final_train.copy(),
        )

        final_thresholds = calculate_base_thresholds(
            final_train,
            0.90,
        )

        final_population = build_trade_dataset(
            final_train,
            final_thresholds,
            1.25,
            12,
            len(windows),
        )

        if not final_population.empty and final_population["target"].nunique() >= 2:
            (
                X_final,
                y_final,
                _,
                _,
                feature_columns,
            ) = prepare_xgb_data(
                final_population,
                final_population,
            )

            final_model = train_xgboost(
                X_final,
                y_final,
            )

            importance = pd.Series(
                final_model.feature_importances_,
                index=feature_columns,
            ).sort_values(ascending=False)

            print(importance.to_string())

    except Exception as exc:
        print(f"Feature importance diagnostic failed: {exc}")

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        "s2_short_xgboost_local_results.csv",
        index=False,
    )

    summary.to_csv(
        "s2_short_xgboost_local_summary.csv",
        index=False,
    )

    consistency.to_csv(
        "s2_short_xgboost_local_consistency.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            "s2_short_xgboost_local_trades.csv",
            index=False,
        )

    print("\nSaved:")

    print("s2_short_xgboost_local_results.csv")

    print("s2_short_xgboost_local_summary.csv")

    print("s2_short_xgboost_local_consistency.csv")

    print("s2_short_xgboost_local_trades.csv")

    print("\n" + "=" * 110)

    print("SPECIALIZED XGBOOST TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
