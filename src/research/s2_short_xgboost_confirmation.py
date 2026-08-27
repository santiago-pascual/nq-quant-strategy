from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 SHORT — FROZEN XGBOOST CONFIRMATION TEST
# ============================================================
#
# THIS IS NOT AN OPTIMIZATION SCRIPT.
#
# EVERYTHING BELOW IS FROZEN BEFORE RUNNING THE TEST.
#
# Base hypothesis:
#
#   HMM State 2
#   + 30-bar bearish return
#   + 30-bar bearish directional pressure
#   + 30-bar bearish close location
#   + 30-bar negative normalized momentum
#
# Frozen local configuration:
#
#   Tail          = 17.5%
#   RR            = 1.30
#   Horizon       = 15 bars
#   XGB threshold = 0.58
#
# The purpose is to answer:
#
#   "Does the XGBoost improvement survive on unseen data?"
#
# We DO NOT search for another optimum here.
#
# ============================================================


# ============================================================
# FROZEN PARAMETERS
# ============================================================

RANDOM_STATE = 42

N_STATES = 3
STATE = 2

STOP_POINTS = 20.0

TAIL_PERCENT = 17.5
TAIL_QUANTILE = 1.0 - (TAIL_PERCENT / 100.0)

RR = 1.30
HORIZON = 15

XGB_PROBABILITY_THRESHOLD = 0.58


# ============================================================
# PROVEN BASE CONDITIONS
# ============================================================

BASE_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


# ============================================================
# SPECIALIZED XGBOOST FEATURES
# ============================================================

XGB_FEATURES = [
    # Existing proven 30-bar conditions
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
    # Shorter-term information from the same concepts
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
    # Volatility
    "realized_vol_5",
    "realized_vol_15",
    "realized_vol_30",
    "realized_vol_60",
    "vol_ratio_5_30",
    "vol_ratio_5_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
    # Directional persistence
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
# FIXED WALK-FORWARD WINDOWS
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


def fit_hmm(train, validation):

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # HMM IS FIT ONLY ON TRAIN.
    # --------------------------------------------------------

    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    train_probs = model.predict_probabilities(train)

    validation_probs = model.predict_probabilities(validation)

    for column in train_probs.columns:
        train[column] = train_probs[column]

        validation[column] = validation_probs[column]

    return train, validation


# ============================================================
# BASE THRESHOLDS
# ============================================================


def calculate_base_thresholds(train):

    state_train = train.loc[train["hmm_state"] == STATE]

    if state_train.empty:
        raise ValueError(f"No State {STATE} observations in training data.")

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

        thresholds[feature] = float(values.quantile(1.0 - TAIL_QUANTILE))

    return thresholds


# ============================================================
# BASE SETUP
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
# TRADE RESOLUTION
# ============================================================


def resolve_trade(
    session,
    entry_position,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = close[entry_position]

    target_points = STOP_POINTS * RR

    target_price = entry_price - target_points

    stop_price = entry_price + STOP_POINTS

    last_position = min(
        entry_position + HORIZON,
        len(session) - 1,
    )

    for i in range(
        entry_position + 1,
        last_position + 1,
    ):
        target_hit = low[i] <= target_price

        stop_hit = high[i] >= stop_price

        # Conservative treatment when both are touched
        # in the same bar.
        if target_hit and stop_hit:
            return {
                "pnl_points": -STOP_POINTS,
                "pnl_R": -1.0,
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        if target_hit:
            return {
                "pnl_points": target_points,
                "pnl_R": RR,
                "reason": "target",
                "exit_position": i,
            }

        if stop_hit:
            return {
                "pnl_points": -STOP_POINTS,
                "pnl_R": -1.0,
                "reason": "stop",
                "exit_position": i,
            }

    # Timeout / mark-to-market exit
    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    pnl_R = pnl_points / STOP_POINTS

    return {
        "pnl_points": pnl_points,
        "pnl_R": pnl_R,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# BUILD TRADE POPULATION
# ============================================================


def build_trade_dataset(
    df,
    thresholds,
    window_number,
):

    records = []

    for session_id, session in df.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= HORIZON:
            continue

        mask = base_setup_mask(
            session,
            thresholds,
        )

        positions = np.flatnonzero(mask.to_numpy())

        i = 0

        while i < len(positions):
            position = positions[i]

            if position >= len(session) - HORIZON:
                break

            row = session.iloc[position]

            result = resolve_trade(
                session,
                position,
            )

            record = {}

            # ------------------------------------------------
            # XGBoost inputs
            # ------------------------------------------------

            for feature in XGB_FEATURES:
                if feature in session.columns:
                    record[feature] = row[feature]

            # HMM probabilities
            for state in range(N_STATES):
                column = f"state_probability_{state}"

                if column in row.index:
                    record[column] = row[column]

            # ------------------------------------------------
            # Distance from the already-proven thresholds.
            #
            # This is critical.
            #
            # We are NOT creating new base conditions.
            #
            # We are measuring how strongly a setup satisfies
            # the existing conditions.
            # ------------------------------------------------

            for feature in BASE_FEATURES:
                value = row[feature]

                threshold = thresholds[feature]

                record[f"{feature}_distance"] = value - threshold

            # ------------------------------------------------
            # Outcome
            # ------------------------------------------------

            record["target"] = int(result["reason"] == "target")

            record["pnl_R"] = result["pnl_R"]

            record["pnl_points"] = result["pnl_points"]

            record["entry_timestamp"] = session.index[position]

            record["exit_timestamp"] = session.index[result["exit_position"]]

            record["holding_bars"] = result["exit_position"] - position

            record["exit_reason"] = result["reason"]

            record["window"] = window_number

            records.append(record)

            # ------------------------------------------------
            # NON-OVERLAPPING EXECUTION
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
# XGBOOST DATA
# ============================================================


def prepare_xgb_data(
    train_trades,
    validation_trades,
):

    excluded = {
        "target",
        "pnl_R",
        "pnl_points",
        "entry_timestamp",
        "exit_timestamp",
        "exit_reason",
        "window",
    }

    feature_columns = [c for c in train_trades.columns if c not in excluded]

    feature_columns = [
        c for c in feature_columns if pd.api.types.is_numeric_dtype(train_trades[c])
    ]

    X_train = train_trades[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_validation = validation_trades[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # TRAIN ONLY imputation.
    medians = X_train.median()

    X_train = X_train.fillna(medians)

    X_validation = X_validation.fillna(medians)

    y_train = train_trades["target"].astype(int)

    y_validation = validation_trades["target"].astype(int)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


# ============================================================
# XGBOOST MODEL
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

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

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
    # Daily statistics
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
        "median_R": pnl.median(),
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
    print("S2 SHORT — FROZEN XGBOOST CONFIRMATION")
    print("=" * 110)

    print("\nTHIS IS A CONFIRMATION TEST.")

    print("NO PARAMETER OPTIMIZATION IS PERFORMED.")

    print("\nFROZEN PARAMETERS:")

    print(f"Tail: {TAIL_PERCENT:.1f}%")

    print(f"RR: {RR:.2f}")

    print(f"Horizon: {HORIZON} bars")

    print(f"XGBoost threshold: {XGB_PROBABILITY_THRESHOLD:.2f}")

    # ========================================================
    # LOAD
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
        # HMM FIT ON TRAIN ONLY
        # ----------------------------------------------------

        (
            train,
            validation,
        ) = fit_hmm(
            train,
            validation,
        )

        # ----------------------------------------------------
        # THRESHOLDS FROM TRAIN ONLY
        # ----------------------------------------------------

        thresholds = calculate_base_thresholds(train)

        print("\nBase thresholds:")

        for feature, value in thresholds.items():
            print(f"  {feature}: {value:.8f}")

        # ----------------------------------------------------
        # BUILD TRAIN POPULATION
        # ----------------------------------------------------

        train_trades = build_trade_dataset(
            train,
            thresholds,
            window_number,
        )

        # ----------------------------------------------------
        # BUILD COMPLETELY OOS POPULATION
        # ----------------------------------------------------

        validation_trades = build_trade_dataset(
            validation,
            thresholds,
            window_number,
        )

        print(f"\nTrain valid setups: {len(train_trades)}")

        print(f"OOS valid setups: {len(validation_trades)}")

        if train_trades.empty or validation_trades.empty:
            print("Skipping window: no valid setups.")

            continue

        if train_trades["target"].nunique() < 2:
            print("Skipping window: training population contains only one class.")

            continue

        # ----------------------------------------------------
        # PREPARE XGBOOST
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model = train_xgboost(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # OOS PREDICTIONS
        # ----------------------------------------------------

        probabilities = model.predict_proba(X_validation)[:, 1]

        validation_trades = validation_trades.copy()

        validation_trades["xgb_probability"] = probabilities

        # ----------------------------------------------------
        # AUC DIAGNOSTIC
        # ----------------------------------------------------

        if y_validation.nunique() >= 2:
            auc = roc_auc_score(
                y_validation,
                probabilities,
            )

        else:
            auc = np.nan

        print(f"OOS AUC: {auc:.4f}")

        # ====================================================
        # BASELINE
        # ====================================================

        baseline = calculate_metrics(validation_trades)

        if baseline:
            baseline.update(
                {
                    "window": window_number,
                    "model": "baseline",
                    "auc": auc,
                }
            )

            all_results.append(baseline)

        # ====================================================
        # XGBOOST FILTER
        # ====================================================

        selected = validation_trades.loc[
            validation_trades["xgb_probability"] >= XGB_PROBABILITY_THRESHOLD
        ].copy()

        if selected.empty:
            print("XGBoost selected ZERO OOS trades.")

            continue

        xgb_metrics = calculate_metrics(selected)

        if xgb_metrics:
            xgb_metrics.update(
                {
                    "window": window_number,
                    "model": "xgboost",
                    "auc": auc,
                }
            )

            all_results.append(xgb_metrics)

        selected["model"] = "xgboost"

        all_trades.append(selected)

        # ====================================================
        # WINDOW COMPARISON
        # ====================================================

        print("\nWINDOW COMPARISON")

        print("-" * 70)

        if baseline:
            print("\nBASELINE")

            for key in [
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "longest_losing_streak",
            ]:
                print(f"{key:28s}: {baseline[key]}")

        if xgb_metrics:
            print("\nXGBOOST")

            for key in [
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "longest_losing_streak",
            ]:
                print(f"{key:28s}: {xgb_metrics[key]}")

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    results = pd.DataFrame(all_results)

    print("\n" + "=" * 110)

    print("COMBINED OOS CONFIRMATION")

    print("=" * 110)

    if results.empty:
        print("No results.")

        return

    print(results.to_string(index=False))

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    print("\n" + "=" * 110)

    print("MODEL SUMMARY")

    print("=" * 110)

    summary = (
        results.groupby("model")
        .agg(
            windows=(
                "window",
                "count",
            ),
            trades=(
                "trades",
                "sum",
            ),
            mean_WR=(
                "win_rate",
                "mean",
            ),
            median_WR=(
                "win_rate",
                "median",
            ),
            mean_R=(
                "mean_R",
                "mean",
            ),
            median_R=(
                "mean_R",
                "median",
            ),
            total_R=(
                "total_R",
                "sum",
            ),
            median_PF=(
                "profit_factor",
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

    print(summary.to_string(index=False))

    # ========================================================
    # WINDOW CONSISTENCY
    # ========================================================

    print("\n" + "=" * 110)

    print("WINDOW CONSISTENCY")

    print("=" * 110)

    consistency = (
        results.groupby("model")
        .agg(
            positive_windows=(
                "mean_R",
                lambda x: int((x > 0).sum()),
            ),
            total_windows=(
                "window",
                "count",
            ),
        )
        .reset_index()
    )

    consistency["all_positive"] = (
        consistency["positive_windows"] == consistency["total_windows"]
    )

    print(consistency.to_string(index=False))

    # ========================================================
    # WIN RATE CHANGE
    # ========================================================

    print("\n" + "=" * 110)

    print("WIN RATE IMPROVEMENT")

    print("=" * 110)

    pivot = results.pivot(
        index="window",
        columns="model",
        values="win_rate",
    )

    if "baseline" in pivot.columns and "xgboost" in pivot.columns:
        pivot["WR_improvement"] = pivot["xgboost"] - pivot["baseline"]

        print(pivot.to_string())

        print(f"\nMean WR improvement: {pivot['WR_improvement'].mean():.4f}")

    # ========================================================
    # TRADE REDUCTION
    # ========================================================

    print("\n" + "=" * 110)

    print("TRADE REDUCTION")

    print("=" * 110)

    if "baseline" in pivot.columns and "xgboost" in pivot.columns:
        baseline_trades = results.loc[
            results["model"] == "baseline",
            "trades",
        ].sum()

        xgb_trades = results.loc[
            results["model"] == "xgboost",
            "trades",
        ].sum()

        reduction = 1 - (xgb_trades / baseline_trades)

        print(f"Baseline trades: {baseline_trades}")

        print(f"XGBoost trades: {xgb_trades}")

        print(f"Trade reduction: {reduction:.2%}")

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        "s2_short_xgboost_confirmation_results.csv",
        index=False,
    )

    summary.to_csv(
        "s2_short_xgboost_confirmation_summary.csv",
        index=False,
    )

    consistency.to_csv(
        "s2_short_xgboost_confirmation_consistency.csv",
        index=False,
    )

    if all_trades:
        trades = pd.concat(
            all_trades,
            ignore_index=True,
        )

        trades.to_csv(
            "s2_short_xgboost_confirmation_trades.csv",
            index=False,
        )

    print("\nSaved confirmation results.")

    print("\n" + "=" * 110)

    print("CONFIRMATION TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
