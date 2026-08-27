from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 SHORT — VOLATILITY ROBUSTNESS TEST
# ============================================================
#
# PURPOSE
# -------
# Test the EXISTING BASELINE strategy across different
# volatility environments.
#
# NO XGBOOST
# NO PARAMETER OPTIMIZATION
# NO NEW STRATEGY
#
# Frozen strategy:
#
#   HMM State 2
#   + past_return_30 <= train-derived lower-tail threshold
#   + directional_pressure_30 <= train-derived threshold
#   + close_location_30 <= train-derived threshold
#   + normalized_momentum_30 <= train-derived threshold
#
# Frozen payoff:
#
#   Stop  = 20 NQ points
#   Target = 1.30R = 26 points
#   Horizon = 15 bars
#
# We then classify each valid setup according to its
# VOLATILITY ENVIRONMENT.
#
# The volatility buckets are calculated using TRAIN DATA ONLY
# in each walk-forward window.
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

N_STATES = 3
STATE = 2

STOP_POINTS = 20.0
RR = 1.30
HORIZON = 15

TAIL_PERCENT = 17.5
TAIL_QUANTILE = 1.0 - TAIL_PERCENT / 100.0

# Volatility buckets:
#
# LOW       = bottom 25%
# NORMAL    = 25% → 75%
# HIGH      = 75% → 90%
# EXTREME   = top 10%
#
VOL_BUCKETS = [
    "LOW",
    "NORMAL",
    "HIGH",
    "EXTREME",
]


# ============================================================
# PROVEN BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


# ============================================================
# PREPARE RTH DATA
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


def fit_hmm(train, validation):

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    # IMPORTANT:
    # Fit only on TRAIN.
    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    return train, validation


# ============================================================
# BASE STRATEGY THRESHOLDS
# ============================================================


def calculate_base_thresholds(train):

    state_train = train.loc[train["hmm_state"] == STATE]

    if state_train.empty:
        raise ValueError(f"No HMM State {STATE} observations in training.")

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
# BASE STRATEGY GATE
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
# VOLATILITY BUCKET THRESHOLDS
# ============================================================
#
# We use realized_vol_30 because:
#
# - it is directly related to the strategy horizon
# - it represents recent market volatility
# - it avoids using future information
#
# Thresholds are calculated from TRAIN ONLY.
# ============================================================


def calculate_volatility_thresholds(train):

    values = (
        train["realized_vol_30"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if values.empty:
        raise ValueError("No valid realized_vol_30 values in training.")

    return {
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "q90": float(values.quantile(0.90)),
    }


# ============================================================
# CLASSIFY VOLATILITY
# ============================================================


def classify_volatility(
    value,
    thresholds,
):

    if pd.isna(value):
        return "UNKNOWN"

    if value <= thresholds["q25"]:
        return "LOW"

    if value <= thresholds["q75"]:
        return "NORMAL"

    if value <= thresholds["q90"]:
        return "HIGH"

    return "EXTREME"


# ============================================================
# RESOLVE SHORT TRADE
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

        # Conservative assumption:
        # if both are touched within the same bar,
        # assume the stop happened first.
        if target_hit and stop_hit:
            return {
                "pnl_R": -1.0,
                "pnl_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        if target_hit:
            return {
                "pnl_R": RR,
                "pnl_points": target_points,
                "reason": "target",
                "exit_position": i,
            }

        if stop_hit:
            return {
                "pnl_R": -1.0,
                "pnl_points": -STOP_POINTS,
                "reason": "stop",
                "exit_position": i,
            }

    # Timeout / mark-to-market
    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    return {
        "pnl_R": pnl_points / STOP_POINTS,
        "pnl_points": pnl_points,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# BUILD VALID TRADES
# ============================================================


def build_trades(
    df,
    thresholds,
    volatility_thresholds,
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

            volatility = row["realized_vol_30"]

            volatility_bucket = classify_volatility(
                volatility,
                volatility_thresholds,
            )

            record = {
                "entry_timestamp": session.index[position],
                "exit_timestamp": session.index[result["exit_position"]],
                "window": window_number,
                "volatility": volatility,
                "volatility_bucket": volatility_bucket,
                "hmm_state": row["hmm_state"],
                "pnl_R": result["pnl_R"],
                "pnl_points": result["pnl_points"],
                "exit_reason": result["reason"],
                "holding_bars": (result["exit_position"] - position),
            }

            # Keep the actual setup characteristics
            # so we can investigate bad trades later.
            for feature in BASE_FEATURES:
                record[feature] = row[feature]

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
# METRICS
# ============================================================


def calculate_metrics(trades):

    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "longest_losing_streak": 0,
            "average_holding_bars": np.nan,
        }

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

    return {
        "trades": len(trades),
        "win_rate": (pnl > 0).mean(),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "profit_factor": pf,
        "max_drawdown_R": drawdown.min(),
        "longest_losing_streak": longest_streak,
        "average_holding_bars": trades["holding_bars"].mean(),
    }


# ============================================================
# DAILY METRICS
# ============================================================


def calculate_daily_metrics(trades):

    if trades.empty:
        return {
            "trades_per_day": 0.0,
            "mean_daily_R": np.nan,
            "profitable_days": 0,
            "losing_days": 0,
            "worst_day_R": np.nan,
            "daily_sharpe": np.nan,
        }

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    active_days = trades["entry_timestamp"].dt.normalize().nunique()

    if active_days > 0:
        trades_per_day = len(trades) / active_days

    else:
        trades_per_day = 0.0

    if len(daily) > 1 and daily.std(ddof=1) > 0:
        daily_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)

    else:
        daily_sharpe = np.nan

    return {
        "trades_per_day": trades_per_day,
        "mean_daily_R": daily.mean(),
        "profitable_days": int((daily > 0).sum()),
        "losing_days": int((daily < 0).sum()),
        "worst_day_R": daily.min(),
        "daily_sharpe": daily_sharpe,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 SHORT — BASELINE VOLATILITY ROBUSTNESS")
    print("=" * 110)

    print("\nTHIS TEST DOES NOT OPTIMIZE THE STRATEGY.")

    print("XGBoost is NOT used.")

    print("\nFROZEN STRATEGY:")

    print("HMM State 2")

    print("17.5% lower-tail four-condition setup")

    print(f"RR = {RR:.2f}")

    print(f"Stop = {STOP_POINTS:.1f} points")

    print(f"Horizon = {HORIZON} bars")

    print("\nVOLATILITY BUCKETS:")

    print("LOW      = bottom 25%")

    print("NORMAL   = 25% → 75%")

    print("HIGH     = 75% → 90%")

    print("EXTREME  = top 10%")

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

    all_trades = []
    window_results = []
    volatility_results = []

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
        # HMM
        # ----------------------------------------------------

        train, validation = fit_hmm(
            train,
            validation,
        )

        # ----------------------------------------------------
        # BASE THRESHOLDS
        # ----------------------------------------------------

        thresholds = calculate_base_thresholds(train)

        # ----------------------------------------------------
        # VOLATILITY THRESHOLDS
        # ----------------------------------------------------

        volatility_thresholds = calculate_volatility_thresholds(train)

        print("\nBase thresholds:")

        for feature, value in thresholds.items():
            print(f"  {feature}: {value:.8f}")

        print("\nVolatility thresholds:")

        print(f"  Q25: {volatility_thresholds['q25']:.8f}")

        print(f"  Q75: {volatility_thresholds['q75']:.8f}")

        print(f"  Q90: {volatility_thresholds['q90']:.8f}")

        # ----------------------------------------------------
        # BUILD OOS TRADES
        # ----------------------------------------------------

        trades = build_trades(
            validation,
            thresholds,
            volatility_thresholds,
            window_number,
        )

        if trades.empty:
            print("\nNo valid OOS trades.")

            continue

        all_trades.append(trades)

        # ----------------------------------------------------
        # OVERALL WINDOW METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(trades)

        daily_metrics = calculate_daily_metrics(trades)

        result = {
            "window": window_number,
            **metrics,
            **daily_metrics,
        }

        window_results.append(result)

        print("\nWINDOW RESULTS")

        for key, value in result.items():
            print(f"{key:28s}: {value}")

        # ----------------------------------------------------
        # VOLATILITY BREAKDOWN
        # ----------------------------------------------------

        print("\nVOLATILITY BREAKDOWN")

        for bucket in VOL_BUCKETS:
            bucket_trades = trades.loc[trades["volatility_bucket"] == bucket]

            bucket_metrics = calculate_metrics(bucket_trades)

            bucket_daily = calculate_daily_metrics(bucket_trades)

            volatility_result = {
                "window": window_number,
                "volatility_bucket": bucket,
                **bucket_metrics,
                **bucket_daily,
            }

            volatility_results.append(volatility_result)

            print(f"\n{bucket}")

            print(f"  trades       : {bucket_metrics['trades']}")

            print(f"  WR           : {bucket_metrics['win_rate']:.4f}")

            print(f"  mean R       : {bucket_metrics['mean_R']:.4f}")

            print(f"  total R      : {bucket_metrics['total_R']:.4f}")

            print(f"  PF           : {bucket_metrics['profit_factor']:.4f}")

            print(f"  max DD       : {bucket_metrics['max_drawdown_R']:.4f}")

            print(f"  trades/day   : {bucket_daily['trades_per_day']:.4f}")

    # ========================================================
    # COMBINE
    # ========================================================

    if not all_trades:
        print("\nNo trades generated.")

        return

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    window_df = pd.DataFrame(window_results)

    volatility_df = pd.DataFrame(volatility_results)

    # ========================================================
    # COMBINED OOS
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS BASELINE")

    print("=" * 110)

    combined = calculate_metrics(trades)

    combined_daily = calculate_daily_metrics(trades)

    combined_result = {
        **combined,
        **combined_daily,
    }

    for key, value in combined_result.items():
        print(f"{key:28s}: {value}")

    # ========================================================
    # COMBINED VOLATILITY
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS — VOLATILITY ENVIRONMENTS")

    print("=" * 110)

    combined_volatility = []

    for bucket in VOL_BUCKETS:
        bucket_trades = trades.loc[trades["volatility_bucket"] == bucket]

        metrics = calculate_metrics(bucket_trades)

        daily = calculate_daily_metrics(bucket_trades)

        row = {
            "volatility_bucket": bucket,
            **metrics,
            **daily,
        }

        combined_volatility.append(row)

        print(f"\n{bucket}")

        for key, value in metrics.items():
            print(f"  {key:24s}: {value}")

        print(f"  trades_per_day: {daily['trades_per_day']}")

    combined_volatility_df = pd.DataFrame(combined_volatility)

    # ========================================================
    # WINDOW TABLE
    # ========================================================

    print("\n" + "=" * 110)

    print("WALK-FORWARD WINDOW COMPARISON")

    print("=" * 110)

    print(window_df.to_string(index=False))

    # ========================================================
    # VOLATILITY TABLE
    # ========================================================

    print("\n" + "=" * 110)

    print("VOLATILITY COMPARISON")

    print("=" * 110)

    print(combined_volatility_df.to_string(index=False))

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    print("\n" + "=" * 110)

    print("EXIT ANALYSIS BY VOLATILITY")

    print("=" * 110)

    exit_table = (
        trades.groupby(
            [
                "volatility_bucket",
                "exit_reason",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(exit_table.to_string())

    # ========================================================
    # BAD TRADE ANALYSIS
    # ========================================================
    #
    # We explicitly inspect losing trades.
    #
    # This is important because later we may want to know
    # whether losses are concentrated in a specific
    # volatility environment.
    #
    # ========================================================

    print("\n" + "=" * 110)

    print("LOSING TRADE DISTRIBUTION")

    print("=" * 110)

    losing_trades = trades.loc[trades["pnl_R"] < 0]

    if losing_trades.empty:
        print("No losing trades.")

    else:
        loss_distribution = (
            losing_trades.groupby("volatility_bucket").size().rename("losing_trades")
        )

        total_distribution = (
            trades.groupby("volatility_bucket").size().rename("total_trades")
        )

        loss_table = pd.concat(
            [
                total_distribution,
                loss_distribution,
            ],
            axis=1,
        ).fillna(0)

        loss_table["loss_rate"] = (
            loss_table["losing_trades"] / loss_table["total_trades"]
        )

        print(loss_table.to_string())

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "s2_short_volatility_trades.csv",
        index=False,
    )

    window_df.to_csv(
        "s2_short_volatility_windows.csv",
        index=False,
    )

    volatility_df.to_csv(
        "s2_short_volatility_by_window.csv",
        index=False,
    )

    combined_volatility_df.to_csv(
        "s2_short_volatility_combined.csv",
        index=False,
    )

    print("\nSaved:")

    print("s2_short_volatility_trades.csv")

    print("s2_short_volatility_windows.csv")

    print("s2_short_volatility_by_window.csv")

    print("s2_short_volatility_combined.csv")

    print("\n" + "=" * 110)

    print("BASELINE VOLATILITY ROBUSTNESS TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
