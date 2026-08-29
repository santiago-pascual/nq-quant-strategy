from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 SHORT — CONTINUOUS VOLATILITY PROFILE
# ============================================================
#
# PURPOSE
# -------
# Analyze the EXISTING frozen S2-short strategy continuously
# across volatility percentiles.
#
# THIS IS NOT AN OPTIMIZATION.
# THIS IS NOT A VOLATILITY FILTER.
# THIS IS NOT XGBOOST.
#
# We want to answer:
#
#   "How does the existing edge behave as volatility changes?"
#
# Frozen strategy:
#
#   HMM State 2
#   + four existing directional conditions
#   + 17.5% lower-tail thresholds
#   + 1.30R target
#   + 20 point stop
#   + 15 bar maximum holding period
#
# Volatility:
#
#   realized_vol_30
#
# For every OOS trade we calculate the volatility percentile
# using TRAINING DATA ONLY.
#
# Then we divide the OOS trades into:
#
#   0-10%
#   10-20%
#   ...
#   90-100%
#
# IMPORTANT:
# The percentile boundaries are calculated independently inside
# each walk-forward training period.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_STATES = 3
STATE = 2

STOP_POINTS = 20.0
RR = 1.30
HORIZON = 15

TAIL_PERCENT = 17.5
TAIL_QUANTILE = 1.0 - TAIL_PERCENT / 100.0

VOLATILITY_FEATURE = "realized_vol_30"

VOL_BUCKETS = [
    (0.00, 0.10, "00-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, 0.30, "20-30%"),
    (0.30, 0.40, "30-40%"),
    (0.40, 0.50, "40-50%"),
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.00, "90-100%"),
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
# PREPARE RTH
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

    missing = [column for column in required if column not in df.columns]

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

    # HMM is fitted ONLY on training data.
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
        raise ValueError(f"No observations in HMM State {STATE}.")

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
# BASE SETUP MASK
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
# TRAIN VOLATILITY DISTRIBUTION
# ============================================================
#
# The OOS volatility percentile MUST be relative to the
# volatility distribution known during training.
#
# We therefore save the TRAIN volatility values and use them
# to calculate an empirical percentile for each OOS trade.
#
# This avoids using the entire dataset to define the buckets.
# ============================================================


def get_train_volatility_distribution(train):

    values = (
        train[VOLATILITY_FEATURE]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values()
        .to_numpy()
    )

    if len(values) == 0:
        raise ValueError("No valid training volatility observations.")

    return values


# ============================================================
# EMPIRICAL TRAIN-BASED PERCENTILE
# ============================================================


def calculate_train_percentile(
    value,
    train_volatility,
):

    if pd.isna(value):
        return np.nan

    # Fraction of training observations <= value.
    percentile = np.searchsorted(
        train_volatility,
        value,
        side="right",
    ) / len(train_volatility)

    return float(
        np.clip(
            percentile,
            0.0,
            1.0,
        )
    )


# ============================================================
# VOLATILITY BUCKET
# ============================================================


def percentile_bucket(percentile):

    if pd.isna(percentile):
        return "UNKNOWN"

    for (
        lower,
        upper,
        label,
    ) in VOL_BUCKETS:
        if percentile >= lower and percentile < upper:
            return label

    # Handle exactly 1.0.
    return "90-100%"


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

        # Conservative same-bar treatment.
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

    # Timeout / mark-to-market.
    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    return {
        "pnl_R": pnl_points / STOP_POINTS,
        "pnl_points": pnl_points,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# BUILD TRADES
# ============================================================


def build_trades(
    df,
    thresholds,
    train_volatility,
    window_number,
):

    records = []

    for (
        session_id,
        session,
    ) in df.groupby(
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

            volatility = row[VOLATILITY_FEATURE]

            percentile = calculate_train_percentile(
                volatility,
                train_volatility,
            )

            bucket = percentile_bucket(percentile)

            record = {
                "entry_timestamp": session.index[position],
                "exit_timestamp": session.index[result["exit_position"]],
                "window": window_number,
                "hmm_state": row["hmm_state"],
                "volatility": volatility,
                "volatility_percentile": percentile,
                "volatility_bucket": bucket,
                "pnl_R": result["pnl_R"],
                "pnl_points": result["pnl_points"],
                "exit_reason": result["reason"],
                "holding_bars": (result["exit_position"] - position),
            }

            # Save the underlying setup characteristics.
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
            "median_R": np.nan,
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
        "median_R": pnl.median(),
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
        }

    daily = trades.set_index("entry_timestamp")["pnl_R"].resample("1D").sum()

    active_days = trades["entry_timestamp"].dt.normalize().nunique()

    if active_days > 0:
        trades_per_day = len(trades) / active_days

    else:
        trades_per_day = 0.0

    return {
        "trades_per_day": trades_per_day,
        "mean_daily_R": daily.mean(),
        "profitable_days": int((daily > 0).sum()),
        "losing_days": int((daily < 0).sum()),
        "worst_day_R": daily.min(),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 SHORT — CONTINUOUS VOLATILITY PROFILE")

    print("=" * 110)

    print("\nNO XGBOOST.")

    print("NO PARAMETER OPTIMIZATION.")

    print("NO VOLATILITY FILTER.")

    print("\nFROZEN STRATEGY:")

    print("HMM State 2")

    print("17.5% lower-tail four-condition setup")

    print(f"RR = {RR:.2f}")

    print(f"Stop = {STOP_POINTS:.1f} points")

    print(f"Horizon = {HORIZON} bars")

    print("\nVOLATILITY:")

    print("realized_vol_30")

    print("Percentiles are calculated from TRAIN data only.")

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

    if VOLATILITY_FEATURE not in df.columns:
        raise KeyError(f"{VOLATILITY_FEATURE} is not available.")

    # ========================================================
    # WINDOWS
    # ========================================================

    windows = generate_windows(df)

    print(f"\nWalk-forward windows: {len(windows)}")

    all_trades = []
    window_results = []

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
        # EXISTING STRATEGY THRESHOLDS
        # ----------------------------------------------------

        thresholds = calculate_base_thresholds(train)

        # ----------------------------------------------------
        # TRAIN VOLATILITY DISTRIBUTION
        # ----------------------------------------------------

        train_volatility = get_train_volatility_distribution(train)

        print("\nBase thresholds:")

        for (
            feature,
            value,
        ) in thresholds.items():
            print(f"  {feature}: {value:.8f}")

        print("\nTrain volatility:")

        print(f"  observations: {len(train_volatility)}")

        print(f"  min: {train_volatility.min():.8f}")

        print(f"  median: {np.median(train_volatility):.8f}")

        print(f"  max: {train_volatility.max():.8f}")

        # ----------------------------------------------------
        # BUILD OOS TRADES
        # ----------------------------------------------------

        trades = build_trades(
            validation,
            thresholds,
            train_volatility,
            window_number,
        )

        if trades.empty:
            print("\nNo valid OOS trades.")

            continue

        all_trades.append(trades)

        # ----------------------------------------------------
        # WINDOW METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(trades)

        daily = calculate_daily_metrics(trades)

        result = {
            "window": window_number,
            **metrics,
            **daily,
        }

        window_results.append(result)

        print("\nWINDOW RESULTS")

        for key, value in result.items():
            print(f"{key:28s}: {value}")

        # ----------------------------------------------------
        # CONTINUOUS VOLATILITY PROFILE
        # ----------------------------------------------------

        print("\nVOLATILITY PERCENTILE PROFILE")

        print("-" * 95)

        for (
            lower,
            upper,
            label,
        ) in VOL_BUCKETS:
            bucket_trades = trades.loc[trades["volatility_bucket"] == label]

            metrics = calculate_metrics(bucket_trades)

            daily = calculate_daily_metrics(bucket_trades)

            print(
                f"{label:10s} | "
                f"trades={metrics['trades']:4d} | "
                f"WR={metrics['win_rate']:.4f} | "
                f"meanR={metrics['mean_R']:.4f} | "
                f"PF={metrics['profit_factor']:.4f} | "
                f"DD={metrics['max_drawdown_R']:.4f} | "
                f"T/day={daily['trades_per_day']:.3f}"
            )

    # ========================================================
    # COMBINED
    # ========================================================

    if not all_trades:
        print("\nNo OOS trades generated.")

        return

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    window_df = pd.DataFrame(window_results)

    # ========================================================
    # COMBINED METRICS
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS BASELINE")

    print("=" * 110)

    combined_metrics = calculate_metrics(trades)

    combined_daily = calculate_daily_metrics(trades)

    for key, value in {
        **combined_metrics,
        **combined_daily,
    }.items():
        print(f"{key:28s}: {value}")

    # ========================================================
    # COMBINED VOLATILITY PROFILE
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS — CONTINUOUS VOLATILITY PROFILE")

    print("=" * 110)

    profile_rows = []

    for (
        lower,
        upper,
        label,
    ) in VOL_BUCKETS:
        bucket_trades = trades.loc[trades["volatility_bucket"] == label]

        metrics = calculate_metrics(bucket_trades)

        daily = calculate_daily_metrics(bucket_trades)

        row = {
            "volatility_bucket": label,
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "mean_R": metrics["mean_R"],
            "median_R": metrics["median_R"],
            "total_R": metrics["total_R"],
            "profit_factor": metrics["profit_factor"],
            "max_drawdown_R": metrics["max_drawdown_R"],
            "longest_losing_streak": metrics["longest_losing_streak"],
            "trades_per_day": daily["trades_per_day"],
            "profitable_days": daily["profitable_days"],
            "losing_days": daily["losing_days"],
            "worst_day_R": daily["worst_day_R"],
        }

        profile_rows.append(row)

        print(f"\n{label}")

        for key, value in row.items():
            if key != "volatility_bucket":
                print(f"  {key:28s}: {value}")

    profile_df = pd.DataFrame(profile_rows)

    # ========================================================
    # POSITIVE / NEGATIVE ZONES
    # ========================================================

    print("\n" + "=" * 110)

    print("VOLATILITY ZONE DIAGNOSTIC")

    print("=" * 110)

    positive = profile_df.loc[profile_df["mean_R"] > 0]

    negative = profile_df.loc[profile_df["mean_R"] <= 0]

    print(f"\nPositive expectancy buckets: {len(positive)} / {len(profile_df)}")

    if not positive.empty:
        print(
            positive[
                [
                    "volatility_bucket",
                    "trades",
                    "win_rate",
                    "mean_R",
                    "profit_factor",
                ]
            ].to_string(index=False)
        )

    print(f"\nNon-positive expectancy buckets: {len(negative)} / {len(profile_df)}")

    if not negative.empty:
        print(
            negative[
                [
                    "volatility_bucket",
                    "trades",
                    "win_rate",
                    "mean_R",
                    "profit_factor",
                ]
            ].to_string(index=False)
        )

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    print("\n" + "=" * 110)

    print("EXIT REASONS BY VOLATILITY PERCENTILE")

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
    # WALK-FORWARD COMPARISON
    # ========================================================

    print("\n" + "=" * 110)

    print("WALK-FORWARD COMPARISON")

    print("=" * 110)

    print(window_df.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "s2_short_volatility_percentile_trades.csv",
        index=False,
    )

    window_df.to_csv(
        "s2_short_volatility_percentile_windows.csv",
        index=False,
    )

    profile_df.to_csv(
        "s2_short_volatility_percentile_profile.csv",
        index=False,
    )

    print("\nSaved:")

    print("s2_short_volatility_percentile_trades.csv")

    print("s2_short_volatility_percentile_windows.csv")

    print("s2_short_volatility_percentile_profile.csv")

    print("\n" + "=" * 110)

    print("CONTINUOUS VOLATILITY PROFILE COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
