from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 SHORT — CONDITIONAL SETUP QUALITY × VOLATILITY
# ============================================================
#
# FROZEN STRATEGY:
#
#   HMM State 2
#   + 17.5% lower-tail conditions
#   + 1.30R
#   + 20-point stop
#   + 15-bar horizon
#
# We DO NOT change the strategy.
#
# PURPOSE:
# --------
# Measure setup quality INSIDE the already-valid S2 population.
#
# Previous test was flawed because every valid setup naturally
# appeared in the 80-100% strength bucket.
#
# This version measures:
#
#   How far beyond the entry threshold
#   did each condition go?
#
# Example:
#
# threshold = -0.20
# value     = -0.30
#
# For a SHORT lower-tail condition, this is stronger.
#
# We calculate a normalized excess score for each condition,
# then combine the four scores into one setup_quality score.
#
# Finally:
#
#       setup_quality × volatility
#
# is analyzed.
#
# NO XGBOOST
# NO PARAMETER OPTIMIZATION
# NO NEW FILTER
# ============================================================


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

N_STATES = 3
TARGET_STATE = 2

TAIL_PERCENT = 17.5

STOP_POINTS = 20.0
RR = 1.30
HORIZON = 15

BASE_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]

VOLATILITY_FEATURE = "realized_vol_30"


# ============================================================
# WALK-FORWARD
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
# PREPARE RTH
# ============================================================


def prepare_rth(df):

    df = df.copy()

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

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
# HMM
# ============================================================


def fit_hmm(train, validation):

    model = VolatilityRegimeModel(
        n_states=N_STATES,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    return train, validation


# ============================================================
# THRESHOLDS
# ============================================================


def calculate_thresholds(train):

    state_train = train.loc[train["hmm_state"] == TARGET_STATE]

    thresholds = {}

    q = TAIL_PERCENT / 100.0

    for feature in BASE_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        thresholds[feature] = float(values.quantile(q))

    return thresholds


# ============================================================
# SCALE FOR SETUP QUALITY
# ============================================================
#
# We need a scale for each feature.
#
# The scale is calculated using TRAINING State-2 data.
#
# We use the distance between the 17.5th percentile threshold
# and the 5th percentile.
#
# Therefore:
#
#   value == threshold
#       -> quality = 0
#
#   value == 5th percentile
#       -> quality = 1
#
# More extreme values are clipped at 1.
#
# This gives us "depth beyond threshold".
# ============================================================


def calculate_quality_scales(train):

    state_train = train.loc[train["hmm_state"] == TARGET_STATE]

    scales = {}

    for feature in BASE_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        threshold = values.quantile(TAIL_PERCENT / 100.0)

        extreme = values.quantile(0.05)

        distance = threshold - extreme

        if distance <= 0:
            distance = np.nan

        scales[feature] = float(distance)

    return scales


# ============================================================
# VALID SETUP
# ============================================================


def get_setup_mask(
    df,
    thresholds,
):

    mask = df["hmm_state"] == TARGET_STATE

    for feature in BASE_FEATURES:
        mask &= df[feature] <= thresholds[feature]

    return mask


# ============================================================
# SETUP QUALITY
# ============================================================


def calculate_setup_quality(
    row,
    thresholds,
    scales,
):

    scores = {}

    for feature in BASE_FEATURES:
        value = row[feature]

        threshold = thresholds[feature]

        scale = scales[feature]

        if pd.isna(value) or pd.isna(scale) or scale <= 0:
            scores[feature] = np.nan
            continue

        # SHORT:
        #
        # lower value = stronger
        #
        # threshold - value
        #
        # gives positive distance once
        # the condition is satisfied.

        excess = threshold - value

        score = excess / scale

        scores[feature] = float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    valid = [x for x in scores.values() if not pd.isna(x)]

    if not valid:
        return np.nan, scores

    quality = float(np.mean(valid))

    return quality, scores


# ============================================================
# EMPIRICAL PERCENTILE
# ============================================================


def empirical_percentile(
    value,
    distribution,
):

    if pd.isna(value):
        return np.nan

    return float(
        np.clip(
            np.searchsorted(
                distribution,
                value,
                side="right",
            )
            / len(distribution),
            0.0,
            1.0,
        )
    )


# ============================================================
# BUCKET
# ============================================================


def quality_bucket(value):

    if pd.isna(value):
        return "UNKNOWN"

    if value < 0.20:
        return "00-20%"

    if value < 0.40:
        return "20-40%"

    if value < 0.60:
        return "40-60%"

    if value < 0.80:
        return "60-80%"

    return "80-100%"


def volatility_bucket(value):

    if pd.isna(value):
        return "UNKNOWN"

    if value < 0.20:
        return "00-20%"

    if value < 0.40:
        return "20-40%"

    if value < 0.60:
        return "40-60%"

    if value < 0.80:
        return "60-80%"

    return "80-100%"


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
    scales,
    volatility_distribution,
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

        setup_mask = get_setup_mask(
            session,
            thresholds,
        )

        positions = np.flatnonzero(setup_mask.to_numpy())

        position_index = 0

        while position_index < len(positions):
            position = positions[position_index]

            if position >= len(session) - HORIZON:
                break

            row = session.iloc[position]

            # --------------------------------------------
            # SETUP QUALITY
            # --------------------------------------------

            quality, scores = calculate_setup_quality(
                row,
                thresholds,
                scales,
            )

            # --------------------------------------------
            # VOLATILITY
            # --------------------------------------------

            volatility = row[VOLATILITY_FEATURE]

            volatility_pct = empirical_percentile(
                volatility,
                volatility_distribution,
            )

            # --------------------------------------------
            # TRADE
            # --------------------------------------------

            result = resolve_trade(
                session,
                position,
            )

            record = {
                "entry_timestamp": session.index[position],
                "exit_timestamp": session.index[result["exit_position"]],
                "window": window_number,
                "hmm_state": row["hmm_state"],
                "setup_quality": quality,
                "setup_quality_bucket": quality_bucket(quality),
                "volatility": volatility,
                "volatility_percentile": volatility_pct,
                "volatility_bucket": volatility_bucket(volatility_pct),
                "pnl_R": result["pnl_R"],
                "pnl_points": result["pnl_points"],
                "exit_reason": result["reason"],
                "holding_bars": (result["exit_position"] - position),
            }

            for feature in BASE_FEATURES:
                record[feature] = row[feature]

                record[feature + "_quality"] = scores.get(
                    feature,
                    np.nan,
                )

            records.append(record)

            # --------------------------------------------
            # NON-OVERLAPPING
            # --------------------------------------------

            exit_position = result["exit_position"]

            future_positions = positions[positions > exit_position]

            if len(future_positions) == 0:
                break

            next_position = future_positions[0]

            position_index = np.searchsorted(
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

    longest = 0
    current = 0

    for value in pnl:
        if value < 0:
            current += 1

            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    return {
        "trades": len(trades),
        "win_rate": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "median_R": float(pnl.median()),
        "total_R": float(pnl.sum()),
        "profit_factor": float(pf),
        "max_drawdown_R": float(drawdown.min()),
        "longest_losing_streak": int(longest),
    }


# ============================================================
# 2D PROFILE
# ============================================================


def build_profile(trades):

    rows = []

    grouped = trades.groupby(
        [
            "volatility_bucket",
            "setup_quality_bucket",
        ],
        observed=True,
    )

    for (
        volatility_bucket_name,
        quality_bucket_name,
    ), group in grouped:
        metrics = calculate_metrics(group)

        rows.append(
            {
                "volatility_bucket": volatility_bucket_name,
                "setup_quality_bucket": quality_bucket_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 SHORT — CONDITIONAL SETUP QUALITY × VOLATILITY")

    print("=" * 110)

    print("\nFROZEN STRATEGY:")

    print("HMM State 2")

    print("17.5% lower-tail four-condition setup")

    print(f"RR = {RR:.2f}")

    print(f"Stop = {STOP_POINTS:.1f} points")

    print(f"Horizon = {HORIZON} bars")

    print("\nDIAGNOSTIC:")

    print("Depth beyond existing thresholds × volatility")

    print("\nNO XGBOOST.")

    print("NO PARAMETER OPTIMIZATION.")

    print("NO NEW FILTER.")

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
        raise KeyError(f"{VOLATILITY_FEATURE} not found.")

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
        # FROZEN STRATEGY THRESHOLDS
        # ----------------------------------------------------

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        volatility_distribution = (
            train[VOLATILITY_FEATURE]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .sort_values()
            .to_numpy()
        )

        print("\nBase thresholds:")

        for (
            feature,
            value,
        ) in thresholds.items():
            print(f"  {feature}: {value:.8f}")

        print("\nQuality scales:")

        for (
            feature,
            value,
        ) in scales.items():
            print(f"  {feature}: {value:.8f}")

        # ----------------------------------------------------
        # OOS TRADES
        # ----------------------------------------------------

        trades = build_trades(
            validation,
            thresholds,
            scales,
            volatility_distribution,
            window_number,
        )

        if trades.empty:
            print("\nNo OOS trades.")

            continue

        all_trades.append(trades)

        # ----------------------------------------------------
        # WINDOW METRICS
        # ----------------------------------------------------

        metrics = calculate_metrics(trades)

        window_result = {
            "window": window_number,
            **metrics,
        }

        window_results.append(window_result)

        print("\nWINDOW RESULTS")

        for (
            key,
            value,
        ) in window_result.items():
            print(f"{key:28s}: {value}")

        # ----------------------------------------------------
        # QUALITY PROFILE
        # ----------------------------------------------------

        print("\nSETUP QUALITY PROFILE")

        print("-" * 105)

        for bucket in [
            "00-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%",
        ]:
            group = trades.loc[trades["setup_quality_bucket"] == bucket]

            metrics = calculate_metrics(group)

            print(
                f"{bucket:10s} | "
                f"trades={metrics['trades']:4d} | "
                f"WR={metrics['win_rate']:.4f} | "
                f"meanR={metrics['mean_R']:.4f} | "
                f"PF={metrics['profit_factor']:.4f} | "
                f"DD={metrics['max_drawdown_R']:.4f}"
            )

        # ----------------------------------------------------
        # 2D PROFILE
        # ----------------------------------------------------

        print("\nVOLATILITY × SETUP QUALITY")

        print("-" * 110)

        profile = build_profile(trades)

        if not profile.empty:
            print(profile.to_string(index=False))

    # ========================================================
    # COMBINED
    # ========================================================

    if not all_trades:
        print("\nNo trades generated.")

        return

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    window_df = pd.DataFrame(window_results)

    # ========================================================
    # BASELINE
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS BASELINE")

    print("=" * 110)

    baseline_metrics = calculate_metrics(trades)

    for (
        key,
        value,
    ) in baseline_metrics.items():
        print(f"{key:28s}: {value}")

    # ========================================================
    # QUALITY PROFILE
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS — SETUP QUALITY")

    print("=" * 110)

    quality_rows = []

    for bucket in [
        "00-20%",
        "20-40%",
        "40-60%",
        "60-80%",
        "80-100%",
    ]:
        group = trades.loc[trades["setup_quality_bucket"] == bucket]

        metrics = calculate_metrics(group)

        quality_rows.append(
            {
                "setup_quality_bucket": bucket,
                **metrics,
            }
        )

    quality_df = pd.DataFrame(quality_rows)

    print(quality_df.to_string(index=False))

    # ========================================================
    # 2D PROFILE
    # ========================================================

    print("\n" + "=" * 110)

    print("COMBINED OOS — VOLATILITY × SETUP QUALITY")

    print("=" * 110)

    profile_df = build_profile(trades)

    print(profile_df.to_string(index=False))

    # ========================================================
    # EXPECTANCY MATRIX
    # ========================================================

    print("\n" + "=" * 110)

    print("EXPECTANCY MATRIX")

    print("=" * 110)

    expectancy_matrix = profile_df.pivot(
        index="setup_quality_bucket",
        columns="volatility_bucket",
        values="mean_R",
    )

    print(expectancy_matrix.to_string())

    # ========================================================
    # WIN RATE MATRIX
    # ========================================================

    print("\n" + "=" * 110)

    print("WIN RATE MATRIX")

    print("=" * 110)

    wr_matrix = profile_df.pivot(
        index="setup_quality_bucket",
        columns="volatility_bucket",
        values="win_rate",
    )

    print(wr_matrix.to_string())

    # ========================================================
    # TRADE COUNT MATRIX
    # ========================================================

    print("\n" + "=" * 110)

    print("TRADE COUNT MATRIX")

    print("=" * 110)

    count_matrix = profile_df.pivot(
        index="setup_quality_bucket",
        columns="volatility_bucket",
        values="trades",
    )

    print(count_matrix.to_string())

    # ========================================================
    # STRONGEST CELLS
    # ========================================================

    print("\n" + "=" * 110)

    print("STRONGEST CELLS — DIAGNOSTIC ONLY")

    print("=" * 110)

    if not profile_df.empty:
        print(
            profile_df.sort_values(
                [
                    "mean_R",
                    "trades",
                ],
                ascending=False,
            )
            .head(15)
            .to_string(index=False)
        )

    # ========================================================
    # WINDOW RESULTS
    # ========================================================

    print("\n" + "=" * 110)

    print("WALK-FORWARD COMPARISON")

    print("=" * 110)

    print(window_df.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "s2_short_conditional_quality_trades.csv",
        index=False,
    )

    window_df.to_csv(
        "s2_short_conditional_quality_windows.csv",
        index=False,
    )

    quality_df.to_csv(
        "s2_short_conditional_quality_profile.csv",
        index=False,
    )

    profile_df.to_csv(
        "s2_short_quality_volatility_profile.csv",
        index=False,
    )

    expectancy_matrix.to_csv("s2_short_quality_volatility_expectancy.csv")

    wr_matrix.to_csv("s2_short_quality_volatility_wr.csv")

    count_matrix.to_csv("s2_short_quality_volatility_trade_count.csv")

    print("\nSaved:")

    print("s2_short_conditional_quality_trades.csv")

    print("s2_short_conditional_quality_windows.csv")

    print("s2_short_conditional_quality_profile.csv")

    print("s2_short_quality_volatility_profile.csv")

    print("s2_short_quality_volatility_expectancy.csv")

    print("s2_short_quality_volatility_wr.csv")

    print("s2_short_quality_volatility_trade_count.csv")

    print("\n" + "=" * 110)

    print("CONDITIONAL SETUP QUALITY TEST COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
