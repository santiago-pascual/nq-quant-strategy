# src/research/s2_narrow_optimization.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 NARROW OPTIMIZATION
# ============================================================
#
# PURPOSE
# -------
# We are NOT reopening the entire strategy parameter space.
#
# The structure is frozen:
#
#   HMM State 2
#   17.5% lower-tail conditions
#   four directional features
#   20-point stop
#   15-bar horizon
#
# We only optimize the narrow region around the configuration
# that has already shown evidence of working.
#
# Variables:
#
#   1. Setup-quality threshold
#   2. RR
#
# RR is restricted to the already interesting neighborhood.
#
# NO:
#   - XGBoost
#   - new indicators
#   - new HMM states
#   - new entry logic
#   - broad parameter sweep
#
# Every candidate is evaluated WALK-FORWARD.
# ============================================================


RANDOM_STATE = 42

N_STATES = 3
TARGET_STATE = 2

TAIL_PERCENT = 17.5

STOP_POINTS = 20.0
HORIZON = 15


# ------------------------------------------------------------
# Narrow parameter surface
# ------------------------------------------------------------

QUALITY_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

RR_VALUES = [
    1.15,
    1.20,
    1.25,
    1.30,
    1.35,
    1.40,
]


BASE_FEATURES = [
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]

VOLATILITY_FEATURE = "realized_vol_30"


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

    model.fit(train)

    train = train.copy()
    validation = validation.copy()

    train["hmm_state"] = model.predict_states(train)

    validation["hmm_state"] = model.predict_states(validation)

    return train, validation


# ============================================================
# BASE THRESHOLDS
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
# QUALITY SCALES
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

        scale = threshold - extreme

        if scale <= 0:
            scale = np.nan

        scales[feature] = float(scale)

    return scales


# ============================================================
# SETUP QUALITY
# ============================================================


def setup_quality(
    row,
    thresholds,
    scales,
):

    scores = []

    for feature in BASE_FEATURES:
        value = row[feature]

        threshold = thresholds[feature]

        scale = scales[feature]

        if pd.isna(value) or pd.isna(scale) or scale <= 0:
            return np.nan

        # Short setup:
        #
        # lower than threshold = stronger
        #

        excess = threshold - value

        score = excess / scale

        score = np.clip(
            score,
            0.0,
            1.0,
        )

        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# VALID BASE SETUP
# ============================================================


def base_setup_mask(
    df,
    thresholds,
):

    mask = df["hmm_state"] == TARGET_STATE

    for feature in BASE_FEATURES:
        mask &= df[feature] <= thresholds[feature]

    return mask


# ============================================================
# TRADE RESOLUTION
# ============================================================


def resolve_trade(
    session,
    entry_position,
    rr,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = close[entry_position]

    target_points = STOP_POINTS * rr

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
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        if target_hit:
            return {
                "pnl_R": rr,
                "reason": "target",
                "exit_position": i,
            }

        if stop_hit:
            return {
                "pnl_R": -1.0,
                "reason": "stop",
                "exit_position": i,
            }

    exit_price = close[last_position]

    pnl_points = entry_price - exit_price

    return {
        "pnl_R": pnl_points / STOP_POINTS,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# BUILD CANDIDATE TRADES
# ============================================================


def build_trades(
    validation,
    thresholds,
    scales,
    quality_threshold,
    rr,
):

    records = []

    for (
        session_id,
        session,
    ) in validation.groupby(
        "_session_id",
        sort=False,
    ):
        session = session.sort_index()

        if len(session) <= HORIZON:
            continue

        valid = base_setup_mask(
            session,
            thresholds,
        )

        positions = np.flatnonzero(valid.to_numpy())

        position_index = 0

        while position_index < len(positions):
            position = positions[position_index]

            if position >= len(session) - HORIZON:
                break

            row = session.iloc[position]

            quality = setup_quality(
                row,
                thresholds,
                scales,
            )

            # --------------------------------------------
            # NARROW QUALITY FILTER
            # --------------------------------------------

            if pd.isna(quality) or quality < quality_threshold:
                position_index += 1
                continue

            result = resolve_trade(
                session,
                position,
                rr,
            )

            records.append(
                {
                    "entry_timestamp": session.index[position],
                    "exit_timestamp": session.index[result["exit_position"]],
                    "quality": quality,
                    "rr": rr,
                    "pnl_R": result["pnl_R"],
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - position),
                }
            )

            # --------------------------------------------
            # NON-OVERLAP
            # --------------------------------------------

            exit_position = result["exit_position"]

            future_positions = positions[positions > exit_position]

            if len(future_positions) == 0:
                break

            position_index = np.searchsorted(
                positions,
                future_positions[0],
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
        "total_R": float(pnl.sum()),
        "profit_factor": float(pf),
        "max_drawdown_R": float(drawdown.min()),
        "longest_losing_streak": int(longest),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 SHORT — NARROW OPTIMIZATION")

    print("=" * 110)

    print("\nFROZEN STRUCTURE:")

    print("HMM State 2")

    print("17.5% lower-tail four-condition setup")

    print("20-point stop")

    print("15-bar horizon")

    print("\nSEARCHING ONLY:")

    print("setup quality threshold")

    print("RR")

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

        train, validation = fit_hmm(
            train,
            validation,
        )

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        candidate_count = 0

        # ----------------------------------------------------
        # TEST EVERY NARROW COMBINATION
        # ----------------------------------------------------

        for quality_threshold in QUALITY_THRESHOLDS:
            for rr in RR_VALUES:
                trades = build_trades(
                    validation,
                    thresholds,
                    scales,
                    quality_threshold,
                    rr,
                )

                metrics = calculate_metrics(trades)

                result = {
                    "window": window_number,
                    "quality_threshold": quality_threshold,
                    "rr": rr,
                    **metrics,
                }

                all_results.append(result)

                candidate_count += 1

        print(f"\nCandidates tested: {candidate_count}")

    # ========================================================
    # RESULTS
    # ========================================================

    results = pd.DataFrame(all_results)

    print("\n" + "=" * 110)

    print("WINDOW RESULTS")

    print("=" * 110)

    print(results.to_string(index=False))

    # ========================================================
    # AGGREGATE CANDIDATES
    # ========================================================

    grouped = results.groupby(
        [
            "quality_threshold",
            "rr",
        ],
        as_index=False,
    ).agg(
        windows=(
            "window",
            "count",
        ),
        total_trades=(
            "trades",
            "sum",
        ),
        mean_WR=(
            "win_rate",
            "mean",
        ),
        mean_R=(
            "mean_R",
            "mean",
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

    # ========================================================
    # POSITIVE WINDOWS
    # ========================================================

    positive_windows = (
        results.assign(positive=lambda x: x["total_R"] > 0)
        .groupby(
            [
                "quality_threshold",
                "rr",
            ]
        )["positive"]
        .sum()
        .reset_index(name="positive_windows")
    )

    grouped = grouped.merge(
        positive_windows,
        on=[
            "quality_threshold",
            "rr",
        ],
        how="left",
    )

    grouped["consistency"] = grouped["positive_windows"] / grouped["windows"]

    # ========================================================
    # SCORE
    # ========================================================
    #
    # We do NOT simply maximize total R.
    #
    # Score favors:
    #
    #   expectancy
    #   consistency
    #   drawdown
    #
    # This is only a ranking aid.
    # Final selection remains a human decision.
    # ========================================================

    grouped["score"] = (
        grouped["mean_R"] * grouped["consistency"] / (1.0 + abs(grouped["worst_DD"]))
    )

    # ========================================================
    # SORT
    # ========================================================

    ranked = grouped.sort_values(
        [
            "score",
            "mean_R",
            "total_R",
        ],
        ascending=False,
    )

    print("\n" + "=" * 110)

    print("NARROW OPTIMIZATION RANKING")

    print("=" * 110)

    print(ranked.head(25).to_string(index=False))

    # ========================================================
    # BEST BY MEAN R
    # ========================================================

    print("\n" + "=" * 110)

    print("TOP EXPECTANCY CANDIDATES")

    print("=" * 110)

    print(
        grouped.sort_values(
            "mean_R",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # BEST BY CONSISTENCY
    # ========================================================

    print("\n" + "=" * 110)

    print("MOST CONSISTENT CANDIDATES")

    print("=" * 110)

    print(
        grouped.sort_values(
            [
                "consistency",
                "mean_R",
            ],
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
        "s2_narrow_optimization_windows.csv",
        index=False,
    )

    ranked.to_csv(
        "s2_narrow_optimization_ranked.csv",
        index=False,
    )

    print("\nSaved:")

    print("s2_narrow_optimization_windows.csv")

    print("s2_narrow_optimization_ranked.csv")

    print("\n" + "=" * 110)

    print("NARROW OPTIMIZATION COMPLETE")

    print("=" * 110)


if __name__ == "__main__":
    main()
