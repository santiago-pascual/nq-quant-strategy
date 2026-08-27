from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 FINAL VALIDATION — CORRECTED EXECUTION MODEL
# ============================================================
#
# FROZEN STRATEGIES
#
# Candidate A:
#   HMM State 2
#   17.5% lower-tail conditions
#   quality >= 0.65
#   RR = 1.25
#   20 point stop
#   15 bar horizon
#
# Candidate B:
#   HMM State 2
#   17.5% lower-tail conditions
#   quality >= 0.75
#   RR = 1.30
#   20 point stop
#   15 bar horizon
#
# NO OPTIMIZATION.
#
# This test evaluates the frozen strategies under:
#
#   - Correct short-side P&L
#   - TopstepX MNQ round-trip fees
#   - Slippage
#   - Volatility regimes
#   - Daily behavior
#   - Walk-forward consistency
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_STATE = 2
TAIL_PERCENT = 17.5

STOP_POINTS = 20.0
HORIZON = 15


# ============================================================
# FROZEN CANDIDATES
# ============================================================

CANDIDATES = {
    "candidate_A": {
        "quality_threshold": 0.65,
        "rr": 1.25,
    },
    "candidate_B": {
        "quality_threshold": 0.75,
        "rr": 1.30,
    },
}


# ============================================================
# MNQ / TOPSTEPX EXECUTION COSTS
# ============================================================
#
# TopstepX current MNQ RT cost:
#
#   Exchange       $0.70
#   NFA             $0.02
#   Commission      $0.50
#   ---------------------
#   Total RT        $1.22
#
# MNQ point value:
#
#   $2 / point
#
# Therefore:
#
#   $1.22 / $2 = 0.61 points
#
# Slippage assumption:
#
#   0.25 points per side
#   = 0.50 points RT
#
# Total expected execution drag:
#
#   0.61 + 0.50
#   = 1.11 points
#
# In R:
#
#   1.11 / 20
#   = 0.0555R
#
# ============================================================

MNQ_POINT_VALUE = 2.00

TOPSTEP_MNQ_RT_FEE_USD = 1.22

SLIPPAGE_POINTS_PER_SIDE = 0.25

TOTAL_SLIPPAGE_POINTS = 2.0 * SLIPPAGE_POINTS_PER_SIDE

TOPSTEP_FEE_POINTS = TOPSTEP_MNQ_RT_FEE_USD / MNQ_POINT_VALUE

TOTAL_EXECUTION_COST_POINTS = TOTAL_SLIPPAGE_POINTS + TOPSTEP_FEE_POINTS

TOTAL_EXECUTION_COST_R = TOTAL_EXECUTION_COST_POINTS / STOP_POINTS


# ============================================================
# FEATURES
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


def fit_hmm(
    train,
    validation,
):

    model = VolatilityRegimeModel(
        n_states=3,
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
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        thresholds[feature] = float(values.quantile(q))

    return thresholds


# ============================================================
# QUALITY SCALES
# ============================================================


def calculate_quality_scales(
    train,
):

    state_train = train.loc[train["hmm_state"] == TARGET_STATE]

    scales = {}

    for feature in BASE_FEATURES:
        values = (
            state_train[feature]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
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


def calculate_quality(
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

        score = (threshold - value) / scale

        score = np.clip(
            score,
            0.0,
            1.0,
        )

        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# BASE SETUP
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
# CORRECT SHORT TRADE RESOLUTION
# ============================================================
#
# IMPORTANT:
#
# For a SHORT:
#
#   Target below entry = PROFIT
#   Stop above entry   = LOSS
#
# Therefore:
#
#   Target = +RR
#   Stop   = -1R
#
# ============================================================


def resolve_short_trade(
    session,
    entry_position,
    rr,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = float(close[entry_position])

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

        # ----------------------------------------------------
        # BOTH HIT SAME BAR
        # ----------------------------------------------------
        #
        # Conservative assumption:
        # assume stop was hit first.
        #

        if target_hit and stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        if target_hit:
            return {
                "raw_points": target_points,
                "reason": "target",
                "exit_position": i,
            }

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "stop",
                "exit_position": i,
            }

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    exit_price = float(close[last_position])

    raw_points = entry_price - exit_price

    return {
        "raw_points": raw_points,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# APPLY EXECUTION COST
# ============================================================
#
# The raw strategy result is measured from the bar data.
#
# Then we subtract:
#
#   entry slippage
#   exit slippage
#   Topstep round-trip fees
#
# This is applied to EVERY completed trade.
#
# ============================================================


def apply_execution_cost(
    raw_points,
):

    net_points = raw_points - TOTAL_EXECUTION_COST_POINTS

    net_R = net_points / STOP_POINTS

    gross_R = raw_points / STOP_POINTS

    return (
        net_points,
        net_R,
        gross_R,
    )


# ============================================================
# GENERATE TRADES
# ============================================================


def generate_trades(
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

        i = 0

        while i < len(positions):
            position = positions[i]

            if position >= len(session) - HORIZON:
                break

            row = session.iloc[position]

            quality = calculate_quality(
                row,
                thresholds,
                scales,
            )

            if pd.isna(quality) or quality < quality_threshold:
                i += 1
                continue

            result = resolve_short_trade(
                session,
                position,
                rr,
            )

            (
                net_points,
                net_R,
                gross_R,
            ) = apply_execution_cost(result["raw_points"])

            records.append(
                {
                    "entry_timestamp": session.index[position],
                    "exit_timestamp": session.index[result["exit_position"]],
                    "session_id": session_id,
                    "quality": quality,
                    "rr": rr,
                    "raw_points": result["raw_points"],
                    "gross_R": gross_R,
                    "slippage_points": TOTAL_SLIPPAGE_POINTS,
                    "topstep_fee_points": TOPSTEP_FEE_POINTS,
                    "total_cost_points": TOTAL_EXECUTION_COST_POINTS,
                    "net_points": net_points,
                    "net_R": net_R,
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - position),
                }
            )

            # ------------------------------------------------
            # NON-OVERLAPPING
            # ------------------------------------------------

            exit_position = result["exit_position"]

            future_positions = positions[positions > exit_position]

            if not len(future_positions):
                break

            i = np.searchsorted(
                positions,
                future_positions[0],
            )

    return pd.DataFrame(records)


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades,
):

    if trades.empty:
        return {
            "trades": 0,
            "WR": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "PF": np.nan,
            "max_DD_R": np.nan,
            "worst_streak": 0,
        }

    pnl = trades["net_R"].astype(float)

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    if gross_loss > 0:
        PF = gross_profit / gross_loss

    else:
        PF = np.inf

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
        "WR": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "total_R": float(pnl.sum()),
        "PF": float(PF),
        "max_DD_R": float(drawdown.min()),
        "worst_streak": int(longest),
    }


# ============================================================
# DAILY ANALYSIS
# ============================================================


def daily_analysis(
    trades,
):

    if trades.empty:
        return {
            "trading_days": 0,
            "profitable_days": 0,
            "losing_days": 0,
            "mean_daily_R": np.nan,
            "median_daily_R": np.nan,
            "worst_day_R": np.nan,
            "best_day_R": np.nan,
            "avg_trades_per_day": np.nan,
        }

    temp = trades.copy()

    temp["date"] = temp["entry_timestamp"].dt.date

    daily = temp.groupby("date")["net_R"].sum()

    return {
        "trading_days": len(daily),
        "profitable_days": int((daily > 0).sum()),
        "losing_days": int((daily < 0).sum()),
        "mean_daily_R": float(daily.mean()),
        "median_daily_R": float(daily.median()),
        "worst_day_R": float(daily.min()),
        "best_day_R": float(daily.max()),
        "avg_trades_per_day": float(len(temp) / len(daily)),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 FINAL VALIDATION — CORRECTED")

    print("=" * 110)

    print("\nFROZEN PARAMETERS.")

    print("NO OPTIMIZATION.")

    # --------------------------------------------------------
    # COST INFORMATION
    # --------------------------------------------------------

    print("\nMNQ EXECUTION MODEL")

    print("-" * 70)

    print(f"MNQ point value: ${MNQ_POINT_VALUE:.2f}")

    print(f"Topstep MNQ RT fee: ${TOPSTEP_MNQ_RT_FEE_USD:.2f}")

    print(f"Topstep fee in points: {TOPSTEP_FEE_POINTS:.2f}")

    print(f"Slippage / side: {SLIPPAGE_POINTS_PER_SIDE:.2f} points")

    print(f"Round-trip slippage: {TOTAL_SLIPPAGE_POINTS:.2f} points")

    print(f"TOTAL COST / TRADE: {TOTAL_EXECUTION_COST_POINTS:.2f} points")

    print(f"TOTAL COST / TRADE: {TOTAL_EXECUTION_COST_R:.4f}R")

    print(f"TOTAL COST / TRADE: ${TOTAL_EXECUTION_COST_POINTS * MNQ_POINT_VALUE:.2f}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    df = prepare_rth(df)

    df = add_directional_features(df)

    print(f"\nRTH observations: {len(df)}")

    print(f"RTH sessions: {df['_session_id'].nunique()}")

    windows = generate_windows(df)

    print(f"Walk-forward windows: {len(windows)}")

    all_window_results = []
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

        train, validation = fit_hmm(
            train,
            validation,
        )

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        for (
            candidate_name,
            params,
        ) in CANDIDATES.items():
            print(f"\n{candidate_name.upper()}")

            print(f"Quality >= {params['quality_threshold']}")

            print(f"RR = {params['rr']}")

            trades = generate_trades(
                validation,
                thresholds,
                scales,
                params["quality_threshold"],
                params["rr"],
            )

            if not trades.empty:
                trades["window"] = window_number

                trades["candidate"] = candidate_name

                all_trades.append(trades)

            m = calculate_metrics(trades)

            daily = daily_analysis(trades)

            result = {
                "candidate": candidate_name,
                "window": window_number,
                **m,
                **daily,
            }

            all_window_results.append(result)

            print(f"Trades: {m['trades']}")

            print(f"WR: {m['WR']:.4f}")

            print(f"Mean R: {m['mean_R']:.4f}")

            print(f"Total R: {m['total_R']:.2f}")

            print(f"PF: {m['PF']:.3f}")

            print(f"Max DD: {m['max_DD_R']:.2f}R")

    # ========================================================
    # COMBINE
    # ========================================================

    window_results = pd.DataFrame(all_window_results)

    if all_trades:
        trades = pd.concat(
            all_trades,
            ignore_index=True,
        )

    else:
        trades = pd.DataFrame()

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 110)

    print("FINAL CANDIDATE COMPARISON")

    print("=" * 110)

    summaries = []

    for candidate in CANDIDATES:
        candidate_trades = trades.loc[trades["candidate"] == candidate]

        m = calculate_metrics(candidate_trades)

        candidate_windows = window_results.loc[window_results["candidate"] == candidate]

        positive_windows = int((candidate_windows["total_R"] > 0).sum())

        summaries.append(
            {
                "candidate": candidate,
                **m,
                "positive_windows": positive_windows,
                "total_windows": len(windows),
            }
        )

    summary = pd.DataFrame(summaries)

    print(summary.to_string(index=False))

    # ========================================================
    # RAW VS NET
    # ========================================================

    print("\n" + "=" * 110)

    print("RAW VS NET P&L")

    print("=" * 110)

    for candidate in CANDIDATES:
        candidate_trades = trades.loc[trades["candidate"] == candidate]

        if candidate_trades.empty:
            continue

        gross_R = candidate_trades["gross_R"].sum()

        net_R = candidate_trades["net_R"].sum()

        total_cost_R = gross_R - net_R

        print(f"\n{candidate.upper()}")

        print(f"Gross R: {gross_R:.2f}")

        print(f"Execution costs: -{total_cost_R:.2f}R")

        print(f"Net R: {net_R:.2f}")

        print(f"Cost per trade: {TOTAL_EXECUTION_COST_R:.4f}R")

    # ========================================================
    # DAILY PROFILE
    # ========================================================

    print("\n" + "=" * 110)

    print("DAILY EXECUTION PROFILE")

    print("=" * 110)

    for candidate in CANDIDATES:
        candidate_trades = trades.loc[trades["candidate"] == candidate]

        daily = daily_analysis(candidate_trades)

        print(f"\n{candidate.upper()}")

        for (
            key,
            value,
        ) in daily.items():
            print(f"{key}: {value}")

    # ========================================================
    # VOLATILITY
    # ========================================================

    print("\n" + "=" * 110)

    print("VOLATILITY REGIME ANALYSIS")

    print("=" * 110)

    volatility_percentile = df["realized_vol_30"].rank(pct=True)

    for candidate in CANDIDATES:
        candidate_trades = trades.loc[trades["candidate"] == candidate].copy()

        if candidate_trades.empty:
            continue

        candidate_trades["vol_percentile"] = volatility_percentile.reindex(
            candidate_trades["entry_timestamp"],
            method="ffill",
        ).to_numpy()

        candidate_trades["vol_bucket"] = pd.cut(
            candidate_trades["vol_percentile"],
            bins=[
                0.0,
                0.20,
                0.40,
                0.60,
                0.80,
                1.0,
            ],
            labels=[
                "0-20",
                "20-40",
                "40-60",
                "60-80",
                "80-100",
            ],
            include_lowest=True,
        )

        print(f"\n{candidate.upper()}")

        rows = []

        for (
            bucket,
            group,
        ) in candidate_trades.groupby(
            "vol_bucket",
            observed=False,
        ):
            m = calculate_metrics(group)

            rows.append(
                {
                    "volatility": str(bucket),
                    "trades": m["trades"],
                    "WR": m["WR"],
                    "mean_R": m["mean_R"],
                    "total_R": m["total_R"],
                    "PF": m["PF"],
                    "DD": m["max_DD_R"],
                }
            )

        print(pd.DataFrame(rows).to_string(index=False))

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    print("\n" + "=" * 110)

    print("EXIT ANALYSIS")

    print("=" * 110)

    for candidate in CANDIDATES:
        candidate_trades = trades.loc[trades["candidate"] == candidate]

        print(f"\n{candidate.upper()}")

        if candidate_trades.empty:
            continue

        exit_stats = candidate_trades.groupby("exit_reason")["net_R"].agg(
            [
                "count",
                "mean",
                "sum",
            ]
        )

        print(exit_stats.to_string())

    # ========================================================
    # SAVE
    # ========================================================

    window_results.to_csv(
        "s2_final_validation_windows.csv",
        index=False,
    )

    summary.to_csv(
        "s2_final_validation_summary.csv",
        index=False,
    )

    trades.to_csv(
        "s2_final_validation_trades.csv",
        index=False,
    )

    print("\n" + "=" * 110)

    print("FINAL VALIDATION COMPLETE")

    print("Saved:")

    print("s2_final_validation_windows.csv")

    print("s2_final_validation_summary.csv")

    print("s2_final_validation_trades.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
