from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# S2 BENCHMARK
# ============================================================
#
# PURPOSE
# -------
# Freeze the current best exploratory S2 configuration as a
# benchmark before performing failure analysis and modifications.
#
# THIS IS NOT PARAMETER OPTIMIZATION.
#
# Frozen:
#
#   HMM state       = 2
#   Lower tail      = 17.5%
#   Quality         >= 0.75
#   Volatility      = 40-60%
#   Stop            = 25 points
#   RR              = 1.75
#   Horizon         = 20 bars
#
# Walk-forward:
#
#   Training window = 2 years
#   Validation      = 3 months
#
# The benchmark will be used later to compare modified versions
# of S2.
#
# ============================================================


RANDOM_STATE = 42

TARGET_STATE = 2
TAIL_PERCENT = 17.5

QUALITY_THRESHOLD = 0.75

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

VOL_LOW = 0.40
VOL_HIGH = 0.60


# ============================================================
# COST MODEL
# ============================================================

MNQ_POINT_VALUE = 2.00

TOPSTEP_MNQ_RT_FEE_USD = 1.22

SLIPPAGE_POINTS_PER_SIDE = 0.25

TOTAL_SLIPPAGE_POINTS = 2.0 * SLIPPAGE_POINTS_PER_SIDE

TOPSTEP_FEE_POINTS = TOPSTEP_MNQ_RT_FEE_USD / MNQ_POINT_VALUE

TOTAL_COST_POINTS = TOTAL_SLIPPAGE_POINTS + TOPSTEP_FEE_POINTS


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
# RESULTS
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PREPARE RTH
# ============================================================


def prepare_rth(df):

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


def calculate_quality_scales(train):

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
# QUALITY
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
# VOLATILITY PERCENTILE
# ============================================================


def add_volatility_percentile(
    train,
    validation,
):

    train = train.copy()
    validation = validation.copy()

    train_values = (
        train["realized_vol_30"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .sort_values()
        .to_numpy()
    )

    def transform(series):

        values = series.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        result = np.full(
            len(values),
            np.nan,
            dtype=float,
        )

        valid = values.notna()

        if len(train_values) > 0:
            result[valid.to_numpy()] = np.searchsorted(
                train_values,
                values[valid].to_numpy(),
                side="right",
            ) / len(train_values)

        return result

    train["vol_percentile"] = transform(train["realized_vol_30"])

    validation["vol_percentile"] = transform(validation["realized_vol_30"])

    return (
        train,
        validation,
    )


# ============================================================
# TRADE RESOLUTION
# ============================================================


def resolve_short_trade(
    session,
    entry_position,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = float(close[entry_position])

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
                "raw_points": -STOP_POINTS,
                "reason": "both_hit_conservative_stop",
                "exit_position": i,
            }

        if target_hit:
            return {
                "raw_points": target_points,
                "reason": "target",
                "exit_position": i,
            }

        if stop_hit:
            return {
                "raw_points": -STOP_POINTS,
                "reason": "stop",
                "exit_position": i,
            }

    exit_price = float(close[last_position])

    return {
        "raw_points": entry_price - exit_price,
        "reason": "timeout",
        "exit_position": last_position,
    }


# ============================================================
# GENERATE BENCHMARK TRADES
# ============================================================


def generate_trades(
    validation,
    thresholds,
    scales,
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

        positions = np.flatnonzero((session["hmm_state"] == TARGET_STATE).to_numpy())

        i = 0

        while i < len(positions):
            position = positions[i]

            if position >= len(session) - HORIZON:
                break

            row = session.iloc[position]

            # --------------------------------------------
            # Base signal
            # --------------------------------------------

            signal = True

            for feature in BASE_FEATURES:
                if pd.isna(row[feature]) or row[feature] > thresholds[feature]:
                    signal = False
                    break

            if not signal:
                i += 1
                continue

            # --------------------------------------------
            # Quality
            # --------------------------------------------

            quality = calculate_quality(
                row,
                thresholds,
                scales,
            )

            if pd.isna(quality) or quality < QUALITY_THRESHOLD:
                i += 1
                continue

            # --------------------------------------------
            # Volatility regime
            # --------------------------------------------

            percentile = row["vol_percentile"]

            if pd.isna(percentile) or percentile < VOL_LOW or percentile >= VOL_HIGH:
                i += 1
                continue

            # --------------------------------------------
            # Resolve trade
            # --------------------------------------------

            result = resolve_short_trade(
                session,
                position,
            )

            raw_points = result["raw_points"]

            net_points = raw_points - TOTAL_COST_POINTS

            net_R = net_points / STOP_POINTS

            records.append(
                {
                    "entry_timestamp": session.index[position],
                    "exit_timestamp": session.index[result["exit_position"]],
                    "session_id": session_id,
                    "quality": quality,
                    "vol_percentile": percentile,
                    "stop_points": STOP_POINTS,
                    "rr": RR,
                    "horizon": HORIZON,
                    "raw_points": raw_points,
                    "net_points": net_points,
                    "net_R": net_R,
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - position),
                }
            )

            # --------------------------------------------
            # No overlapping trades
            # --------------------------------------------

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
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "median_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "longest_losing_streak": 0,
        }

    pnl = trades["net_R"].astype(float)

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = pnl.cumsum()

    drawdown = equity - equity.cummax()

    max_drawdown = drawdown.min()

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
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "median_R": float(pnl.median()),
        "total_R": float(pnl.sum()),
        "profit_factor": float(profit_factor),
        "max_drawdown_R": float(max_drawdown),
        "longest_losing_streak": int(longest),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 BENCHMARK — FROZEN CONFIGURATION")
    print("=" * 110)

    print()
    print("Frozen configuration")
    print("-" * 110)

    print(f"HMM state       : {TARGET_STATE}")

    print(f"Lower tail      : {TAIL_PERCENT}%")

    print(f"Quality         : >= {QUALITY_THRESHOLD}")

    print(f"Volatility      : 40-60%")

    print(f"Stop            : {STOP_POINTS} points")

    print(f"RR              : {RR}")

    print(f"Horizon         : {HORIZON} bars")

    print()
    print("Loading expanded dataset...")

    df = load_data()

    df = add_directional_features(df)

    df = prepare_rth(df)

    windows = generate_windows(df)

    print(f"Walk-forward windows: {len(windows)}")

    all_trades = []

    window_results = []

    for window_number, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        print(f"Processing window {window_number}/{len(windows)}...")

        train = df.loc[train_start:validation_start].copy()

        validation = df.loc[validation_start:validation_end].copy()

        if train.empty or validation.empty:
            continue

        # --------------------------------------------
        # Fit HMM on TRAIN only
        # --------------------------------------------

        train, validation = fit_hmm(
            train,
            validation,
        )

        # --------------------------------------------
        # Learn signal thresholds from TRAIN
        # --------------------------------------------

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        # --------------------------------------------
        # Transform volatility using TRAIN only
        # --------------------------------------------

        (
            train,
            validation,
        ) = add_volatility_percentile(
            train,
            validation,
        )

        # --------------------------------------------
        # Generate OOS trades
        # --------------------------------------------

        trades = generate_trades(
            validation,
            thresholds,
            scales,
        )

        if not trades.empty:
            trades["window"] = window_number

            trades["validation_start"] = validation_start

            trades["validation_end"] = validation_end

            all_trades.append(trades)

        metrics = calculate_metrics(trades)

        metrics["window"] = window_number

        metrics["validation_start"] = validation_start

        metrics["validation_end"] = validation_end

        window_results.append(metrics)

    # ========================================================
    # COMBINE
    # ========================================================

    if all_trades:
        trades = pd.concat(
            all_trades,
            ignore_index=True,
        )

    else:
        trades = pd.DataFrame()

    windows_df = pd.DataFrame(window_results)

    combined = calculate_metrics(trades)

    combined["windows_with_trades"] = int((windows_df["trades"] > 0).sum())

    combined["positive_windows"] = int((windows_df["total_R"] > 0).sum())

    combined["negative_windows"] = int((windows_df["total_R"] < 0).sum())

    combined["positive_window_pct"] = (
        combined["positive_windows"] / len(windows_df)
        if len(windows_df) > 0
        else np.nan
    )

    combined["mean_window_R"] = float(windows_df["total_R"].mean())

    combined["median_window_R"] = float(windows_df["total_R"].median())

    combined["worst_window_R"] = float(windows_df["total_R"].min())

    combined["best_window_R"] = float(windows_df["total_R"].max())

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 110)
    print("BENCHMARK RESULTS BY WINDOW")
    print("=" * 110)

    print(
        windows_df[
            [
                "window",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "longest_losing_streak",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("COMBINED BENCHMARK")
    print("=" * 110)

    for key, value in combined.items():
        print(f"{key:28s}: {value}")

    # ========================================================
    # SAVE
    # ========================================================

    trades_path = RESULTS_DIR / "s2_benchmark_trades.csv"

    windows_path = RESULTS_DIR / "s2_benchmark_by_window.csv"

    summary_path = RESULTS_DIR / "s2_benchmark_summary.csv"

    trades.to_csv(
        trades_path,
        index=False,
    )

    windows_df.to_csv(
        windows_path,
        index=False,
    )

    pd.DataFrame([combined]).to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(trades_path)
    print(windows_path)
    print(summary_path)

    print()
    print("S2 BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
