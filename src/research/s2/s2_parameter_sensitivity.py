from __future__ import annotations

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.models.regime import VolatilityRegimeModel
from src.research.direction_features import add_directional_features


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_STATE = 2
TAIL_PERCENT = 17.5

# Keep the strongest existing S2 candidate frozen.
QUALITY_THRESHOLD = 0.75

# Parameter grid.
STOP_GRID = [15.0, 20.0, 25.0]
RR_GRID = [1.00, 1.30, 1.50, 1.75]
HORIZON_GRID = [10, 15, 20]

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
# VOLATILITY BUCKETS
# ============================================================

VOL_BUCKETS = [
    (0.00, 0.20, "0-20"),
    (0.20, 0.40, "20-40"),
    (0.40, 0.60, "40-60"),
    (0.60, 0.80, "60-80"),
    (0.80, 1.00, "80-100"),
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


def fit_hmm(train, validation):

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
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values()
        .to_numpy()
    )

    def transform(series):

        values = series.replace(
            [np.inf, -np.inf],
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

    return train, validation


# ============================================================
# VOLATILITY BUCKET
# ============================================================


def volatility_bucket(
    percentile,
):

    if pd.isna(percentile):
        return None

    for low, high, label in VOL_BUCKETS:
        if percentile >= low and percentile < high:
            return label

    if percentile >= 1.0:
        return "80-100"

    return None


# ============================================================
# PARAMETERIZED SHORT TRADE
# ============================================================


def resolve_short_trade(
    session,
    entry_position,
    stop_points,
    rr,
    horizon,
):

    close = session["close"].to_numpy(dtype=float)

    high = session["high"].to_numpy(dtype=float)

    low = session["low"].to_numpy(dtype=float)

    entry_price = float(close[entry_position])

    target_points = stop_points * rr

    target_price = entry_price - target_points

    stop_price = entry_price + stop_points

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

        # Same conservative treatment
        # as the original S2 engine.
        if target_hit and stop_hit:
            return {
                "raw_points": -stop_points,
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
                "raw_points": -stop_points,
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
# GENERATE PARAMETERIZED TRADES
# ============================================================


def generate_trades(
    df,
    thresholds,
    scales,
    stop_points,
    rr,
    horizon,
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

        if len(session) <= horizon:
            continue

        valid = base_setup_mask(
            session,
            thresholds,
        )

        positions = np.flatnonzero(valid.to_numpy())

        i = 0

        while i < len(positions):
            position = positions[i]

            if position >= len(session) - horizon:
                break

            row = session.iloc[position]

            quality = calculate_quality(
                row,
                thresholds,
                scales,
            )

            if pd.isna(quality) or quality < QUALITY_THRESHOLD:
                i += 1
                continue

            result = resolve_short_trade(
                session=session,
                entry_position=position,
                stop_points=stop_points,
                rr=rr,
                horizon=horizon,
            )

            raw_points = result["raw_points"]

            net_points = raw_points - TOTAL_COST_POINTS

            # IMPORTANT:
            # Normalize by the tested stop.
            net_R = net_points / stop_points

            records.append(
                {
                    "entry_timestamp": session.index[position],
                    "exit_timestamp": session.index[result["exit_position"]],
                    "session_id": session_id,
                    "quality": quality,
                    "rr": rr,
                    "stop_points": stop_points,
                    "horizon": horizon,
                    "raw_points": raw_points,
                    "net_points": net_points,
                    "net_R": net_R,
                    "exit_reason": result["reason"],
                    "holding_bars": (result["exit_position"] - position),
                    "vol_percentile": row.get(
                        "vol_percentile",
                        np.nan,
                    ),
                    "vol_bucket": volatility_bucket(
                        row.get(
                            "vol_percentile",
                            np.nan,
                        )
                    ),
                }
            )

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

    PF = gross_profit / gross_loss if gross_loss > 0 else np.inf

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
        "trades": len(pnl),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "WR": (pnl > 0).mean(),
        "mean_R": pnl.mean(),
        "total_R": pnl.sum(),
        "PF": PF,
        "max_DD_R": drawdown.min(),
        "worst_streak": longest,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 PARAMETER SENSITIVITY")
    print("=" * 110)

    print()
    print("Frozen signal configuration:")
    print(f"Quality >= {QUALITY_THRESHOLD}")
    print(f"HMM state = {TARGET_STATE}")
    print(f"Lower tail = {TAIL_PERCENT}%")

    print()
    print("Parameter grid:")
    print(f"Stops    : {STOP_GRID}")
    print(f"RR       : {RR_GRID}")
    print(f"Horizons : {HORIZON_GRID}")

    df = load_data()

    df = add_directional_features(df)

    df = prepare_rth(df)

    windows = generate_windows(df)

    print()
    print(f"Walk-forward windows: {len(windows)}")

    parameter_records = []
    window_records = []

    combinations = list(
        product(
            STOP_GRID,
            RR_GRID,
            HORIZON_GRID,
        )
    )

    print()
    print(f"Parameter combinations: {len(combinations)}")

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for window_id, (
        train_start,
        validation_start,
        validation_end,
    ) in enumerate(
        windows,
        start=1,
    ):
        print(f"Processing window {window_id}/{len(windows)}...")

        train = df.loc[(df.index >= train_start) & (df.index < validation_start)].copy()

        validation = df.loc[
            (df.index >= validation_start) & (df.index < validation_end)
        ].copy()

        if train.empty or validation.empty:
            continue

        # ----------------------------------------------------
        # FIT ONLY ON TRAIN
        # ----------------------------------------------------

        train, validation = fit_hmm(
            train,
            validation,
        )

        thresholds = calculate_thresholds(train)

        scales = calculate_quality_scales(train)

        train, validation = add_volatility_percentile(
            train,
            validation,
        )

        # ----------------------------------------------------
        # PARAMETER GRID
        # ----------------------------------------------------

        for (
            stop_points,
            rr,
            horizon,
        ) in combinations:
            trades = generate_trades(
                validation,
                thresholds,
                scales,
                stop_points,
                rr,
                horizon,
            )

            metrics = calculate_metrics(trades)

            window_records.append(
                {
                    "window": window_id,
                    "stop_points": stop_points,
                    "rr": rr,
                    "horizon": horizon,
                    **metrics,
                }
            )

            parameter_records.append(
                {
                    "window": window_id,
                    "train_start": train_start,
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "stop_points": stop_points,
                    "rr": rr,
                    "horizon": horizon,
                    **metrics,
                }
            )

    # ========================================================
    # OUTPUTS
    # ========================================================

    results_dir = Path(__file__).resolve().parent / "results" / "s2_extended"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_window = pd.DataFrame(parameter_records)

    by_window_file = results_dir / "s2_parameter_sensitivity_by_window.csv"

    by_window.to_csv(
        by_window_file,
        index=False,
    )

    # ========================================================
    # COMBINED PARAMETER SUMMARY
    # ========================================================

    summary_records = []

    for (
        stop_points,
        rr,
        horizon,
    ), group in by_window.groupby(
        [
            "stop_points",
            "rr",
            "horizon",
        ],
        sort=True,
    ):
        active = group[group["trades"] > 0]

        if active.empty:
            continue

        positive_windows = (active["total_R"] > 0).sum()

        summary_records.append(
            {
                "stop_points": stop_points,
                "rr": rr,
                "horizon": horizon,
                "windows_with_trades": len(active),
                "positive_windows": positive_windows,
                "positive_window_pct": positive_windows / len(active),
                "total_trades": active["trades"].sum(),
                "total_R": active["total_R"].sum(),
                "mean_window_R": active["total_R"].mean(),
                "median_window_R": active["total_R"].median(),
                "mean_R": active["mean_R"].mean(),
                "median_PF": active["PF"].median(),
                "worst_window_R": active["total_R"].min(),
                "best_window_R": active["total_R"].max(),
                "mean_max_DD_R": active["max_DD_R"].mean(),
                "worst_max_DD_R": active["max_DD_R"].min(),
            }
        )

    summary = pd.DataFrame(summary_records)

    summary_file = results_dir / "s2_parameter_sensitivity_summary.csv"

    summary.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # PRINT TOP CANDIDATES
    # ========================================================

    print()
    print("=" * 110)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("=" * 110)

    if not summary.empty:
        display_columns = [
            "stop_points",
            "rr",
            "horizon",
            "windows_with_trades",
            "positive_window_pct",
            "total_trades",
            "total_R",
            "mean_window_R",
            "median_PF",
            "worst_window_R",
        ]

        ranked = summary.sort_values(
            [
                "positive_window_pct",
                "mean_window_R",
                "median_PF",
            ],
            ascending=False,
        )

        print(ranked[display_columns].head(20).to_string(index=False))

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(by_window_file)
    print(summary_file)

    print()
    print("S2 PARAMETER SENSITIVITY COMPLETE")


if __name__ == "__main__":
    main()
