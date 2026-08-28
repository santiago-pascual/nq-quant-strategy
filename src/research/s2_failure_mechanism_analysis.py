from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data


# ============================================================
# S2 FAILURE MECHANISM ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Understand WHY the frozen S2 benchmark wins or loses.
#
# IMPORTANT:
# This script is diagnostic.
# It does NOT modify the strategy.
# It does NOT optimize parameters.
# It does NOT create an OOS filter.
#
# Benchmark:
#   HMM state       = 2
#   Lower tail      = 17.5%
#   Quality         >= 0.75
#   Volatility      = 40-60%
#   Stop            = 25 points
#   RR              = 1.75
#   Horizon         = 20 bars
#
# ============================================================


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BENCHMARK_FILE = RESULTS_DIR / "s2_benchmark_trades.csv"


STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# ============================================================
# LOAD BENCHMARK
# ============================================================


def load_benchmark():

    trades = pd.read_csv(BENCHMARK_FILE)

    trades["entry_timestamp"] = pd.to_datetime(
        trades["entry_timestamp"],
        utc=True,
    )

    trades["exit_timestamp"] = pd.to_datetime(
        trades["exit_timestamp"],
        utc=True,
    )

    return trades


# ============================================================
# PREPARE MARKET DATA
# ============================================================


def prepare_market():

    df = load_data()

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")

    df["_timestamp_et"] = timestamps

    df = df.sort_values("_timestamp_et").copy()

    return df


# ============================================================
# ENTRY FEATURE ATTACHMENT
# ============================================================


def attach_entry_features(
    trades,
    market,
):

    entry = market[
        [
            "_timestamp_et",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "market_period",
            "past_return_1",
            "past_return_3",
            "past_return_5",
            "past_return_10",
            "past_return_15",
            "past_return_30",
            "realized_vol_5",
            "realized_vol_15",
            "realized_vol_30",
            "realized_vol_60",
            "vol_ratio_5_30",
            "vol_ratio_5_60",
            "variance_ratio_5_30",
            "variance_ratio_5_60",
        ]
    ].copy()

    entry = entry.rename(columns={"_timestamp_et": "entry_timestamp"})

    trades = trades.copy()

    # Benchmark timestamps are UTC.
    # Market timestamps are converted to UTC
    # before merging.

    entry["entry_timestamp"] = entry["entry_timestamp"].dt.tz_convert("UTC")

    trades = trades.merge(
        entry,
        on="entry_timestamp",
        how="left",
        suffixes=("", "_entry"),
    )

    return trades


# ============================================================
# INTRATRADE PATH
# ============================================================


def analyze_trade_path(
    trade,
    market,
):

    entry_ts = trade["entry_timestamp"]

    exit_ts = trade["exit_timestamp"]

    entry_price = float(trade["close"])

    window = market.loc[
        (market["_timestamp_et"].dt.tz_convert("UTC") > entry_ts)
        & (market["_timestamp_et"].dt.tz_convert("UTC") <= exit_ts)
    ].copy()

    if window.empty:
        return {
            "mfe_points": np.nan,
            "mae_points": np.nan,
            "mfe_R": np.nan,
            "mae_R": np.nan,
            "time_to_mfe": np.nan,
            "time_to_mae": np.nan,
            "max_favorable_bar": np.nan,
            "max_adverse_bar": np.nan,
            "first_bar_return_points": np.nan,
            "worst_close_to_close_points": np.nan,
            "best_close_to_close_points": np.nan,
        }

    highs = window["high"].to_numpy(dtype=float)

    lows = window["low"].to_numpy(dtype=float)

    closes = window["close"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # SHORT TRADE
    # --------------------------------------------------------

    favorable = entry_price - lows
    adverse = highs - entry_price

    mfe_points = float(np.max(favorable))

    mae_points = float(np.max(adverse))

    mfe_idx = int(np.argmax(favorable))

    mae_idx = int(np.argmax(adverse))

    mfe_R = mfe_points / STOP_POINTS
    mae_R = mae_points / STOP_POINTS

    time_to_mfe = mfe_idx + 1
    time_to_mae = mae_idx + 1

    first_bar_return_points = entry_price - closes[0]

    close_path = entry_price - closes

    worst_close_to_close_points = float(np.min(close_path))

    best_close_to_close_points = float(np.max(close_path))

    return {
        "mfe_points": mfe_points,
        "mae_points": mae_points,
        "mfe_R": mfe_R,
        "mae_R": mae_R,
        "time_to_mfe": time_to_mfe,
        "time_to_mae": time_to_mae,
        "max_favorable_bar": mfe_idx + 1,
        "max_adverse_bar": mae_idx + 1,
        "first_bar_return_points": first_bar_return_points,
        "worst_close_to_close_points": worst_close_to_close_points,
        "best_close_to_close_points": best_close_to_close_points,
    }


# ============================================================
# BUILD PATH FEATURES
# ============================================================


def build_path_features(
    trades,
    market,
):

    records = []

    for _, trade in trades.iterrows():
        result = analyze_trade_path(
            trade,
            market,
        )

        records.append(result)

    path = pd.DataFrame(
        records,
        index=trades.index,
    )

    return pd.concat(
        [
            trades,
            path,
        ],
        axis=1,
    )


# ============================================================
# FAILURE CLASSIFICATION
# ============================================================


def classify_failure(
    row,
):

    reason = row["exit_reason"]

    if reason == "target":
        return "TARGET"

    if reason == "stop":
        return "STOP"

    if reason == "both_hit_conservative_stop":
        return "BOTH_HIT_STOP"

    if reason == "timeout":
        if row["raw_points"] > 0:
            return "TIMEOUT_WIN"

        if row["raw_points"] < 0:
            return "TIMEOUT_LOSS"

        return "TIMEOUT_FLAT"

    return "OTHER"


# ============================================================
# EXIT TYPE SUMMARY
# ============================================================


def exit_type_analysis(
    trades,
):

    summary = (
        trades.groupby("failure_type")
        .agg(
            trades=("net_R", "size"),
            win_rate=(
                "net_R",
                lambda x: (x > 0).mean(),
            ),
            mean_R=("net_R", "mean"),
            median_R=("net_R", "median"),
            total_R=("net_R", "sum"),
            mean_MFE_R=("mfe_R", "mean"),
            mean_MAE_R=("mae_R", "mean"),
            median_MFE_R=("mfe_R", "median"),
            median_MAE_R=("mae_R", "median"),
            mean_time_to_MFE=(
                "time_to_mfe",
                "mean",
            ),
            mean_time_to_MAE=(
                "time_to_mae",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# WIN / LOSS MECHANICS
# ============================================================


def outcome_analysis(
    trades,
):

    trades = trades.copy()

    trades["outcome"] = np.where(
        trades["net_R"] > 0,
        "WIN",
        "LOSS",
    )

    summary = (
        trades.groupby("outcome")
        .agg(
            trades=("net_R", "size"),
            mean_R=("net_R", "mean"),
            median_R=("net_R", "median"),
            mean_MFE_R=("mfe_R", "mean"),
            median_MFE_R=("mfe_R", "median"),
            mean_MAE_R=("mae_R", "mean"),
            median_MAE_R=("mae_R", "median"),
            mean_time_to_MFE=(
                "time_to_mfe",
                "mean",
            ),
            median_time_to_MFE=(
                "time_to_mfe",
                "median",
            ),
            mean_time_to_MAE=(
                "time_to_mae",
                "mean",
            ),
            median_time_to_MAE=(
                "time_to_mae",
                "median",
            ),
            mean_first_bar_return=(
                "first_bar_return_points",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def mfe_mae_buckets(
    trades,
):

    df = trades.copy()

    df["mfe_bucket"] = pd.cut(
        df["mfe_R"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
            1.00,
            1.50,
            2.00,
            np.inf,
        ],
    )

    df["mae_bucket"] = pd.cut(
        df["mae_R"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
            1.00,
            1.50,
            2.00,
            np.inf,
        ],
    )

    mfe_summary = (
        df.groupby(
            "mfe_bucket",
            observed=True,
        )
        .agg(
            trades=("net_R", "size"),
            win_rate=(
                "net_R",
                lambda x: (x > 0).mean(),
            ),
            mean_R=("net_R", "mean"),
            total_R=("net_R", "sum"),
        )
        .reset_index()
    )

    mae_summary = (
        df.groupby(
            "mae_bucket",
            observed=True,
        )
        .agg(
            trades=("net_R", "size"),
            win_rate=(
                "net_R",
                lambda x: (x > 0).mean(),
            ),
            mean_R=("net_R", "mean"),
            total_R=("net_R", "sum"),
        )
        .reset_index()
    )

    return (
        mfe_summary,
        mae_summary,
    )


# ============================================================
# STOP DIAGNOSTICS
# ============================================================


def stop_analysis(
    trades,
):

    stops = trades.loc[trades["failure_type"] == "STOP"].copy()

    if stops.empty:
        return pd.DataFrame(
            [
                {
                    "stop_trades": 0,
                    "mean_MFE_R": np.nan,
                    "median_MFE_R": np.nan,
                    "mean_MAE_R": np.nan,
                    "median_MAE_R": np.nan,
                    "mean_time_to_MFE": np.nan,
                    "median_time_to_MFE": np.nan,
                    "mean_time_to_MAE": np.nan,
                    "median_time_to_MAE": np.nan,
                    "mean_first_bar_return": np.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "stop_trades": len(stops),
                "mean_MFE_R": stops["mfe_R"].mean(),
                "median_MFE_R": stops["mfe_R"].median(),
                "mean_MAE_R": stops["mae_R"].mean(),
                "median_MAE_R": stops["mae_R"].median(),
                "mean_time_to_MFE": stops["time_to_mfe"].mean(),
                "median_time_to_MFE": stops["time_to_mfe"].median(),
                "mean_time_to_MAE": stops["time_to_mae"].mean(),
                "median_time_to_MAE": stops["time_to_mae"].median(),
                "mean_first_bar_return": (stops["first_bar_return_points"].mean()),
            }
        ]
    )

    if "time_to_MFE" in trades.columns and "time_to_mfe" not in trades.columns:
        trades["time_to_mfe"] = trades["time_to_MFE"]

    if "time_to_MAE" in trades.columns and "time_to_mae" not in trades.columns:
        trades["time_to_mae"] = trades["time_to_MAE"]


def add_time_features(
    trades,
):

    trades = trades.copy()

    eastern = trades["entry_timestamp"].dt.tz_convert("America/New_York")

    trades["entry_hour"] = eastern.dt.hour

    trades["entry_minute"] = eastern.dt.minute

    trades["entry_time_minutes"] = eastern.dt.hour * 60 + eastern.dt.minute

    trades["time_bucket"] = pd.cut(
        trades["entry_time_minutes"],
        bins=[
            570,
            600,
            630,
            660,
            690,
            720,
            750,
            780,
            810,
            840,
            870,
            900,
            930,
            960,
        ],
        right=False,
        labels=[
            "09:30-10:00",
            "10:00-10:30",
            "10:30-11:00",
            "11:00-11:30",
            "11:30-12:00",
            "12:00-12:30",
            "12:30-13:00",
            "13:00-13:30",
            "13:30-14:00",
            "14:00-14:30",
            "14:30-15:00",
            "15:00-15:30",
            "15:30-16:00",
        ],
    )

    return trades


# ============================================================
# TIME × OUTCOME
# ============================================================


def time_analysis(
    trades,
):

    summary = (
        trades.groupby(
            "time_bucket",
            observed=True,
        )
        .agg(
            trades=("net_R", "size"),
            wins=(
                "net_R",
                lambda x: (x > 0).sum(),
            ),
            win_rate=(
                "net_R",
                lambda x: (x > 0).mean(),
            ),
            mean_R=("net_R", "mean"),
            total_R=("net_R", "sum"),
            mean_MFE_R=("mfe_R", "mean"),
            mean_MAE_R=("mae_R", "mean"),
        )
        .reset_index()
    )

    return summary


# ============================================================
# MOMENTUM × OUTCOME
# ============================================================


def momentum_analysis(
    trades,
):

    features = [
        "past_return_5",
        "past_return_10",
        "past_return_15",
        "past_return_30",
    ]

    rows = []

    for feature in features:
        valid = (
            trades[feature]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if valid.empty:
            continue

        q = valid.quantile(
            [
                0.25,
                0.50,
                0.75,
            ]
        )

        for label, mask in [
            (
                "BOTTOM25",
                trades[feature] <= q.iloc[0],
            ),
            (
                "MIDDLE50",
                (trades[feature] > q.iloc[0]) & (trades[feature] <= q.iloc[2]),
            ),
            (
                "TOP25",
                trades[feature] > q.iloc[2],
            ),
        ]:
            subset = trades.loc[mask]

            if subset.empty:
                continue

            rows.append(
                {
                    "feature": feature,
                    "bucket": label,
                    "trades": len(subset),
                    "win_rate": (subset["net_R"] > 0).mean(),
                    "mean_R": subset["net_R"].mean(),
                    "total_R": subset["net_R"].sum(),
                    "mean_MFE_R": subset["mfe_R"].mean(),
                    "mean_MAE_R": subset["mae_R"].mean(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# VOLATILITY × OUTCOME
# ============================================================


def volatility_analysis(
    trades,
):

    features = [
        "vol_ratio_5_30",
        "vol_ratio_5_60",
        "variance_ratio_5_30",
        "variance_ratio_5_60",
    ]

    rows = []

    for feature in features:
        values = (
            trades[feature]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if values.empty:
            continue

        q25 = values.quantile(0.25)

        q75 = values.quantile(0.75)

        for label, mask in [
            (
                "BOTTOM25",
                trades[feature] <= q25,
            ),
            (
                "MIDDLE50",
                (trades[feature] > q25) & (trades[feature] <= q75),
            ),
            (
                "TOP25",
                trades[feature] > q75,
            ),
        ]:
            subset = trades.loc[mask]

            if subset.empty:
                continue

            rows.append(
                {
                    "feature": feature,
                    "bucket": label,
                    "trades": len(subset),
                    "win_rate": (subset["net_R"] > 0).mean(),
                    "mean_R": subset["net_R"].mean(),
                    "total_R": subset["net_R"].sum(),
                    "mean_MFE_R": subset["mfe_R"].mean(),
                    "mean_MAE_R": subset["mae_R"].mean(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 FAILURE MECHANISM ANALYSIS")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print("  HMM state       = 2")
    print("  Lower tail      = 17.5%")
    print("  Quality         >= 0.75")
    print("  Volatility      = 40-60%")
    print("  Stop            = 25 points")
    print("  RR              = 1.75")
    print("  Horizon         = 20 bars")

    print()
    print("Loading benchmark...")

    trades = load_benchmark()

    print(f"Benchmark trades: {len(trades)}")

    print()
    print("Loading expanded market data...")

    market = prepare_market()

    print(f"Market observations: {len(market)}")

    print()
    print("Attaching entry features...")

    trades = attach_entry_features(
        trades,
        market,
    )

    missing = trades["close"].isna().sum()

    print(f"Trades without entry features: {missing}")

    if missing > 0:
        raise RuntimeError("Some benchmark trades could not be matched to market data.")

    print()
    print("Analyzing intratrade paths...")

    trades = build_path_features(
        trades,
        market,
    )

    trades["failure_type"] = trades.apply(
        classify_failure,
        axis=1,
    )

    trades = add_time_features(trades)

    # --------------------------------------------------------
    # OUTPUT 1
    # --------------------------------------------------------

    exit_summary = exit_type_analysis(trades)

    print()
    print("=" * 110)
    print("1. EXIT / FAILURE MECHANISM")
    print("=" * 110)

    print(exit_summary.to_string(index=False))

    # --------------------------------------------------------
    # OUTPUT 2
    # --------------------------------------------------------

    outcome_summary = outcome_analysis(trades)

    print()
    print("=" * 110)
    print("2. WIN vs LOSS MECHANICS")
    print("=" * 110)

    print(outcome_summary.to_string(index=False))

    # --------------------------------------------------------
    # OUTPUT 3
    # --------------------------------------------------------

    mfe_summary, mae_summary = mfe_mae_buckets(trades)

    print()
    print("=" * 110)
    print("3. MFE BUCKETS")
    print("=" * 110)

    print(mfe_summary.to_string(index=False))

    print()
    print("=" * 110)
    print("4. MAE BUCKETS")
    print("=" * 110)

    print(mae_summary.to_string(index=False))

    # --------------------------------------------------------
    # OUTPUT 4
    # --------------------------------------------------------

    stop_summary = stop_analysis(trades)

    print()
    print("=" * 110)
    print("5. STOP MECHANICS")
    print("=" * 110)

    if stop_summary.empty:
        print("No stop trades.")
    else:
        print(stop_summary.to_string(index=False))

    # --------------------------------------------------------
    # OUTPUT 5
    # --------------------------------------------------------

    time_summary = time_analysis(trades)

    print()
    print("=" * 110)
    print("6. TIME OF DAY")
    print("=" * 110)

    print(time_summary.to_string(index=False))

    # --------------------------------------------------------
    # OUTPUT 6
    # --------------------------------------------------------

    momentum_summary = momentum_analysis(trades)

    print()
    print("=" * 110)
    print("7. MOMENTUM")
    print("=" * 110)

    print(momentum_summary.to_string(index=False))

    # --------------------------------------------------------
    # OUTPUT 7
    # --------------------------------------------------------

    volatility_summary = volatility_analysis(trades)

    print()
    print("=" * 110)
    print("8. VOLATILITY EXPANSION")
    print("=" * 110)

    print(volatility_summary.to_string(index=False))

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    trades.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_trades.csv",
        index=False,
    )

    exit_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_exit_summary.csv",
        index=False,
    )

    outcome_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_outcome_summary.csv",
        index=False,
    )

    mfe_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_mfe.csv",
        index=False,
    )

    mae_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_mae.csv",
        index=False,
    )

    stop_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_stop.csv",
        index=False,
    )

    time_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_time.csv",
        index=False,
    )

    momentum_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_momentum.csv",
        index=False,
    )

    volatility_summary.to_csv(
        RESULTS_DIR / "s2_failure_mechanism_volatility.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(RESULTS_DIR / "s2_failure_mechanism_trades.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_exit_summary.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_outcome_summary.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_mfe.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_mae.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_stop.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_time.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_momentum.csv")

    print(RESULTS_DIR / "s2_failure_mechanism_volatility.csv")

    print()
    print("S2 FAILURE MECHANISM ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
