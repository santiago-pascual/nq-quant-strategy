from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data


# ============================================================
# S2 FAILURE ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Investigate WHY S2 trades win or lose.
#
# IMPORTANT
# ---------
# This script is DESCRIPTIVE.
#
# It does NOT:
#   - optimize parameters
#   - create a new strategy
#   - select a filter
#   - modify S2
#
# We first identify patterns in failures.
# Only afterwards do we formulate hypotheses and test them OOS.
#
# ============================================================


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRADES_FILE = RESULTS_DIR / "s2_selective_execution_B_trades.csv"


# ============================================================
# PARAMETERS
# ============================================================

RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60

VOL_BUCKETS = [
    (0.00, 0.20, "0-20"),
    (0.20, 0.40, "20-40"),
    (0.40, 0.60, "40-60"),
    (0.60, 0.80, "60-80"),
    (0.80, 1.00, "80-100"),
]


# ============================================================
# LOAD TRADES
# ============================================================


def load_trades():

    print("Loading S2 trades...")

    trades = pd.read_csv(
        TRADES_FILE,
        parse_dates=[
            "entry_timestamp",
            "exit_timestamp",
        ],
    )

    print(f"Trades loaded: {len(trades):,}")

    return trades


# ============================================================
# LOAD MARKET DATA
# ============================================================


def load_market_data():

    print("Loading expanded MNQ dataset...")

    df = load_data()

    df["timestamp ET"] = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    if df["timestamp ET"].dt.tz is None:
        df["timestamp ET"] = df["timestamp ET"].dt.tz_localize("America/New_York")

    else:
        df["timestamp ET"] = df["timestamp ET"].dt.tz_convert("America/New_York")

    return df


# ============================================================
# PREPARE TRADE FEATURES
# ============================================================


def add_entry_features(
    trades,
    market,
):

    # --------------------------------------------------------
    # Normalize trade timestamps
    # --------------------------------------------------------
    #
    # CSV trade timestamps may contain mixed DST offsets:
    #   -04:00
    #   -05:00
    #
    # Parse through UTC first, then convert to New York.
    #

    trades["entry_timestamp"] = pd.to_datetime(
        trades["entry_timestamp"],
        errors="coerce",
        utc=True,
    )

    trades["entry_timestamp"] = trades["entry_timestamp"].dt.tz_convert(
        "America/New_York"
    )

    # --------------------------------------------------------
    # Normalize market timestamps
    # --------------------------------------------------------

    market["timestamp ET"] = pd.to_datetime(
        market["timestamp ET"],
        errors="coerce",
        utc=True,
    )

    market["timestamp ET"] = market["timestamp ET"].dt.tz_convert("America/New_York")

    market = market.set_index("timestamp ET")

    # --------------------------------------------------------
    # Direct entry-time features
    # --------------------------------------------------------

    feature_columns = [
        "realized_vol_5",
        "realized_vol_15",
        "realized_vol_30",
        "realized_vol_60",
        "vol_ratio_5_30",
        "vol_ratio_5_60",
        "variance_5",
        "variance_30",
        "variance_60",
        "variance_ratio_5_30",
        "variance_ratio_5_60",
        "past_return_1",
        "past_return_3",
        "past_return_5",
        "past_return_10",
        "past_return_15",
        "past_return_30",
        "minutes_since_rth_open",
        "minutes_until_rth_close",
        "rth_progress",
    ]

    available = [column for column in feature_columns if column in market.columns]

    entry_features = market[available].copy()

    entry_features = entry_features.rename(
        columns={column: f"entry_{column}" for column in available}
    )

    trades = trades.merge(
        entry_features,
        left_on="entry_timestamp",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Entry clock time
    # --------------------------------------------------------

    trades["entry_hour"] = trades["entry_timestamp"].dt.hour

    trades["entry_minute"] = trades["entry_timestamp"].dt.minute

    trades["entry_minutes"] = trades["entry_hour"] * 60 + trades["entry_minute"]

    trades["entry_minutes_from_rth_open"] = trades["entry_minutes"] - RTH_OPEN_MINUTE

    # --------------------------------------------------------
    # RTH time bucket
    # --------------------------------------------------------

    def time_bucket(minutes):

        if pd.isna(minutes):
            return "unknown"

        if minutes < 30:
            return "09:30-10:00"

        if minutes < 60:
            return "10:00-10:30"

        if minutes < 120:
            return "10:30-11:30"

        if minutes < 210:
            return "11:30-13:00"

        if minutes < 300:
            return "13:00-14:30"

        if minutes < 390:
            return "14:30-15:30"

        return "15:30-16:00"

    trades["entry_time_bucket"] = trades["entry_minutes_from_rth_open"].apply(
        time_bucket
    )

    return trades


# ============================================================
# VOLATILITY BUCKET
# ============================================================


def add_volatility_bucket(
    trades,
):

    trades = trades.copy()

    # --------------------------------------------------------
    # Prefer volatility percentile already stored in S2 trades
    # --------------------------------------------------------

    if "vol_percentile" in trades.columns:
        column = "vol_percentile"

    elif "entry_vol_percentile" in trades.columns:
        column = "entry_vol_percentile"

    else:
        trades["entry_vol_bucket"] = "unknown"

        return trades

    def bucket(value):

        if pd.isna(value):
            return "unknown"

        for low, high, label in VOL_BUCKETS:
            if value >= low and value < high:
                return label

        if value >= 1.0:
            return "80-100"

        return "unknown"

    trades["entry_vol_bucket"] = trades[column].apply(bucket)

    return trades


def add_outcome_features(
    trades,
):

    trades = trades.copy()

    trades["outcome"] = np.where(
        trades["net_R"] > 0,
        "WIN",
        np.where(
            trades["net_R"] < 0,
            "LOSS",
            "FLAT",
        ),
    )

    trades["is_win"] = trades["net_R"] > 0

    trades["is_loss"] = trades["net_R"] < 0

    trades["abs_R"] = trades["net_R"].abs()

    return trades


# ============================================================
# SUMMARY HELPER
# ============================================================


def summarize_group(
    grouped,
):

    result = grouped.agg(
        trades=("net_R", "size"),
        wins=("is_win", "sum"),
        losses=("is_loss", "sum"),
        mean_R=("net_R", "mean"),
        median_R=("net_R", "median"),
        total_R=("net_R", "sum"),
        mean_holding_bars=(
            "holding_bars",
            "mean",
        ),
        median_holding_bars=(
            "holding_bars",
            "median",
        ),
    ).reset_index()

    result["win_rate"] = result["wins"] / result["trades"]

    return result


# ============================================================
# ANALYSIS 1 — VOLATILITY
# ============================================================


def analyze_volatility(
    trades,
):

    print()
    print("=" * 110)
    print("1. VOLATILITY ANALYSIS")
    print("=" * 110)

    result = summarize_group(
        trades.groupby(
            "entry_vol_bucket",
            dropna=False,
        )
    )

    result = result.sort_values(
        "total_R",
        ascending=False,
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 2 — QUALITY
# ============================================================


def analyze_quality(
    trades,
):

    print()
    print("=" * 110)
    print("2. QUALITY ANALYSIS")
    print("=" * 110)

    if "quality" not in trades.columns:
        print("Quality column unavailable.")

        return pd.DataFrame()

    bins = [
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.01,
    ]

    labels = [
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]

    trades = trades.copy()

    trades["quality_bucket"] = pd.cut(
        trades["quality"],
        bins=bins,
        labels=labels,
        right=False,
    )

    result = summarize_group(
        trades.groupby(
            "quality_bucket",
            observed=False,
        )
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 3 — TIME OF DAY
# ============================================================


def analyze_time(
    trades,
):

    print()
    print("=" * 110)
    print("3. TIME-OF-DAY ANALYSIS")
    print("=" * 110)

    result = summarize_group(
        trades.groupby(
            "entry_time_bucket",
            dropna=False,
        )
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 4 — EXIT REASON
# ============================================================


def analyze_exit_reason(
    trades,
):

    print()
    print("=" * 110)
    print("4. EXIT REASON ANALYSIS")
    print("=" * 110)

    result = summarize_group(
        trades.groupby(
            "exit_reason",
            dropna=False,
        )
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 5 — HOLDING TIME
# ============================================================


def analyze_holding_time(
    trades,
):

    print()
    print("=" * 110)
    print("5. HOLDING TIME ANALYSIS")
    print("=" * 110)

    trades = trades.copy()

    bins = [
        -1,
        2,
        5,
        10,
        15,
        20,
        30,
        1000,
    ]

    labels = [
        "0-2",
        "3-5",
        "6-10",
        "11-15",
        "16-20",
        "21-30",
        "31+",
    ]

    trades["holding_bucket"] = pd.cut(
        trades["holding_bars"],
        bins=bins,
        labels=labels,
    )

    result = summarize_group(
        trades.groupby(
            "holding_bucket",
            observed=False,
        )
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 6 — WIN VS LOSS FEATURE DISTRIBUTIONS
# ============================================================


def analyze_feature_distributions(
    trades,
):

    print()
    print("=" * 110)
    print("6. WIN vs LOSS FEATURE DISTRIBUTIONS")
    print("=" * 110)

    features = [
        "quality",
        "entry_realized_vol_5",
        "entry_realized_vol_15",
        "entry_realized_vol_30",
        "entry_realized_vol_60",
        "entry_vol_ratio_5_30",
        "entry_vol_ratio_5_60",
        "entry_variance_ratio_5_30",
        "entry_variance_ratio_5_60",
        "entry_past_return_5",
        "entry_past_return_15",
        "entry_past_return_30",
        "entry_minutes_from_rth_open",
    ]

    available = [feature for feature in features if feature in trades.columns]

    rows = []

    for feature in available:
        win_values = trades.loc[
            trades["is_win"],
            feature,
        ].dropna()

        loss_values = trades.loc[
            trades["is_loss"],
            feature,
        ].dropna()

        if len(win_values) == 0 or len(loss_values) == 0:
            continue

        rows.append(
            {
                "feature": feature,
                "win_n": len(win_values),
                "loss_n": len(loss_values),
                "win_mean": win_values.mean(),
                "loss_mean": loss_values.mean(),
                "win_median": win_values.median(),
                "loss_median": loss_values.median(),
                "win_q25": win_values.quantile(0.25),
                "win_q75": win_values.quantile(0.75),
                "loss_q25": loss_values.quantile(0.25),
                "loss_q75": loss_values.quantile(0.75),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["mean_difference"] = result["win_mean"] - result["loss_mean"]

        result["median_difference"] = result["win_median"] - result["loss_median"]

        result = result.sort_values(
            "mean_difference",
            key=lambda x: x.abs(),
            ascending=False,
        )

        print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 7 — WORST LOSSES
# ============================================================


def analyze_worst_losses(
    trades,
):

    print()
    print("=" * 110)
    print("7. WORST LOSSES")
    print("=" * 110)

    columns = [
        "entry_timestamp",
        "exit_timestamp",
        "quality",
        "entry_vol_bucket",
        "entry_time_bucket",
        "net_R",
        "raw_points",
        "net_points",
        "exit_reason",
        "holding_bars",
    ]

    available = [column for column in columns if column in trades.columns]

    result = (
        trades.loc[
            trades["is_loss"],
            available,
        ]
        .sort_values(
            "net_R",
            ascending=True,
        )
        .head(30)
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# ANALYSIS 8 — FAILURE INTERACTION
# ============================================================


def analyze_regime_time_interaction(
    trades,
):

    print()
    print("=" * 110)
    print("8. VOLATILITY × TIME INTERACTION")
    print("=" * 110)

    result = summarize_group(
        trades.groupby(
            [
                "entry_vol_bucket",
                "entry_time_bucket",
            ],
            dropna=False,
        )
    )

    result = result.sort_values(
        "total_R",
        ascending=False,
    )

    print(result.to_string(index=False))

    return result


# ============================================================
# SAVE
# ============================================================


def save_result(
    result,
    filename,
):

    path = RESULTS_DIR / filename

    result.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 FAILURE ANALYSIS — EXPANDED DATASET")
    print("=" * 110)

    trades = load_trades()

    market = load_market_data()

    trades = add_entry_features(
        trades,
        market,
    )

    trades = add_volatility_bucket(
        trades,
    )

    trades = add_outcome_features(
        trades,
    )

    print()
    print("=" * 110)
    print("DATASET")
    print("=" * 110)

    print(f"Trades: {len(trades):,}")

    print(f"Wins: {trades['is_win'].sum():,}")

    print(f"Losses: {trades['is_loss'].sum():,}")

    print(f"Overall WR: {trades['is_win'].mean():.2%}")

    print(f"Total R: {trades['net_R'].sum():.4f}")

    # --------------------------------------------------------
    # Run analyses
    # --------------------------------------------------------

    volatility = analyze_volatility(trades)

    quality = analyze_quality(trades)

    time = analyze_time(trades)

    exit_reason = analyze_exit_reason(trades)

    holding = analyze_holding_time(trades)

    distributions = analyze_feature_distributions(trades)

    worst_losses = analyze_worst_losses(trades)

    interaction = analyze_regime_time_interaction(trades)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_result(
        volatility,
        "s2_failure_volatility.csv",
    )

    save_result(
        quality,
        "s2_failure_quality.csv",
    )

    save_result(
        time,
        "s2_failure_time.csv",
    )

    save_result(
        exit_reason,
        "s2_failure_exit_reason.csv",
    )

    save_result(
        holding,
        "s2_failure_holding.csv",
    )

    save_result(
        distributions,
        "s2_failure_feature_distributions.csv",
    )

    save_result(
        worst_losses,
        "s2_failure_worst_losses.csv",
    )

    save_result(
        interaction,
        "s2_failure_volatility_time.csv",
    )

    # --------------------------------------------------------
    # Save enriched trades
    # --------------------------------------------------------

    enriched_path = RESULTS_DIR / "s2_failure_analysis_trades.csv"

    trades.to_csv(
        enriched_path,
        index=False,
    )

    print()
    print(f"Saved enriched trades: {enriched_path}")

    print()
    print("=" * 110)
    print("S2 FAILURE ANALYSIS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
