from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"

BENCHMARK_FILE = RESULTS_DIR / "s2_benchmark_trades.csv"


# ============================================================
# FROZEN BENCHMARK
# ============================================================

# State 2
# Lower tail = 17.5%
# Quality >= 0.75
# Volatility = 40-60%
# Stop = 25
# RR = 1.75
# Horizon = 20
#
# NOTHING BELOW CHANGES THE BENCHMARK.
# ============================================================


# ============================================================
# TIME WINDOWS
# ============================================================

TIME_BUCKETS = {
    "09:30-10:00": (0, 30),
    "10:00-10:30": (30, 60),
    "10:30-11:30": (60, 120),
    "11:30-13:00": (120, 210),
    "13:00-14:30": (210, 300),
    "14:30-15:30": (300, 360),
    "15:30-16:00": (360, 390),
}


# ============================================================
# MARKET FEATURES TO RECOVER
# ============================================================

MARKET_FEATURES = [
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
    "variance_5",
    "variance_30",
    "variance_60",
    "variance_ratio_5_30",
    "variance_ratio_5_60",
]


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict:

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
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "median_R": float(pnl.median()),
        "total_R": float(pnl.sum()),
        "profit_factor": float(profit_factor),
        "max_drawdown_R": float(drawdown.min()),
        "longest_losing_streak": int(longest_streak),
    }


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================


def normalize_timestamp(
    series: pd.Series,
) -> pd.Series:

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).dt.tz_convert("America/New_York")


# ============================================================
# RECOVER ENTRY FEATURES
# ============================================================


def attach_entry_features(
    trades: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:

    trades = trades.copy()

    market = market.copy()

    trades["_entry_ts"] = normalize_timestamp(trades["entry_timestamp"])

    market["_market_ts"] = normalize_timestamp(market["timestamp ET"])

    market = market.sort_values("_market_ts").drop_duplicates(
        "_market_ts",
        keep="last",
    )

    columns = [
        "_market_ts",
        *[feature for feature in MARKET_FEATURES if feature in market.columns],
    ]

    entry_features = market[columns].copy()

    entry_features = entry_features.rename(
        columns={
            feature: f"entry_{feature}"
            for feature in MARKET_FEATURES
            if feature in entry_features.columns
        }
    )

    # Exact timestamp match.
    #
    # The benchmark entries are generated from the same
    # one-minute dataset, so we expect an exact match.
    #

    trades = trades.merge(
        entry_features,
        left_on="_entry_ts",
        right_on="_market_ts",
        how="left",
        validate="one_to_one",
    )

    trades = trades.drop(
        columns=[
            "_entry_ts",
            "_market_ts",
        ]
    )

    return trades


# ============================================================
# TIME FEATURES
# ============================================================


def add_time_features(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    trades = trades.copy()

    ts = normalize_timestamp(trades["entry_timestamp"])

    trades["entry_timestamp_et"] = ts

    minutes = ts.dt.hour * 60 + ts.dt.minute

    trades["minutes_from_rth_open"] = minutes - (9 * 60 + 30)

    return trades


# ============================================================
# FEATURE QUANTILE FILTER
# ============================================================


def make_quantile_filter(
    series: pd.Series,
    quantile: float,
    direction: str,
) -> pd.Series:

    threshold = series.quantile(quantile)

    if direction == "low":
        return series <= threshold

    if direction == "high":
        return series >= threshold

    raise ValueError(f"Unknown direction: {direction}")


# ============================================================
# TIME FILTERS
# ============================================================


def time_mask(
    trades: pd.DataFrame,
    bucket: str,
) -> pd.Series:

    low, high = TIME_BUCKETS[bucket]

    return (trades["minutes_from_rth_open"] >= low) & (
        trades["minutes_from_rth_open"] < high
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 FAILURE FILTER TEST — ENTRY FEATURE DISCOVERY")
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

    # ========================================================
    # LOAD BENCHMARK
    # ========================================================

    print("Loading frozen benchmark...")

    trades = pd.read_csv(BENCHMARK_FILE)

    print(
        "Benchmark trades:",
        len(trades),
    )

    # ========================================================
    # LOAD MARKET DATA
    # ========================================================

    print("Loading expanded MNQ dataset...")

    market = load_data()

    print(
        "Expanded observations:",
        len(market),
    )

    # ========================================================
    # ATTACH FEATURES
    # ========================================================

    print("Attaching entry features...")

    trades = attach_entry_features(
        trades,
        market,
    )

    trades = add_time_features(trades)

    # ========================================================
    # MATCH QUALITY CHECK
    # ========================================================

    feature_columns = [
        f"entry_{feature}"
        for feature in MARKET_FEATURES
        if f"entry_{feature}" in trades.columns
    ]

    print()
    print(
        "Entry features recovered:",
        len(feature_columns),
    )

    if feature_columns:
        missing = trades[feature_columns].isna().all(axis=1).sum()

        print(
            "Trades without entry features:",
            missing,
        )

    # ========================================================
    # BASELINE
    # ========================================================

    experiments = []

    experiments.append(
        (
            "BASELINE",
            pd.Series(
                True,
                index=trades.index,
            ),
        )
    )

    # ========================================================
    # TIME ANALYSIS
    # ========================================================

    for bucket in TIME_BUCKETS:
        experiments.append(
            (
                f"TIME_{bucket}",
                time_mask(
                    trades,
                    bucket,
                ),
            )
        )

    # Exclude previously weak periods
    experiments.append(
        (
            "EXCLUDE_10_00_10_30",
            ~time_mask(
                trades,
                "10:00-10:30",
            ),
        )
    )

    experiments.append(
        (
            "EXCLUDE_10_00_10_30_14_30_PLUS",
            (
                ~time_mask(
                    trades,
                    "10:00-10:30",
                )
                & ~time_mask(
                    trades,
                    "14:30-15:30",
                )
                & ~time_mask(
                    trades,
                    "15:30-16:00",
                )
            ),
        )
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    momentum_features = [
        "entry_past_return_5",
        "entry_past_return_10",
        "entry_past_return_15",
        "entry_past_return_30",
    ]

    for feature in momentum_features:
        if feature not in trades.columns:
            continue

        series = trades[feature]

        # SHORT strategy:
        # investigate stronger negative momentum.

        experiments.append(
            (
                f"{feature}_BOTTOM25",
                make_quantile_filter(
                    series,
                    0.25,
                    "low",
                ),
            )
        )

        experiments.append(
            (
                f"{feature}_TOP25",
                make_quantile_filter(
                    series,
                    0.75,
                    "high",
                ),
            )
        )

    # ========================================================
    # VOLATILITY STRUCTURE
    # ========================================================

    volatility_features = [
        "entry_vol_ratio_5_30",
        "entry_vol_ratio_5_60",
        "entry_variance_ratio_5_30",
        "entry_variance_ratio_5_60",
    ]

    for feature in volatility_features:
        if feature not in trades.columns:
            continue

        series = trades[feature]

        experiments.append(
            (
                f"{feature}_BOTTOM25",
                make_quantile_filter(
                    series,
                    0.25,
                    "low",
                ),
            )
        )

        experiments.append(
            (
                f"{feature}_TOP25",
                make_quantile_filter(
                    series,
                    0.75,
                    "high",
                ),
            )
        )

    # ========================================================
    # QUALITY
    # ========================================================

    if "quality" in trades.columns:
        experiments.append(
            (
                "QUALITY_BOTTOM25",
                make_quantile_filter(
                    trades["quality"],
                    0.25,
                    "low",
                ),
            )
        )

        experiments.append(
            (
                "QUALITY_TOP25",
                make_quantile_filter(
                    trades["quality"],
                    0.75,
                    "high",
                ),
            )
        )

    # ========================================================
    # VOLATILITY × TIME
    # ========================================================

    if "vol_percentile" in trades.columns:
        medium_vol = (trades["vol_percentile"] >= 0.40) & (
            trades["vol_percentile"] < 0.60
        )

        higher_vol = (trades["vol_percentile"] >= 0.60) & (
            trades["vol_percentile"] < 0.80
        )

        experiments.append(
            (
                "VOL_40_60",
                medium_vol,
            )
        )

        experiments.append(
            (
                "VOL_60_80",
                higher_vol,
            )
        )

        afternoon = (trades["minutes_from_rth_open"] >= 210) & (
            trades["minutes_from_rth_open"] < 300
        )

        experiments.append(
            (
                "VOL_40_60_AND_13_00_14_30",
                medium_vol & afternoon,
            )
        )

        experiments.append(
            (
                "VOL_60_80_AND_13_00_14_30",
                higher_vol & afternoon,
            )
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    summary_rows = []
    window_rows = []

    for name, mask in experiments:
        mask = mask.fillna(False).astype(bool)

        subset = trades.loc[mask].copy()

        metrics = calculate_metrics(subset)

        window_metrics = []

        for window, group in subset.groupby(
            "window",
            sort=True,
        ):
            wm = calculate_metrics(group)

            window_metrics.append(wm)

            window_rows.append(
                {
                    "filter": name,
                    "window": window,
                    **wm,
                }
            )

        total_Rs = [x["total_R"] for x in window_metrics]

        positive_windows = sum(value > 0 for value in total_Rs)

        windows_with_trades = len(window_metrics)

        summary_rows.append(
            {
                "filter": name,
                **metrics,
                "windows_with_trades": windows_with_trades,
                "positive_windows": positive_windows,
                "positive_window_pct": (
                    positive_windows / windows_with_trades
                    if windows_with_trades
                    else np.nan
                ),
                "mean_window_R": (np.mean(total_Rs) if total_Rs else np.nan),
                "worst_window_R": (np.min(total_Rs) if total_Rs else np.nan),
                "best_window_R": (np.max(total_Rs) if total_Rs else np.nan),
            }
        )

    # ========================================================
    # SAVE
    # ========================================================

    summary = pd.DataFrame(summary_rows).sort_values(
        "total_R",
        ascending=False,
    )

    by_window = pd.DataFrame(window_rows)

    summary_file = RESULTS_DIR / "s2_failure_filter_summary.csv"

    window_file = RESULTS_DIR / "s2_failure_filter_by_window.csv"

    enriched_file = RESULTS_DIR / "s2_benchmark_trades_enriched.csv"

    summary.to_csv(
        summary_file,
        index=False,
    )

    by_window.to_csv(
        window_file,
        index=False,
    )

    trades.to_csv(
        enriched_file,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 110)
    print("FAILURE FILTER RESULTS")
    print("=" * 110)

    print(
        summary[
            [
                "filter",
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "positive_window_pct",
                "worst_window_R",
                "best_window_R",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(summary_file)
    print(window_file)
    print(enriched_file)

    print()
    print("S2 FAILURE FILTER TEST COMPLETE")


if __name__ == "__main__":
    main()
