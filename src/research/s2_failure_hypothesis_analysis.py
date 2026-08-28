from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# S2 FAILURE HYPOTHESIS ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Investigate whether S2 failures have identifiable
# characteristics BEFORE the trade is entered.
#
# IMPORTANT
# ---------
# This script is DIAGNOSTIC.
#
# It does NOT modify S2.
# It does NOT optimize parameters.
# It does NOT select a new strategy.
#
# The goal is to generate hypotheses for later testing.
#
# Benchmark remains frozen:
#
#   HMM state       = 2
#   lower tail      = 17.5%
#   quality        >= 0.75
#   volatility      = 40-60%
#   stop            = 25 points
#   RR              = 1.75
#   horizon         = 20 bars
#
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "s2_extended"

TRADES_FILE = RESULTS_DIR / "s2_benchmark_trades.csv"

OUTPUT_FILE = RESULTS_DIR / "s2_failure_hypothesis_analysis.csv"


# ============================================================
# HELPERS
# ============================================================


def load_trades():

    print("Loading frozen benchmark trades...")

    trades = pd.read_csv(TRADES_FILE)

    print(f"Trades loaded: {len(trades)}")

    return trades


def classify_outcome(
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

    return trades


# ============================================================
# FEATURE CANDIDATES
# ============================================================


FEATURES = [
    "quality",
    "entry_vol_percentile",
    "entry_minutes_from_rth_open",
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
]


# ============================================================
# UNIVARIATE WIN/LOSS COMPARISON
# ============================================================


def analyze_feature(
    trades,
    feature,
):

    if feature not in trades.columns:
        return None

    data = trades[
        [
            "outcome",
            feature,
        ]
    ].copy()

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    data = data.dropna(subset=[feature])

    wins = data.loc[
        data["outcome"] == "WIN",
        feature,
    ]

    losses = data.loc[
        data["outcome"] == "LOSS",
        feature,
    ]

    if len(wins) == 0 or len(losses) == 0:
        return None

    return {
        "feature": feature,
        "win_n": len(wins),
        "loss_n": len(losses),
        "win_mean": wins.mean(),
        "loss_mean": losses.mean(),
        "win_median": wins.median(),
        "loss_median": losses.median(),
        "win_q25": wins.quantile(0.25),
        "win_q75": wins.quantile(0.75),
        "loss_q25": losses.quantile(0.25),
        "loss_q75": losses.quantile(0.75),
        "mean_difference": wins.mean() - losses.mean(),
        "median_difference": wins.median() - losses.median(),
    }


def run_univariate_analysis(
    trades,
):

    records = []

    for feature in FEATURES:
        result = analyze_feature(
            trades,
            feature,
        )

        if result is not None:
            records.append(result)

    return pd.DataFrame(records)


# ============================================================
# QUANTILE ANALYSIS
# ============================================================
#
# Instead of asking only:
#
#   "Are winners different from losers?"
#
# ask:
#
#   "Does performance change materially across
#    the distribution of the variable?"
#
# ============================================================


def quantile_analysis(
    trades,
    feature,
    n_bins=5,
):

    if feature not in trades.columns:
        return pd.DataFrame()

    data = trades[
        [
            feature,
            "net_R",
            "outcome",
        ]
    ].copy()

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    data = data.dropna(subset=[feature])

    if len(data) < n_bins * 5:
        return pd.DataFrame()

    try:
        data["bucket"] = pd.qcut(
            data[feature],
            q=n_bins,
            duplicates="drop",
        )

    except ValueError:
        return pd.DataFrame()

    result = (
        data.groupby(
            "bucket",
            observed=False,
        )
        .agg(
            trades=("net_R", "size"),
            wins=("outcome", lambda x: (x == "WIN").sum()),
            losses=("outcome", lambda x: (x == "LOSS").sum()),
            mean_R=("net_R", "mean"),
            median_R=("net_R", "median"),
            total_R=("net_R", "sum"),
            win_rate=(
                "outcome",
                lambda x: (x == "WIN").mean(),
            ),
        )
        .reset_index()
    )

    result.insert(
        0,
        "feature",
        feature,
    )

    return result


def run_quantile_analysis(
    trades,
):

    frames = []

    for feature in FEATURES:
        result = quantile_analysis(
            trades,
            feature,
        )

        if not result.empty:
            frames.append(result)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# VOLATILITY × TIME
# ============================================================


def volatility_time_analysis(
    trades,
):

    required = [
        "entry_vol_bucket",
        "entry_time_bucket",
        "net_R",
        "outcome",
    ]

    if not all(column in trades.columns for column in required):
        return pd.DataFrame()

    result = (
        trades.groupby(
            [
                "entry_vol_bucket",
                "entry_time_bucket",
            ],
            observed=False,
        )
        .agg(
            trades=("net_R", "size"),
            wins=(
                "outcome",
                lambda x: (x == "WIN").sum(),
            ),
            losses=(
                "outcome",
                lambda x: (x == "LOSS").sum(),
            ),
            mean_R=("net_R", "mean"),
            median_R=("net_R", "median"),
            total_R=("net_R", "sum"),
            win_rate=(
                "outcome",
                lambda x: (x == "WIN").mean(),
            ),
        )
        .reset_index()
    )

    return result.sort_values(
        "mean_R",
        ascending=False,
    )


# ============================================================
# QUALITY × VOLATILITY
# ============================================================


def quality_volatility_analysis(
    trades,
):

    required = [
        "quality",
        "entry_vol_bucket",
        "net_R",
        "outcome",
    ]

    if not all(column in trades.columns for column in required):
        return pd.DataFrame()

    data = trades.copy()

    data["quality_bucket"] = pd.cut(
        data["quality"],
        bins=[
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.01,
        ],
        right=False,
    )

    result = (
        data.groupby(
            [
                "entry_vol_bucket",
                "quality_bucket",
            ],
            observed=False,
        )
        .agg(
            trades=("net_R", "size"),
            wins=(
                "outcome",
                lambda x: (x == "WIN").sum(),
            ),
            losses=(
                "outcome",
                lambda x: (x == "LOSS").sum(),
            ),
            mean_R=("net_R", "mean"),
            total_R=("net_R", "sum"),
            win_rate=(
                "outcome",
                lambda x: (x == "WIN").mean(),
            ),
        )
        .reset_index()
    )

    return result.sort_values(
        "mean_R",
        ascending=False,
    )


# ============================================================
# MOMENTUM × VOLATILITY
# ============================================================


def momentum_volatility_analysis(
    trades,
):

    required = [
        "entry_past_return_30",
        "entry_vol_bucket",
        "net_R",
        "outcome",
    ]

    if not all(column in trades.columns for column in required):
        return pd.DataFrame()

    data = trades.copy()

    data["momentum_bucket"] = pd.qcut(
        data["entry_past_return_30"],
        q=5,
        duplicates="drop",
    )

    result = (
        data.groupby(
            [
                "entry_vol_bucket",
                "momentum_bucket",
            ],
            observed=False,
        )
        .agg(
            trades=("net_R", "size"),
            wins=(
                "outcome",
                lambda x: (x == "WIN").sum(),
            ),
            losses=(
                "outcome",
                lambda x: (x == "LOSS").sum(),
            ),
            mean_R=("net_R", "mean"),
            total_R=("net_R", "sum"),
            win_rate=(
                "outcome",
                lambda x: (x == "WIN").mean(),
            ),
        )
        .reset_index()
    )

    return result.sort_values(
        "mean_R",
        ascending=False,
    )


# ============================================================
# STOP FAILURE ANALYSIS
# ============================================================


def stop_failure_analysis(
    trades,
):

    stops = trades[trades["exit_reason"] == "stop"].copy()

    if stops.empty:
        return pd.DataFrame()

    records = []

    for feature in FEATURES:
        if feature not in stops.columns:
            continue

        values = (
            stops[feature]
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

        records.append(
            {
                "feature": feature,
                "stop_n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S2 FAILURE HYPOTHESIS ANALYSIS")
    print("=" * 110)

    trades = load_trades()

    trades = classify_outcome(trades)

    print()
    print("Frozen benchmark:")
    print("  HMM state       = 2")
    print("  Lower tail      = 17.5%")
    print("  Quality         >= 0.75")
    print("  Volatility      = 40-60%")
    print("  Stop            = 25 points")
    print("  RR              = 1.75")
    print("  Horizon         = 20 bars")

    # --------------------------------------------------------
    # 1. UNIVARIATE
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("1. WIN vs LOSS FEATURE COMPARISON")
    print("=" * 110)

    feature_results = run_univariate_analysis(trades)

    if not feature_results.empty:
        feature_results["abs_mean_difference"] = feature_results[
            "mean_difference"
        ].abs()

        feature_results = feature_results.sort_values(
            "abs_mean_difference",
            ascending=False,
        )

        print(feature_results.to_string(index=False))

    # --------------------------------------------------------
    # 2. QUANTILES
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("2. FEATURE QUANTILE ANALYSIS")
    print("=" * 110)

    quantile_results = run_quantile_analysis(trades)

    if not quantile_results.empty:
        print(quantile_results.to_string(index=False))

    # --------------------------------------------------------
    # 3. VOL × TIME
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("3. VOLATILITY × TIME")
    print("=" * 110)

    vol_time = volatility_time_analysis(trades)

    if not vol_time.empty:
        print(vol_time.to_string(index=False))

    # --------------------------------------------------------
    # 4. QUALITY × VOL
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("4. QUALITY × VOLATILITY")
    print("=" * 110)

    quality_vol = quality_volatility_analysis(trades)

    if not quality_vol.empty:
        print(quality_vol.to_string(index=False))

    # --------------------------------------------------------
    # 5. MOMENTUM × VOL
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("5. MOMENTUM × VOLATILITY")
    print("=" * 110)

    momentum_vol = momentum_volatility_analysis(trades)

    if not momentum_vol.empty:
        print(momentum_vol.to_string(index=False))

    # --------------------------------------------------------
    # 6. STOP FAILURES
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("6. STOP FAILURE FEATURE PROFILE")
    print("=" * 110)

    stop_results = stop_failure_analysis(trades)

    if not stop_results.empty:
        print(stop_results.to_string(index=False))

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_results.to_csv(
        RESULTS_DIR / "s2_failure_hypothesis_features.csv",
        index=False,
    )

    quantile_results.to_csv(
        RESULTS_DIR / "s2_failure_hypothesis_quantiles.csv",
        index=False,
    )

    vol_time.to_csv(
        RESULTS_DIR / "s2_failure_hypothesis_vol_time.csv",
        index=False,
    )

    quality_vol.to_csv(
        RESULTS_DIR / "s2_failure_hypothesis_quality_vol.csv",
        index=False,
    )

    momentum_vol.to_csv(
        RESULTS_DIR / "s2_failure_hypothesis_momentum_vol.csv",
        index=False,
    )

    stop_results.to_csv(
        RESULTS_DIR / "s2_failure_hypothesis_stop_profile.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(RESULTS_DIR / "s2_failure_hypothesis_features.csv")

    print(RESULTS_DIR / "s2_failure_hypothesis_quantiles.csv")

    print(RESULTS_DIR / "s2_failure_hypothesis_vol_time.csv")

    print(RESULTS_DIR / "s2_failure_hypothesis_quality_vol.csv")

    print(RESULTS_DIR / "s2_failure_hypothesis_momentum_vol.csv")

    print(RESULTS_DIR / "s2_failure_hypothesis_stop_profile.csv")

    print()
    print("S2 FAILURE HYPOTHESIS ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
