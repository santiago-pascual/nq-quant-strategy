from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data


# ============================================================
# S2 EARLY FAILURE TEST
# ============================================================
#
# OBJECTIVE
# ------------------------------------------------------------
# Determine whether the first N bars after entry contain
# enough information to distinguish future winners from
# future failures.
#
# IMPORTANT
# ------------------------------------------------------------
# This is a DISCOVERY experiment.
#
# We do NOT modify the benchmark yet.
#
# We first identify whether early behavior contains signal.
#
# FROZEN BENCHMARK
# ------------------------------------------------------------
# HMM state       = 2
# Lower tail      = 17.5%
# Quality         >= 0.75
# Volatility      = 40-60%
# Stop            = 25 points
# RR              = 1.75
# Horizon         = 20 bars
#
# ============================================================


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BENCHMARK_FILE = RESULTS_DIR / "s2_benchmark_trades_enriched.csv"


STOP_POINTS = 25.0
HORIZON = 20


EARLY_HORIZONS = [
    1,
    2,
    3,
    5,
    8,
]


# ============================================================
# HELPERS
# ============================================================


def safe_numeric(series):
    """
    Convert a Series to numeric safely.

    Boolean values are converted to 0/1.
    Invalid and infinite values become NaN.
    """

    result = series.copy()

    if pd.api.types.is_bool_dtype(result):
        result = result.astype(float)

    else:
        result = pd.to_numeric(
            result,
            errors="coerce",
        )

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return result


# ============================================================
# LOAD BENCHMARK
# ============================================================


def load_benchmark():

    trades = pd.read_csv(BENCHMARK_FILE)

    trades["entry_timestamp"] = pd.to_datetime(
        trades["entry_timestamp"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert("America/New_York")

    trades["exit_timestamp"] = pd.to_datetime(
        trades["exit_timestamp"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert("America/New_York")

    return trades


# ============================================================
# LOAD MARKET
# ============================================================


def load_market():

    df = load_data()

    timestamps = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert("America/New_York")

    df = df.copy()

    df["_timestamp"] = timestamps

    df = df.sort_values("_timestamp")

    df = df.set_index("_timestamp")

    return df


# ============================================================
# ATTACH EARLY PATH
# ============================================================


def calculate_early_features(
    trades,
    market,
):

    records = []

    close = market["close"].to_numpy(dtype=float)

    high = market["high"].to_numpy(dtype=float)

    low = market["low"].to_numpy(dtype=float)

    timestamps = market.index

    timestamp_to_position = {timestamp: i for i, timestamp in enumerate(timestamps)}

    for _, trade in trades.iterrows():
        entry_timestamp = trade["entry_timestamp"]

        position = timestamp_to_position.get(entry_timestamp)

        if position is None:
            records.append({})

            continue

        # ----------------------------------------------------
        # ENTRY PRICE
        # ----------------------------------------------------

        if "entry_price" in trade.index:
            entry_price = float(trade["entry_price"])

        elif "close" in trade.index:
            entry_price = float(trade["close"])

        else:
            entry_price = float(market.iloc[position]["close"])

        row = {
            "entry_timestamp": entry_timestamp,
        }

        # ----------------------------------------------------
        # EARLY HORIZONS
        # ----------------------------------------------------

        for horizon in EARLY_HORIZONS:
            end_position = min(
                position + horizon,
                len(market) - 1,
            )

            path_close = close[position + 1 : end_position + 1]

            path_high = high[position + 1 : end_position + 1]

            path_low = low[position + 1 : end_position + 1]

            if len(path_close) == 0:
                continue

            # ------------------------------------------------
            # SHORT TRADE
            # ------------------------------------------------

            favorable_move = entry_price - np.min(path_low)

            adverse_move = np.max(path_high) - entry_price

            close_move = entry_price - path_close[-1]

            max_favorable_R = favorable_move / STOP_POINTS

            max_adverse_R = adverse_move / STOP_POINTS

            close_move_R = close_move / STOP_POINTS

            range_R = (np.max(path_high) - np.min(path_low)) / STOP_POINTS

            row[f"early_{horizon}_MFE_R"] = max_favorable_R

            row[f"early_{horizon}_MAE_R"] = max_adverse_R

            row[f"early_{horizon}_close_R"] = close_move_R

            row[f"early_{horizon}_range_R"] = range_R

            row[f"early_{horizon}_favorable"] = max_favorable_R > max_adverse_R

        records.append(row)

    early = pd.DataFrame(records)

    early = early.reset_index(drop=True)

    trades = trades.reset_index(drop=True)

    # --------------------------------------------------------
    # Make sure row counts match.
    # --------------------------------------------------------

    if len(early) != len(trades):
        raise ValueError("Early-feature rows do not match benchmark trade rows.")

    early = early.drop(
        columns=["entry_timestamp"],
        errors="ignore",
    )

    return pd.concat(
        [
            trades,
            early,
        ],
        axis=1,
    )


# ============================================================
# OUTCOME LABEL
# ============================================================


def classify_outcome(
    trades,
):

    trades = trades.copy()

    trades["outcome"] = np.where(
        trades["net_R"] > 0,
        "WIN",
        "LOSS",
    )

    return trades


# ============================================================
# QUANTILE ANALYSIS
# ============================================================


def analyze_feature(
    trades,
    feature,
):

    if feature not in trades.columns:
        return pd.DataFrame()

    data = trades[
        [
            feature,
            "net_R",
        ]
    ].copy()

    data[feature] = safe_numeric(data[feature])

    data["net_R"] = safe_numeric(data["net_R"])

    data = data.dropna()

    if len(data) < 20:
        return pd.DataFrame()

    try:
        data["bucket"] = pd.qcut(
            data[feature],
            q=5,
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
            trades=(
                "net_R",
                "size",
            ),
            mean_R=(
                "net_R",
                "mean",
            ),
            median_R=(
                "net_R",
                "median",
            ),
            total_R=(
                "net_R",
                "sum",
            ),
            win_rate=(
                "net_R",
                lambda x: (x > 0).mean(),
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


# ============================================================
# THRESHOLD TEST
# ============================================================


def threshold_test(
    trades,
    feature,
):

    if feature not in trades.columns:
        return pd.DataFrame()

    data = trades[
        [
            feature,
            "net_R",
        ]
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Convert booleans BEFORE all numeric operations.
    # --------------------------------------------------------

    data[feature] = safe_numeric(data[feature])

    data["net_R"] = safe_numeric(data["net_R"])

    data = data.dropna()

    if len(data) < 20:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Detect boolean feature from ORIGINAL dataframe.
    # --------------------------------------------------------

    is_boolean = pd.api.types.is_bool_dtype(trades[feature])

    # --------------------------------------------------------
    # BOOLEAN FEATURES
    # --------------------------------------------------------

    if is_boolean:
        lower = data[data[feature] <= 0]

        higher = data[data[feature] > 0]

        records = []

        for label, subset in [
            ("FALSE", lower),
            ("TRUE", higher),
        ]:
            if len(subset) == 0:
                continue

            records.append(
                {
                    "feature": feature,
                    "threshold": 0.5,
                    "side": label,
                    "trades": len(subset),
                    "win_rate": (subset["net_R"] > 0).mean(),
                    "mean_R": subset["net_R"].mean(),
                    "total_R": subset["net_R"].sum(),
                }
            )

        return pd.DataFrame(records)

    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    thresholds = np.quantile(
        data[feature].to_numpy(dtype=float),
        np.arange(
            0.10,
            1.00,
            0.05,
        ),
    )

    records = []

    for threshold in thresholds:
        lower = data[data[feature] <= threshold]

        higher = data[data[feature] > threshold]

        for label, subset in [
            (
                "LOWER_OR_EQUAL",
                lower,
            ),
            (
                "ABOVE",
                higher,
            ),
        ]:
            if len(subset) == 0:
                continue

            records.append(
                {
                    "feature": feature,
                    "threshold": float(threshold),
                    "side": label,
                    "trades": len(subset),
                    "win_rate": (subset["net_R"] > 0).mean(),
                    "mean_R": subset["net_R"].mean(),
                    "total_R": subset["net_R"].sum(),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# WIN / LOSS COMPARISON
# ============================================================


def compare_win_loss(
    trades,
    feature_columns,
):

    comparison_records = []

    for feature in feature_columns:
        if feature not in trades.columns:
            continue

        for outcome in [
            "WIN",
            "LOSS",
        ]:
            subset = trades.loc[
                trades["outcome"] == outcome,
                feature,
            ]

            # ------------------------------------------------
            # Convert booleans to numeric.
            # ------------------------------------------------

            subset_numeric = safe_numeric(subset).dropna()

            if len(subset_numeric) == 0:
                comparison_records.append(
                    {
                        "feature": feature,
                        "outcome": outcome,
                        "trades": 0,
                        "mean": np.nan,
                        "median": np.nan,
                        "q25": np.nan,
                        "q50": np.nan,
                        "q75": np.nan,
                    }
                )

                continue

            comparison_records.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "trades": len(subset_numeric),
                    "mean": subset_numeric.mean(),
                    "median": subset_numeric.median(),
                    "q25": subset_numeric.quantile(0.25),
                    "q50": subset_numeric.quantile(0.50),
                    "q75": subset_numeric.quantile(0.75),
                }
            )

    return pd.DataFrame(comparison_records)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 EARLY FAILURE TEST")

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
    # LOAD
    # ========================================================

    print("Loading benchmark...")

    trades = load_benchmark()

    print(f"Benchmark trades: {len(trades)}")

    print()

    print("Loading expanded market data...")

    market = load_market()

    print(f"Market observations: {len(market)}")

    print()

    # ========================================================
    # ENTRY PRICE
    # ========================================================

    if "entry_price" not in trades.columns:
        entry_prices = market["close"].reindex(trades["entry_timestamp"])

        trades["entry_price"] = entry_prices.to_numpy()

    # ========================================================
    # EARLY FEATURES
    # ========================================================

    print("Calculating early-path features...")

    trades = calculate_early_features(
        trades,
        market,
    )

    trades = classify_outcome(trades)

    # ========================================================
    # DISCOVERY FEATURES
    # ========================================================

    feature_columns = []

    for horizon in EARLY_HORIZONS:
        feature_columns.extend(
            [
                f"early_{horizon}_MFE_R",
                f"early_{horizon}_MAE_R",
                f"early_{horizon}_close_R",
                f"early_{horizon}_range_R",
                f"early_{horizon}_favorable",
            ]
        )

    # ========================================================
    # QUANTILES
    # ========================================================

    print()

    print("=" * 110)

    print("1. EARLY FEATURE QUANTILE ANALYSIS")

    print("=" * 110)

    quantile_results = []

    for feature in feature_columns:
        if feature not in trades.columns:
            continue

        result = analyze_feature(
            trades,
            feature,
        )

        if not result.empty:
            quantile_results.append(result)

    if quantile_results:
        quantiles = pd.concat(
            quantile_results,
            ignore_index=True,
        )

        print(quantiles.to_string(index=False))

    else:
        quantiles = pd.DataFrame()

    # ========================================================
    # THRESHOLDS
    # ========================================================

    print()

    print("=" * 110)

    print("2. EARLY FAILURE THRESHOLD ANALYSIS")

    print("=" * 110)

    threshold_results = []

    for feature in feature_columns:
        if feature not in trades.columns:
            continue

        result = threshold_test(
            trades,
            feature,
        )

        if not result.empty:
            threshold_results.append(result)

    if threshold_results:
        thresholds = pd.concat(
            threshold_results,
            ignore_index=True,
        )

        print(thresholds.to_string(index=False))

    else:
        thresholds = pd.DataFrame()

    # ========================================================
    # WIN / LOSS
    # ========================================================

    print()

    print("=" * 110)

    print("3. WIN vs LOSS EARLY BEHAVIOR")

    print("=" * 110)

    comparison = compare_win_loss(
        trades,
        feature_columns,
    )

    if not comparison.empty:
        print(comparison.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        RESULTS_DIR / "s2_early_failure_trades.csv",
        index=False,
    )

    quantiles.to_csv(
        RESULTS_DIR / "s2_early_failure_quantiles.csv",
        index=False,
    )

    thresholds.to_csv(
        RESULTS_DIR / "s2_early_failure_thresholds.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR / "s2_early_failure_comparison.csv",
        index=False,
    )

    # ========================================================
    # FILES
    # ========================================================

    print()

    print("=" * 110)

    print("FILES SAVED")

    print("=" * 110)

    print(RESULTS_DIR / "s2_early_failure_trades.csv")

    print(RESULTS_DIR / "s2_early_failure_quantiles.csv")

    print(RESULTS_DIR / "s2_early_failure_thresholds.csv")

    print(RESULTS_DIR / "s2_early_failure_comparison.csv")

    print()

    print("S2 EARLY FAILURE TEST COMPLETE")


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()
