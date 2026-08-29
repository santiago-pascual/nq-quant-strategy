from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.databento_loader import load_databento_mnq


# ============================================================
# S3 — EARLY FAILURE MECHANISM TEST
# ============================================================
#
# OBJECTIVE
# ------------------------------------------------------------
# Determine whether S2 losing trades can be identified early
# enough to justify an additional exit / failure filter.
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
# IMPORTANT
# ------------------------------------------------------------
# This script does NOT optimize the original S2 signal.
#
# It only studies whether the path immediately after entry
# contains information capable of identifying future failures.
#
# MARKET DATA
# ------------------------------------------------------------
# Uses the Databento-expanded MNQ dataset through:
#
#     load_databento_mnq()
#
# Therefore it must use the complete:
#
#     2,577,661 observations
#
# and NOT Dataset_NQ_1min_2022_2025.csv.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BENCHMARK_PATH = (
    RESULTS_DIR
    / "s2_benchmark_trades.csv"
)


# ============================================================
# FROZEN PARAMETERS
# ============================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

MNQ_POINT_VALUE = 2.00
TOPSTEP_MNQ_RT_FEE_USD = 1.22

SLIPPAGE_POINTS_PER_SIDE = 0.25

TOTAL_SLIPPAGE_POINTS = (
    2.0 * SLIPPAGE_POINTS_PER_SIDE
)

TOPSTEP_FEE_POINTS = (
    TOPSTEP_MNQ_RT_FEE_USD
    / MNQ_POINT_VALUE
)

TOTAL_COST_POINTS = (
    TOTAL_SLIPPAGE_POINTS
    + TOPSTEP_FEE_POINTS
)


# ============================================================
# EARLY OBSERVATION BARS
# ============================================================

DECISION_BARS = [
    1,
    2,
    3,
    5,
    8,
    10,
    15,
]


# ============================================================
# THRESHOLDS
# ============================================================

MAE_THRESHOLDS = [
    0.25,
    0.50,
    0.75,
    1.00,
]

PROGRESS_THRESHOLDS = [
    0.00,
    -0.25,
    -0.50,
    -0.75,
]


# ============================================================
# HELPERS
# ============================================================


def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return np.nan


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(trades):

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
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
        }

    pnl = (
        pd.to_numeric(
            trades["net_R"],
            errors="coerce",
        )
        .dropna()
    )

    if pnl.empty:

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
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
        }

    wins = pnl[pnl > 0]

    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    equity = pnl.cumsum()

    drawdown = (
        equity
        - equity.cummax()
    )

    result = {
        "trades": int(len(pnl)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "mean_R": float(pnl.mean()),
        "median_R": float(pnl.median()),
        "total_R": float(pnl.sum()),
        "profit_factor": float(profit_factor),
        "max_drawdown_R": float(drawdown.min()),
    }

    # --------------------------------------------------------
    # Walk-forward window statistics
    # --------------------------------------------------------

    if "window" in trades.columns:

        window_pnl = (
            trades
            .groupby("window")["net_R"]
            .sum()
        )

        if len(window_pnl) > 0:

            result[
                "positive_window_pct"
            ] = float(
                (window_pnl > 0).mean()
            )

            result[
                "worst_window_R"
            ] = float(
                window_pnl.min()
            )

            result[
                "best_window_R"
            ] = float(
                window_pnl.max()
            )

        else:

            result[
                "positive_window_pct"
            ] = np.nan

            result[
                "worst_window_R"
            ] = np.nan

            result[
                "best_window_R"
            ] = np.nan

    else:

        result[
            "positive_window_pct"
        ] = np.nan

        result[
            "worst_window_R"
        ] = np.nan

        result[
            "best_window_R"
        ] = np.nan

    return result


# ============================================================
# LOAD BENCHMARK
# ============================================================


def load_benchmark():

    print(
        "Loading frozen benchmark..."
    )

    if not BENCHMARK_PATH.exists():

        raise FileNotFoundError(
            f"Benchmark file not found:\n"
            f"{BENCHMARK_PATH}"
        )

    trades = pd.read_csv(
        BENCHMARK_PATH
    )

    print(
        f"Benchmark trades: {len(trades)}"
    )

    # --------------------------------------------------------
    # Normalize timestamps.
    #
    # utc=True is intentional because benchmark timestamps
    # contain timezone information and can otherwise produce
    # mixed-timezone errors.
    # --------------------------------------------------------

    trades["entry_timestamp"] = (
        pd.to_datetime(
            trades["entry_timestamp"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    trades["exit_timestamp"] = (
        pd.to_datetime(
            trades["exit_timestamp"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(
            "America/New_York"
        )
    )

    trades["net_R"] = pd.to_numeric(
        trades["net_R"],
        errors="coerce",
    )

    return trades


# ============================================================
# NORMALIZE MARKET
# ============================================================


def normalize_market(market):

    market = market.copy()

    if "timestamp ET" in market.columns:

        timestamps = pd.to_datetime(
            market["timestamp ET"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(
            "America/New_York"
        )

        market["timestamp ET"] = timestamps

        market = market.set_index(
            "timestamp ET"
        )

    else:

        if not isinstance(
            market.index,
            pd.DatetimeIndex,
        ):

            raise ValueError(
                "Market data has no DatetimeIndex "
                "and no 'timestamp ET' column."
            )

        index = pd.to_datetime(
            market.index,
            errors="coerce",
            utc=True,
        ).tz_convert(
            "America/New_York"
        )

        market.index = index

    market.index.name = (
        "timestamp ET"
    )

    market = market.sort_index()

    return market


# ============================================================
# LOAD COMPLETE DATABENTO MARKET
# ============================================================


def load_market():

    print(
        "Loading COMPLETE Databento MNQ dataset..."
    )

    market = load_databento_mnq()

    market = normalize_market(
        market
    )

    print(
        f"Expanded observations: "
        f"{len(market)}"
    )

    print(
        f"First timestamp: "
        f"{market.index.min()}"
    )

    print(
        f"Last timestamp: "
        f"{market.index.max()}"
    )

    # --------------------------------------------------------
    # Safety check.
    #
    # We explicitly want the expanded dataset.
    # --------------------------------------------------------

    if len(market) < 2_000_000:

        raise RuntimeError(
            "The loaded market dataset contains "
            f"only {len(market):,} rows. "
            "Expected the expanded Databento dataset "
            "with approximately 2.58 million observations."
        )

    return market


# ============================================================
# MARKET POSITION LOOKUP
# ============================================================


def build_timestamp_lookup(market):

    return {
        timestamp: position
        for position, timestamp
        in enumerate(market.index)
    }


# ============================================================
# ATTACH EARLY PATH FEATURES
# ============================================================


def attach_early_features(
    trades,
    market,
):

    print(
        "Attaching early intratrade features..."
    )

    lookup = build_timestamp_lookup(
        market
    )

    opens = market[
        "open"
    ].to_numpy(
        dtype=float
    )

    highs = market[
        "high"
    ].to_numpy(
        dtype=float
    )

    lows = market[
        "low"
    ].to_numpy(
        dtype=float
    )

    closes = market[
        "close"
    ].to_numpy(
        dtype=float
    )

    feature_records = []

    missing = 0

    for _, trade in trades.iterrows():

        entry_timestamp = (
            trade["entry_timestamp"]
        )

        if entry_timestamp not in lookup:

            missing += 1

            feature_records.append({})

            continue

        entry_position = lookup[
            entry_timestamp
        ]

        entry_price = closes[
            entry_position
        ]

        features = {}

        # ====================================================
        # Examine path after entry.
        # ====================================================

        for bar in DECISION_BARS:

            start = (
                entry_position + 1
            )

            end = min(
                entry_position + bar,
                len(market) - 1,
            )

            if end < start:

                continue

            path_highs = highs[
                start:end + 1
            ]

            path_lows = lows[
                start:end + 1
            ]

            path_closes = closes[
                start:end + 1
            ]

            # ------------------------------------------------
            # SHORT POSITION
            #
            # Adverse:
            # price rises above entry.
            #
            # Favorable:
            # price falls below entry.
            # ------------------------------------------------

            max_adverse_points = (
                np.max(path_highs)
                - entry_price
            )

            max_favorable_points = (
                entry_price
                - np.min(path_lows)
            )

            final_close_points = (
                entry_price
                - path_closes[-1]
            )

            features[
                f"early_{bar}_MAE_R"
            ] = (
                max_adverse_points
                / STOP_POINTS
            )

            features[
                f"early_{bar}_MFE_R"
            ] = (
                max_favorable_points
                / STOP_POINTS
            )

            features[
                f"early_{bar}_close_R"
            ] = (
                final_close_points
                / STOP_POINTS
            )

            features[
                f"early_{bar}_adverse_points"
            ] = max_adverse_points

            features[
                f"early_{bar}_favorable_points"
            ] = max_favorable_points

        feature_records.append(
            features
        )

    feature_df = pd.DataFrame(
        feature_records,
        index=trades.index,
    )

    result = pd.concat(
        [
            trades.reset_index(
                drop=True
            ),
            feature_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    print(
        f"Trades without entry features: "
        f"{missing}"
    )

    if missing > 0:

        raise RuntimeError(
            f"{missing} benchmark trades could not "
            "be matched to the complete market dataset."
        )

    return result


# ============================================================
# EARLY EXIT
# ============================================================


def calculate_early_exit_R(
    trade,
    market,
    decision_bar,
):

    entry_timestamp = (
        trade["entry_timestamp"]
    )

    try:

        position = (
            market.index.get_loc(
                entry_timestamp
            )
        )

    except KeyError:

        return np.nan

    exit_position = min(
        position + decision_bar,
        len(market) - 1,
    )

    if exit_position <= position:

        return np.nan

    entry_price = float(
        market.iloc[
            position
        ]["close"]
    )

    exit_price = float(
        market.iloc[
            exit_position
        ]["close"]
    )

    # --------------------------------------------------------
    # SHORT P&L
    # --------------------------------------------------------

    raw_points = (
        entry_price
        - exit_price
    )

    net_points = (
        raw_points
        - TOTAL_COST_POINTS
    )

    return (
        net_points
        / STOP_POINTS
    )


# ============================================================
# APPLY RULE
# ============================================================


def apply_early_rule(
    trades,
    market,
    rule_type,
    decision_bar,
    threshold_a,
    threshold_b=None,
):

    result = trades.copy()

    new_R = []

    triggered = []

    reasons = []

    mae_column = (
        f"early_{decision_bar}_MAE_R"
    )

    close_column = (
        f"early_{decision_bar}_close_R"
    )

    for _, trade in result.iterrows():

        mae = safe_float(
            trade.get(
                mae_column,
                np.nan,
            )
        )

        close_R = safe_float(
            trade.get(
                close_column,
                np.nan,
            )
        )

        should_exit = False

        reason = "BENCHMARK"

        # ====================================================
        # MAE RULE
        # ====================================================

        if rule_type == "MAE":

            if (
                pd.notna(mae)
                and mae >= threshold_a
            ):

                should_exit = True

                reason = (
                    f"MAE>={threshold_a:.2f}R"
                )

        # ====================================================
        # PROGRESS RULE
        # ====================================================

        elif rule_type == "PROGRESS":

            if (
                pd.notna(close_R)
                and close_R <= threshold_a
            ):

                should_exit = True

                reason = (
                    f"close_R<={threshold_a:.2f}R"
                )

        # ====================================================
        # COMBINED RULE
        # ====================================================

        elif rule_type == "COMBINED":

            if (
                pd.notna(mae)
                and pd.notna(close_R)
                and mae >= threshold_a
                and close_R <= threshold_b
            ):

                should_exit = True

                reason = (
                    f"MAE>={threshold_a:.2f}R"
                    f"_AND_"
                    f"close_R<={threshold_b:.2f}R"
                )

        # ====================================================
        # EXECUTE EARLY EXIT
        # ====================================================

        if should_exit:

            early_R = (
                calculate_early_exit_R(
                    trade,
                    market,
                    decision_bar,
                )
            )

            if pd.notna(early_R):

                new_R.append(
                    early_R
                )

                triggered.append(
                    True
                )

                reasons.append(
                    reason
                )

                continue

        # ====================================================
        # KEEP BENCHMARK OUTCOME
        # ====================================================

        new_R.append(
            float(
                trade["net_R"]
            )
        )

        triggered.append(
            False
        )

        reasons.append(
            "BENCHMARK"
        )

    result[
        "original_net_R"
    ] = result["net_R"]

    result[
        "net_R"
    ] = new_R

    result[
        "early_exit_triggered"
    ] = triggered

    result[
        "early_exit_reason"
    ] = reasons

    return result


# ============================================================
# RUN TEST
# ============================================================


def run_test(
    trades,
    market,
    rule_type,
    decision_bar,
    threshold_a,
    threshold_b=None,
):

    tested = apply_early_rule(
        trades=trades,
        market=market,
        rule_type=rule_type,
        decision_bar=decision_bar,
        threshold_a=threshold_a,
        threshold_b=threshold_b,
    )

    metrics = calculate_metrics(
        tested
    )

    metrics[
        "rule_type"
    ] = rule_type

    metrics[
        "decision_bar"
    ] = decision_bar

    metrics[
        "threshold_a"
    ] = threshold_a

    metrics[
        "threshold_b"
    ] = (
        threshold_b
        if threshold_b is not None
        else np.nan
    )

    metrics[
        "triggered_trades"
    ] = int(
        tested[
            "early_exit_triggered"
        ].sum()
    )

    return (
        metrics,
        tested,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "=" * 110
    )

    print(
        "S3 EARLY FAILURE MECHANISM TEST — COMPLETE DATASET"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Frozen benchmark:"
    )

    print(
        f"  Stop            = {STOP_POINTS} points"
    )

    print(
        f"  RR              = {RR}"
    )

    print(
        f"  Horizon         = {HORIZON} bars"
    )

    print()

    # ========================================================
    # LOAD
    # ========================================================

    trades = load_benchmark()

    market = load_market()

    print()

    # ========================================================
    # ATTACH FEATURES
    # ========================================================

    trades = attach_early_features(
        trades,
        market,
    )

    print()

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = calculate_metrics(
        trades
    )

    baseline[
        "rule_type"
    ] = "BASELINE"

    baseline[
        "decision_bar"
    ] = np.nan

    baseline[
        "threshold_a"
    ] = np.nan

    baseline[
        "threshold_b"
    ] = np.nan

    baseline[
        "triggered_trades"
    ] = 0

    results = [
        baseline
    ]

    tested_trade_sets = {}

    # ========================================================
    # COUNT TESTS
    # ========================================================

    total_tests = (
        len(DECISION_BARS)
        * len(MAE_THRESHOLDS)
    )

    total_tests += (
        len(DECISION_BARS)
        * len(PROGRESS_THRESHOLDS)
    )

    total_tests += (
        len(DECISION_BARS)
        * len(MAE_THRESHOLDS)
        * len(PROGRESS_THRESHOLDS)
    )

    completed = 0

    print(
        f"Testing {total_tests} early-failure rules..."
    )

    # ========================================================
    # MAE RULES
    # ========================================================

    for bar in DECISION_BARS:

        for threshold in MAE_THRESHOLDS:

            completed += 1

            print(
                f"Processing "
                f"{completed}/{total_tests}...",
                end="\r",
            )

            metrics, tested = run_test(
                trades,
                market,
                "MAE",
                bar,
                threshold,
            )

            results.append(
                metrics
            )

            tested_trade_sets[
                (
                    "MAE",
                    bar,
                    threshold,
                    np.nan,
                )
            ] = tested

    # ========================================================
    # PROGRESS RULES
    # ========================================================

    for bar in DECISION_BARS:

        for threshold in (
            PROGRESS_THRESHOLDS
        ):

            completed += 1

            print(
                f"Processing "
                f"{completed}/{total_tests}...",
                end="\r",
            )

            metrics, tested = run_test(
                trades,
                market,
                "PROGRESS",
                bar,
                threshold,
            )

            results.append(
                metrics
            )

            tested_trade_sets[
                (
                    "PROGRESS",
                    bar,
                    threshold,
                    np.nan,
                )
            ] = tested

    # ========================================================
    # COMBINED RULES
    # ========================================================

    for bar in DECISION_BARS:

        for mae_threshold in (
            MAE_THRESHOLDS
        ):

            for progress_threshold in (
                PROGRESS_THRESHOLDS
            ):

                completed += 1

                print(
                    f"Processing "
                    f"{completed}/{total_tests}...",
                    end="\r",
                )

                metrics, tested = run_test(
                    trades,
                    market,
                    "COMBINED",
                    bar,
                    mae_threshold,
                    progress_threshold,
                )

                results.append(
                    metrics
                )

                tested_trade_sets[
                    (
                        "COMBINED",
                        bar,
                        mae_threshold,
                        progress_threshold,
                    )
                ] = tested

    print()
    print()

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        results
    )

    non_baseline = summary[
        summary["rule_type"]
        != "BASELINE"
    ].copy()

    non_baseline = (
        non_baseline
        .sort_values(
            [
                "total_R",
                "profit_factor",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    display_columns = [
        "rule_type",
        "decision_bar",
        "threshold_a",
        "threshold_b",
        "triggered_trades",
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

    # ========================================================
    # PRINT BASELINE
    # ========================================================

    print(
        "=" * 110
    )

    print(
        "BASELINE"
    )

    print(
        "=" * 110
    )

    print(
        pd.DataFrame(
            [baseline]
        )[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    # ========================================================
    # PRINT TOP RULES
    # ========================================================

    print(
        "=" * 110
    )

    print(
        "TOP EARLY FAILURE RULES"
    )

    print(
        "=" * 110
    )

    print(
        non_baseline[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        RESULTS_DIR
        / "s3_early_failure_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    if not non_baseline.empty:

        best = (
            non_baseline.iloc[0]
        )

        key = (
            best["rule_type"],
            int(
                best["decision_bar"]
            ),
            float(
                best["threshold_a"]
            ),
            (
                float(
                    best["threshold_b"]
                )
                if pd.notna(
                    best["threshold_b"]
                )
                else np.nan
            ),
        )

        best_trades = (
            tested_trade_sets[key]
        )

        best_path = (
            RESULTS_DIR
            / "s3_best_early_failure_rule_trades.csv"
        )

        best_trades.to_csv(
            best_path,
            index=False,
        )

        print()

        print(
            "=" * 110
        )

        print(
            "BEST CANDIDATE"
        )

        print(
            "=" * 110
        )

        print(
            f"Rule             : "
            f"{best['rule_type']}"
        )

        print(
            f"Decision bar     : "
            f"{best['decision_bar']}"
        )

        print(
            f"Threshold A      : "
            f"{best['threshold_a']}"
        )

        print(
            f"Threshold B      : "
            f"{best['threshold_b']}"
        )

        print(
            f"Triggered trades : "
            f"{best['triggered_trades']}"
        )

        print(
            f"Win rate         : "
            f"{best['win_rate']:.4f}"
        )

        print(
            f"Mean R           : "
            f"{best['mean_R']:.4f}"
        )

        print(
            f"Total R          : "
            f"{best['total_R']:.4f}"
        )

        print(
            f"Profit Factor    : "
            f"{best['profit_factor']:.4f}"
        )

        print(
            f"Max DD           : "
            f"{best['max_drawdown_R']:.4f}"
        )

    # ========================================================
    # SAVE BY WINDOW
    # ========================================================

    window_rows = []

    for key, tested in (
        tested_trade_sets.items()
    ):

        rule_type = key[0]
        decision_bar = key[1]
        threshold_a = key[2]
        threshold_b = key[3]

        if "window" not in tested.columns:

            continue

        for window, group in (
            tested.groupby("window")
        ):

            metrics = calculate_metrics(
                group
            )

            window_rows.append(
                {
                    "rule_type": rule_type,
                    "decision_bar": decision_bar,
                    "threshold_a": threshold_a,
                    "threshold_b": threshold_b,
                    "window": window,
                    "trades": metrics["trades"],
                    "win_rate": metrics["win_rate"],
                    "mean_R": metrics["mean_R"],
                    "total_R": metrics["total_R"],
                    "profit_factor": metrics[
                        "profit_factor"
                    ],
                    "max_drawdown_R": metrics[
                        "max_drawdown_R"
                    ],
                }
            )

    if window_rows:

        by_window = pd.DataFrame(
            window_rows
        )

        by_window_path = (
            RESULTS_DIR
            / "s3_early_failure_by_window.csv"
        )

        by_window.to_csv(
            by_window_path,
            index=False,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 110
    )

    print(
        summary_path
    )

    if not non_baseline.empty:

        print(
            RESULTS_DIR
            / "s3_best_early_failure_rule_trades.csv"
        )

    if window_rows:

        print(
            RESULTS_DIR
            / "s3_early_failure_by_window.csv"
        )

    print()

    print(
        "S3 EARLY FAILURE MECHANISM TEST COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()