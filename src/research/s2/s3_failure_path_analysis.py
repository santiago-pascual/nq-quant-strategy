from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# S3 FAILURE PATH ANALYSIS
# ============================================================
#
# OBJECTIVE
#
# Understand WHY S2 trades fail.
#
# Frozen benchmark:
#   Stop       = 25 points
#   RR         = 1.75
#   Horizon    = 20 bars
#
# We do NOT optimize the strategy here.
#
# We classify trades according to their intratrade path:
#
#   EARLY_ADVERSE + LOSS
#   EARLY_ADVERSE + WIN
#   EARLY_OK      + LOSS
#   EARLY_OK      + WIN
#
# Then measure:
#
#   - MAE at different bars
#   - MFE at different bars
#   - close excursion
#   - final outcome
#   - time to MFE
#   - time to MAE
#   - exit reason
#   - recovery after early adverse excursion
#
# IMPORTANT:
#
# This is diagnostic research.
# It does NOT modify the benchmark.
#
# ============================================================


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2_extended"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BENCHMARK_FILE = RESULTS_DIR / "s2_benchmark_trades.csv"


# ============================================================
# FROZEN PARAMETERS
# ============================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# ============================================================
# EARLY DECISION GRID
# ============================================================

DECISION_BARS = [
    1,
    2,
    3,
    5,
    8,
    10,
]


# ============================================================
# EARLY MAE THRESHOLDS
# ============================================================

MAE_THRESHOLDS_R = [
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# HELPERS
# ============================================================


def safe_numeric(
    series,
):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def parse_timestamps(
    series,
):
    """
    Parse timestamps into timezone-aware
    America/New_York timestamps.

    Handles mixed timezone CSV values safely.
    """

    values = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return values.dt.tz_convert("America/New_York")


def calculate_trade_path(
    entry_price,
    highs,
    lows,
    closes,
):
    """
    Calculate intratrade path in R.

    Short strategy:

        MAE_R = (high - entry) / STOP

        MFE_R = (entry - low) / STOP

        close_R = (entry - close) / STOP

    Positive values are favorable.
    """

    highs = np.asarray(
        highs,
        dtype=float,
    )

    lows = np.asarray(
        lows,
        dtype=float,
    )

    closes = np.asarray(
        closes,
        dtype=float,
    )

    mae_r = (highs - entry_price) / STOP_POINTS

    mfe_r = (entry_price - lows) / STOP_POINTS

    close_r = (entry_price - closes) / STOP_POINTS

    return (
        mae_r,
        mfe_r,
        close_r,
    )


# ============================================================
# LOAD BENCHMARK
# ============================================================


def load_benchmark():

    print("Loading frozen benchmark...")

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"Benchmark not found:\n{BENCHMARK_FILE}")

    trades = pd.read_csv(BENCHMARK_FILE)

    trades["entry_timestamp"] = parse_timestamps(trades["entry_timestamp"])

    trades["exit_timestamp"] = parse_timestamps(trades["exit_timestamp"])

    trades["net_R"] = safe_numeric(trades["net_R"])

    trades["raw_points"] = safe_numeric(trades["raw_points"])

    trades["net_points"] = safe_numeric(trades["net_points"])

    trades["quality"] = safe_numeric(trades["quality"])

    trades["window"] = pd.to_numeric(
        trades["window"],
        errors="coerce",
    )

    print(f"Benchmark trades: {len(trades)}")

    return trades


# ============================================================
# LOAD MARKET DATA
# ============================================================


def load_market():

    print("Loading complete market data...")

    from src.data_loader import load_data

    market = load_data()

    market = market.copy()

    market["timestamp ET"] = parse_timestamps(market["timestamp ET"])

    market = market.sort_values("timestamp ET")

    market = market.drop_duplicates(
        subset=["timestamp ET"],
        keep="last",
    )

    market = market.set_index("timestamp ET")

    return market


# ============================================================
# ATTACH PATHS
# ============================================================


def attach_paths(
    trades,
    market,
):

    print("Attaching intratrade paths...")

    records = []

    missing = 0

    for idx, trade in trades.iterrows():
        entry_ts = trade["entry_timestamp"]

        session_id = str(trade["session_id"])

        entry_price = None

        if entry_ts in market.index:
            entry_row = market.loc[entry_ts]

            if isinstance(
                entry_row,
                pd.DataFrame,
            ):
                entry_row = entry_row.iloc[-1]

            entry_price = float(entry_row["close"])

        else:
            missing += 1

            continue

        session_market = market.loc[
            market["session_date"].astype(str) == session_id
        ].copy()

        if session_market.empty:
            missing += 1

            continue

        session_market = session_market.sort_index()

        positions = np.flatnonzero(session_market.index == entry_ts)

        if len(positions) == 0:
            missing += 1

            continue

        entry_position = positions[0]

        future = session_market.iloc[
            entry_position : min(
                entry_position + HORIZON + 1,
                len(session_market),
            )
        ]

        if len(future) < 2:
            missing += 1

            continue

        highs = future["high"].to_numpy(dtype=float)

        lows = future["low"].to_numpy(dtype=float)

        closes = future["close"].to_numpy(dtype=float)

        (
            mae_r,
            mfe_r,
            close_r,
        ) = calculate_trade_path(
            entry_price,
            highs,
            lows,
            closes,
        )

        record = {
            "trade_index": idx,
            "entry_price": entry_price,
            "path_length": len(future),
        }

        # ----------------------------------------
        # Path features
        # ----------------------------------------

        for bar in range(
            1,
            len(future),
        ):
            record[f"mae_{bar}R"] = float(np.max(mae_r[1 : bar + 1]))

            record[f"mfe_{bar}R"] = float(np.max(mfe_r[1 : bar + 1]))

            record[f"close_{bar}R"] = float(close_r[bar])

        # ----------------------------------------
        # Final path metrics
        # ----------------------------------------

        record["max_MAE_R"] = float(np.max(mae_r[1:]))

        record["max_MFE_R"] = float(np.max(mfe_r[1:]))

        record["final_close_R"] = float(close_r[-1])

        # ----------------------------------------
        # Time to extremes
        # ----------------------------------------

        if len(mfe_r) > 1:
            mfe_position = np.argmax(mfe_r[1:]) + 1

            mae_position = np.argmax(mae_r[1:]) + 1

            record["time_to_max_MFE"] = int(mfe_position)

            record["time_to_max_MAE"] = int(mae_position)

        else:
            record["time_to_max_MFE"] = np.nan

            record["time_to_max_MAE"] = np.nan

        # ----------------------------------------
        # Outcome
        # ----------------------------------------

        net_r = float(trade["net_R"])

        record["outcome"] = "WIN" if net_r > 0 else "LOSS"

        record["exit_reason"] = trade.get(
            "exit_reason",
            "",
        )

        record["net_R"] = net_r

        record["window"] = trade.get(
            "window",
            np.nan,
        )

        records.append(record)

    paths = pd.DataFrame(records)

    if paths.empty:
        raise RuntimeError("No intratrade paths recovered.")

    print(f"Paths recovered: {len(paths)}")

    print(f"Missing paths: {missing}")

    return paths


# ============================================================
# MERGE
# ============================================================


def merge_trade_information(
    trades,
    paths,
):

    info_columns = [
        "trade_index",
        "entry_price",
        "path_length",
        "max_MAE_R",
        "max_MFE_R",
        "final_close_R",
        "time_to_max_MFE",
        "time_to_max_MAE",
        "outcome",
        "exit_reason",
        "net_R",
        "window",
    ]

    path_columns = [
        c
        for c in paths.columns
        if c.startswith("mae_") or c.startswith("mfe_") or c.startswith("close_")
    ]

    info_columns += path_columns

    merged = trades.reset_index(drop=True).copy()

    merged["trade_index"] = merged.index

    merged = merged.merge(
        paths[info_columns],
        on="trade_index",
        how="inner",
        suffixes=(
            "",
            "_path",
        ),
    )

    return merged


# ============================================================
# FOUR-QUADRANT ANALYSIS
# ============================================================


def quadrant_analysis(
    trades,
):

    print()
    print("=" * 110)
    print("1. EARLY MAE × FINAL OUTCOME")
    print("=" * 110)

    rows = []

    for bar in DECISION_BARS:
        mae_col = f"mae_{bar}R"

        if mae_col not in trades.columns:
            continue

        threshold = 0.75

        adverse = trades[mae_col] >= threshold

        outcome = trades["outcome"]

        for group_name, mask in [
            (
                "EARLY_ADVERSE",
                adverse,
            ),
            (
                "EARLY_OK",
                ~adverse,
            ),
        ]:
            subset = trades.loc[mask]

            if subset.empty:
                continue

            for final_outcome in [
                "WIN",
                "LOSS",
            ]:
                group = subset.loc[outcome.loc[subset.index] == final_outcome]

                if group.empty:
                    continue

                rows.append(
                    {
                        "decision_bar": bar,
                        "early_state": group_name,
                        "outcome": final_outcome,
                        "trades": len(group),
                        "pct_of_all": (len(group) / len(trades)),
                        "mean_R": group["net_R"].mean(),
                        "median_R": group["net_R"].median(),
                        "mean_MFE_R": group["max_MFE_R"].mean(),
                        "mean_MAE_R": group["max_MAE_R"].mean(),
                    }
                )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "s3_failure_path_quadrants.csv",
        index=False,
    )

    return result


# ============================================================
# EARLY MAE THRESHOLD ANALYSIS
# ============================================================


def early_mae_analysis(
    trades,
):

    print()
    print("=" * 110)
    print("2. EARLY MAE THRESHOLD ANALYSIS")
    print("=" * 110)

    rows = []

    for bar in DECISION_BARS:
        col = f"mae_{bar}R"

        if col not in trades.columns:
            continue

        for threshold in MAE_THRESHOLDS_R:
            adverse = trades[col] >= threshold

            group = trades.loc[adverse]

            if group.empty:
                continue

            wins = (group["net_R"] > 0).sum()

            losses = (group["net_R"] < 0).sum()

            rows.append(
                {
                    "decision_bar": bar,
                    "threshold_R": threshold,
                    "triggered_trades": len(group),
                    "trigger_pct": len(group) / len(trades),
                    "win_rate": wins / len(group),
                    "mean_R": group["net_R"].mean(),
                    "total_R": group["net_R"].sum(),
                    "mean_final_close_R": group["final_close_R"].mean(),
                    "mean_max_MFE_R": group["max_MFE_R"].mean(),
                    "mean_max_MAE_R": group["max_MAE_R"].mean(),
                    "mean_time_to_MFE": group["time_to_max_MFE"].mean(),
                    "mean_time_to_MAE": group["time_to_max_MAE"].mean(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "s3_early_mae_threshold_analysis.csv",
        index=False,
    )

    return result


# ============================================================
# RECOVERY ANALYSIS
# ============================================================


def recovery_analysis(
    trades,
):

    print()
    print("=" * 110)
    print("3. RECOVERY AFTER EARLY ADVERSE EXCURSION")
    print("=" * 110)

    rows = []

    for bar in DECISION_BARS:
        mae_col = f"mae_{bar}R"

        if mae_col not in trades.columns:
            continue

        subset = trades.loc[trades[mae_col] >= 0.75].copy()

        if subset.empty:
            continue

        for outcome in [
            "WIN",
            "LOSS",
        ]:
            group = subset.loc[subset["outcome"] == outcome]

            if group.empty:
                continue

            rows.append(
                {
                    "decision_bar": bar,
                    "outcome": outcome,
                    "trades": len(group),
                    "mean_early_MAE_R": group[mae_col].mean(),
                    "median_early_MAE_R": group[mae_col].median(),
                    "mean_max_MFE_R": group["max_MFE_R"].mean(),
                    "median_max_MFE_R": group["max_MFE_R"].median(),
                    "mean_final_close_R": group["final_close_R"].mean(),
                    "mean_time_to_MFE": group["time_to_max_MFE"].mean(),
                    "mean_time_to_MAE": group["time_to_max_MAE"].mean(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "s3_recovery_after_adverse.csv",
        index=False,
    )

    return result


# ============================================================
# EXIT REASON × EARLY MAE
# ============================================================


def exit_reason_analysis(
    trades,
):

    print()
    print("=" * 110)
    print("4. EXIT REASON × EARLY MAE")
    print("=" * 110)

    rows = []

    for bar in DECISION_BARS:
        col = f"mae_{bar}R"

        if col not in trades.columns:
            continue

        temp = trades.copy()

        temp["early_adverse"] = temp[col] >= 0.75

        grouped = temp.groupby(
            [
                "early_adverse",
                "exit_reason",
            ],
            dropna=False,
        )

        for (
            (
                adverse,
                exit_reason,
            ),
            group,
        ) in grouped:
            rows.append(
                {
                    "decision_bar": bar,
                    "early_adverse": adverse,
                    "exit_reason": exit_reason,
                    "trades": len(group),
                    "win_rate": (group["net_R"] > 0).mean(),
                    "mean_R": group["net_R"].mean(),
                    "total_R": group["net_R"].sum(),
                    "mean_MFE_R": group["max_MFE_R"].mean(),
                    "mean_MAE_R": group["max_MAE_R"].mean(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "s3_exit_reason_early_mae.csv",
        index=False,
    )

    return result


# ============================================================
# PATH PROFILE
# ============================================================


def path_profile(
    trades,
):

    print()
    print("=" * 110)
    print("5. WINNER vs LOSER PATH PROFILE")
    print("=" * 110)

    rows = []

    for outcome in [
        "WIN",
        "LOSS",
    ]:
        subset = trades.loc[trades["outcome"] == outcome]

        if subset.empty:
            continue

        for bar in DECISION_BARS:
            mae_col = f"mae_{bar}R"

            mfe_col = f"mfe_{bar}R"

            close_col = f"close_{bar}R"

            if (
                mae_col not in trades.columns
                or mfe_col not in trades.columns
                or close_col not in trades.columns
            ):
                continue

            rows.append(
                {
                    "outcome": outcome,
                    "bar": bar,
                    "trades": len(subset),
                    "mean_MAE_R": subset[mae_col].mean(),
                    "median_MAE_R": subset[mae_col].median(),
                    "mean_MFE_R": subset[mfe_col].mean(),
                    "median_MFE_R": subset[mfe_col].median(),
                    "mean_close_R": subset[close_col].mean(),
                    "median_close_R": subset[close_col].median(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "s3_winner_loser_path_profile.csv",
        index=False,
    )

    return result


# ============================================================
# WINDOW ROBUSTNESS
# ============================================================


def window_analysis(
    trades,
):

    print()
    print("=" * 110)
    print("6. EARLY MAE RULE — WINDOW ROBUSTNESS")
    print("=" * 110)

    rows = []

    rule_bar = 8
    rule_threshold = 0.75

    col = f"mae_{rule_bar}R"

    if col not in trades.columns:
        return pd.DataFrame()

    trades = trades.copy()

    trades["early_adverse"] = trades[col] >= rule_threshold

    for window, group in trades.groupby(
        "window",
        dropna=False,
    ):
        baseline = group

        filtered = group.loc[~group["early_adverse"]]

        for label, subset in [
            (
                "BASELINE",
                baseline,
            ),
            (
                "EARLY_FILTER",
                filtered,
            ),
        ]:
            if subset.empty:
                rows.append(
                    {
                        "window": window,
                        "rule": label,
                        "trades": 0,
                        "win_rate": np.nan,
                        "mean_R": np.nan,
                        "total_R": 0.0,
                    }
                )

                continue

            rows.append(
                {
                    "window": window,
                    "rule": label,
                    "trades": len(subset),
                    "win_rate": (subset["net_R"] > 0).mean(),
                    "mean_R": subset["net_R"].mean(),
                    "total_R": subset["net_R"].sum(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    result.to_csv(
        RESULTS_DIR / "s3_early_mae_window_robustness.csv",
        index=False,
    )

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S3 FAILURE PATH ANALYSIS")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop            = {STOP_POINTS} points")
    print(f"  RR              = {RR}")
    print(f"  Horizon         = {HORIZON} bars")

    trades = load_benchmark()

    market = load_market()

    paths = attach_paths(
        trades,
        market,
    )

    trades = merge_trade_information(
        trades,
        paths,
    )

    print()
    print(f"Final enriched trades: {len(trades)}")

    if len(trades) < len(trades):
        raise RuntimeError("Trade count changed unexpectedly.")

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    quadrant_analysis(trades)

    early_mae_analysis(trades)

    recovery_analysis(trades)

    exit_reason_analysis(trades)

    path_profile(trades)

    window_analysis(trades)

    # --------------------------------------------------------
    # Save master enriched dataset
    # --------------------------------------------------------

    output_file = RESULTS_DIR / "s3_failure_path_enriched.csv"

    trades.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(output_file)

    print(RESULTS_DIR / "s3_failure_path_quadrants.csv")

    print(RESULTS_DIR / "s3_early_mae_threshold_analysis.csv")

    print(RESULTS_DIR / "s3_recovery_after_adverse.csv")

    print(RESULTS_DIR / "s3_exit_reason_early_mae.csv")

    print(RESULTS_DIR / "s3_winner_loser_path_profile.csv")

    print(RESULTS_DIR / "s3_early_mae_window_robustness.csv")

    print()
    print("S3 FAILURE PATH ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
