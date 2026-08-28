"""
S7 — MAE RECOVERY BOUNDARY ANALYSIS

Purpose
-------
Identify the MAE level at which a trade becomes increasingly unlikely
to recover.

This is a DISCOVERY analysis.

It does NOT optimize a trading rule yet.

Core question
-------------
As a short trade reaches increasing adverse excursion levels:

    MAE >= 0.25R
    MAE >= 0.50R
    MAE >= 0.75R
    MAE >= 1.00R
    MAE >= 1.25R

what percentage subsequently:

    1. returns to break-even,
    2. reaches +0.50R,
    3. reaches +1.00R,
    4. reaches the original +1.75R target,
    5. eventually loses?

The purpose is to identify a potential MAE "point of no return"
before designing the actual filter.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"

# Prefer the enriched failure-mechanism dataset because it contains
# the complete intratrade path.
INPUT_CANDIDATES = [
    RESULTS_DIR / "s2_benchmark_trades_enriched.csv",
    RESULTS_DIR / "s4_adverse_recovery_enriched.csv",
    RESULTS_DIR / "s2_failure_mechanism_trades.csv",
]

STOP_R = 1.0
TARGET_R = 1.75

# MAE levels to investigate.
MAE_LEVELS = [
    0.25,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.25,
]

# Recovery thresholds.
RECOVERY_LEVELS = [
    0.00,
    -0.25,
    -0.50,
    -0.75,
    -1.00,
    -1.25,
    -1.50,
    -1.75,
]

OUTPUT_SUMMARY = RESULTS_DIR / "s7_mae_recovery_boundary_summary.csv"
OUTPUT_PATHS = RESULTS_DIR / "s7_mae_recovery_boundary_paths.csv"
OUTPUT_TRANSITIONS = RESULTS_DIR / "s7_mae_recovery_transitions.csv"


# ============================================================
# UTILITIES
# ============================================================


def find_input_file():
    """
    Find the first enriched dataset that actually contains
    the per-bar intratrade path columns required by S7.
    """

    for path in INPUT_CANDIDATES:
        if not path.exists():
            continue

        try:
            columns = pd.read_csv(
                path,
                nrows=0,
            ).columns.tolist()
        except Exception:
            continue

        has_close_path = any(col.startswith("close_") for col in columns)

        has_mae_path = any(col.startswith("mae_") for col in columns)

        if has_close_path and has_mae_path:
            return path

    raise FileNotFoundError(
        "Could not find a valid enriched dataset containing "
        "close_* and mae_* path columns.\n\n"
        "Checked:\n" + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def detect_path_columns(df):
    """
    Detect per-bar path columns.

    Expected structure:

        close_1R
        close_2R
        ...
        close_20R

    and optionally:

        mae_1R
        mfe_1R
        ...
    """

    close_cols = {}

    for col in df.columns:
        if not col.startswith("close_"):
            continue

        try:
            bar = int(col.split("_")[1].replace("R", ""))
        except (ValueError, IndexError):
            continue

        close_cols[bar] = col

    if not close_cols:
        raise RuntimeError(
            "No per-bar close path columns found. "
            "Expected columns such as close_1R, close_2R, ..."
        )

    return dict(sorted(close_cols.items()))


def detect_mae_columns(df):
    """
    Detect cumulative MAE path columns.
    """

    mae_cols = {}

    for col in df.columns:
        if not col.startswith("mae_"):
            continue

        try:
            bar = int(col.split("_")[1].replace("R", ""))
        except (ValueError, IndexError):
            continue

        mae_cols[bar] = col

    return dict(sorted(mae_cols.items()))


def detect_mfe_columns(df):
    """
    Detect cumulative MFE path columns.
    """

    mfe_cols = {}

    for col in df.columns:
        if not col.startswith("mfe_"):
            continue

        try:
            bar = int(col.split("_")[1].replace("R", ""))
        except (ValueError, IndexError):
            continue

        mfe_cols[bar] = col

    return dict(sorted(mfe_cols.items()))


# ============================================================
# PATH ANALYSIS
# ============================================================


def first_mae_crossing(
    row,
    mae_cols,
    threshold,
):
    """
    Return first bar where cumulative MAE reaches threshold.

    MAE is represented as a positive adverse excursion.

    Example:

        MAE = 0.80R

    means price moved 0.80R against the short.
    """

    for bar, col in mae_cols.items():
        value = row[col]

        if pd.isna(value):
            continue

        if float(value) >= threshold:
            return bar

    return np.nan


def future_recovery(
    row,
    close_cols,
    crossing_bar,
    recovery_close_level,
):
    """
    Determine whether price subsequently reaches a recovery level.

    For a SHORT:

        close_R > 0   = adverse
        close_R = 0   = break-even
        close_R < 0   = favorable

    Therefore:

        recovery_close_level = 0.00
            -> reaches break-even

        recovery_close_level = -0.50
            -> reaches +0.50R favorable

        recovery_close_level = -1.75
            -> reaches original target

    Returns:

        recovered: bool
        recovery_bar: first bar where recovery occurs
    """

    if pd.isna(crossing_bar):
        return False, np.nan

    crossing_bar = int(crossing_bar)

    for bar, col in close_cols.items():
        if bar <= crossing_bar:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        if float(value) <= recovery_close_level:
            return True, bar

    return False, np.nan


def future_max_favorable(
    row,
    close_cols,
    crossing_bar,
):
    """
    Maximum favorable CLOSE excursion after the MAE crossing.

    For a short:

        favorable = -close_R
    """

    if pd.isna(crossing_bar):
        return np.nan

    crossing_bar = int(crossing_bar)

    values = []

    for bar, col in close_cols.items():
        if bar <= crossing_bar:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        values.append(-float(value))

    if not values:
        return np.nan

    return max(values)


def future_min_adverse(
    row,
    close_cols,
    crossing_bar,
):
    """
    Maximum adverse CLOSE excursion after the MAE crossing.

    For a short:

        adverse = close_R
    """

    if pd.isna(crossing_bar):
        return np.nan

    crossing_bar = int(crossing_bar)

    values = []

    for bar, col in close_cols.items():
        if bar <= crossing_bar:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        values.append(float(value))

    if not values:
        return np.nan

    return max(values)


# ============================================================
# BUILD BOUNDARY DATA
# ============================================================


def analyze_threshold(
    df,
    close_cols,
    mae_cols,
    threshold,
):
    """
    Analyze all trades that reach a particular MAE threshold.
    """

    rows = []

    for idx, row in df.iterrows():
        crossing_bar = first_mae_crossing(
            row,
            mae_cols,
            threshold,
        )

        if pd.isna(crossing_bar):
            continue

        recovered_0, recovery_bar_0 = future_recovery(
            row,
            close_cols,
            crossing_bar,
            0.0,
        )

        recovered_05, recovery_bar_05 = future_recovery(
            row,
            close_cols,
            crossing_bar,
            -0.50,
        )

        recovered_10, recovery_bar_10 = future_recovery(
            row,
            close_cols,
            crossing_bar,
            -1.00,
        )

        recovered_target, recovery_bar_target = future_recovery(
            row,
            close_cols,
            crossing_bar,
            -TARGET_R,
        )

        future_mfe = future_max_favorable(
            row,
            close_cols,
            crossing_bar,
        )

        future_adverse = future_min_adverse(
            row,
            close_cols,
            crossing_bar,
        )

        final_R = pd.to_numeric(
            pd.Series([row.get("net_R", np.nan)]),
            errors="coerce",
        ).iloc[0]

        outcome = row.get("outcome", np.nan)

        exit_reason = row.get(
            "exit_reason",
            row.get("exit_reason_path", np.nan),
        )

        rows.append(
            {
                "trade_index": idx,
                "entry_timestamp": row.get(
                    "entry_timestamp",
                    np.nan,
                ),
                "window": row.get(
                    "window",
                    row.get("window_path", np.nan),
                ),
                "mae_threshold": threshold,
                "mae_crossing_bar": crossing_bar,
                "recovered_BE": recovered_0,
                "recovered_0_5R": recovered_05,
                "recovered_1R": recovered_10,
                "recovered_target": recovered_target,
                "recovery_bar_BE": recovery_bar_0,
                "recovery_bar_0_5R": recovery_bar_05,
                "recovery_bar_1R": recovery_bar_10,
                "recovery_bar_target": recovery_bar_target,
                "future_max_favorable_close_R": future_mfe,
                "future_max_adverse_close_R": future_adverse,
                "final_R": final_R,
                "outcome": outcome,
                "exit_reason": exit_reason,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================


def summarize_thresholds(
    path_df,
):
    """
    Convert individual threshold observations into a compact
    threshold-level summary.
    """

    summaries = []

    for threshold, group in path_df.groupby(
        "mae_threshold",
        sort=True,
    ):
        n = len(group)

        if n == 0:
            continue

        final_R = numeric(group["final_R"])

        wins = final_R > 0
        losses = final_R < 0

        recovered_BE = group["recovered_BE"].astype(bool)
        recovered_05 = group["recovered_0_5R"].astype(bool)
        recovered_10 = group["recovered_1R"].astype(bool)
        recovered_target = group["recovered_target"].astype(bool)

        stop_mask = group["exit_reason"].astype(str).str.upper().eq("STOP")

        summaries.append(
            {
                "mae_threshold_R": threshold,
                "trades_reaching_threshold": n,
                "final_wins": int(wins.sum()),
                "final_losses": int(losses.sum()),
                "final_win_rate": float(wins.mean()),
                "mean_final_R": float(final_R.mean()),
                "median_final_R": float(final_R.median()),
                "total_final_R": float(final_R.sum()),
                "break_even_recovery_pct": float(recovered_BE.mean()),
                "recovery_0_5R_pct": float(recovered_05.mean()),
                "recovery_1R_pct": float(recovered_10.mean()),
                "recovery_target_pct": float(recovered_target.mean()),
                "stop_pct": float(stop_mask.mean()),
                "mean_future_MFE_close_R": float(
                    numeric(group["future_max_favorable_close_R"]).mean()
                ),
                "median_future_MFE_close_R": float(
                    numeric(group["future_max_favorable_close_R"]).median()
                ),
                "mean_future_adverse_close_R": float(
                    numeric(group["future_max_adverse_close_R"]).mean()
                ),
                "mean_crossing_bar": float(numeric(group["mae_crossing_bar"]).mean()),
                "median_crossing_bar": float(
                    numeric(group["mae_crossing_bar"]).median()
                ),
            }
        )

    return pd.DataFrame(summaries)


# ============================================================
# TRANSITION ANALYSIS
# ============================================================


def build_transition_analysis(
    df,
    close_cols,
    mae_cols,
):
    """
    Analyze what happens as the same trade crosses progressively
    larger MAE thresholds.

    This is useful for identifying the deterioration curve.
    """

    rows = []

    for idx, row in df.iterrows():
        crossed = []

        for threshold in MAE_LEVELS:
            crossing_bar = first_mae_crossing(
                row,
                mae_cols,
                threshold,
            )

            if pd.isna(crossing_bar):
                continue

            crossed.append(
                (
                    threshold,
                    int(crossing_bar),
                )
            )

        if not crossed:
            continue

        for j in range(len(crossed)):
            threshold, bar = crossed[j]

            next_threshold = crossed[j + 1][0] if j + 1 < len(crossed) else np.nan

            next_bar = crossed[j + 1][1] if j + 1 < len(crossed) else np.nan

            future_mfe = future_max_favorable(
                row,
                close_cols,
                bar,
            )

            rows.append(
                {
                    "trade_index": idx,
                    "mae_threshold_R": threshold,
                    "crossing_bar": bar,
                    "next_mae_threshold_R": next_threshold,
                    "next_crossing_bar": next_bar,
                    "bars_until_next_threshold": (
                        next_bar - bar if not pd.isna(next_bar) else np.nan
                    ),
                    "future_max_favorable_close_R": future_mfe,
                    "final_R": row.get(
                        "net_R",
                        np.nan,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PRINT
# ============================================================


def print_summary(summary):
    print()
    print("=" * 110)
    print("S7 MAE RECOVERY BOUNDARY SUMMARY")
    print("=" * 110)

    columns = [
        "mae_threshold_R",
        "trades_reaching_threshold",
        "final_win_rate",
        "mean_final_R",
        "total_final_R",
        "break_even_recovery_pct",
        "recovery_0_5R_pct",
        "recovery_1R_pct",
        "recovery_target_pct",
        "stop_pct",
        "mean_crossing_bar",
    ]

    display = summary[columns].copy()

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_key_observation(summary):
    print()
    print("=" * 110)
    print("KEY MAE TRANSITION OBSERVATION")
    print("=" * 110)

    for _, row in summary.iterrows():
        threshold = row["mae_threshold_R"]
        n = int(row["trades_reaching_threshold"])
        be = row["break_even_recovery_pct"]
        target = row["recovery_target_pct"]
        wr = row["final_win_rate"]
        mean_R = row["mean_final_R"]

        print(
            f"MAE >= {threshold:.2f}R | "
            f"N={n:4d} | "
            f"BE recovery={be:.3f} | "
            f"Target recovery={target:.3f} | "
            f"Final WR={wr:.3f} | "
            f"Mean R={mean_R:.4f}"
        )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S7 MAE RECOVERY BOUNDARY ANALYSIS")
    print("=" * 110)

    print()
    print("This is a DISCOVERY test.")
    print("No trading filter is being optimized yet.")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    input_file = find_input_file()

    print()
    print("Loading enriched dataset...")
    print(input_file)

    df = pd.read_csv(input_file)

    print(f"Trades loaded: {len(df)}")

    # --------------------------------------------------------
    # PATH COLUMNS
    # --------------------------------------------------------

    close_cols = detect_path_columns(df)
    mae_cols = detect_mae_columns(df)
    mfe_cols = detect_mfe_columns(df)

    print()
    print("Detected path columns:")
    print(f"  Close path bars : {len(close_cols)}")
    print(f"  MAE path bars   : {len(mae_cols)}")
    print(f"  MFE path bars   : {len(mfe_cols)}")

    print(f"  Close range     : {min(close_cols)} -> {max(close_cols)}")

    if not mae_cols:
        raise RuntimeError("MAE path columns were not found.")

    # --------------------------------------------------------
    # NORMALIZE FINAL R
    # --------------------------------------------------------

    if "net_R" not in df.columns:
        if "net_R_path" in df.columns:
            df["net_R"] = numeric(df["net_R_path"])
        else:
            raise RuntimeError("No net_R or net_R_path column found.")

    # --------------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------------

    all_paths = []

    print()
    print("=" * 110)
    print("ANALYZING MAE THRESHOLDS")
    print("=" * 110)

    for threshold in MAE_LEVELS:
        print(f"Testing MAE >= {threshold:.2f}R...")

        result = analyze_threshold(
            df,
            close_cols,
            mae_cols,
            threshold,
        )

        all_paths.append(result)

        print(f"  Trades reaching threshold: {len(result)}")

    path_df = pd.concat(
        all_paths,
        ignore_index=True,
    )

    summary = summarize_thresholds(path_df)

    transitions = build_transition_analysis(
        df,
        close_cols,
        mae_cols,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_summary(summary)

    print_key_observation(summary)

    # --------------------------------------------------------
    # FIND CANDIDATE BOUNDARY
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("POTENTIAL BOUNDARY")
    print("=" * 110)

    if len(summary) >= 2:
        previous_be = None

        for _, row in summary.iterrows():
            threshold = row["mae_threshold_R"]

            be = row["break_even_recovery_pct"]

            target = row["recovery_target_pct"]

            if previous_be is not None:
                deterioration = previous_be - be

                if deterioration > 0.10:
                    print(
                        f"MAE {threshold:.2f}R: "
                        f"recovery deterioration "
                        f"={deterioration:.3f}"
                    )

            previous_be = be

    print()
    print("IMPORTANT:")
    print("This output identifies a candidate MAE boundary.")
    print("It does NOT yet prove that exiting at that boundary")
    print("will improve the strategy.")
    print("The next step will be a temporal OOS filter test.")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    path_df.to_csv(
        OUTPUT_PATHS,
        index=False,
    )

    transitions.to_csv(
        OUTPUT_TRANSITIONS,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(OUTPUT_SUMMARY)
    print(OUTPUT_PATHS)
    print(OUTPUT_TRANSITIONS)

    print()
    print("S7 MAE RECOVERY BOUNDARY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
