from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

OUTPUT_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"

# MAE thresholds to investigate.
# These are DISCOVERY thresholds, not a trading rule.
MAE_THRESHOLDS = [
    0.25,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

# Early decision bars.
DECISION_BARS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
]

# A trade is considered a final failure if it does not
# finish positive.
#
# We separately report STOP / TIMEOUT_LOSS / WIN / TARGET.
FAILURE_R_THRESHOLD = 0.0


# ============================================================
# HELPERS
# ============================================================


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return np.nan


def detect_path_columns(df):
    mae_cols = {}
    close_cols = {}
    mfe_cols = {}

    for column in df.columns:
        name = str(column)

        if name.startswith("mae_") and name.endswith("R"):
            try:
                bar = int(name[4:-1])
                mae_cols[bar] = name
            except ValueError:
                pass

        elif name.startswith("close_") and name.endswith("R"):
            try:
                bar = int(name[6:-1])
                close_cols[bar] = name
            except ValueError:
                pass

        elif name.startswith("mfe_") and name.endswith("R"):
            try:
                bar = int(name[4:-1])
                mfe_cols[bar] = name
            except ValueError:
                pass

    return mae_cols, close_cols, mfe_cols


def classify_final_outcome(row):
    """
    Uses the already-computed benchmark outcome where available.

    Fallback:
        net_R > 0  -> WIN
        net_R <= 0 -> LOSS
    """

    if "outcome" in row.index:
        value = str(row["outcome"]).strip().upper()

        if value:
            return value

    net_R = safe_float(row.get("net_R"))

    if np.isfinite(net_R):
        if net_R > FAILURE_R_THRESHOLD:
            return "WIN"

        return "LOSS"

    return "UNKNOWN"


def calculate_binary_stats(triggered, failed):
    """
    triggered: boolean Series
    failed: boolean Series
    """

    n = int(triggered.sum())

    if n == 0:
        return {
            "triggered": 0,
            "failed": 0,
            "survived": 0,
            "failure_rate": np.nan,
            "survival_rate": np.nan,
        }

    failed_n = int((triggered & failed).sum())

    survived_n = n - failed_n

    return {
        "triggered": n,
        "failed": failed_n,
        "survived": survived_n,
        "failure_rate": failed_n / n,
        "survival_rate": survived_n / n,
    }


# ============================================================
# BUILD EARLY MAE MATRIX
# ============================================================


def build_early_mae_matrix(
    df,
    mae_cols,
    decision_bars,
):
    records = []

    for index, row in df.iterrows():
        record = {
            "source_index": index,
            "trade_index": row.get(
                "trade_index",
                index,
            ),
            "window": row.get(
                "window",
                np.nan,
            ),
            "entry_timestamp": row.get(
                "entry_timestamp",
                np.nan,
            ),
            "net_R": safe_float(row.get("net_R")),
            "outcome": classify_final_outcome(row),
            "exit_reason": row.get(
                "exit_reason",
                row.get(
                    "exit_reason_path",
                    "",
                ),
            ),
        }

        for bar in decision_bars:
            if bar in mae_cols:
                record[f"mae_at_{bar}"] = safe_float(row[mae_cols[bar]])

            else:
                record[f"mae_at_{bar}"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# THRESHOLD / TIME ANALYSIS
# ============================================================


def threshold_time_analysis(
    df,
    mae_cols,
    decision_bars,
    threshold,
):
    rows = []

    for bar in decision_bars:
        if bar not in mae_cols:
            continue

        mae = pd.to_numeric(
            df[mae_cols[bar]],
            errors="coerce",
        )

        valid = mae.notna()

        triggered = valid & (mae >= threshold)

        failed = df["final_failure"]

        stats = calculate_binary_stats(
            triggered,
            failed,
        )

        triggered_df = df.loc[triggered].copy()

        if len(triggered_df):
            final_R = pd.to_numeric(
                triggered_df["net_R"],
                errors="coerce",
            )

            wins = final_R[final_R > 0]

            losses = final_R[final_R <= 0]

            gross_profit = wins.sum()

            gross_loss = -losses.sum()

            if gross_loss > 0:
                pf = gross_profit / gross_loss
            elif gross_profit > 0:
                pf = np.inf
            else:
                pf = np.nan

            mean_R = final_R.mean()

            total_R = final_R.sum()

            target_n = int((triggered_df["final_target"]).sum())

            stop_n = int((triggered_df["final_stop"]).sum())

            timeout_n = int((triggered_df["final_timeout"]).sum())

            recovery_n = int((triggered_df["final_recovery"]).sum())

        else:
            pf = np.nan
            mean_R = np.nan
            total_R = 0.0

            target_n = 0
            stop_n = 0
            timeout_n = 0
            recovery_n = 0

        rows.append(
            {
                "mae_threshold_R": threshold,
                "decision_bar": bar,
                **stats,
                "mean_final_R": mean_R,
                "total_final_R": total_R,
                "profit_factor": pf,
                "target_n": target_n,
                "stop_n": stop_n,
                "timeout_n": timeout_n,
                "recovery_n": recovery_n,
                "target_rate": (
                    target_n / stats["triggered"] if stats["triggered"] > 0 else np.nan
                ),
                "stop_rate": (
                    stop_n / stats["triggered"] if stats["triggered"] > 0 else np.nan
                ),
                "timeout_rate": (
                    timeout_n / stats["triggered"] if stats["triggered"] > 0 else np.nan
                ),
                "recovery_rate": (
                    recovery_n / stats["triggered"]
                    if stats["triggered"] > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# OUTCOME CROSS-TAB
# ============================================================


def outcome_cross_tab(
    df,
    mae_cols,
    decision_bars,
    threshold,
):
    rows = []

    for bar in decision_bars:
        if bar not in mae_cols:
            continue

        mae = pd.to_numeric(
            df[mae_cols[bar]],
            errors="coerce",
        )

        triggered = mae.notna() & (mae >= threshold)

        cohort = df.loc[triggered].copy()

        if cohort.empty:
            continue

        counts = cohort["outcome"].value_counts()

        total = len(cohort)

        for outcome, count in counts.items():
            rows.append(
                {
                    "mae_threshold_R": threshold,
                    "decision_bar": bar,
                    "outcome": outcome,
                    "trades": int(count),
                    "pct": count / total,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAE PATH DIAGNOSTICS
# ============================================================


def path_transition_analysis(
    df,
    mae_cols,
    close_cols,
    decision_bars,
):
    rows = []

    for bar in decision_bars:
        if bar not in mae_cols:
            continue

        mae = pd.to_numeric(
            df[mae_cols[bar]],
            errors="coerce",
        )

        valid = mae.notna()

        for threshold in MAE_THRESHOLDS:
            triggered = valid & (mae >= threshold)

            cohort = df.loc[triggered].copy()

            if cohort.empty:
                continue

            # ----------------------------------------------
            # Recovery definitions
            # ----------------------------------------------

            recovery_to_zero = 0
            recovery_to_target = 0
            eventual_stop = 0

            for _, row in cohort.iterrows():
                start_mae = safe_float(row[mae_cols[bar]])

                if not np.isfinite(start_mae):
                    continue

                # Look forward through available path.
                for future_bar in sorted(mae_cols):
                    if future_bar <= bar:
                        continue

                    future_mae = safe_float(row[mae_cols[future_bar]])

                    if np.isfinite(future_mae):
                        if future_mae < threshold:
                            recovery_to_zero += 1
                            break

                # Target / final target.
                outcome = str(row["outcome"]).upper()

                if "TARGET" in outcome or outcome == "WIN":
                    recovery_to_target += 1

                if "STOP" in outcome or outcome == "LOSS":
                    eventual_stop += 1

            n = len(cohort)

            rows.append(
                {
                    "mae_threshold_R": threshold,
                    "decision_bar": bar,
                    "triggered_trades": n,
                    "later_mae_recovery_below_threshold": recovery_to_zero,
                    "recovery_below_threshold_pct": recovery_to_zero / n,
                    "eventual_target_or_win": recovery_to_target,
                    "eventual_target_or_win_pct": recovery_to_target / n,
                    "eventual_stop_or_loss": eventual_stop,
                    "eventual_stop_or_loss_pct": eventual_stop / n,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S10 MAE EARLY-WARNING ANALYSIS")
    print("=" * 110)

    print()
    print("Purpose:")
    print("Determine whether early adverse movement predicts eventual trade failure.")
    print()
    print("IMPORTANT: This is a DISCOVERY analysis.")
    print("No trading exit is being optimized.")
    print()

    print(f"MAE thresholds: {MAE_THRESHOLDS}")

    print(f"Decision bars  : {DECISION_BARS}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    # --------------------------------------------------------
    # PATH COLUMNS
    # --------------------------------------------------------

    mae_cols, close_cols, mfe_cols = detect_path_columns(df)

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_cols)}")
    print(f"  Close bars : {len(close_cols)}")
    print(f"  MFE bars   : {len(mfe_cols)}")

    if not mae_cols:
        raise RuntimeError("No MAE path columns found.")

    # --------------------------------------------------------
    # FINAL OUTCOME LABELS
    # --------------------------------------------------------

    df["outcome"] = df.apply(
        classify_final_outcome,
        axis=1,
    )

    df["final_failure"] = (
        pd.to_numeric(
            df["net_R"],
            errors="coerce",
        )
        <= FAILURE_R_THRESHOLD
    )

    df["final_target"] = (
        df["exit_reason"].astype(str).str.upper().str.contains("TARGET")
    )

    df["final_stop"] = df["exit_reason"].astype(str).str.upper().str.contains("STOP")

    df["final_timeout"] = (
        df["exit_reason"].astype(str).str.upper().str.contains("TIMEOUT")
    )

    df["final_recovery"] = ~df["final_stop"] & (
        pd.to_numeric(
            df["net_R"],
            errors="coerce",
        )
        > 0
    )

    print()
    print("Final outcome distribution:")

    print(df["outcome"].value_counts(dropna=False).to_string())

    # --------------------------------------------------------
    # BASIC EARLY MAE MATRIX
    # --------------------------------------------------------

    early_matrix = build_early_mae_matrix(
        df,
        mae_cols,
        DECISION_BARS,
    )

    # --------------------------------------------------------
    # THRESHOLD × TIME
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("MAE THRESHOLD × DECISION TIME")
    print("=" * 110)

    all_rows = []

    for threshold in MAE_THRESHOLDS:
        print(f"Analyzing MAE >= {threshold:.2f}R...")

        result = threshold_time_analysis(
            df,
            mae_cols,
            DECISION_BARS,
            threshold,
        )

        all_rows.append(result)

    threshold_time = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # Print the most important columns.
    print()

    print(
        threshold_time[
            [
                "mae_threshold_R",
                "decision_bar",
                "triggered",
                "failed",
                "failure_rate",
                "survival_rate",
                "mean_final_R",
                "total_final_R",
                "target_rate",
                "stop_rate",
                "timeout_rate",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # FIND MOST DISCRIMINATIVE OBSERVATIONS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("STRONGEST FAILURE DISCRIMINATION")
    print("=" * 110)

    ranked = threshold_time[threshold_time["triggered"] >= 10].copy()

    if not ranked.empty:
        ranked = ranked.sort_values(
            [
                "failure_rate",
                "triggered",
            ],
            ascending=[
                False,
                False,
            ],
        )

        print(
            ranked[
                [
                    "mae_threshold_R",
                    "decision_bar",
                    "triggered",
                    "failed",
                    "failure_rate",
                    "survival_rate",
                    "mean_final_R",
                    "target_rate",
                    "stop_rate",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # OUTCOME CROSS-TABS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("OUTCOME DISTRIBUTION AFTER MAE CROSSING")
    print("=" * 110)

    outcome_frames = []

    for threshold in MAE_THRESHOLDS:
        result = outcome_cross_tab(
            df,
            mae_cols,
            DECISION_BARS,
            threshold,
        )

        outcome_frames.append(result)

    outcome_table = pd.concat(
        outcome_frames,
        ignore_index=True,
    )

    print(outcome_table.to_string(index=False))

    # --------------------------------------------------------
    # RECOVERY TRANSITIONS
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("MAE RECOVERY TRANSITIONS")
    print("=" * 110)

    recovery_table = path_transition_analysis(
        df,
        mae_cols,
        close_cols,
        DECISION_BARS,
    )

    print(
        recovery_table[
            [
                "mae_threshold_R",
                "decision_bar",
                "triggered_trades",
                "recovery_below_threshold_pct",
                "eventual_target_or_win_pct",
                "eventual_stop_or_loss_pct",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # SPECIFIC 0.80R REPORT
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("0.80R EARLY-WARNING REPORT")
    print("=" * 110)

    report = threshold_time[threshold_time["mae_threshold_R"].eq(0.80)].copy()

    if not report.empty:
        print(
            report[
                [
                    "mae_threshold_R",
                    "decision_bar",
                    "triggered",
                    "failed",
                    "survival_rate",
                    "failure_rate",
                    "mean_final_R",
                    "total_final_R",
                    "target_rate",
                    "stop_rate",
                    "timeout_rate",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_path = OUTPUT_DIR / "s10_mae_early_warning_threshold_time.csv"

    outcome_path = OUTPUT_DIR / "s10_mae_early_warning_outcomes.csv"

    recovery_path = OUTPUT_DIR / "s10_mae_early_warning_recovery.csv"

    matrix_path = OUTPUT_DIR / "s10_mae_early_warning_matrix.csv"

    ranked_path = OUTPUT_DIR / "s10_mae_early_warning_ranked.csv"

    threshold_time.to_csv(
        threshold_path,
        index=False,
    )

    outcome_table.to_csv(
        outcome_path,
        index=False,
    )

    recovery_table.to_csv(
        recovery_path,
        index=False,
    )

    early_matrix.to_csv(
        matrix_path,
        index=False,
    )

    if not ranked.empty:
        ranked.to_csv(
            ranked_path,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            ranked_path,
            index=False,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(threshold_path)
    print(outcome_path)
    print(recovery_path)
    print(matrix_path)
    print(ranked_path)

    print()
    print("=" * 110)
    print("S10 MAE EARLY-WARNING ANALYSIS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
