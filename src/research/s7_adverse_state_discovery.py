from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# S7 ADVERSE STATE DISCOVERY
# ============================================================
#
# OBJECTIVE
# ------------------------------------------------------------
# We are NOT optimizing an exit.
#
# We want to understand what differentiates:
#
#   1. RECOVERY
#      Adverse trade reaches MAE >= 0.75R
#      but subsequently recovers.
#
#   2. FAILURE
#      Adverse trade reaches MAE >= 0.75R
#      and subsequently deteriorates.
#
#   3. EARLY FAILURE
#      Adverse pressure develops rapidly.
#
# The purpose is to discover STATE VARIABLES that may later
# become:
#
#   - regime filters
#   - trade vetoes
#   - dynamic risk filters
#   - complementary models
#
# IMPORTANT:
# This script is DISCOVERY ONLY.
# No OOS optimization is performed here.
#
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results" / "s2_extended"

INPUT_FILE = RESULTS_DIR / "s2_failure_mechanism_trades.csv"

# Fallback if the previous analysis used another name.
FALLBACK_FILES = [
    RESULTS_DIR / "s3_early_failure_enriched_trades.csv",
    RESULTS_DIR / "s2_failure_mechanism_enriched.csv",
    RESULTS_DIR / "s3_best_early_failure_rule_trades.csv",
    RESULTS_DIR / "s2_benchmark_trades_enriched.csv",
]

ADVERSE_TRIGGER_R = 0.75

# ------------------------------------------------------------
# State classification
# ------------------------------------------------------------

RECOVERY_MFE_R = 1.50
FAILURE_FINAL_CLOSE_R = -0.50

# Bars used for state snapshots.
DECISION_BARS = [
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]


# ============================================================
# UTILITIES
# ============================================================


def find_input_file() -> Path:

    if INPUT_FILE.exists():
        return INPUT_FILE

    for path in FALLBACK_FILES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the S7 input dataset.\n"
        "Checked:\n" + "\n".join(str(x) for x in [INPUT_FILE] + FALLBACK_FILES)
    )


def numeric(df, column):

    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def percentile_rank(series):

    return series.rank(
        pct=True,
        method="average",
    )


# ============================================================
# CLASSIFY ADVERSE TRADES
# ============================================================


def classify_states(df):

    df = df.copy()

    df["max_MAE_R"] = numeric(
        df,
        "max_MAE_R",
    )

    df["max_MFE_R"] = numeric(
        df,
        "max_MFE_R",
    )

    df["final_close_R"] = numeric(
        df,
        "final_close_R",
    )

    adverse = df["max_MAE_R"] >= ADVERSE_TRIGGER_R

    adverse_df = df.loc[adverse].copy()

    # --------------------------------------------------------
    # Recovery:
    #
    # The trade suffered meaningful adverse excursion but
    # eventually achieved strong favorable excursion.
    # --------------------------------------------------------

    adverse_df["state"] = np.where(
        adverse_df["max_MFE_R"] >= RECOVERY_MFE_R,
        "RECOVERY",
        np.where(
            adverse_df["final_close_R"] <= FAILURE_FINAL_CLOSE_R,
            "FAILURE",
            "AMBIGUOUS",
        ),
    )

    return adverse_df


# ============================================================
# PATH SNAPSHOTS
# ============================================================


def build_snapshot_features(df):

    df = df.copy()

    for bar in DECISION_BARS:
        mae_col = f"mae_{bar}R"
        mfe_col = f"mfe_{bar}R"
        close_col = f"close_{bar}R"

        if mae_col in df.columns:
            df[f"snapshot_mae_{bar}"] = numeric(
                df,
                mae_col,
            )

        if mfe_col in df.columns:
            df[f"snapshot_mfe_{bar}"] = numeric(
                df,
                mfe_col,
            )

        if close_col in df.columns:
            df[f"snapshot_close_{bar}"] = numeric(
                df,
                close_col,
            )

        # ----------------------------------------------------
        # Adverse pressure relative to favorable excursion
        # ----------------------------------------------------

        if f"snapshot_mae_{bar}" in df.columns and f"snapshot_mfe_{bar}" in df.columns:
            df[f"adverse_favorable_ratio_{bar}"] = df[f"snapshot_mae_{bar}"] / (
                df[f"snapshot_mfe_{bar}"].abs() + 0.05
            )

        # ----------------------------------------------------
        # Recovery distance
        # ----------------------------------------------------

        if (
            f"snapshot_mae_{bar}" in df.columns
            and f"snapshot_close_{bar}" in df.columns
        ):
            df[f"recovery_gap_{bar}"] = (
                df[f"snapshot_close_{bar}"] + df[f"snapshot_mae_{bar}"]
            )

    # --------------------------------------------------------
    # Path acceleration
    # --------------------------------------------------------

    for b1, b2 in zip(
        DECISION_BARS[:-1],
        DECISION_BARS[1:],
    ):
        c1 = f"snapshot_close_{b1}"
        c2 = f"snapshot_close_{b2}"

        if c1 in df.columns and c2 in df.columns:
            df[f"close_change_{b1}_{b2}"] = df[c2] - df[c1]

    return df


# ============================================================
# STATE COMPARISON
# ============================================================


def state_comparison(df):

    feature_cols = []

    for column in df.columns:
        if (
            column.startswith("snapshot_")
            or column.startswith("adverse_favorable_ratio_")
            or column.startswith("recovery_gap_")
            or column.startswith("close_change_")
        ):
            feature_cols.append(column)

    rows = []

    for feature in feature_cols:
        recovery = numeric(
            df.loc[df["state"] == "RECOVERY"],
            feature,
        ).dropna()

        failure = numeric(
            df.loc[df["state"] == "FAILURE"],
            feature,
        ).dropna()

        if len(recovery) == 0 or len(failure) == 0:
            continue

        recovery_mean = recovery.mean()
        failure_mean = failure.mean()

        pooled_scale = pd.concat([recovery, failure]).std()

        effect = recovery_mean - failure_mean

        standardized = effect / pooled_scale if pooled_scale > 0 else np.nan

        rows.append(
            {
                "feature": feature,
                "recovery_n": len(recovery),
                "failure_n": len(failure),
                "recovery_mean": recovery_mean,
                "failure_mean": failure_mean,
                "recovery_median": recovery.median(),
                "failure_median": failure.median(),
                "mean_difference": effect,
                "standardized_difference": standardized,
                "abs_standardized_difference": abs(standardized)
                if pd.notna(standardized)
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "abs_standardized_difference",
            ascending=False,
        )

    return result


# ============================================================
# QUANTILE DISCRIMINATION
# ============================================================


def quantile_analysis(
    df,
    features,
):

    rows = []

    for feature in features:
        values = numeric(
            df,
            feature,
        )

        valid = values.notna()

        if valid.sum() < 30:
            continue

        try:
            buckets = pd.qcut(
                values[valid],
                q=4,
                duplicates="drop",
            )
        except ValueError:
            continue

        temp = df.loc[
            valid,
            ["state", "net_R"],
        ].copy()

        temp["bucket"] = buckets.values

        grouped = (
            temp.groupby(
                "bucket",
                observed=True,
            )
            .agg(
                trades=("net_R", "size"),
                recovery_rate=(
                    "state",
                    lambda x: np.mean(x == "RECOVERY"),
                ),
                mean_R=("net_R", "mean"),
            )
            .reset_index()
        )

        grouped.insert(
            0,
            "feature",
            feature,
        )

        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# TEMPORAL STATE TRANSITIONS
# ============================================================


def transition_analysis(df):

    rows = []

    for bar in DECISION_BARS:
        close_col = f"snapshot_close_{bar}"

        mae_col = f"snapshot_mae_{bar}"

        mfe_col = f"snapshot_mfe_{bar}"

        if (
            close_col not in df.columns
            or mae_col not in df.columns
            or mfe_col not in df.columns
        ):
            continue

        temp = df.copy()

        temp["close_state"] = pd.cut(
            temp[close_col],
            bins=[
                -np.inf,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                np.inf,
            ],
        )

        temp["mae_state"] = pd.cut(
            temp[mae_col],
            bins=[
                -np.inf,
                0.25,
                0.50,
                0.75,
                1.00,
                1.50,
                np.inf,
            ],
        )

        grouped = (
            temp.groupby(
                [
                    "close_state",
                    "mae_state",
                ],
                observed=True,
            )
            .agg(
                trades=("state", "size"),
                recovery_rate=(
                    "state",
                    lambda x: np.mean(x == "RECOVERY"),
                ),
                failure_rate=(
                    "state",
                    lambda x: np.mean(x == "FAILURE"),
                ),
                mean_R=("net_R", "mean"),
            )
            .reset_index()
        )

        grouped.insert(
            0,
            "decision_bar",
            bar,
        )

        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# SIMPLE STATE SCORE
# ============================================================


def state_score_analysis(df):

    candidates = []

    for bar in DECISION_BARS:
        close_col = f"snapshot_close_{bar}"
        mae_col = f"snapshot_mae_{bar}"
        mfe_col = f"snapshot_mfe_{bar}"

        if not all(
            x in df.columns
            for x in [
                close_col,
                mae_col,
                mfe_col,
            ]
        ):
            continue

        temp = df[
            [
                "state",
                "net_R",
                close_col,
                mae_col,
                mfe_col,
            ]
        ].copy()

        # ----------------------------------------------------
        # Construct interpretable state scores.
        #
        # Higher score should indicate recovery potential.
        # ----------------------------------------------------

        temp["recovery_score"] = (
            temp[close_col] + 0.50 * temp[mfe_col] - 0.50 * temp[mae_col]
        )

        try:
            temp["score_bucket"] = pd.qcut(
                temp["recovery_score"],
                q=5,
                duplicates="drop",
            )
        except ValueError:
            continue

        grouped = (
            temp.groupby(
                "score_bucket",
                observed=True,
            )
            .agg(
                trades=("state", "size"),
                recovery_rate=(
                    "state",
                    lambda x: np.mean(x == "RECOVERY"),
                ),
                failure_rate=(
                    "state",
                    lambda x: np.mean(x == "FAILURE"),
                ),
                mean_R=("net_R", "mean"),
            )
            .reset_index()
        )

        grouped.insert(
            0,
            "decision_bar",
            bar,
        )

        candidates.append(grouped)

    if not candidates:
        return pd.DataFrame()

    return pd.concat(
        candidates,
        ignore_index=True,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)
    print("S7 ADVERSE STATE DISCOVERY")
    print("=" * 110)

    input_file = find_input_file()

    print()
    print("Input:")
    print(input_file)

    df = pd.read_csv(input_file)

    print()
    print("Trades loaded:", len(df))

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required = [
        "net_R",
        "max_MAE_R",
        "max_MFE_R",
        "final_close_R",
    ]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    # --------------------------------------------------------
    # Classify adverse trades
    # --------------------------------------------------------

    adverse = classify_states(df)

    print()
    print("=" * 110)
    print("ADVERSE STATE COHORT")
    print("=" * 110)

    print(
        "Benchmark trades:",
        len(df),
    )

    print(
        "Adverse trades:",
        len(adverse),
    )

    print("\nState distribution:")

    print(adverse["state"].value_counts().to_string())

    # --------------------------------------------------------
    # Build path features
    # --------------------------------------------------------

    adverse = build_snapshot_features(adverse)

    # --------------------------------------------------------
    # 1. Recovery vs failure
    # --------------------------------------------------------

    comparison = state_comparison(adverse)

    print()
    print("=" * 110)
    print("1. RECOVERY vs FAILURE FEATURE DISCRIMINATION")
    print("=" * 110)

    if comparison.empty:
        print("No comparable features found.")
    else:
        print(comparison.head(30).to_string(index=False))

    # --------------------------------------------------------
    # 2. Quantile analysis
    # --------------------------------------------------------

    features = [
        x
        for x in adverse.columns
        if (
            x.startswith("snapshot_")
            or x.startswith("adverse_favorable_ratio_")
            or x.startswith("recovery_gap_")
            or x.startswith("close_change_")
        )
    ]

    quantiles = quantile_analysis(
        adverse,
        features,
    )

    print()
    print("=" * 110)
    print("2. FEATURE QUANTILE DISCRIMINATION")
    print("=" * 110)

    if quantiles.empty:
        print("No quantile results.")
    else:
        print(quantiles.head(50).to_string(index=False))

    # --------------------------------------------------------
    # 3. Temporal transitions
    # --------------------------------------------------------

    transitions = transition_analysis(adverse)

    print()
    print("=" * 110)
    print("3. TEMPORAL STATE TRANSITIONS")
    print("=" * 110)

    if transitions.empty:
        print("No transition results.")
    else:
        print(transitions.head(60).to_string(index=False))

    # --------------------------------------------------------
    # 4. Composite state score
    # --------------------------------------------------------

    scores = state_score_analysis(adverse)

    print()
    print("=" * 110)
    print("4. COMPOSITE RECOVERY STATE SCORE")
    print("=" * 110)

    if scores.empty:
        print("No score results.")
    else:
        print(scores.head(60).to_string(index=False))

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    adverse.to_csv(
        RESULTS_DIR / "s7_adverse_state_dataset.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR / "s7_state_feature_discrimination.csv",
        index=False,
    )

    quantiles.to_csv(
        RESULTS_DIR / "s7_state_feature_quantiles.csv",
        index=False,
    )

    transitions.to_csv(
        RESULTS_DIR / "s7_state_transitions.csv",
        index=False,
    )

    scores.to_csv(
        RESULTS_DIR / "s7_recovery_state_scores.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(RESULTS_DIR / "s7_adverse_state_dataset.csv")

    print(RESULTS_DIR / "s7_state_feature_discrimination.csv")

    print(RESULTS_DIR / "s7_state_feature_quantiles.csv")

    print(RESULTS_DIR / "s7_state_transitions.csv")

    print(RESULTS_DIR / "s7_recovery_state_scores.csv")

    print()
    print("S7 ADVERSE STATE DISCOVERY COMPLETE")


if __name__ == "__main__":
    main()
