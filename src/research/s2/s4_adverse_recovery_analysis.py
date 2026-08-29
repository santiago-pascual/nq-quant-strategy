from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "src" / "research" / "results" / "s2_extended"

# This is the file produced by the failure-mechanism analysis.
BENCHMARK_PATH = RESULTS_DIR / "s3_failure_path_enriched.csv"


# ============================================================================
# FROZEN BENCHMARK
# ============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

DECISION_BAR = 8
ADVERSE_THRESHOLD_R = 0.75


# ============================================================================
# HELPERS
# ============================================================================


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def print_separator(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def get_path(df, metric, bar):
    """
    Dataset naming convention:

        mae_1R
        mfe_1R
        close_1R

        ...

        mae_20R
        mfe_20R
        close_20R
    """

    column = f"{metric}_{bar}R"

    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return numeric(df[column])


# ============================================================================
# LOAD
# ============================================================================


def load_data():

    print("Loading S3 enriched failure-path dataset...")

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"\nRequired file not found:\n{BENCHMARK_PATH}\n")

    df = pd.read_csv(BENCHMARK_PATH)

    print(f"Trades loaded: {len(df)}")

    return df


# ============================================================================
# BUILD FEATURES
# ============================================================================


def build_features(df):

    df = df.copy()

    print_separator("BUILDING S4 PATH FEATURES")

    # ----------------------------------------------------------------------
    # Recover path observations
    # ----------------------------------------------------------------------

    for bar in range(1, DECISION_BAR + 1):
        df[f"mae_{bar}"] = get_path(
            df,
            "mae",
            bar,
        )

        df[f"mfe_{bar}"] = get_path(
            df,
            "mfe",
            bar,
        )

        df[f"close_{bar}"] = get_path(
            df,
            "close",
            bar,
        )

    # ----------------------------------------------------------------------
    # Sanity check
    # ----------------------------------------------------------------------

    print(f"Decision-bar MAE available: {df['mae_8'].notna().sum()}")

    print(f"Decision-bar MFE available: {df['mfe_8'].notna().sum()}")

    print(f"Decision-bar close available: {df['close_8'].notna().sum()}")

    # ----------------------------------------------------------------------
    # MAE trajectory
    # ----------------------------------------------------------------------

    df["mae_change_1_8"] = df["mae_8"] - df["mae_1"]

    df["mae_change_2_8"] = df["mae_8"] - df["mae_2"]

    df["mae_change_3_8"] = df["mae_8"] - df["mae_3"]

    df["mae_change_5_8"] = df["mae_8"] - df["mae_5"]

    # ----------------------------------------------------------------------
    # MFE trajectory
    # ----------------------------------------------------------------------

    df["mfe_change_1_8"] = df["mfe_8"] - df["mfe_1"]

    df["mfe_change_2_8"] = df["mfe_8"] - df["mfe_2"]

    df["mfe_change_3_8"] = df["mfe_8"] - df["mfe_3"]

    df["mfe_change_5_8"] = df["mfe_8"] - df["mfe_5"]

    # ----------------------------------------------------------------------
    # Close trajectory
    # ----------------------------------------------------------------------

    df["close_change_1_8"] = df["close_8"] - df["close_1"]

    df["close_change_2_8"] = df["close_8"] - df["close_2"]

    df["close_change_3_8"] = df["close_8"] - df["close_3"]

    df["close_change_5_8"] = df["close_8"] - df["close_5"]

    # ----------------------------------------------------------------------
    # Excursion relationship
    # ----------------------------------------------------------------------

    df["mfe_minus_mae_8"] = df["mfe_8"] - df["mae_8"]

    df["mfe_to_mae_8"] = df["mfe_8"] / df["mae_8"].replace(
        0,
        np.nan,
    )

    # ----------------------------------------------------------------------
    # MAE speed
    # ----------------------------------------------------------------------

    df["mae_speed_1_8"] = df["mae_8"] / 8.0

    df["mae_speed_3_8"] = (df["mae_8"] - df["mae_3"]) / 5.0

    df["mae_speed_5_8"] = (df["mae_8"] - df["mae_5"]) / 3.0

    # ----------------------------------------------------------------------
    # MFE speed
    # ----------------------------------------------------------------------

    df["mfe_speed_1_8"] = df["mfe_8"] / 8.0

    df["mfe_speed_3_8"] = (df["mfe_8"] - df["mfe_3"]) / 5.0

    df["mfe_speed_5_8"] = (df["mfe_8"] - df["mfe_5"]) / 3.0

    # ----------------------------------------------------------------------
    # Early path dispersion
    # ----------------------------------------------------------------------

    mae_columns = [
        f"mae_{bar}"
        for bar in range(
            1,
            DECISION_BAR + 1,
        )
    ]

    mfe_columns = [
        f"mfe_{bar}"
        for bar in range(
            1,
            DECISION_BAR + 1,
        )
    ]

    close_columns = [
        f"close_{bar}"
        for bar in range(
            1,
            DECISION_BAR + 1,
        )
    ]

    df["mae_path_std_8"] = df[mae_columns].std(axis=1)

    df["mfe_path_std_8"] = df[mfe_columns].std(axis=1)

    df["close_path_std_8"] = df[close_columns].std(axis=1)

    # ----------------------------------------------------------------------
    # Final outcome
    # ----------------------------------------------------------------------

    df["final_profitable"] = numeric(df["net_R"]) > 0

    # ----------------------------------------------------------------------
    # Early adverse event
    # ----------------------------------------------------------------------

    df["early_adverse"] = df["mae_8"] >= ADVERSE_THRESHOLD_R

    return df


# ============================================================================
# ADVERSE COHORT
# ============================================================================


def create_adverse_cohort(df):

    print_separator("DEFINING ADVERSE COHORT")

    adverse = df.loc[df["early_adverse"]].copy()

    adverse["recovery_group"] = np.where(
        adverse["final_profitable"],
        "RECOVERY",
        "FAILURE",
    )

    recovery_count = (adverse["recovery_group"] == "RECOVERY").sum()

    failure_count = (adverse["recovery_group"] == "FAILURE").sum()

    print(f"Total benchmark trades : {len(df)}")

    print(f"Adverse trades          : {len(adverse)}")

    print(f"Recovery trades         : {recovery_count}")

    print(f"Failure trades          : {failure_count}")

    if len(adverse) == 0:
        print()
        print(
            "ERROR: No trades reached MAE >= "
            f"{ADVERSE_THRESHOLD_R}R by bar {DECISION_BAR}."
        )

        print()
        print("Available MAE statistics:")

        print(df["mae_8"].describe().to_string())

        raise RuntimeError("No adverse cohort.")

    return adverse


# ============================================================================
# GROUP SUMMARY
# ============================================================================


def group_summary(adverse):

    print_separator("1. RECOVERY vs FAILURE SUMMARY")

    rows = []

    for group in [
        "RECOVERY",
        "FAILURE",
    ]:
        subset = adverse.loc[adverse["recovery_group"] == group]

        if subset.empty:
            continue

        rows.append(
            {
                "group": group,
                "trades": len(subset),
                "pct_adverse": (len(subset) / len(adverse)),
                "mean_R": numeric(subset["net_R"]).mean(),
                "median_R": numeric(subset["net_R"]).median(),
                "mean_MAE_8": subset["mae_8"].mean(),
                "median_MAE_8": subset["mae_8"].median(),
                "mean_MFE_8": subset["mfe_8"].mean(),
                "median_MFE_8": subset["mfe_8"].median(),
                "mean_close_8": subset["close_8"].mean(),
                "median_close_8": subset["close_8"].median(),
                "mean_MFE_minus_MAE": subset["mfe_minus_mae_8"].mean(),
                "median_MFE_minus_MAE": subset["mfe_minus_mae_8"].median(),
            }
        )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================================
# FEATURE COMPARISON
# ============================================================================


def feature_comparison(adverse):

    print_separator("2. RECOVERY vs FAILURE FEATURE COMPARISON")

    exclude = {
        "net_R",
        "net_points",
        "raw_points",
        "final_profitable",
        "early_adverse",
        "recovery_group",
        "outcome",
    }

    features = [
        column
        for column in adverse.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(adverse[column])
    ]

    rows = []

    for feature in features:
        recovery = numeric(
            adverse.loc[
                adverse["recovery_group"] == "RECOVERY",
                feature,
            ]
        ).dropna()

        failure = numeric(
            adverse.loc[
                adverse["recovery_group"] == "FAILURE",
                feature,
            ]
        ).dropna()

        if len(recovery) < 5:
            continue

        if len(failure) < 5:
            continue

        recovery_mean = recovery.mean()
        failure_mean = failure.mean()

        recovery_median = recovery.median()
        failure_median = failure.median()

        pooled_std = np.sqrt((recovery.var() + failure.var()) / 2.0)

        if pooled_std > 0:
            standardized_difference = (recovery_mean - failure_mean) / pooled_std

        else:
            standardized_difference = np.nan

        rows.append(
            {
                "feature": feature,
                "recovery_n": len(recovery),
                "failure_n": len(failure),
                "recovery_mean": recovery_mean,
                "failure_mean": failure_mean,
                "recovery_median": recovery_median,
                "failure_median": failure_median,
                "mean_difference": (recovery_mean - failure_mean),
                "median_difference": (recovery_median - failure_median),
                "abs_standardized_difference": abs(standardized_difference),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "abs_standardized_difference",
            ascending=False,
        )

    print(result.head(40).to_string(index=False))

    return result


# ============================================================================
# QUANTILE ANALYSIS
# ============================================================================


def quantile_analysis(adverse):

    print_separator("3. FEATURE QUANTILE ANALYSIS")

    features = [
        "mae_8",
        "mfe_8",
        "close_8",
        "mae_change_1_8",
        "mae_change_3_8",
        "mae_change_5_8",
        "mfe_change_1_8",
        "mfe_change_3_8",
        "mfe_change_5_8",
        "close_change_1_8",
        "close_change_3_8",
        "close_change_5_8",
        "mfe_minus_mae_8",
        "mfe_to_mae_8",
        "mae_speed_1_8",
        "mae_speed_3_8",
        "mae_speed_5_8",
        "mfe_speed_1_8",
        "mfe_speed_3_8",
        "mfe_speed_5_8",
        "mae_path_std_8",
        "mfe_path_std_8",
        "close_path_std_8",
        "quality",
        "vol_percentile",
        "entry_past_return_5",
        "entry_past_return_10",
        "entry_past_return_15",
        "entry_past_return_30",
        "entry_vol_ratio_5_30",
        "entry_vol_ratio_5_60",
        "entry_variance_ratio_5_30",
        "entry_variance_ratio_5_60",
    ]

    rows = []

    for feature in features:
        if feature not in adverse.columns:
            continue

        values = numeric(adverse[feature])

        if values.notna().sum() < 20:
            continue

        temp = adverse.copy()

        temp["_value"] = values

        try:
            temp["_bucket"] = pd.qcut(
                temp["_value"],
                q=4,
                duplicates="drop",
            )

        except ValueError:
            continue

        for bucket, subset in temp.groupby(
            "_bucket",
            observed=False,
        ):
            if len(subset) == 0:
                continue

            pnl = numeric(subset["net_R"])

            recovery_rate = (subset["recovery_group"] == "RECOVERY").mean()

            rows.append(
                {
                    "feature": feature,
                    "bucket": str(bucket),
                    "trades": len(subset),
                    "recovery_trades": int(
                        (subset["recovery_group"] == "RECOVERY").sum()
                    ),
                    "recovery_rate": recovery_rate,
                    "mean_R": pnl.mean(),
                    "median_R": pnl.median(),
                    "total_R": pnl.sum(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================================
# MAE / MFE STRUCTURE
# ============================================================================


def excursion_structure(adverse):

    print_separator("4. MAE / MFE STRUCTURE")

    rows = []

    for mae_threshold in [
        0.75,
        1.00,
        1.25,
    ]:
        for mfe_threshold in [
            0.25,
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
        ]:
            subset = adverse.loc[
                (adverse["mae_8"] >= mae_threshold)
                & (adverse["mfe_8"] <= mfe_threshold)
            ]

            if len(subset) < 5:
                continue

            recovery_rate = (subset["recovery_group"] == "RECOVERY").mean()

            pnl = numeric(subset["net_R"])

            rows.append(
                {
                    "mae_min": mae_threshold,
                    "mfe_max": mfe_threshold,
                    "trades": len(subset),
                    "recovery_rate": recovery_rate,
                    "mean_R": pnl.mean(),
                    "total_R": pnl.sum(),
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            [
                "recovery_rate",
                "trades",
            ],
            ascending=[
                True,
                False,
            ],
        )

    print(result.to_string(index=False))

    return result


# ============================================================================
# TIME-TO-ADVERSE
# ============================================================================


def adverse_timing_analysis(df):

    print_separator("5. TIME TO FIRST ADVERSE LEVEL")

    rows = []

    for threshold in [
        0.50,
        0.75,
        1.00,
    ]:
        timing = pd.Series(
            np.nan,
            index=df.index,
        )

        for bar in range(
            1,
            DECISION_BAR + 1,
        ):
            mask = timing.isna() & (df[f"mae_{bar}"] >= threshold)

            timing.loc[mask] = bar

        temp = df.copy()

        temp["time_to_adverse"] = timing

        temp = temp.loc[temp["time_to_adverse"].notna()]

        if temp.empty:
            continue

        temp["group"] = np.where(
            temp["final_profitable"],
            "RECOVERY",
            "FAILURE",
        )

        for group in [
            "RECOVERY",
            "FAILURE",
        ]:
            subset = temp.loc[temp["group"] == group]

            if subset.empty:
                continue

            rows.append(
                {
                    "threshold_R": threshold,
                    "group": group,
                    "trades": len(subset),
                    "mean_time_to_adverse": subset["time_to_adverse"].mean(),
                    "median_time_to_adverse": subset["time_to_adverse"].median(),
                }
            )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================================
# WINDOW ROBUSTNESS
# ============================================================================


def window_analysis(adverse):

    print_separator("6. RECOVERY vs FAILURE BY WINDOW")

    if "window" not in adverse.columns:
        print("WARNING: window column unavailable.")

        return pd.DataFrame()

    rows = []

    for window, subset in adverse.groupby("window"):
        recovery = subset.loc[subset["recovery_group"] == "RECOVERY"]

        failure = subset.loc[subset["recovery_group"] == "FAILURE"]

        rows.append(
            {
                "window": window,
                "adverse_trades": len(subset),
                "recovery_trades": len(recovery),
                "failure_trades": len(failure),
                "recovery_pct": (
                    len(recovery) / len(subset) if len(subset) else np.nan
                ),
                "recovery_mean_R": (
                    numeric(recovery["net_R"]).mean() if len(recovery) else np.nan
                ),
                "failure_mean_R": (
                    numeric(failure["net_R"]).mean() if len(failure) else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 110)
    print("S4 ADVERSE RECOVERY ANALYSIS")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop            = {STOP_POINTS} points")
    print(f"  RR              = {RR}")
    print(f"  Horizon         = {HORIZON} bars")
    print(f"  Decision bar    = {DECISION_BAR}")
    print(f"  Adverse MAE     >= {ADVERSE_THRESHOLD_R}R")

    trades = load_data()

    print_separator("VALIDATING DATASET")

    required = [
        "net_R",
        "mae_8R",
        "mfe_8R",
        "close_8R",
    ]

    missing = [column for column in required if column not in trades.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    trades = build_features(trades)

    adverse = create_adverse_cohort(trades)

    summary = group_summary(adverse)

    comparison = feature_comparison(adverse)

    quantiles = quantile_analysis(adverse)

    excursion = excursion_structure(adverse)

    timing = adverse_timing_analysis(trades)

    windows = window_analysis(adverse)

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "enriched": RESULTS_DIR / "s4_adverse_recovery_enriched.csv",
        "summary": RESULTS_DIR / "s4_adverse_recovery_summary.csv",
        "comparison": RESULTS_DIR / "s4_recovery_feature_comparison.csv",
        "quantiles": RESULTS_DIR / "s4_recovery_feature_quantiles.csv",
        "excursion": RESULTS_DIR / "s4_recovery_excursion_structure.csv",
        "timing": RESULTS_DIR / "s4_recovery_adverse_timing.csv",
        "windows": RESULTS_DIR / "s4_recovery_window_analysis.csv",
    }

    adverse.to_csv(
        paths["enriched"],
        index=False,
    )

    summary.to_csv(
        paths["summary"],
        index=False,
    )

    comparison.to_csv(
        paths["comparison"],
        index=False,
    )

    quantiles.to_csv(
        paths["quantiles"],
        index=False,
    )

    excursion.to_csv(
        paths["excursion"],
        index=False,
    )

    timing.to_csv(
        paths["timing"],
        index=False,
    )

    windows.to_csv(
        paths["windows"],
        index=False,
    )

    print_separator("FILES SAVED")

    for path in paths.values():
        print(path)

    print()
    print("S4 ADVERSE RECOVERY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
