from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ======================================================================================
# CONFIGURATION
# ======================================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

OUTPUT_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"

STOP_R = 1.00
RR = 1.75
HORIZON = 20

MAE_BOUNDARIES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

TIME_BUCKETS = [
    ("BAR_1_2", 1, 2),
    ("BAR_3_4", 3, 4),
    ("BAR_5_6", 5, 6),
    ("BAR_7_10", 7, 10),
    ("BAR_11_20", 11, 20),
]

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))

# Candidate rules.
#
# Example:
#   MAE >= 0.80R reached by bar 2
#
# means:
#   if MAE reaches 0.80R at or before bar 2,
#   classify the trade as "early invalidation".
#
# We intentionally test only a compact set.
MAX_DECISION_BARS = [2, 3, 4, 5, 6, 8, 10]

MIN_TRIGGERED_DEVELOPMENT = 10


# ======================================================================================
# UTILITY
# ======================================================================================


def print_header(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def normalise_window(value) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    try:
        return float(text)
    except ValueError:
        pass

    digits = "".join(ch for ch in text if ch.isdigit())

    if digits:
        try:
            return float(digits)
        except ValueError:
            return np.nan

    return np.nan


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def detect_path_columns(
    df: pd.DataFrame,
    prefix: str,
) -> dict[int, str]:
    found: dict[int, str] = {}

    for column in df.columns:
        if not str(column).startswith(prefix):
            continue

        suffix = str(column)[len(prefix) :]

        if suffix.endswith("R"):
            suffix = suffix[:-1]

        try:
            bar = int(suffix)
        except ValueError:
            continue

        found[bar] = column

    return dict(sorted(found.items()))


def first_crossing_bar(
    row: pd.Series,
    path_columns: dict[int, str],
    threshold: float,
) -> float:
    """
    Returns the first bar at which MAE reaches threshold.

    MAE is represented as a positive adverse excursion in R.

    Returns NaN if threshold is never reached.
    """

    for bar, column in path_columns.items():
        value = pd.to_numeric(row[column], errors="coerce")

        if pd.notna(value) and float(value) >= threshold:
            return float(bar)

    return np.nan


def value_at_bar(
    row: pd.Series,
    path_columns: dict[int, str],
    bar: int,
) -> float:
    column = path_columns.get(bar)

    if column is None:
        return np.nan

    value = pd.to_numeric(row[column], errors="coerce")

    if pd.isna(value):
        return np.nan

    return float(value)


def first_crossing_recovery_bar(
    row: pd.Series,
    close_columns: dict[int, str],
    crossing_bar: float,
    recovery_level: float,
) -> float:
    """
    After the MAE boundary is reached, find the first bar at which
    close_R recovers to the specified level.

    recovery_level is expressed in R:
        0.00 = breakeven
       -0.20 = recovery to -0.20R
        0.25 = +0.25R
        0.50 = +0.50R
    """

    if pd.isna(crossing_bar):
        return np.nan

    start_bar = int(crossing_bar)

    for bar, column in close_columns.items():
        if bar < start_bar:
            continue

        value = pd.to_numeric(row[column], errors="coerce")

        if pd.notna(value) and float(value) >= recovery_level:
            return float(bar)

    return np.nan


def classify_time_bucket(crossing_bar: float) -> str:
    if pd.isna(crossing_bar):
        return "NO_CROSS"

    bar = int(crossing_bar)

    for label, start, end in TIME_BUCKETS:
        if start <= bar <= end:
            return label

    return "OUTSIDE"


# ======================================================================================
# METRICS
# ======================================================================================


def profit_factor(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)

    if series.empty:
        return np.nan

    gross_profit = series[series > 0].sum()
    gross_loss = -series[series < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return float(gross_profit / gross_loss)


def max_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float)

    if series.empty:
        return 0.0

    cumulative = series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    return float(drawdown.min())


def summarize_returns(
    df: pd.DataFrame,
    r_column: str = "strategy_R",
) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
        }

    returns = pd.to_numeric(df[r_column], errors="coerce").fillna(0.0)

    wins = int((returns > 0).sum())
    losses = int((returns <= 0).sum())

    return {
        "trades": int(len(df)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(df)),
        "mean_R": float(returns.mean()),
        "total_R": float(returns.sum()),
        "profit_factor": profit_factor(returns),
        "max_drawdown_R": max_drawdown(returns),
    }


# ======================================================================================
# DATA LOADING
# ======================================================================================


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found:\n{INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise RuntimeError("Input dataset is empty.")

    return df


# ======================================================================================
# PATH ENRICHMENT
# ======================================================================================


def build_crossing_features(
    df: pd.DataFrame,
    mae_columns: dict[int, str],
    close_columns: dict[int, str],
) -> pd.DataFrame:

    result = df.copy()

    for threshold in MAE_BOUNDARIES:
        key = f"{threshold:.2f}".replace(".", "_")

        result[f"cross_bar_mae_{key}"] = result.apply(
            lambda row: first_crossing_bar(
                row,
                mae_columns,
                threshold,
            ),
            axis=1,
        )

        result[f"cross_bucket_mae_{key}"] = result[f"cross_bar_mae_{key}"].apply(
            classify_time_bucket
        )

        for recovery_level in [-0.20, 0.00, 0.25, 0.50, 0.75, 1.00]:
            recovery_key = (
                f"{recovery_level:+.2f}".replace("+", "p")
                .replace("-", "m")
                .replace(".", "_")
            )

            result[f"recovery_{key}_{recovery_key}"] = result.apply(
                lambda row: first_crossing_recovery_bar(
                    row,
                    close_columns,
                    row[f"cross_bar_mae_{key}"],
                    recovery_level,
                ),
                axis=1,
            )

    return result


# ======================================================================================
# 1. DISCOVERY: MAE × TIME
# ======================================================================================


def discovery_analysis(
    df: pd.DataFrame,
    close_columns: dict[int, str],
) -> pd.DataFrame:

    print_header("1. MAE × TIME FAILURE EVIDENCE")

    rows = []

    final_r = numeric_series(df, "final_close_R")

    for threshold in MAE_BOUNDARIES:
        threshold_key = f"{threshold:.2f}".replace(".", "_")
        crossing_col = f"cross_bar_mae_{threshold_key}"
        bucket_col = f"cross_bucket_mae_{threshold_key}"

        for bucket_label, start_bar, end_bar in TIME_BUCKETS:
            mask = (
                df[crossing_col].notna()
                & (df[crossing_col] >= start_bar)
                & (df[crossing_col] <= end_bar)
            )

            cohort = df.loc[mask].copy()

            if cohort.empty:
                continue

            cohort_final = pd.to_numeric(
                cohort["final_close_R"],
                errors="coerce",
            )

            positive = cohort_final > 0
            negative = cohort_final <= 0

            recovery_0 = 0
            recovery_025 = 0
            recovery_050 = 0
            recovery_075 = 0
            recovery_100 = 0

            for value in cohort[f"recovery_{threshold_key}_m0_20"].notna():
                recovery_0 += int(value)

            for value in cohort[f"recovery_{threshold_key}_p0_25"].notna():
                recovery_025 += int(value)

            for value in cohort[f"recovery_{threshold_key}_p0_50"].notna():
                recovery_050 += int(value)

            for value in cohort[f"recovery_{threshold_key}_p0_75"].notna():
                recovery_075 += int(value)

            for value in cohort[f"recovery_{threshold_key}_p1_00"].notna():
                recovery_100 += int(value)

            rows.append(
                {
                    "mae_boundary_R": threshold,
                    "time_bucket": bucket_label,
                    "start_bar": start_bar,
                    "end_bar": end_bar,
                    "trades": len(cohort),
                    "finished_positive": int(positive.sum()),
                    "finished_negative": int(negative.sum()),
                    "failure_pct": float(negative.mean()),
                    "survival_pct": float(positive.mean()),
                    "mean_final_R": float(cohort_final.mean()),
                    "total_final_R": float(cohort_final.sum()),
                    "recovery_0R_pct": recovery_0 / len(cohort),
                    "recovery_0_25R_pct": recovery_025 / len(cohort),
                    "recovery_0_50R_pct": recovery_050 / len(cohort),
                    "recovery_0_75R_pct": recovery_075 / len(cohort),
                    "recovery_1R_pct": recovery_100 / len(cohort),
                    "mean_crossing_bar": float(
                        pd.to_numeric(
                            cohort[crossing_col],
                            errors="coerce",
                        ).mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No MAE × time cohorts could be constructed.")

    print(result.to_string(index=False))

    return result


# ======================================================================================
# RULE APPLICATION
# ======================================================================================


def apply_rule(
    df: pd.DataFrame,
    mae_boundary: float,
    max_decision_bar: int,
) -> pd.DataFrame:
    """
    Counterfactual early-invalidation rule.

    If the trade reaches MAE >= boundary by max_decision_bar,
    the strategy exits at the boundary.

    Otherwise the original final result is retained.

    This is intentionally simple:
        - no recovery logic
        - no re-entry
        - no parameter optimization after the fact
    """

    threshold_key = f"{mae_boundary:.2f}".replace(".", "_")

    crossing_col = f"cross_bar_mae_{threshold_key}"

    result = df.copy()

    crossing = pd.to_numeric(
        result[crossing_col],
        errors="coerce",
    )

    triggered = crossing.notna() & (crossing <= max_decision_bar)

    result["triggered"] = triggered

    original_r = pd.to_numeric(
        result["final_close_R"],
        errors="coerce",
    ).fillna(0.0)

    # Conservative boundary execution:
    # once the MAE boundary is crossed, we model the exit at -boundary R.
    strategy_r = original_r.copy()
    strategy_r.loc[triggered] = -float(mae_boundary)

    result["strategy_R"] = strategy_r

    return result


# ======================================================================================
# TEMPORAL EVALUATION
# ======================================================================================


def evaluate_rule(
    df: pd.DataFrame,
    mae_boundary: float,
    max_decision_bar: int,
    windows: list[int],
) -> tuple[dict, pd.DataFrame]:

    subset = df[df["_window_numeric"].isin(windows)].copy()

    strategy = apply_rule(
        subset,
        mae_boundary,
        max_decision_bar,
    )

    summary = summarize_returns(
        strategy,
        "strategy_R",
    )

    benchmark = summarize_returns(
        strategy.assign(strategy_R=strategy["final_close_R"]),
        "strategy_R",
    )

    triggered = int(strategy["triggered"].sum())

    summary["triggered_trades"] = triggered
    summary["benchmark_R"] = benchmark["total_R"]
    summary["delta_R"] = summary["total_R"] - benchmark["total_R"]

    summary["delta_mean_R"] = summary["mean_R"] - benchmark["mean_R"]

    summary["delta_win_rate"] = summary["win_rate"] - benchmark["win_rate"]

    summary["delta_max_DD_R"] = summary["max_drawdown_R"] - benchmark["max_drawdown_R"]

    return summary, strategy


# ======================================================================================
# DEVELOPMENT SEARCH
# ======================================================================================


def development_search(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("2. DEVELOPMENT SEARCH — MAE × TIME INVALIDATION")

    rows = []

    for threshold in MAE_BOUNDARIES:
        for decision_bar in MAX_DECISION_BARS:
            summary, strategy = evaluate_rule(
                df,
                threshold,
                decision_bar,
                DEVELOPMENT_WINDOWS,
            )

            if summary["triggered_trades"] < MIN_TRIGGERED_DEVELOPMENT:
                continue

            rows.append(
                {
                    "mae_boundary_R": threshold,
                    "decision_bar": decision_bar,
                    "development_trades": summary["trades"],
                    "triggered_trades": summary["triggered_trades"],
                    "development_WR": summary["win_rate"],
                    "development_mean_R": summary["mean_R"],
                    "development_R": summary["total_R"],
                    "development_PF": summary["profit_factor"],
                    "development_DD": summary["max_drawdown_R"],
                    "benchmark_R": summary["benchmark_R"],
                    "development_delta_R": summary["delta_R"],
                    "development_delta_WR": summary["delta_win_rate"],
                    "development_delta_DD": summary["delta_max_DD_R"],
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No valid development rules survived the minimum trigger count."
        )

    # Primary objective:
    # maximize development delta_R.
    #
    # Secondary objectives:
    # minimize drawdown and prefer more triggered observations.
    result = result.sort_values(
        by=[
            "development_delta_R",
            "development_delta_DD",
            "triggered_trades",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    print(result.to_string(index=False))

    return result


# ======================================================================================
# WINDOW ANALYSIS
# ======================================================================================


def window_analysis(
    df: pd.DataFrame,
    mae_boundary: float,
    decision_bar: int,
    windows: list[int],
) -> pd.DataFrame:

    rows = []

    for window in windows:
        subset = df[df["_window_numeric"] == window].copy()

        if subset.empty:
            continue

        strategy = apply_rule(
            subset,
            mae_boundary,
            decision_bar,
        )

        benchmark_r = pd.to_numeric(
            strategy["final_close_R"],
            errors="coerce",
        ).fillna(0.0)

        strategy_r = pd.to_numeric(
            strategy["strategy_R"],
            errors="coerce",
        ).fillna(0.0)

        rows.append(
            {
                "window": window,
                "trades": len(strategy),
                "triggered_trades": int(strategy["triggered"].sum()),
                "benchmark_R": float(benchmark_r.sum()),
                "strategy_R": float(strategy_r.sum()),
                "delta_R": float(strategy_r.sum() - benchmark_r.sum()),
                "benchmark_WR": float((benchmark_r > 0).mean()),
                "strategy_WR": float((strategy_r > 0).mean()),
                "benchmark_PF": profit_factor(benchmark_r),
                "strategy_PF": profit_factor(strategy_r),
                "benchmark_DD": max_drawdown(benchmark_r),
                "strategy_DD": max_drawdown(strategy_r),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================================
# OOS AUDIT
# ======================================================================================


def trigger_audit(
    df: pd.DataFrame,
    mae_boundary: float,
    decision_bar: int,
) -> pd.DataFrame:

    subset = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    strategy = apply_rule(
        subset,
        mae_boundary,
        decision_bar,
    )

    threshold_key = f"{mae_boundary:.2f}".replace(".", "_")

    crossing_col = f"cross_bar_mae_{threshold_key}"

    audit = strategy.loc[strategy["triggered"]].copy()

    if audit.empty:
        return pd.DataFrame()

    audit["crossing_bar"] = audit[crossing_col]

    audit["original_R"] = pd.to_numeric(
        audit["final_close_R"],
        errors="coerce",
    )

    audit["strategy_R"] = pd.to_numeric(
        audit["strategy_R"],
        errors="coerce",
    )

    audit["saved_R"] = audit["strategy_R"] - audit["original_R"]

    columns = [
        column
        for column in [
            "trade_index",
            "window",
            "crossing_bar",
            "original_R",
            "strategy_R",
            "saved_R",
        ]
        if column in audit.columns
    ]

    return audit[columns].sort_values(by=["window", "trade_index"])


# ======================================================================================
# MAIN
# ======================================================================================


def main() -> None:

    print_header("S16 MAE × TIME INVALIDATION — TEMPORAL OOS TEST")

    print(
        """
CORE HYPOTHESIS:
A sufficiently large MAE is associated with trade failure,
but the predictive power may depend on HOW QUICKLY that MAE occurs.

This test asks:

    P(final loss | MAE >= X R by bar N)

and then tests the strongest compact MAE × TIME rule temporally OOS.

No recovery optimization.
No re-entry.
No post-OOS parameter adjustment.
"""
    )

    print("Frozen benchmark:")
    print(f"  Original stop = {STOP_R:.2f}R")
    print(f"  RR            = {RR}")
    print(f"  Horizon       = {HORIZON} bars")

    print()
    print("MAE boundaries:")
    print("  " + ", ".join(f"{x:.2f}R" for x in MAE_BOUNDARIES))

    print()
    print("Decision bars:")
    print("  " + ", ".join(str(x) for x in MAX_DECISION_BARS))

    print()
    print(f"Development windows : {DEVELOPMENT_WINDOWS}")
    print(f"Holdout windows     : {HOLDOUT_WINDOWS}")

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    print_header("LOADING ENRICHED DATASET")

    print(INPUT_PATH)

    df = load_dataset()

    print(f"Trades loaded: {len(df)}")

    required = [
        "final_close_R",
        "window",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------

    mae_columns = detect_path_columns(
        df,
        "mae_",
    )

    close_columns = detect_path_columns(
        df,
        "close_",
    )

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_columns)}")

    if mae_columns:
        print(f"  MAE range  : {min(mae_columns)} -> {max(mae_columns)}")

    print(f"  Close bars : {len(close_columns)}")

    if close_columns:
        print(f"  Close range: {min(close_columns)} -> {max(close_columns)}")

    if not mae_columns:
        raise RuntimeError("No per-bar MAE columns found.")

    if not close_columns:
        raise RuntimeError("No per-bar close columns found.")

    # ------------------------------------------------------------------
    # NORMALISE WINDOW
    # ------------------------------------------------------------------

    df["_window_numeric"] = df["window"].apply(normalise_window)

    # ------------------------------------------------------------------
    # BUILD CROSSINGS
    # ------------------------------------------------------------------

    print_header("BUILDING MAE × TIME CROSSING FEATURES")

    df = build_crossing_features(
        df,
        mae_columns,
        close_columns,
    )

    print("Crossing features built.")

    # ------------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------------

    discovery_df = discovery_analysis(
        df,
        close_columns,
    )

    # ------------------------------------------------------------------
    # DEVELOPMENT
    # ------------------------------------------------------------------

    development_df = development_search(df)

    best = development_df.iloc[0]

    best_boundary = float(best["mae_boundary_R"])

    best_decision_bar = int(best["decision_bar"])

    print_header("SELECTED DEVELOPMENT RULE")

    print(f"MAE boundary       : {best_boundary:.2f}R")
    print(f"Decision bar       : {best_decision_bar}")
    print(f"Triggered trades   : {int(best['triggered_trades'])}")
    print(f"Development WR     : {best['development_WR']:.4f}")
    print(f"Development R      : {best['development_R']:.4f}")
    print(f"Development ΔR     : {best['development_delta_R']:.4f}")
    print(f"Development PF     : {best['development_PF']}")
    print(f"Development DD     : {best['development_DD']:.4f}")

    # ------------------------------------------------------------------
    # OOS
    # ------------------------------------------------------------------

    print_header("3. HOLDOUT OOS TEST")

    print("Frozen OOS rule:")
    print(f"  MAE boundary = {best_boundary:.2f}R")
    print(f"  Decision bar = {best_decision_bar}")

    oos_summary, oos_strategy = evaluate_rule(
        df,
        best_boundary,
        best_decision_bar,
        HOLDOUT_WINDOWS,
    )

    benchmark_oos = summarize_returns(
        oos_strategy.assign(strategy_R=oos_strategy["final_close_R"]),
        "strategy_R",
    )

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades       : {benchmark_oos['trades']}")
    print(f"  Win rate     : {benchmark_oos['win_rate']:.4f}")
    print(f"  Mean R       : {benchmark_oos['mean_R']:.4f}")
    print(f"  Total R      : {benchmark_oos['total_R']:.4f}")
    print(f"  PF           : {benchmark_oos['profit_factor']}")
    print(f"  Max DD       : {benchmark_oos['max_drawdown_R']:.4f}")

    print()
    print("MAE × TIME FILTER HOLDOUT")

    print(f"  Trades       : {oos_summary['trades']}")
    print(f"  Triggered    : {oos_summary['triggered_trades']}")
    print(f"  Win rate     : {oos_summary['win_rate']:.4f}")
    print(f"  Mean R       : {oos_summary['mean_R']:.4f}")
    print(f"  Total R      : {oos_summary['total_R']:.4f}")
    print(f"  PF           : {oos_summary['profit_factor']}")
    print(f"  Max DD       : {oos_summary['max_drawdown_R']:.4f}")

    print()
    print("OOS IMPROVEMENT")

    print(f"  Delta R      : {oos_summary['delta_R']:.4f}")
    print(f"  Delta Mean R : {oos_summary['delta_mean_R']:.4f}")
    print(f"  Delta WR     : {oos_summary['delta_win_rate']:.4f}")
    print(f"  Delta Max DD : {oos_summary['delta_max_DD_R']:.4f}")

    # ------------------------------------------------------------------
    # WINDOW BY WINDOW
    # ------------------------------------------------------------------

    print_header("4. WINDOW-BY-WINDOW OOS")

    oos_by_window = window_analysis(
        df,
        best_boundary,
        best_decision_bar,
        HOLDOUT_WINDOWS,
    )

    print(oos_by_window.to_string(index=False))

    # ------------------------------------------------------------------
    # CROSSING DISTRIBUTION
    # ------------------------------------------------------------------

    print_header("5. OOS CROSSING-TIME DISTRIBUTION")

    threshold_key = f"{best_boundary:.2f}".replace(".", "_")

    crossing_col = f"cross_bar_mae_{threshold_key}"

    crossing_oos = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    crossing_oos = crossing_oos[crossing_oos[crossing_col].notna()]

    distribution = (
        crossing_oos[crossing_col]
        .value_counts()
        .sort_index()
        .rename_axis("crossing_bar")
        .reset_index(name="trades")
    )

    print(distribution.to_string(index=False))

    # ------------------------------------------------------------------
    # COUNTERFACTUAL RECOVERY
    # ------------------------------------------------------------------

    print_header("6. COUNTERFACTUAL RECOVERY OF EARLY INVALIDATIONS")

    triggered = oos_strategy[oos_strategy["triggered"]].copy()

    if triggered.empty:
        recovery_summary = pd.DataFrame(
            columns=[
                "recovery_level_R",
                "trades",
                "recovered_trades",
                "recovery_pct",
            ]
        )
    else:
        recovery_rows = []

        for recovery_level in [
            -0.20,
            0.00,
            0.25,
            0.50,
            0.75,
            1.00,
        ]:
            recovery_col = f"recovery_{threshold_key}_" + (
                f"{recovery_level:+.2f}".replace("+", "p")
                .replace("-", "m")
                .replace(".", "_")
            )

            if recovery_col not in triggered.columns:
                continue

            recovered = triggered[recovery_col].notna()

            recovery_rows.append(
                {
                    "recovery_level_R": recovery_level,
                    "trades": len(triggered),
                    "recovered_trades": int(recovered.sum()),
                    "recovery_pct": float(recovered.mean()),
                }
            )

        recovery_summary = pd.DataFrame(recovery_rows)

    if not recovery_summary.empty:
        print(recovery_summary.to_string(index=False))
    else:
        print("No recovery observations available.")

    # ------------------------------------------------------------------
    # TRIGGER AUDIT
    # ------------------------------------------------------------------

    print_header("7. TRIGGER AUDIT")

    audit = trigger_audit(
        df,
        best_boundary,
        best_decision_bar,
    )

    if audit.empty:
        print("No OOS trades were triggered.")
    else:
        print(f"Triggered OOS trades: {len(audit)}")

        original_winners = int((audit["original_R"] > 0).sum())

        original_losers = int((audit["original_R"] <= 0).sum())

        print(f"Original winners among triggered: {original_winners}")

        print(f"Original losers among triggered : {original_losers}")

        print()

        print(audit.to_string(index=False))

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_path = OUTPUT_DIR / "s16_mae_time_evidence.csv"

    development_path = OUTPUT_DIR / "s16_mae_time_development.csv"

    oos_path = OUTPUT_DIR / "s16_mae_time_oos_trades.csv"

    window_path = OUTPUT_DIR / "s16_mae_time_oos_by_window.csv"

    distribution_path = OUTPUT_DIR / "s16_mae_time_crossing_distribution.csv"

    recovery_path = OUTPUT_DIR / "s16_mae_time_recovery.csv"

    audit_path = OUTPUT_DIR / "s16_mae_time_trigger_audit.csv"

    discovery_df.to_csv(
        evidence_path,
        index=False,
    )

    development_df.to_csv(
        development_path,
        index=False,
    )

    oos_strategy.to_csv(
        oos_path,
        index=False,
    )

    oos_by_window.to_csv(
        window_path,
        index=False,
    )

    distribution.to_csv(
        distribution_path,
        index=False,
    )

    recovery_summary.to_csv(
        recovery_path,
        index=False,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print_header("FILES SAVED")

    print(evidence_path)
    print(development_path)
    print(oos_path)
    print(window_path)
    print(distribution_path)
    print(recovery_path)
    print(audit_path)

    print()
    print("=" * 110)
    print("S16 MAE × TIME INVALIDATION OOS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
