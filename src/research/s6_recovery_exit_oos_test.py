"""
S6 RECOVERY EXIT OOS TEST

Goal
----
Validate whether the S5 recovery mechanism survives an unseen
temporal holdout.

Frozen benchmark:
    HMM state       = 2
    Lower tail      = 17.5%
    Quality         >= 0.75
    Volatility      = 40-60%
    Stop            = 25 points
    RR              = 1.75
    Horizon         = 20 bars

S5 hypothesis:
    If a trade reaches >= 0.75R MAE by bar 8 and has not recovered
    sufficiently by a later decision bar, exit early.

Primary candidate discovered in S5:
    CLOSE at decision bar 12 < +0.25R

IMPORTANT
---------
S5 was exploratory and used the complete adverse cohort.

S6 explicitly prevents using the holdout period to select the rule.

Development:
    Windows 1-11

Holdout:
    Windows 12-22

The rule is selected ONLY on development windows.
Then it is frozen and evaluated on the holdout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "src" / "research" / "results" / "s2_extended"

INPUT_FILE = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "s6_recovery_exit_oos_summary.csv"

OUTPUT_TRADES = RESULTS_DIR / "s6_recovery_exit_oos_trades.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "s6_recovery_exit_oos_by_window.csv"


# =============================================================================
# FROZEN BENCHMARK
# =============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

ADVERSE_BAR = 8
ADVERSE_THRESHOLD = 0.75

COST_POINTS = 1.11
COST_R = COST_POINTS / STOP_POINTS


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# RULE GRID
# =============================================================================

DECISION_BARS = [
    8,
    10,
    12,
    14,
    16,
]

CLOSE_THRESHOLDS = [
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
]

MFE_THRESHOLDS = [
    0.25,
    0.50,
    0.75,
    1.00,
]


# =============================================================================
# UTILITY
# =============================================================================


def calculate_profit_factor(r: pd.Series) -> float:

    r = pd.to_numeric(
        r,
        errors="coerce",
    ).dropna()

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss <= 0:
        if gross_profit > 0:
            return float("inf")

        return np.nan

    return gross_profit / gross_loss


def calculate_max_drawdown(r: pd.Series) -> float:

    r = pd.to_numeric(
        r,
        errors="coerce",
    ).fillna(0.0)

    equity = r.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    if len(drawdown) == 0:
        return 0.0

    return float(drawdown.min())


def calculate_metrics(
    df: pd.DataFrame,
    column: str,
) -> dict:

    if len(df) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
        }

    r = pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0.0)

    window_results = df.groupby("window")[column].sum()

    return {
        "trades": len(df),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate": float((r > 0).mean()),
        "mean_R": float(r.mean()),
        "total_R": float(r.sum()),
        "profit_factor": calculate_profit_factor(r),
        "max_drawdown_R": calculate_max_drawdown(r),
        "positive_window_pct": (
            float((window_results > 0).mean()) if len(window_results) else np.nan
        ),
        "worst_window_R": (
            float(window_results.min()) if len(window_results) else np.nan
        ),
        "best_window_R": (
            float(window_results.max()) if len(window_results) else np.nan
        ),
    }


# =============================================================================
# LOAD
# =============================================================================


def load_data() -> pd.DataFrame:

    print("Loading S4 enriched benchmark...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    numeric_columns = [
        "net_R",
        "window",
        "mae_8R",
        "mfe_8R",
        "close_8R",
        "mae_10R",
        "mfe_10R",
        "close_10R",
        "mae_12R",
        "mfe_12R",
        "close_12R",
        "mae_14R",
        "mfe_14R",
        "close_14R",
        "mae_16R",
        "mfe_16R",
        "close_16R",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


# =============================================================================
# VALIDATION
# =============================================================================


def validate_columns(
    df: pd.DataFrame,
):

    required = [
        "net_R",
        "window",
        "mae_8R",
    ]

    for bar in DECISION_BARS:
        required.extend(
            [
                f"mfe_{bar}R",
                f"close_{bar}R",
                f"mae_{bar}R",
            ]
        )

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n" + "\n".join(f"  - {x}" for x in missing)
        )


# =============================================================================
# BUILD ADVERSE COHORT
# =============================================================================


def add_adverse_flag(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["adverse"] = df[f"mae_{ADVERSE_BAR}R"] >= ADVERSE_THRESHOLD

    return df


# =============================================================================
# EXIT RULE
# =============================================================================


def build_abort_mask(
    df: pd.DataFrame,
    decision_bar: int,
    close_threshold: float,
) -> pd.Series:

    close_col = f"close_{decision_bar}R"

    return df["adverse"] & (df[close_col] < close_threshold)


# =============================================================================
# APPLY RULE
# =============================================================================


def apply_rule(
    df: pd.DataFrame,
    decision_bar: int,
    close_threshold: float,
) -> pd.DataFrame:

    result = df.copy()

    abort_mask = build_abort_mask(
        result,
        decision_bar,
        close_threshold,
    )

    result["rule_triggered"] = abort_mask

    result["decision"] = np.where(
        abort_mask,
        "EARLY_EXIT",
        "HOLD",
    )

    # -------------------------------------------------------------------------
    # IMPORTANT
    #
    # close_XR represents the favorable excursion of the SHORT position
    # at that bar.
    #
    # We convert it into a net R exit by subtracting transaction costs.
    #
    # We only replace the original benchmark result when the rule actually
    # triggers.
    # -------------------------------------------------------------------------

    close_col = f"close_{decision_bar}R"

    early_exit_R = result[close_col] - COST_R

    result["strategy_R"] = np.where(
        abort_mask,
        early_exit_R,
        result["net_R"],
    )

    return result


# =============================================================================
# DEVELOPMENT SCORING
# =============================================================================


def score_rule(
    df: pd.DataFrame,
    decision_bar: int,
    close_threshold: float,
) -> dict:

    development = df.loc[df["window"].isin(DEVELOPMENT_WINDOWS)].copy()

    result = apply_rule(
        development,
        decision_bar,
        close_threshold,
    )

    metrics = calculate_metrics(
        result,
        "strategy_R",
    )

    triggered = int(result["rule_triggered"].sum())

    return {
        "decision_bar": decision_bar,
        "close_threshold": close_threshold,
        "triggered_trades": triggered,
        **metrics,
    }


# =============================================================================
# RULE SELECTION
# =============================================================================


def select_best_rule(
    df: pd.DataFrame,
) -> pd.Series:

    candidates = []

    for bar in DECISION_BARS:
        for threshold in CLOSE_THRESHOLDS:
            row = score_rule(
                df,
                bar,
                threshold,
            )

            # Avoid tiny samples.
            if row["triggered_trades"] < 10:
                continue

            candidates.append(row)

    results = pd.DataFrame(candidates)

    if results.empty:
        raise RuntimeError("No valid recovery rules.")

    # -------------------------------------------------------------------------
    # Robust selection:
    #
    # We don't simply maximize total R.
    #
    # First prioritize PF, then positive-window percentage,
    # then total R.
    # -------------------------------------------------------------------------

    results = results.sort_values(
        [
            "profit_factor",
            "positive_window_pct",
            "total_R",
        ],
        ascending=False,
    )

    return results.iloc[0], results


# =============================================================================
# WINDOW METRICS
# =============================================================================


def window_analysis(
    df: pd.DataFrame,
    decision_bar: int,
    close_threshold: float,
) -> pd.DataFrame:

    result = apply_rule(
        df,
        decision_bar,
        close_threshold,
    )

    rows = []

    for window, group in result.groupby("window"):
        r = group["strategy_R"].astype(float)

        benchmark_r = group["net_R"].astype(float)

        rows.append(
            {
                "window": window,
                "trades": len(group),
                "triggered_trades": int(group["rule_triggered"].sum()),
                "benchmark_R": float(benchmark_r.sum()),
                "strategy_R": float(r.sum()),
                "delta_R": float(r.sum() - benchmark_r.sum()),
                "benchmark_PF": calculate_profit_factor(benchmark_r),
                "strategy_PF": calculate_profit_factor(r),
                "benchmark_DD": calculate_max_drawdown(benchmark_r),
                "strategy_DD": calculate_max_drawdown(r),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S6 RECOVERY EXIT — TEMPORAL OOS TEST")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop            = {STOP_POINTS} points")
    print(f"  RR              = {RR}")
    print(f"  Horizon         = {HORIZON} bars")
    print(f"  Adverse trigger = MAE >= {ADVERSE_THRESHOLD}R at bar {ADVERSE_BAR}")

    print()
    print(f"Development windows : {DEVELOPMENT_WINDOWS}")

    print(f"Holdout windows     : {HOLDOUT_WINDOWS}")

    df = load_data()

    validate_columns(df)

    df = add_adverse_flag(df)

    print()
    print("=" * 110)
    print("ADVERSE COHORT")
    print("=" * 110)

    print(
        "Total trades       :",
        len(df),
    )

    print(
        "Adverse trades     :",
        int(df["adverse"].sum()),
    )

    # =========================================================================
    # DEVELOPMENT
    # =========================================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    best, development_results = select_best_rule(df)

    print()
    print(development_results.head(20).to_string(index=False))

    print()
    print("SELECTED DEVELOPMENT RULE")

    print(f"Decision bar     : {int(best['decision_bar'])}")

    print(f"Close threshold  : {best['close_threshold']}")

    print(f"Triggered trades : {int(best['triggered_trades'])}")

    print(f"Development PF   : {best['profit_factor']:.4f}")

    print(f"Development R    : {best['total_R']:.4f}")

    print(f"Development DD   : {best['max_drawdown_R']:.4f}")

    # =========================================================================
    # FREEZE RULE
    # =========================================================================

    selected_bar = int(best["decision_bar"])

    selected_threshold = float(best["close_threshold"])

    # =========================================================================
    # FULL BENCHMARK
    # =========================================================================

    benchmark_metrics = calculate_metrics(
        df,
        "net_R",
    )

    # =========================================================================
    # HOLDOUT
    # =========================================================================

    holdout = df.loc[df["window"].isin(HOLDOUT_WINDOWS)].copy()

    holdout_result = apply_rule(
        holdout,
        selected_bar,
        selected_threshold,
    )

    holdout_metrics = calculate_metrics(
        holdout_result,
        "strategy_R",
    )

    holdout_benchmark_metrics = calculate_metrics(
        holdout,
        "net_R",
    )

    # =========================================================================
    # PRINT HOLDOUT
    # =========================================================================

    print()
    print("=" * 110)
    print("HOLDOUT OOS RESULT")
    print("=" * 110)

    print("Frozen rule:")

    print(f"  Decision bar    = {selected_bar}")

    print(f"  Close threshold = {selected_threshold}")

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades          : {holdout_benchmark_metrics['trades']}")

    print(f"  Win rate        : {holdout_benchmark_metrics['win_rate']:.4f}")

    print(f"  Total R         : {holdout_benchmark_metrics['total_R']:.4f}")

    print(f"  PF              : {holdout_benchmark_metrics['profit_factor']:.4f}")

    print(f"  Max DD          : {holdout_benchmark_metrics['max_drawdown_R']:.4f}")

    print()
    print("RECOVERY EXIT HOLDOUT")

    print(f"  Trades          : {holdout_metrics['trades']}")

    print(f"  Win rate        : {holdout_metrics['win_rate']:.4f}")

    print(f"  Total R         : {holdout_metrics['total_R']:.4f}")

    print(f"  PF              : {holdout_metrics['profit_factor']:.4f}")

    print(f"  Max DD          : {holdout_metrics['max_drawdown_R']:.4f}")

    print()
    print("IMPROVEMENT")

    delta_R = holdout_metrics["total_R"] - holdout_benchmark_metrics["total_R"]

    delta_PF = (
        holdout_metrics["profit_factor"] - holdout_benchmark_metrics["profit_factor"]
    )

    delta_DD = (
        holdout_metrics["max_drawdown_R"] - holdout_benchmark_metrics["max_drawdown_R"]
    )

    print(f"  Delta R         : {delta_R:+.4f}")

    print(f"  Delta PF        : {delta_PF:+.4f}")

    print(f"  Delta Max DD    : {delta_DD:+.4f}")

    # =========================================================================
    # WINDOW ANALYSIS
    # =========================================================================

    window_results = window_analysis(
        df,
        selected_bar,
        selected_threshold,
    )

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW HOLDOUT")
    print("=" * 110)

    print(
        window_results[window_results["window"].isin(HOLDOUT_WINDOWS)].to_string(
            index=False
        )
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        [
            {
                "selected_decision_bar": selected_bar,
                "selected_close_threshold": selected_threshold,
                "development_trades": best["trades"],
                "development_total_R": best["total_R"],
                "development_PF": best["profit_factor"],
                "development_max_DD": best["max_drawdown_R"],
                "holdout_benchmark_R": holdout_benchmark_metrics["total_R"],
                "holdout_benchmark_PF": holdout_benchmark_metrics["profit_factor"],
                "holdout_benchmark_DD": holdout_benchmark_metrics["max_drawdown_R"],
                "holdout_strategy_R": holdout_metrics["total_R"],
                "holdout_strategy_PF": holdout_metrics["profit_factor"],
                "holdout_strategy_DD": holdout_metrics["max_drawdown_R"],
                "holdout_delta_R": delta_R,
                "holdout_delta_PF": delta_PF,
                "holdout_delta_DD": delta_DD,
                "holdout_triggered_trades": int(holdout_result["rule_triggered"].sum()),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    holdout_result.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    window_results.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(OUTPUT_SUMMARY)

    print(OUTPUT_TRADES)

    print(OUTPUT_WINDOWS)

    print()
    print("S6 RECOVERY EXIT OOS TEST COMPLETE")


if __name__ == "__main__":
    main()
