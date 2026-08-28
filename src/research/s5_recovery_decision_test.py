"""
S5 RECOVERY DECISION TEST

Purpose
-------
Test whether early post-adverse price behavior can distinguish
recovering trades from failing trades.

Frozen benchmark:
    HMM state       = 2
    Lower tail      = 17.5%
    Quality         >= 0.75
    Volatility      = 40-60%
    Stop            = 25 points
    RR              = 1.75
    Horizon         = 20 bars

Important
---------
This is a RESEARCH TEST.

We do NOT change the production strategy here.

We first identify trades that reach an adverse MAE threshold
and then retrospectively test decision rules based only on
information available up to the decision bar.

The main question:

    "Once a trade reaches >= 0.75R adverse excursion,
     can early recovery behavior tell us whether to HOLD or ABORT?"
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "src" / "research" / "results" / "s2_extended"

INPUT_FILE = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "s5_recovery_decision_summary.csv"
OUTPUT_TRADES = RESULTS_DIR / "s5_recovery_decision_trades.csv"
OUTPUT_WINDOWS = RESULTS_DIR / "s5_recovery_decision_by_window.csv"


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

ADVERSE_THRESHOLD = 0.75
DECISION_BARS = [6, 8, 10, 12]

# Candidate thresholds
CLOSE_THRESHOLDS = [-0.75, -0.50, -0.25, 0.00, 0.25]
MFE_THRESHOLDS = [0.25, 0.50, 0.75, 1.00]
MAE_THRESHOLDS = [0.75, 1.00]

RECOVERY_CLOSE_THRESHOLDS = [0.00, 0.25, 0.50]
RECOVERY_MFE_THRESHOLDS = [0.50, 0.75, 1.00]

# Minimum sample size for a candidate rule.
MIN_TRIGGERED = 15


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert selected columns to numeric when present.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def calculate_profit_factor(r: pd.Series) -> float:
    """
    Profit factor = gross profits / gross losses.
    """
    r = pd.to_numeric(r, errors="coerce").dropna()

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return np.nan

    return gross_profit / gross_loss


def calculate_max_drawdown(r: pd.Series) -> float:
    """
    Maximum drawdown of cumulative R.
    """
    r = pd.to_numeric(r, errors="coerce").fillna(0.0)

    equity = r.cumsum()
    running_max = equity.cummax()

    drawdown = equity - running_max

    if len(drawdown) == 0:
        return 0.0

    return float(drawdown.min())


def calculate_longest_losing_streak(r: pd.Series) -> int:
    """
    Longest consecutive sequence of negative R trades.
    """
    longest = 0
    current = 0

    for value in pd.to_numeric(r, errors="coerce").fillna(0.0):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


# =============================================================================
# LOAD DATA
# =============================================================================


def load_data() -> pd.DataFrame:
    print("Loading S4 enriched dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    trades = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(trades)}")

    trades = safe_numeric(
        trades,
        [
            "net_R",
            "quality",
            "vol_percentile",
            "window",
            "max_MAE_R",
            "max_MFE_R",
            "final_close_R",
            "time_to_max_MAE",
            "time_to_max_MFE",
        ],
    )

    return trades


# =============================================================================
# REQUIRED PATH FEATURES
# =============================================================================


def validate_path_features(trades: pd.DataFrame) -> None:
    required = [
        "net_R",
        "window",
        "mae_6R",
        "mfe_6R",
        "close_6R",
        "mae_8R",
        "mfe_8R",
        "close_8R",
        "mae_10R",
        "mfe_10R",
        "close_10R",
        "mae_12R",
        "mfe_12R",
        "close_12R",
    ]

    missing = [col for col in required if col not in trades.columns]

    if missing:
        raise RuntimeError(
            "Missing required path features:\n" + "\n".join(f"  - {x}" for x in missing)
        )


# =============================================================================
# ADVERSE COHORT
# =============================================================================


def build_adverse_cohort(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Define the adverse cohort exactly as S4:

        MAE at decision bar >= 0.75R

    We use bar 8 as the primary adverse definition.
    """

    trades = trades.copy()

    trades["adverse"] = trades["mae_8R"] >= ADVERSE_THRESHOLD

    adverse = trades.loc[trades["adverse"]].copy()

    print()
    print("=" * 110)
    print("ADVERSE COHORT")
    print("=" * 110)

    print(f"Total benchmark trades : {len(trades)}")
    print(f"Adverse trades         : {len(adverse)}")

    return adverse


# =============================================================================
# ORIGINAL OUTCOME
# =============================================================================


def classify_original_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recovery:
        adverse at bar 8 AND final trade result > 0

    Failure:
        adverse at bar 8 AND final trade result <= 0
    """

    df = df.copy()

    df["original_outcome"] = np.where(
        df["net_R"] > 0,
        "RECOVERY",
        "FAILURE",
    )

    return df


# =============================================================================
# DECISION RULES
# =============================================================================


def rule_close_below(
    df: pd.DataFrame,
    bar: int,
    threshold: float,
) -> pd.Series:
    """
    Abort if close_R at decision bar is below threshold.
    """
    col = f"close_{bar}R"

    return df[col] < threshold


def rule_mfe_below(
    df: pd.DataFrame,
    bar: int,
    threshold: float,
) -> pd.Series:
    """
    Abort if MFE_R at decision bar is below threshold.
    """
    col = f"mfe_{bar}R"

    return df[col] < threshold


def rule_mae_above(
    df: pd.DataFrame,
    bar: int,
    threshold: float,
) -> pd.Series:
    """
    Abort if MAE_R at decision bar is above threshold.
    """
    col = f"mae_{bar}R"

    return df[col] >= threshold


def rule_close_and_mfe(
    df: pd.DataFrame,
    bar: int,
    close_threshold: float,
    mfe_threshold: float,
) -> pd.Series:
    """
    Abort if both:
        close_R < close threshold
        MFE_R < MFE threshold
    """

    close_col = f"close_{bar}R"
    mfe_col = f"mfe_{bar}R"

    return (df[close_col] < close_threshold) & (df[mfe_col] < mfe_threshold)


def rule_recovery_confirmation(
    df: pd.DataFrame,
    bar: int,
    close_threshold: float,
    mfe_threshold: float,
) -> pd.Series:
    """
    Abort unless BOTH recovery conditions are satisfied.

    Therefore:

        ABORT =
            close_R < close_threshold
            OR
            MFE_R < mfe_threshold

    HOLD requires both.
    """

    close_col = f"close_{bar}R"
    mfe_col = f"mfe_{bar}R"

    return (df[close_col] < close_threshold) | (df[mfe_col] < mfe_threshold)


# =============================================================================
# APPLY DECISION
# =============================================================================


def apply_decision_rule(
    adverse: pd.DataFrame,
    abort_mask: pd.Series,
) -> pd.DataFrame:
    """
    Apply a hypothetical decision:

        ABORT:
            replace eventual outcome with zero

        HOLD:
            keep original outcome

    Why zero?

    This models exiting the position at the decision point
    without introducing an additional execution assumption.

    The exact execution-price model will be tested separately
    if this screening stage produces a promising rule.
    """

    result = adverse.copy()

    result["abort"] = abort_mask.fillna(False)

    result["decision"] = np.where(
        result["abort"],
        "ABORT",
        "HOLD",
    )

    result["decision_net_R"] = np.where(
        result["abort"],
        0.0,
        result["net_R"],
    )

    return result


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(
    df: pd.DataFrame,
    r_column: str,
) -> dict:
    """
    Calculate portfolio-level metrics.
    """

    if len(df) == 0:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "positive_window_pct": np.nan,
            "worst_window_R": np.nan,
            "best_window_R": np.nan,
            "longest_losing_streak": 0,
        }

    r = pd.to_numeric(
        df[r_column],
        errors="coerce",
    ).fillna(0.0)

    wins = (r > 0).sum()

    window_results = df.groupby("window")[r_column].sum()

    return {
        "trades": len(df),
        "win_rate": wins / len(df),
        "mean_R": r.mean(),
        "total_R": r.sum(),
        "profit_factor": calculate_profit_factor(r),
        "max_drawdown_R": calculate_max_drawdown(r),
        "positive_window_pct": (
            (window_results > 0).mean() if len(window_results) else np.nan
        ),
        "worst_window_R": (window_results.min() if len(window_results) else np.nan),
        "best_window_R": (window_results.max() if len(window_results) else np.nan),
        "longest_losing_streak": (calculate_longest_losing_streak(r)),
    }


# =============================================================================
# BASELINE
# =============================================================================


def build_baseline(adverse: pd.DataFrame) -> dict:
    """
    Baseline = do nothing.

    Every adverse trade remains in the original strategy.
    """

    metrics = calculate_metrics(
        adverse,
        "net_R",
    )

    return {
        "rule_type": "BASELINE_ADVERSE_COHORT",
        "decision_bar": np.nan,
        "threshold_a": np.nan,
        "threshold_b": np.nan,
        "triggered_trades": 0,
        **metrics,
    }


# =============================================================================
# TEST SINGLE RULE
# =============================================================================


def evaluate_rule(
    adverse: pd.DataFrame,
    rule_type: str,
    decision_bar: int,
    threshold_a: float,
    threshold_b: float | None,
    abort_mask: pd.Series,
) -> tuple[dict, pd.DataFrame]:

    result = apply_decision_rule(
        adverse,
        abort_mask,
    )

    metrics = calculate_metrics(
        result,
        "decision_net_R",
    )

    row = {
        "rule_type": rule_type,
        "decision_bar": decision_bar,
        "threshold_a": threshold_a,
        "threshold_b": (threshold_b if threshold_b is not None else np.nan),
        "triggered_trades": int(result["abort"].sum()),
        **metrics,
    }

    return row, result


# =============================================================================
# GENERATE RULE GRID
# =============================================================================


def generate_rules() -> list[dict]:
    rules = []

    # -------------------------------------------------------------------------
    # CLOSE RULES
    # -------------------------------------------------------------------------

    for bar in DECISION_BARS:
        for threshold in CLOSE_THRESHOLDS:
            rules.append(
                {
                    "rule_type": "CLOSE_BELOW",
                    "decision_bar": bar,
                    "threshold_a": threshold,
                    "threshold_b": None,
                }
            )

    # -------------------------------------------------------------------------
    # MFE RULES
    # -------------------------------------------------------------------------

    for bar in DECISION_BARS:
        for threshold in MFE_THRESHOLDS:
            rules.append(
                {
                    "rule_type": "MFE_BELOW",
                    "decision_bar": bar,
                    "threshold_a": threshold,
                    "threshold_b": None,
                }
            )

    # -------------------------------------------------------------------------
    # MAE RULES
    # -------------------------------------------------------------------------

    for bar in DECISION_BARS:
        for threshold in MAE_THRESHOLDS:
            rules.append(
                {
                    "rule_type": "MAE_ABOVE",
                    "decision_bar": bar,
                    "threshold_a": threshold,
                    "threshold_b": None,
                }
            )

    # -------------------------------------------------------------------------
    # CLOSE + MFE
    # -------------------------------------------------------------------------

    for bar in DECISION_BARS:
        for close_threshold in CLOSE_THRESHOLDS:
            for mfe_threshold in MFE_THRESHOLDS:
                rules.append(
                    {
                        "rule_type": "CLOSE_AND_MFE",
                        "decision_bar": bar,
                        "threshold_a": close_threshold,
                        "threshold_b": mfe_threshold,
                    }
                )

    # -------------------------------------------------------------------------
    # RECOVERY CONFIRMATION
    # -------------------------------------------------------------------------

    for bar in DECISION_BARS:
        for close_threshold in RECOVERY_CLOSE_THRESHOLDS:
            for mfe_threshold in RECOVERY_MFE_THRESHOLDS:
                rules.append(
                    {
                        "rule_type": "RECOVERY_CONFIRMATION",
                        "decision_bar": bar,
                        "threshold_a": close_threshold,
                        "threshold_b": mfe_threshold,
                    }
                )

    return rules


# =============================================================================
# BUILD MASK
# =============================================================================


def build_abort_mask(
    df: pd.DataFrame,
    rule: dict,
) -> pd.Series:

    rule_type = rule["rule_type"]
    bar = int(rule["decision_bar"])

    a = rule["threshold_a"]
    b = rule["threshold_b"]

    if rule_type == "CLOSE_BELOW":
        return rule_close_below(
            df,
            bar,
            a,
        )

    if rule_type == "MFE_BELOW":
        return rule_mfe_below(
            df,
            bar,
            a,
        )

    if rule_type == "MAE_ABOVE":
        return rule_mae_above(
            df,
            bar,
            a,
        )

    if rule_type == "CLOSE_AND_MFE":
        return rule_close_and_mfe(
            df,
            bar,
            a,
            b,
        )

    if rule_type == "RECOVERY_CONFIRMATION":
        return rule_recovery_confirmation(
            df,
            bar,
            a,
            b,
        )

    raise ValueError(f"Unknown rule type: {rule_type}")


# =============================================================================
# WINDOW ANALYSIS
# =============================================================================


def calculate_window_metrics(
    result: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for window, group in result.groupby("window"):
        r = pd.to_numeric(
            group["decision_net_R"],
            errors="coerce",
        ).fillna(0.0)

        rows.append(
            {
                "window": window,
                "trades": len(group),
                "triggered_trades": int(group["abort"].sum()),
                "total_R": r.sum(),
                "mean_R": r.mean(),
                "win_rate": ((r > 0).mean() if len(r) else np.nan),
                "profit_factor": calculate_profit_factor(r),
                "max_drawdown_R": calculate_max_drawdown(r),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S5 RECOVERY DECISION TEST")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop            = {STOP_POINTS} points")
    print(f"  RR              = {RR}")
    print(f"  Horizon         = {HORIZON} bars")
    print(f"  Adverse trigger = MAE >= {ADVERSE_THRESHOLD}R at bar 8")

    trades = load_data()

    validate_path_features(trades)

    adverse = build_adverse_cohort(trades)

    adverse = classify_original_outcome(adverse)

    print()
    print("=" * 110)
    print("ORIGINAL ADVERSE COHORT")
    print("=" * 110)

    recovery_count = adverse["original_outcome"].eq("RECOVERY").sum()

    failure_count = adverse["original_outcome"].eq("FAILURE").sum()

    print(f"Recovery trades : {recovery_count}")
    print(f"Failure trades  : {failure_count}")

    # -------------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------------

    summary_rows = [build_baseline(adverse)]

    # -------------------------------------------------------------------------
    # TEST RULES
    # -------------------------------------------------------------------------

    rules = generate_rules()

    print()
    print(f"Testing {len(rules)} recovery decision rules...")

    all_rule_trades = []

    for i, rule in enumerate(rules, start=1):
        abort_mask = build_abort_mask(
            adverse,
            rule,
        )

        triggered = int(abort_mask.fillna(False).sum())

        # Ignore rules with extremely tiny samples.
        if triggered < MIN_TRIGGERED:
            continue

        row, result = evaluate_rule(
            adverse,
            rule["rule_type"],
            rule["decision_bar"],
            rule["threshold_a"],
            rule["threshold_b"],
            abort_mask,
        )

        summary_rows.append(row)

        result = result.copy()

        result["rule_type"] = rule["rule_type"]
        result["decision_bar"] = rule["decision_bar"]
        result["threshold_a"] = rule["threshold_a"]
        result["threshold_b"] = (
            rule["threshold_b"] if rule["threshold_b"] is not None else np.nan
        )

        all_rule_trades.append(result)

        if i % 25 == 0 or i == len(rules):
            print(f"Processing {i}/{len(rules)}...")

    summary = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # RANK RULES
    # -------------------------------------------------------------------------

    ranked = summary[summary["rule_type"] != "BASELINE_ADVERSE_COHORT"].copy()

    ranked = ranked.sort_values(
        [
            "total_R",
            "profit_factor",
            "positive_window_pct",
        ],
        ascending=False,
    )

    # -------------------------------------------------------------------------
    # BEST RULE
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BASELINE — ADVERSE COHORT")
    print("=" * 110)

    baseline = summary.iloc[0]

    print(
        baseline[
            [
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "positive_window_pct",
            ]
        ].to_string()
    )

    print()
    print("=" * 110)
    print("TOP RECOVERY DECISION RULES")
    print("=" * 110)

    if len(ranked):
        print(ranked.head(30).to_string(index=False))

    # -------------------------------------------------------------------------
    # BEST CANDIDATE
    # -------------------------------------------------------------------------

    if len(ranked):
        best = ranked.iloc[0]

        print()
        print("=" * 110)
        print("BEST CANDIDATE")
        print("=" * 110)

        print(f"Rule             : {best['rule_type']}")

        print(f"Decision bar     : {best['decision_bar']}")

        print(f"Threshold A      : {best['threshold_a']}")

        print(f"Threshold B      : {best['threshold_b']}")

        print(f"Triggered trades : {best['triggered_trades']}")

        print(f"Total R          : {best['total_R']:.4f}")

        print(f"Profit Factor    : {best['profit_factor']:.4f}")

        print(f"Max DD           : {best['max_drawdown_R']:.4f}")

        print(f"Positive windows : {best['positive_window_pct']:.4f}")

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SAVE ALL RULE TRADES
    # -------------------------------------------------------------------------

    if all_rule_trades:
        rule_trades = pd.concat(
            all_rule_trades,
            ignore_index=True,
        )

        rule_trades.to_csv(
            OUTPUT_TRADES,
            index=False,
        )

    # -------------------------------------------------------------------------
    # WINDOW ANALYSIS FOR BEST RULE
    # -------------------------------------------------------------------------

    if len(ranked) and all_rule_trades:
        best_rule = {
            "rule_type": best["rule_type"],
            "decision_bar": int(best["decision_bar"]),
            "threshold_a": best["threshold_a"],
            "threshold_b": (
                None if pd.isna(best["threshold_b"]) else best["threshold_b"]
            ),
        }

        best_mask = build_abort_mask(
            adverse,
            best_rule,
        )

        best_result = apply_decision_rule(
            adverse,
            best_mask,
        )

        window_metrics = calculate_window_metrics(best_result)

        window_metrics["rule_type"] = best_rule["rule_type"]

        window_metrics["decision_bar"] = best_rule["decision_bar"]

        window_metrics["threshold_a"] = best_rule["threshold_a"]

        window_metrics["threshold_b"] = (
            best_rule["threshold_b"] if best_rule["threshold_b"] is not None else np.nan
        )

        window_metrics.to_csv(
            OUTPUT_WINDOWS,
            index=False,
        )

    # -------------------------------------------------------------------------
    # FILES
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(OUTPUT_SUMMARY)
    print(OUTPUT_TRADES)
    print(OUTPUT_WINDOWS)

    print()
    print("S5 RECOVERY DECISION TEST COMPLETE")


if __name__ == "__main__":
    main()
