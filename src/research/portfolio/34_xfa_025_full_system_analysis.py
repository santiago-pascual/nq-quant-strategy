from __future__ import annotations

"""
34_xfa_025_full_system_analysis.py

Analyze the 0.25% XFA scenario from the corrected full-system replay.

Input:
    adaptive_orb_xfa_results_corrected.csv

The source contains:
    105 rows
    5 risk scenarios
    21 payout policies per scenario

Risk scenarios:
    0.25%
    0.50%
    0.75%
    1.00%
    0.50_to_1.00

Payout grid per risk scenario:
    20 days x 7 payout amounts
    21 days x 7 payout amounts
    22 days x 7 payout amounts

Total:
    21 policies per risk scenario
    105 rows total

This script isolates ONLY:
    policy == "0.25%"

and compares:
    STRICT_025_CORRECTED
vs
    ADAPTIVE_ORB_060_CORRECTED

No simulation is rerun.
No strategy parameter is modified.
No optimization is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "funded"

INPUT_FILE = RESULTS_DIR / "adaptive_orb_xfa_results_corrected.csv"

OUTPUT_POLICY = RESULTS_DIR / "adaptive_orb_xfa_025_policy_comparison.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "adaptive_orb_xfa_025_summary.csv"

OUTPUT_INTERVAL = RESULTS_DIR / "adaptive_orb_xfa_025_by_interval.csv"

OUTPUT_AMOUNT = RESULTS_DIR / "adaptive_orb_xfa_025_by_amount.csv"


# ============================================================================
# EXPECTED SOURCE
# ============================================================================

EXPECTED_STRICT_SCENARIO = "STRICT_025_CORRECTED"

EXPECTED_ADAPTIVE_SCENARIO = "ADAPTIVE_ORB_060_CORRECTED"

TARGET_POLICY = "0.25%"

TARGET_RISK_PERCENT = 0.25
TARGET_RISK_DOLLARS = 125.0

EXPECTED_POLICIES = 21

EXPECTED_INTERVALS = {
    20,
    21,
    22,
}

EXPECTED_PAYOUT_AMOUNTS = {
    500.0,
    750.0,
    1000.0,
    1250.0,
    1500.0,
    1750.0,
    2000.0,
}


# ============================================================================
# HELPERS
# ============================================================================


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise KeyError(
            "\nMissing required columns:\n"
            f"{missing}\n\n"
            "Available columns:\n"
            f"{list(df.columns)}"
        )


def assert_close(
    actual: pd.Series,
    expected: pd.Series,
    name: str,
    tolerance: float = 1e-8,
) -> None:

    actual = pd.to_numeric(
        actual,
        errors="coerce",
    )

    expected = pd.to_numeric(
        expected,
        errors="coerce",
    )

    difference = (actual - expected).abs()

    max_difference = difference.max()

    if pd.isna(max_difference):
        raise AssertionError(f"{name}: unable to calculate difference.")

    if max_difference > tolerance:
        raise AssertionError(f"{name} mismatch. Maximum difference = {max_difference}")


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


def policy_label(row: pd.Series) -> str:

    interval = int(row["payout_interval"])

    amount = float(row["payout_amount"])

    return f"{interval}d/${amount:,.0f}"


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("XFA 0.25% FULL-SYSTEM ANALYSIS")
print("=" * 80)

print(f"Input: {INPUT_FILE}")


# ============================================================================
# LOAD
# ============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "\nCorrected XFA replay not found:\n"
        f"{INPUT_FILE}\n\n"
        "Run 32_adaptive_orb_xfa_replay_corrected.py first."
    )


df = pd.read_csv(INPUT_FILE)

print(f"Loaded rows: {len(df):,}")


# ============================================================================
# REQUIRED SOURCE COLUMNS
# ============================================================================

required_columns = [
    "scenario_strict",
    "policy",
    "payout_interval",
    "payout_amount",
    "survival_rate_strict",
    "failure_rate_strict",
    "survived_to_max_trades_rate_strict",
    "median_payouts_strict",
    "median_total_withdrawn_strict",
    "p95_total_withdrawn_strict",
    "median_final_balance_strict",
    "median_net_value_strict",
    "median_max_DD_strict",
    "p95_DD_strict",
    "median_trades_strict",
    "mean_trades_strict",
    "median_winning_days_strict",
    "scenario_adaptive",
    "survival_rate_adaptive",
    "failure_rate_adaptive",
    "survived_to_max_trades_rate_adaptive",
    "median_payouts_adaptive",
    "median_total_withdrawn_adaptive",
    "p95_total_withdrawn_adaptive",
    "median_final_balance_adaptive",
    "median_net_value_adaptive",
    "median_max_DD_adaptive",
    "p95_DD_adaptive",
    "median_trades_adaptive",
    "mean_trades_adaptive",
    "median_winning_days_adaptive",
    "delta_survival",
    "delta_median_net_value",
    "delta_median_DD",
    "delta_median_trades",
]

require_columns(
    df,
    required_columns,
)


# ============================================================================
# SOURCE SCENARIO AUDIT
# ============================================================================

print("\n" + "=" * 80)
print("SOURCE SCENARIO AUDIT")
print("=" * 80)

strict_scenarios = df["scenario_strict"].value_counts()

adaptive_scenarios = df["scenario_adaptive"].value_counts()

print("\nSTRICT:")
print(strict_scenarios.to_string())

print("\nADAPTIVE:")
print(adaptive_scenarios.to_string())


if set(df["scenario_strict"].astype(str)) != {EXPECTED_STRICT_SCENARIO}:
    raise AssertionError(
        "Unexpected STRICT scenario.\n"
        f"Expected: {EXPECTED_STRICT_SCENARIO}\n"
        f"Found:\n{strict_scenarios}"
    )


if set(df["scenario_adaptive"].astype(str)) != {EXPECTED_ADAPTIVE_SCENARIO}:
    raise AssertionError(
        "Unexpected ADAPTIVE scenario.\n"
        f"Expected: {EXPECTED_ADAPTIVE_SCENARIO}\n"
        f"Found:\n{adaptive_scenarios}"
    )


print("\nScenario identifiers PASS.")


# ============================================================================
# POLICY GRID AUDIT BEFORE FILTER
# ============================================================================

print("\n" + "=" * 80)
print("FULL SOURCE POLICY GRID")
print("=" * 80)

policy_counts = df["policy"].value_counts()

print(policy_counts.to_string())


# ============================================================================
# FILTER ONLY 0.25%
# ============================================================================

print("\n" + "=" * 80)
print("0.25% FILTER")
print("=" * 80)

target = df.loc[df["policy"].astype(str).str.strip().eq(TARGET_POLICY)].copy()

print(f"Target policy: {TARGET_POLICY}")

print(f"Rows after filter: {len(target):,}")

# HARD GATE:
# The source contains 21 policies for every risk scenario.
if len(target) != EXPECTED_POLICIES:
    raise AssertionError(
        f"0.25% filter produced {len(target)} rows; "
        f"expected exactly {EXPECTED_POLICIES}."
    )

# Show exactly what survived the filter.
print("\nFiltered policies:")

print(
    target[
        [
            "policy",
            "payout_interval",
            "payout_amount",
        ]
    ]
    .sort_values(
        [
            "payout_interval",
            "payout_amount",
        ]
    )
    .to_string(index=False)
)


# ============================================================================
# POLICY GRID VALIDATION
# ============================================================================

target["payout_interval"] = pd.to_numeric(
    target["payout_interval"],
    errors="coerce",
)

target["payout_amount"] = pd.to_numeric(
    target["payout_amount"],
    errors="coerce",
)

actual_intervals = set(target["payout_interval"].dropna().astype(int))

actual_amounts = set(target["payout_amount"].dropna().astype(float))

print(f"\nIntervals: {sorted(actual_intervals)}")

print(f"Payout amounts: {sorted(actual_amounts)}")

if actual_intervals != EXPECTED_INTERVALS:
    raise AssertionError(
        "Unexpected payout intervals.\n"
        f"Expected: {sorted(EXPECTED_INTERVALS)}\n"
        f"Found: {sorted(actual_intervals)}"
    )

if actual_amounts != EXPECTED_PAYOUT_AMOUNTS:
    raise AssertionError(
        "Unexpected payout amounts.\n"
        f"Expected: {sorted(EXPECTED_PAYOUT_AMOUNTS)}\n"
        f"Found: {sorted(actual_amounts)}"
    )


# ============================================================================
# DUPLICATE CHECK — FILTERED DATA ONLY
# ============================================================================

duplicate_mask = target.duplicated(
    subset=[
        "policy",
        "payout_interval",
        "payout_amount",
    ],
    keep=False,
)

if duplicate_mask.any():
    raise AssertionError(
        "Duplicate 0.25% policy detected:\n"
        + target.loc[
            duplicate_mask,
            [
                "policy",
                "payout_interval",
                "payout_amount",
            ],
        ].to_string(index=False)
    )

print("\n0.25% policy grid: PASS")
# ============================================================================
# NUMERIC CONVERSION
# ============================================================================

numeric_columns = [
    "survival_rate_strict",
    "failure_rate_strict",
    "survived_to_max_trades_rate_strict",
    "median_payouts_strict",
    "median_total_withdrawn_strict",
    "p95_total_withdrawn_strict",
    "median_final_balance_strict",
    "median_net_value_strict",
    "median_max_DD_strict",
    "p95_DD_strict",
    "median_trades_strict",
    "mean_trades_strict",
    "median_winning_days_strict",
    "survival_rate_adaptive",
    "failure_rate_adaptive",
    "survived_to_max_trades_rate_adaptive",
    "median_payouts_adaptive",
    "median_total_withdrawn_adaptive",
    "p95_total_withdrawn_adaptive",
    "median_final_balance_adaptive",
    "median_net_value_adaptive",
    "median_max_DD_adaptive",
    "p95_DD_adaptive",
    "median_trades_adaptive",
    "mean_trades_adaptive",
    "median_winning_days_adaptive",
    "delta_survival",
    "delta_median_net_value",
    "delta_median_DD",
    "delta_median_trades",
]


for column in numeric_columns:
    target[column] = pd.to_numeric(
        target[column],
        errors="coerce",
    )


# ============================================================================
# DELTA RECONCILIATION
# ============================================================================

print("\n" + "=" * 80)
print("SOURCE DELTA AUDIT")
print("=" * 80)


calculated_delta_survival = (
    target["survival_rate_adaptive"] - target["survival_rate_strict"]
)

calculated_delta_net_value = (
    target["median_net_value_adaptive"] - target["median_net_value_strict"]
)

calculated_delta_DD = target["median_max_DD_adaptive"] - target["median_max_DD_strict"]

calculated_delta_trades = (
    target["median_trades_adaptive"] - target["median_trades_strict"]
)


assert_close(
    target["delta_survival"],
    calculated_delta_survival,
    "delta_survival",
)

assert_close(
    target["delta_median_net_value"],
    calculated_delta_net_value,
    "delta_median_net_value",
)

assert_close(
    target["delta_median_DD"],
    calculated_delta_DD,
    "delta_median_DD",
)

assert_close(
    target["delta_median_trades"],
    calculated_delta_trades,
    "delta_median_trades",
)


print("Stored vs independently calculated deltas: PASS")

# ============================================================================
# DIRECTIONAL COMPARISON FLAGS
# ============================================================================

target["adaptive_better_survival"] = target["delta_survival"] > 0

target["adaptive_better_net_value"] = target["delta_median_net_value"] > 0

# Drawdown is negative. Therefore:
#   delta_DD > 0  => adaptive DD is less negative => better
#   delta_DD < 0  => adaptive DD is more negative => worse
target["adaptive_better_median_DD"] = target["delta_median_DD"] > 0

target["adaptive_more_trades"] = target["delta_median_trades"] > 0


# ============================================================================
# DERIVED METRICS
# ============================================================================

target["delta_survival_pp"] = target["delta_survival"] * 100.0

target["delta_failure_pp"] = (
    target["failure_rate_adaptive"] - target["failure_rate_strict"]
) * 100.0


target["delta_total_withdrawn"] = (
    target["median_total_withdrawn_adaptive"] - target["median_total_withdrawn_strict"]
)


target["delta_p95_withdrawn"] = (
    target["p95_total_withdrawn_adaptive"] - target["p95_total_withdrawn_strict"]
)


target["delta_final_balance"] = (
    target["median_final_balance_adaptive"] - target["median_final_balance_strict"]
)


target["delta_payouts"] = (
    target["median_payouts_adaptive"] - target["median_payouts_strict"]
)


target["delta_winning_days"] = (
    target["median_winning_days_adaptive"] - target["median_winning_days_strict"]
)


# ============================================================================
# DRAWDOWN
# ============================================================================

target["strict_abs_DD"] = target["median_max_DD_strict"].abs()

target["adaptive_abs_DD"] = target["median_max_DD_adaptive"].abs()

target["strict_abs_p95_DD"] = target["p95_DD_strict"].abs()

target["adaptive_abs_p95_DD"] = target["p95_DD_adaptive"].abs()

target["delta_abs_DD"] = target["adaptive_abs_DD"] - target["strict_abs_DD"]

target["delta_abs_p95_DD"] = target["adaptive_abs_p95_DD"] - target["strict_abs_p95_DD"]


# ============================================================================
# ECONOMIC EFFICIENCY
# ============================================================================

target["strict_net_value_per_DD"] = safe_ratio(
    target["median_net_value_strict"],
    target["strict_abs_DD"],
)

target["adaptive_net_value_per_DD"] = safe_ratio(
    target["median_net_value_adaptive"],
    target["adaptive_abs_DD"],
)

target["delta_net_value_per_DD"] = (
    target["adaptive_net_value_per_DD"] - target["strict_net_value_per_DD"]
)


target["strict_net_value_per_trade"] = safe_ratio(
    target["median_net_value_strict"],
    target["median_trades_strict"],
)

target["adaptive_net_value_per_trade"] = safe_ratio(
    target["median_net_value_adaptive"],
    target["median_trades_adaptive"],
)

target["delta_net_value_per_trade"] = (
    target["adaptive_net_value_per_trade"] - target["strict_net_value_per_trade"]
)


# ============================================================================
# CLASSIFICATIONS
# ============================================================================

target["adaptive_higher_survival"] = target["delta_survival"] > 0

target["adaptive_higher_net_value"] = target["delta_median_net_value"] > 0

target["adaptive_better_median_DD"] = target["delta_median_DD"] > 0

target["adaptive_lower_absolute_DD"] = target["delta_abs_DD"] < 0

target["adaptive_higher_survival_and_net"] = (
    target["adaptive_higher_survival"] & target["adaptive_higher_net_value"]
)

target["adaptive_improves_survival_net_and_DD"] = (
    target["adaptive_higher_survival"]
    & target["adaptive_higher_net_value"]
    & target["adaptive_better_median_DD"]
)


# ============================================================================
# SORT
# ============================================================================

target = target.sort_values(
    [
        "payout_interval",
        "payout_amount",
    ]
).reset_index(drop=True)


# ============================================================================
# SUMMARY
# ============================================================================

summary_rows = []


def add_summary(
    metric: str,
    value,
) -> None:

    summary_rows.append(
        {
            "metric": metric,
            "value": value,
        }
    )


n_policies = len(target)


add_summary(
    "risk_percent",
    TARGET_RISK_PERCENT,
)

add_summary(
    "target_risk_dollars",
    TARGET_RISK_DOLLARS,
)

add_summary(
    "policies_compared",
    n_policies,
)

add_summary(
    "strict_scenario",
    EXPECTED_STRICT_SCENARIO,
)

add_summary(
    "adaptive_scenario",
    EXPECTED_ADAPTIVE_SCENARIO,
)


# Counts.

add_summary(
    "adaptive_higher_survival_count",
    int(target["adaptive_higher_survival"].sum()),
)

add_summary(
    "adaptive_higher_net_value_count",
    int(target["adaptive_higher_net_value"].sum()),
)

add_summary(
    "adaptive_better_median_DD_count",
    int(target["adaptive_better_median_DD"].sum()),
)

add_summary(
    "adaptive_lower_absolute_DD_count",
    int(target["adaptive_lower_absolute_DD"].sum()),
)

add_summary(
    "adaptive_higher_survival_and_net_count",
    int(target["adaptive_higher_survival_and_net"].sum()),
)

add_summary(
    "adaptive_improves_all_three_count",
    int(target["adaptive_improves_survival_net_and_DD"].sum()),
)


# Aggregate metrics.

summary_metrics = [
    "delta_survival_pp",
    "delta_failure_pp",
    "delta_payouts",
    "delta_total_withdrawn",
    "delta_p95_withdrawn",
    "delta_final_balance",
    "delta_median_net_value",
    "delta_median_DD",
    "delta_median_trades",
    "delta_winning_days",
    "delta_abs_DD",
    "delta_abs_p95_DD",
    "delta_net_value_per_DD",
    "delta_net_value_per_trade",
]


for metric in summary_metrics:
    add_summary(
        f"mean_{metric}",
        target[metric].mean(),
    )

    add_summary(
        f"median_{metric}",
        target[metric].median(),
    )


# ============================================================================
# EXTREMES
# ============================================================================

max_survival_idx = target["delta_survival_pp"].idxmax()

min_survival_idx = target["delta_survival_pp"].idxmin()

max_net_value_idx = target["delta_median_net_value"].idxmax()

min_net_value_idx = target["delta_median_net_value"].idxmin()

best_DD_idx = target["delta_median_DD"].idxmax()

worst_DD_idx = target["delta_median_DD"].idxmin()


def add_extreme(
    prefix: str,
    idx,
    metric: str,
) -> None:

    row = target.loc[idx]

    add_summary(
        f"{prefix}_policy",
        policy_label(row),
    )

    add_summary(
        f"{prefix}_value",
        row[metric],
    )


add_extreme(
    "largest_survival_delta",
    max_survival_idx,
    "delta_survival_pp",
)

add_extreme(
    "smallest_survival_delta",
    min_survival_idx,
    "delta_survival_pp",
)

add_extreme(
    "largest_net_value_delta",
    max_net_value_idx,
    "delta_median_net_value",
)

add_extreme(
    "smallest_net_value_delta",
    min_net_value_idx,
    "delta_median_net_value",
)

add_extreme(
    "best_DD_delta",
    best_DD_idx,
    "delta_median_DD",
)

add_extreme(
    "worst_DD_delta",
    worst_DD_idx,
    "delta_median_DD",
)


summary = pd.DataFrame(summary_rows)


# ============================================================================
# BY PAYOUT INTERVAL
# ============================================================================

interval_rows = []


for interval, group in target.groupby("payout_interval"):
    interval_rows.append(
        {
            "payout_interval": int(interval),
            "policies": len(group),
            "mean_delta_survival_pp": group["delta_survival_pp"].mean(),
            "median_delta_survival_pp": group["delta_survival_pp"].median(),
            "mean_delta_net_value": group["delta_median_net_value"].mean(),
            "median_delta_net_value": group["delta_median_net_value"].median(),
            "mean_delta_DD": group["delta_median_DD"].mean(),
            "median_delta_DD": group["delta_median_DD"].median(),
            "mean_delta_trades": group["delta_median_trades"].mean(),
            "adaptive_higher_survival_count": int(
                group["adaptive_higher_survival"].sum()
            ),
            "adaptive_higher_net_value_count": int(
                group["adaptive_higher_net_value"].sum()
            ),
            "adaptive_better_DD_count": int(group["adaptive_better_median_DD"].sum()),
        }
    )


by_interval = pd.DataFrame(interval_rows).sort_values("payout_interval")


# ============================================================================
# BY PAYOUT AMOUNT
# ============================================================================

amount_rows = []


for amount, group in target.groupby("payout_amount"):
    amount_rows.append(
        {
            "payout_amount": float(amount),
            "policies": len(group),
            "mean_delta_survival_pp": group["delta_survival_pp"].mean(),
            "median_delta_survival_pp": group["delta_survival_pp"].median(),
            "mean_delta_net_value": group["delta_median_net_value"].mean(),
            "median_delta_net_value": group["delta_median_net_value"].median(),
            "mean_delta_DD": group["delta_median_DD"].mean(),
            "median_delta_DD": group["delta_median_DD"].median(),
            "mean_delta_trades": group["delta_median_trades"].mean(),
            "adaptive_higher_survival_count": int(
                group["adaptive_higher_survival"].sum()
            ),
            "adaptive_higher_net_value_count": int(
                group["adaptive_higher_net_value"].sum()
            ),
            "adaptive_better_DD_count": int(group["adaptive_better_median_DD"].sum()),
        }
    )


by_amount = pd.DataFrame(amount_rows).sort_values("payout_amount")


# ============================================================================
# SAVE
# ============================================================================

target.to_csv(
    OUTPUT_POLICY,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)

by_interval.to_csv(
    OUTPUT_INTERVAL,
    index=False,
)

by_amount.to_csv(
    OUTPUT_AMOUNT,
    index=False,
)


# ============================================================================
# CONSOLE REPORT
# ============================================================================

print("\n" + "=" * 80)
print("0.25% FULL-SYSTEM POLICY COMPARISON")
print("=" * 80)

display_columns = [
    "payout_interval",
    "payout_amount",
    "survival_rate_strict",
    "survival_rate_adaptive",
    "delta_survival_pp",
    "median_net_value_strict",
    "median_net_value_adaptive",
    "delta_median_net_value",
    "median_max_DD_strict",
    "median_max_DD_adaptive",
    "delta_median_DD",
    "median_trades_strict",
    "median_trades_adaptive",
    "delta_median_trades",
]


print(target[display_columns].to_string(index=False))


print("\n" + "=" * 80)
print("0.25% AGGREGATE")
print("=" * 80)

print(f"Policies compared: {n_policies}")

print(
    "Adaptive higher survival: "
    f"{int(target['adaptive_higher_survival'].sum())}"
    f"/{n_policies}"
)

print(
    "Adaptive higher median net value: "
    f"{int(target['adaptive_higher_net_value'].sum())}"
    f"/{n_policies}"
)

print(
    "Adaptive better median DD: "
    f"{int(target['adaptive_better_median_DD'].sum())}"
    f"/{n_policies}"
)

print(
    "Adaptive lower absolute DD: "
    f"{int(target['adaptive_lower_absolute_DD'].sum())}"
    f"/{n_policies}"
)

print(
    "Adaptive higher survival AND net value: "
    f"{int(target['adaptive_higher_survival_and_net'].sum())}"
    f"/{n_policies}"
)

print(
    "Adaptive improves survival + net value + DD: "
    f"{int(target['adaptive_improves_survival_net_and_DD'].sum())}"
    f"/{n_policies}"
)

print(f"\nMean survival delta: {target['delta_survival_pp'].mean():+.4f} pp")

print(f"Median survival delta: {target['delta_survival_pp'].median():+.4f} pp")

print(
    f"\nMean median net-value delta: ${target['delta_median_net_value'].mean():+,.2f}"
)

print(
    f"Median median net-value delta: ${target['delta_median_net_value'].median():+,.2f}"
)

print(f"\nMean median-DD delta: ${target['delta_median_DD'].mean():+,.2f}")

print(f"Median median-DD delta: ${target['delta_median_DD'].median():+,.2f}")

print(f"\nMean median-trades delta: {target['delta_median_trades'].mean():+,.2f}")

print(f"Median median-trades delta: {target['delta_median_trades'].median():+,.2f}")


# ============================================================================
# EXTREMES
# ============================================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE EXTREMES")
print("=" * 80)


def print_extreme(
    title: str,
    idx,
) -> None:

    row = target.loc[idx]

    print(f"\n{title}")

    print(f"Policy: {policy_label(row)}")

    print(f"Survival delta: {row['delta_survival_pp']:+.4f} pp")

    print(f"Net-value delta: ${row['delta_median_net_value']:+,.2f}")

    print(f"Median DD delta: ${row['delta_median_DD']:+,.2f}")

    print(f"Median trades delta: {row['delta_median_trades']:+,.2f}")


print_extreme(
    "Largest survival delta",
    max_survival_idx,
)

print_extreme(
    "Smallest survival delta",
    min_survival_idx,
)

print_extreme(
    "Largest net-value delta",
    max_net_value_idx,
)

print_extreme(
    "Smallest net-value delta",
    min_net_value_idx,
)

print_extreme(
    "Best median-DD delta",
    best_DD_idx,
)

print_extreme(
    "Worst median-DD delta",
    worst_DD_idx,
)


# ============================================================================
# OUTPUTS
# ============================================================================

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print(OUTPUT_POLICY)

print(OUTPUT_SUMMARY)

print(OUTPUT_INTERVAL)

print(OUTPUT_AMOUNT)


# ============================================================================
# FINAL AUDIT
# ============================================================================

assert len(target) == EXPECTED_POLICIES

assert set(target["payout_interval"].astype(int)) == EXPECTED_INTERVALS

assert set(target["payout_amount"].astype(float)) == EXPECTED_PAYOUT_AMOUNTS

assert target["survival_rate_strict"].notna().all()

assert target["survival_rate_adaptive"].notna().all()

assert target["median_net_value_strict"].notna().all()

assert target["median_net_value_adaptive"].notna().all()

assert target["median_max_DD_strict"].notna().all()

assert target["median_max_DD_adaptive"].notna().all()


print("\n" + "=" * 80)

print("XFA 0.25% FULL-SYSTEM ANALYSIS: PASS")

print("=" * 80)
