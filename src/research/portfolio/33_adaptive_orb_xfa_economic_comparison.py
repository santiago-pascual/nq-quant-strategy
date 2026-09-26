from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================================
# CONFIG
# ======================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "portfolio"
    / "22_full_system_funded_simulation.py"
)

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "funded"

CORRECTED_RESULTS = RESULTS_DIR / "adaptive_orb_xfa_results_corrected.csv"

OUTPUT_COMPARISON = RESULTS_DIR / "adaptive_orb_xfa_economic_comparison.csv"

OUTPUT_POLICY = RESULTS_DIR / "adaptive_orb_xfa_policy_comparison.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "adaptive_orb_xfa_economic_summary.csv"


# ======================================================================================
# LOAD ORIGINAL XFA MODULE
# ======================================================================================


def load_module():
    spec = importlib.util.spec_from_file_location(
        "full_system_funded_simulation",
        MODULE_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# HELPERS
# ======================================================================================


def pct(x: float) -> float:
    return 100.0 * x


def pp(x: float) -> float:
    return 100.0 * x


def money(x: float) -> str:
    return f"${x:,.2f}"


def get_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"None of {candidates} found.\nAvailable columns:\n{list(df.columns)}"
    )


# ======================================================================================
# MAIN
# ======================================================================================


def main() -> None:

    print("=" * 110)
    print("ADAPTIVE ORB XFA — ECONOMIC COMPARISON")
    print("=" * 110)

    if not CORRECTED_RESULTS.exists():
        raise FileNotFoundError(
            f"Missing corrected replay output:\n{CORRECTED_RESULTS}"
        )

    module = load_module()

    df = pd.read_csv(CORRECTED_RESULTS)

    print()
    print("=" * 110)
    print("1. LOAD CORRECTED XFA RESULTS")
    print("=" * 110)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print()

    print("Columns:")
    for column in df.columns:
        print(f"  {column}")

    # ==================================================================================
    # DISCOVER COLUMN NAMES
    # ==================================================================================

    payout_interval_col = get_column(
        df,
        [
            "payout_interval",
            "interval",
        ],
    )

    payout_amount_col = get_column(
        df,
        [
            "payout_amount",
            "amount",
        ],
    )

    strict_survival_col = get_column(
        df,
        [
            "survival_rate_strict",
        ],
    )

    adaptive_survival_col = get_column(
        df,
        [
            "survival_rate_adaptive",
        ],
    )

    strict_net_col = get_column(
        df,
        [
            "median_net_value_strict",
        ],
    )

    adaptive_net_col = get_column(
        df,
        [
            "median_net_value_adaptive",
        ],
    )

    strict_dd_col = get_column(
        df,
        [
            "median_max_DD_strict",
        ],
    )

    adaptive_dd_col = get_column(
        df,
        [
            "median_max_DD_adaptive",
        ],
    )

    strict_trades_col = get_column(
        df,
        [
            "median_trades_strict",
        ],
    )

    adaptive_trades_col = get_column(
        df,
        [
            "median_trades_adaptive",
        ],
    )

    # ==================================================================================
    # NORMALIZE
    # ==================================================================================

    df = df.copy()

    numeric_columns = [
        payout_interval_col,
        payout_amount_col,
        strict_survival_col,
        adaptive_survival_col,
        strict_net_col,
        adaptive_net_col,
        strict_dd_col,
        adaptive_dd_col,
        strict_trades_col,
        adaptive_trades_col,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # ==================================================================================
    # DERIVED ECONOMIC METRICS
    # ==================================================================================

    df["delta_survival_pp"] = (
        df[adaptive_survival_col] - df[strict_survival_col]
    ) * 100.0

    df["delta_median_net_value"] = df[adaptive_net_col] - df[strict_net_col]

    df["delta_median_DD"] = df[adaptive_dd_col] - df[strict_dd_col]

    # More negative DD means worse.
    df["adaptive_DD_worsening"] = df[strict_dd_col] - df[adaptive_dd_col]

    df["delta_median_trades"] = df[adaptive_trades_col] - df[strict_trades_col]

    # ==================================================================================
    # ECONOMIC SCORECARD
    # ==================================================================================

    print()
    print("=" * 110)
    print("2. FULL POLICY COMPARISON")
    print("=" * 110)

    display_columns = [
        payout_interval_col,
        payout_amount_col,
        strict_survival_col,
        adaptive_survival_col,
        "delta_survival_pp",
        strict_net_col,
        adaptive_net_col,
        "delta_median_net_value",
        strict_dd_col,
        adaptive_dd_col,
        "adaptive_DD_worsening",
        strict_trades_col,
        adaptive_trades_col,
        "delta_median_trades",
    ]

    print(df[display_columns].to_string(index=False))

    # ==================================================================================
    # 0.25% ONLY
    # ==================================================================================

    # The corrected replay currently contains the 0.25% risk comparison.
    # Keep the filter explicit in case future versions add other risk levels.

    risk_columns = [column for column in df.columns if "risk" in column.lower()]

    print()
    print("=" * 110)
    print("3. 0.25% POLICY SET")
    print("=" * 110)

    if risk_columns:
        print(f"Risk columns detected: {risk_columns}")

    # ==================================================================================
    # POLICY TABLE
    # ==================================================================================

    policy_df = df[
        [
            payout_interval_col,
            payout_amount_col,
            strict_survival_col,
            adaptive_survival_col,
            "delta_survival_pp",
            strict_net_col,
            adaptive_net_col,
            "delta_median_net_value",
            strict_dd_col,
            adaptive_dd_col,
            "adaptive_DD_worsening",
            strict_trades_col,
            adaptive_trades_col,
            "delta_median_trades",
        ]
    ].copy()

    policy_df = policy_df.rename(
        columns={
            payout_interval_col: "payout_interval",
            payout_amount_col: "payout_amount",
            strict_survival_col: "strict_survival",
            adaptive_survival_col: "adaptive_survival",
            strict_net_col: "strict_median_net_value",
            adaptive_net_col: "adaptive_median_net_value",
            strict_dd_col: "strict_median_DD",
            adaptive_dd_col: "adaptive_median_DD",
            strict_trades_col: "strict_median_trades",
            adaptive_trades_col: "adaptive_median_trades",
        }
    )
    policy_df["delta_median_DD"] = (
        policy_df["adaptive_median_DD"] - policy_df["strict_median_DD"]
    )

    # ==================================================================================
    # IDENTIFY POLICIES WHERE ADAPTIVE IMPROVES BOTH SURVIVAL AND NET VALUE
    # ==================================================================================

    policy_df["adaptive_higher_survival"] = (
        policy_df["adaptive_survival"] > policy_df["strict_survival"]
    )

    policy_df["adaptive_higher_net_value"] = (
        policy_df["adaptive_median_net_value"] > policy_df["strict_median_net_value"]
    )

    policy_df["adaptive_better_DD"] = (
        policy_df["adaptive_median_DD"] > policy_df["strict_median_DD"]
    )

    policy_df["adaptive_improves_survival_and_net"] = (
        policy_df["adaptive_higher_survival"] & policy_df["adaptive_higher_net_value"]
    )

    policy_df["adaptive_improves_all_three"] = (
        policy_df["adaptive_higher_survival"]
        & policy_df["adaptive_higher_net_value"]
        & policy_df["adaptive_better_DD"]
    )

    # ==================================================================================
    # COUNTS
    # ==================================================================================

    print()
    print("=" * 110)
    print("4. POLICY COUNTS")
    print("=" * 110)

    total_policies = len(policy_df)

    higher_survival = int(policy_df["adaptive_higher_survival"].sum())

    higher_net = int(policy_df["adaptive_higher_net_value"].sum())

    lower_dd = int(policy_df["adaptive_better_DD"].sum())

    better_survival_and_net = int(policy_df["adaptive_improves_survival_and_net"].sum())

    better_all = int(policy_df["adaptive_improves_all_three"].sum())

    print(f"Total policies:                         {total_policies}")
    print(f"Adaptive higher survival:              {higher_survival}/{total_policies}")
    print(f"Adaptive higher median net value:      {higher_net}/{total_policies}")
    print(f"Adaptive lower median DD:               {lower_dd}/{total_policies}")
    print(
        f"Adaptive higher survival + net value:   "
        f"{better_survival_and_net}/{total_policies}"
    )
    print(f"Adaptive improves all three:            {better_all}/{total_policies}")

    # ==================================================================================
    # AGGREGATE DELTAS
    # ==================================================================================

    print()
    print("=" * 110)
    print("5. AGGREGATE DELTAS")
    print("=" * 110)

    print(
        f"Mean survival delta:        {policy_df['delta_survival_pp'].mean():+.3f} pp"
    )

    print(
        f"Median survival delta:      {policy_df['delta_survival_pp'].median():+.3f} pp"
    )

    print(
        f"Mean net-value delta:        "
        f"{money(policy_df['delta_median_net_value'].mean())}"
    )

    print(
        f"Median net-value delta:      "
        f"{money(policy_df['delta_median_net_value'].median())}"
    )

    print(f"Mean DD delta:               {money(policy_df['delta_median_DD'].mean())}")

    print(
        f"Median DD delta:             {money(policy_df['delta_median_DD'].median())}"
    )

    print(
        f"Mean trade-count delta:      {policy_df['delta_median_trades'].mean():+.2f}"
    )

    # ==================================================================================
    # MOST RELEVANT $500 POLICIES
    # ==================================================================================

    payout_500 = policy_df[policy_df["payout_amount"] == 500.0].copy()

    print()
    print("=" * 110)
    print("6. $500 PAYOUT POLICIES")
    print("=" * 110)

    if len(payout_500):
        print(
            payout_500[
                [
                    "payout_interval",
                    "strict_survival",
                    "adaptive_survival",
                    "delta_survival_pp",
                    "strict_median_net_value",
                    "adaptive_median_net_value",
                    "delta_median_net_value",
                    "strict_median_DD",
                    "adaptive_median_DD",
                    "delta_median_trades",
                ]
            ].to_string(index=False)
        )

    # ==================================================================================
    # $750 POLICIES
    # ==================================================================================

    payout_750 = policy_df[policy_df["payout_amount"] == 750.0].copy()

    print()
    print("=" * 110)
    print("7. $750 PAYOUT POLICIES")
    print("=" * 110)

    if len(payout_750):
        print(
            payout_750[
                [
                    "payout_interval",
                    "strict_survival",
                    "adaptive_survival",
                    "delta_survival_pp",
                    "strict_median_net_value",
                    "adaptive_median_net_value",
                    "delta_median_net_value",
                    "strict_median_DD",
                    "adaptive_median_DD",
                    "delta_median_trades",
                ]
            ].to_string(index=False)
        )

    # ==================================================================================
    # MAX NET VALUE DELTA
    # ==================================================================================

    max_net_idx = policy_df["delta_median_net_value"].idxmax()

    min_net_idx = policy_df["delta_median_net_value"].idxmin()

    max_survival_idx = policy_df["delta_survival_pp"].idxmax()

    min_survival_idx = policy_df["delta_survival_pp"].idxmin()

    print()
    print("=" * 110)
    print("8. EXTREMES")
    print("=" * 110)

    print("Largest median net-value delta:")
    print(policy_df.loc[max_net_idx].to_string())

    print()
    print("Smallest median net-value delta:")
    print(policy_df.loc[min_net_idx].to_string())

    print()
    print("Largest survival delta:")
    print(policy_df.loc[max_survival_idx].to_string())

    print()
    print("Smallest survival delta:")
    print(policy_df.loc[min_survival_idx].to_string())

    # ==================================================================================
    # SUMMARY
    # ==================================================================================

    summary = pd.DataFrame(
        [
            {
                "metric": "total_policies",
                "value": float(total_policies),
            },
            {
                "metric": "adaptive_higher_survival_count",
                "value": float(higher_survival),
            },
            {
                "metric": "adaptive_higher_net_value_count",
                "value": float(higher_net),
            },
            {
                "metric": "adaptive_lower_DD_count",
                "value": float(lower_dd),
            },
            {
                "metric": "adaptive_higher_survival_and_net_count",
                "value": float(better_survival_and_net),
            },
            {
                "metric": "adaptive_improves_all_three_count",
                "value": float(better_all),
            },
            {
                "metric": "mean_survival_delta_pp",
                "value": float(policy_df["delta_survival_pp"].mean()),
            },
            {
                "metric": "median_survival_delta_pp",
                "value": float(policy_df["delta_survival_pp"].median()),
            },
            {
                "metric": "mean_median_net_value_delta",
                "value": float(policy_df["delta_median_net_value"].mean()),
            },
            {
                "metric": "median_median_net_value_delta",
                "value": float(policy_df["delta_median_net_value"].median()),
            },
            {
                "metric": "mean_median_DD_delta",
                "value": float(policy_df["delta_median_DD"].mean()),
            },
            {
                "metric": "median_median_DD_delta",
                "value": float(policy_df["delta_median_DD"].median()),
            },
        ]
    )

    # ==================================================================================
    # SAVE
    # ==================================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy_df.to_csv(
        OUTPUT_POLICY,
        index=False,
    )

    df.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()
    print("=" * 110)
    print("9. OUTPUTS")
    print("=" * 110)

    print(f"Policy comparison: {OUTPUT_POLICY}")
    print(f"Full comparison:   {OUTPUT_COMPARISON}")
    print(f"Summary:           {OUTPUT_SUMMARY}")

    print()
    print("=" * 110)
    print("ADAPTIVE ORB XFA — ECONOMIC COMPARISON: PASS")
    print("=" * 110)


if __name__ == "__main__":
    main()
