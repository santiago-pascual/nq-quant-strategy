from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIG
# ============================================================================

POINT_VALUE_MNQ = 2.0

ACCOUNT_SIZE = 50_000.0

BASE_RISK_FRACTION = 0.0025
BASE_RISK_DOLLARS = ACCOUNT_SIZE * BASE_RISK_FRACTION

TOLERANCE_050_DOLLARS = ACCOUNT_SIZE * 0.005

TOLERANCE_060_DOLLARS = ACCOUNT_SIZE * 0.006

OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp("2026-06-19 23:59:59.999999", tz="UTC")

EXPECTED_ORB_TRADES = 1442


# ============================================================================
# PATHS
# ============================================================================

ORB_RESULTS = PROJECT_ROOT / "src" / "research" / "results" / "orb"

INPUT_FILE = ORB_RESULTS / "orb_reconciliation_trades.csv"

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "paper"

OUTPUT_TRADES = OUTPUT_DIR / "orb_risk_tolerance_trades.csv"

OUTPUT_SUMMARY = OUTPUT_DIR / "orb_risk_tolerance_summary.csv"


# ============================================================================
# SCENARIOS
# ============================================================================

SCENARIOS = (
    "STRICT_025",
    "MAX_050",
    "ADAPTIVE_MAX_060",
    "FORCE_1_IF_SUBUNIT",
)


# ============================================================================
# HELPERS
# ============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(f"ORB artifact missing columns: {missing}")


# ============================================================================
# LOAD ORB
# ============================================================================


def load_orb() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"ORB artifact not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    require_columns(
        df,
        [
            "entry_timestamp",
            "entry_price",
            "stop_price",
            "net_R",
        ],
    )

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    df["entry_price"] = pd.to_numeric(
        df["entry_price"],
        errors="raise",
    )

    df["stop_price"] = pd.to_numeric(
        df["stop_price"],
        errors="raise",
    )

    df["net_R"] = pd.to_numeric(
        df["net_R"],
        errors="raise",
    )

    df["stop_points"] = (df["entry_price"] - df["stop_price"]).abs()

    df = df[
        (df["entry_timestamp"] >= OOS_START) & (df["entry_timestamp"] <= OOS_END)
    ].copy()

    df = df.sort_values(
        "entry_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    if len(df) != EXPECTED_ORB_TRADES:
        raise AssertionError(
            f"Expected {EXPECTED_ORB_TRADES} ORB OOS trades, got {len(df)}"
        )

    if df["entry_timestamp"].duplicated().any():
        raise AssertionError("ORB OOS contains duplicate entry timestamps.")

    return df


# ============================================================================
# THEORETICAL SIZING
# ============================================================================


def calculate_base_sizing(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["risk_per_contract"] = out["stop_points"] * POINT_VALUE_MNQ

    out["theoretical_quantity"] = BASE_RISK_DOLLARS / out["risk_per_contract"]

    out["base_quantity"] = np.floor(out["theoretical_quantity"] + 1e-12).astype(int)

    out["one_contract_risk"] = out["risk_per_contract"]

    out["one_contract_risk_fraction"] = out["one_contract_risk"] / ACCOUNT_SIZE

    return out


# ============================================================================
# SCENARIO LOGIC
# ============================================================================


def apply_scenario(
    df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:

    out = df.copy()

    theoretical = out["theoretical_quantity"]

    risk_per_contract = out["risk_per_contract"]

    base_quantity = out["base_quantity"]

    if scenario == "STRICT_025":
        # Current production rule.
        #
        # <1 contract -> reject.
        # >=1 -> floor.

        quantity = base_quantity.copy()

        reason = np.where(
            quantity == 0,
            "rejected_below_025_risk_budget",
            "standard_floor_sizing",
        )

    elif scenario == "MAX_050":
        # Normal floor sizing first.
        #
        # If theoretical quantity < 1, allow
        # one contract only when its actual
        # risk <= 0.50%.

        quantity = base_quantity.copy()

        eligible = (theoretical < 1.0) & (
            risk_per_contract <= TOLERANCE_050_DOLLARS + 1e-9
        )

        quantity = np.where(
            eligible,
            1,
            quantity,
        )

        reason = np.where(
            eligible,
            "forced_one_contract_within_050",
            np.where(
                quantity == 0,
                "rejected_above_050",
                "standard_floor_sizing",
            ),
        )

    elif scenario == "MAX_060":
        # Same logic, but 0.60% maximum
        # effective risk for the special
        # one-contract ORB case.

        quantity = base_quantity.copy()

        eligible = (theoretical < 1.0) & (
            risk_per_contract <= TOLERANCE_060_DOLLARS + 1e-9
        )

        quantity = np.where(
            eligible,
            1,
            quantity,
        )

        reason = np.where(
            eligible,
            "forced_one_contract_within_060",
            np.where(
                quantity == 0,
                "rejected_above_060",
                "standard_floor_sizing",
            ),
        )

    elif scenario == "FORCE_1_IF_SUBUNIT":
        # EXPERIMENTAL / deliberately aggressive test.
        #
        # If theoretical quantity is between
        # 0 and 1, enter one MNQ regardless
        # of actual risk.
        #
        # This is NOT production policy.

        quantity = np.where(
            (theoretical > 0.0) & (theoretical < 1.0),
            1,
            base_quantity,
        )

        reason = np.where(
            (theoretical > 0.0) & (theoretical < 1.0),
            "forced_one_contract_unrestricted",
            np.where(
                quantity == 0,
                "rejected_invalid_size",
                "standard_floor_sizing",
            ),
        )
    elif scenario == "ADAPTIVE_MAX_060":
        # Adaptive ORB sizing:
        #
        # 1. Normal target risk is 0.25% ($125).
        # 2. If normal floor sizing gives >= 1 contract,
        #    use the normal floor.
        # 3. If normal sizing gives < 1 contract,
        #    allow exactly 1 MNQ only if its actual
        #    risk is <= $300 (0.60%).
        # 4. If 1 MNQ would risk more than $300,
        #    reject the trade.
        #
        # This prevents pathological cases such as
        # the $1,247 one-contract ORB trade.

        quantity = base_quantity.copy()

        subunit = theoretical < 1.0

        adaptive_eligible = subunit & (
            risk_per_contract <= TOLERANCE_060_DOLLARS + 1e-9
        )

        quantity = np.where(
            adaptive_eligible,
            1,
            quantity,
        )

        reason = np.where(
            adaptive_eligible,
            "adaptive_one_contract_within_060",
            np.where(
                quantity == 0,
                "rejected_above_060",
                "standard_floor_sizing",
            ),
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    out["scenario"] = scenario

    out["quantity"] = pd.Series(
        quantity,
        index=out.index,
    ).astype(int)

    out["decision_reason"] = reason

    out["approved"] = out["quantity"] >= 1

    out["actual_risk"] = out["quantity"] * risk_per_contract

    out["actual_risk_fraction"] = out["actual_risk"] / ACCOUNT_SIZE

    # Since net_R is expressed relative to
    # the strategy's 1R risk, dollar P&L is:
    #
    #     net_R × actual dollar risk
    #
    out["pnl_dollars"] = out["net_R"] * out["actual_risk"]

    out["pnl_if_base_025"] = out["net_R"] * BASE_RISK_DOLLARS

    return out


# ============================================================================
# METRICS
# ============================================================================


def calculate_max_drawdown(
    pnl: pd.Series,
) -> float:

    equity = pnl.cumsum()

    if equity.empty:
        return 0.0

    running_peak = equity.cummax()

    drawdown = equity - running_peak

    return float(drawdown.min())


def calculate_profit_factor(
    pnl: pd.Series,
) -> float:

    gross_profit = float(pnl[pnl > 0].sum())

    gross_loss = float(-pnl[pnl < 0].sum())

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def calculate_expectancy(
    pnl: pd.Series,
) -> float:

    if pnl.empty:
        return 0.0

    return float(pnl.mean())


def summarize_scenario(
    df: pd.DataFrame,
    scenario: str,
) -> dict[str, object]:

    trades = len(df)

    approved = df[df["approved"]].copy()

    rejected = df[~df["approved"]].copy()

    pnl = approved["pnl_dollars"]

    wins = int((pnl > 0).sum())

    losses = int((pnl < 0).sum())

    total_pnl = float(pnl.sum())

    win_rate = wins / len(pnl) if len(pnl) > 0 else 0.0

    return {
        "scenario": scenario,
        "trades": trades,
        "approved": len(approved),
        "rejected": len(rejected),
        "approval_rate": (len(approved) / trades if trades else 0.0),
        "rejection_rate": (len(rejected) / trades if trades else 0.0),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "expectancy_per_executed_trade": (calculate_expectancy(pnl)),
        "profit_factor": (calculate_profit_factor(pnl)),
        "max_drawdown": (calculate_max_drawdown(pnl)),
        "mean_actual_risk": (
            float(approved["actual_risk"].mean()) if len(approved) else 0.0
        ),
        "max_actual_risk": (
            float(approved["actual_risk"].max()) if len(approved) else 0.0
        ),
        "mean_risk_fraction": (
            float(approved["actual_risk_fraction"].mean()) if len(approved) else 0.0
        ),
        "max_risk_fraction": (
            float(approved["actual_risk_fraction"].max()) if len(approved) else 0.0
        ),
    }


# ============================================================================
# VALIDATION
# ============================================================================


def validate_scenario(
    df: pd.DataFrame,
    scenario: str,
) -> None:

    if scenario == "FORCE_1_IF_SUBUNIT":
        return

    approved = df[df["approved"]]

    if (approved["quantity"] < 1).any():
        raise AssertionError(f"{scenario}: approved trade has zero contracts.")

    if scenario == "STRICT_025":
        if (approved["actual_risk"] > BASE_RISK_DOLLARS + 1e-9).any():
            raise AssertionError("STRICT_025 exceeded $125 risk.")

    elif scenario == "MAX_050":
        if (approved["actual_risk"] > TOLERANCE_050_DOLLARS + 1e-9).any():
            raise AssertionError("MAX_050 exceeded $250 risk.")

    elif scenario == "MAX_060":
        if (approved["actual_risk"] > TOLERANCE_060_DOLLARS + 1e-9).any():
            raise AssertionError("MAX_060 exceeded $300 risk.")


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    banner("ORB RISK TOLERANCE REPLAY")

    print("IMPORTANT: experiment applies ONLY to ORB.")

    print(f"Base risk: {BASE_RISK_FRACTION:.2%} (${BASE_RISK_DOLLARS:.2f})")

    print(f"0.50% tolerance: ${TOLERANCE_050_DOLLARS:.2f}")

    print(f"0.60% tolerance: ${TOLERANCE_060_DOLLARS:.2f}")

    df = load_orb()

    print()
    print(f"ORB OOS trades loaded: {len(df):,}")

    df = calculate_base_sizing(df)

    all_results: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        banner(f"SCENARIO: {scenario}")

        scenario_df = apply_scenario(
            df,
            scenario,
        )

        validate_scenario(
            scenario_df,
            scenario,
        )

        summary = summarize_scenario(
            scenario_df,
            scenario,
        )

        summaries.append(summary)

        all_results.append(scenario_df)

        print(f"Approved: {summary['approved']:,}")

        print(f"Rejected: {summary['rejected']:,}")

        print(f"Approval rate: {summary['approval_rate']:.2%}")

        print(f"Total P&L: ${summary['total_pnl']:,.2f}")

        print(f"Expectancy: ${summary['expectancy_per_executed_trade']:,.2f}")

        print(f"Profit factor: {summary['profit_factor']:.4f}")

        print(f"Max DD: ${summary['max_drawdown']:,.2f}")

        print(f"Mean risk: ${summary['mean_actual_risk']:,.2f}")

        print(f"Max risk: ${summary['max_actual_risk']:,.2f}")

    summary_df = pd.DataFrame(summaries)

    trades_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    banner("SCENARIO COMPARISON")

    print(
        summary_df[
            [
                "scenario",
                "approved",
                "rejected",
                "approval_rate",
                "total_pnl",
                "expectancy_per_executed_trade",
                "profit_factor",
                "max_drawdown",
                "mean_actual_risk",
                "max_actual_risk",
            ]
        ].to_string(index=False)
    )

    banner("SUB-UNIT ORB TRADES")

    subunit = df[df["theoretical_quantity"] < 1].copy()

    print(f"ORB trades with theoretical quantity < 1: {len(subunit):,}")

    print()

    print("Risk distribution for those trades:")

    print(subunit["one_contract_risk"].describe().to_string())

    print()

    print("Maximum one-contract risk:")

    print(f"${subunit['one_contract_risk'].max():,.2f}")

    print()

    print("Output:")

    print(OUTPUT_TRADES)

    print(OUTPUT_SUMMARY)

    print()
    print("ORB RISK TOLERANCE REPLAY: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
