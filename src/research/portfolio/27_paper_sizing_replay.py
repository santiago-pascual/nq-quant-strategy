from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk import XFA_50K_PRODUCTION_POLICY


# ============================================================================
# CONFIG
# ============================================================================

OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59.999999",
    tz="UTC",
)

POINT_VALUE_MNQ = 2.0

EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}

EXPECTED_TOTAL = 3255


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MR_RESULTS = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

S2R_RESULTS = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

ORB_RESULTS = PROJECT_ROOT / "src" / "research" / "results" / "orb"

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "paper"

OUTPUT_TRADES = OUTPUT_DIR / "paper_sizing_replay.csv"

OUTPUT_SUMMARY = OUTPUT_DIR / "paper_sizing_summary.csv"


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
    name: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def normalize_timestamp(
    series: pd.Series,
) -> pd.Series:
    ts = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if ts.isna().any():
        raise ValueError("Timestamp normalization produced NaT values.")

    return ts


# ============================================================================
# LOAD MRL1 + MRS2
# ============================================================================


def load_mr_stream() -> pd.DataFrame:
    path = MR_RESULTS / "research_08aa_modular_reproduction_trades.csv"

    if not path.exists():
        raise FileNotFoundError(f"MR artifact not found:\n{path}")

    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "strategy_name",
            "side",
            "entry_timestamp",
            "entry_price",
            "stop_points",
            "r_multiple",
        ],
        str(path),
    )

    df["entry_timestamp"] = normalize_timestamp(df["entry_timestamp"])

    df["strategy_name"] = df["strategy_name"].astype(str).str.strip().str.upper()

    df["side"] = df["side"].astype(str).str.strip().str.upper()

    df["entry_price"] = pd.to_numeric(
        df["entry_price"],
        errors="raise",
    )

    df["stop_points"] = pd.to_numeric(
        df["stop_points"],
        errors="raise",
    )

    df["r_multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="raise",
    )

    df = df[df["strategy_name"].isin(["MRL1", "MRS2"])].copy()

    return df[
        [
            "strategy_name",
            "side",
            "entry_timestamp",
            "entry_price",
            "stop_points",
            "r_multiple",
        ]
    ].copy()


# ============================================================================
# LOAD S2R
# ============================================================================


def load_s2r_stream() -> pd.DataFrame:
    path = S2R_RESULTS / "s2r_modular_authoritative_reproduction.csv"

    if not path.exists():
        raise FileNotFoundError(f"S2R authoritative artifact not found:\n{path}")

    df = pd.read_csv(path)

    # IMPORTANT:
    #
    # The authoritative S2R artifact intentionally does NOT expose
    # strategy_name, entry_price, or r_multiple.
    #
    # Its real schema contains:
    #   entry_timestamp
    #   stop_points
    #   net_R
    #   ...
    #
    # We normalize those fields without changing the source artifact.

    require_columns(
        df,
        [
            "entry_timestamp",
            "stop_points",
            "net_R",
            "exit_reason",
        ],
        str(path),
    )

    df["entry_timestamp"] = normalize_timestamp(df["entry_timestamp"])

    df["strategy_name"] = "S2R"

    df["side"] = "SHORT"

    df["stop_points"] = pd.to_numeric(
        df["stop_points"],
        errors="raise",
    )

    df["r_multiple"] = pd.to_numeric(
        df["net_R"],
        errors="raise",
    )

    return df[
        [
            "strategy_name",
            "side",
            "entry_timestamp",
            "stop_points",
            "r_multiple",
        ]
    ].copy()


# ============================================================================
# LOAD ORB
# ============================================================================


def load_orb_stream() -> pd.DataFrame:
    path = ORB_RESULTS / "orb_reconciliation_trades.csv"

    if not path.exists():
        raise FileNotFoundError(f"ORB artifact not found:\n{path}")

    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "entry_timestamp",
            "direction",
            "entry_price",
            "stop_price",
            "net_R",
        ],
        str(path),
    )

    df["entry_timestamp"] = normalize_timestamp(df["entry_timestamp"])

    df["strategy_name"] = "ORB"

    df["side"] = df["direction"].astype(str).str.strip().str.upper()

    df["entry_price"] = pd.to_numeric(
        df["entry_price"],
        errors="raise",
    )

    df["stop_price"] = pd.to_numeric(
        df["stop_price"],
        errors="raise",
    )

    df["stop_points"] = (df["entry_price"] - df["stop_price"]).abs()

    df["r_multiple"] = pd.to_numeric(
        df["net_R"],
        errors="raise",
    )

    return df[
        [
            "strategy_name",
            "side",
            "entry_timestamp",
            "entry_price",
            "stop_price",
            "stop_points",
            "r_multiple",
        ]
    ].copy()


# ============================================================================
# LOAD COMPLETE FROZEN OOS
# ============================================================================


def load_frozen_oos() -> pd.DataFrame:
    banner("LOADING FROZEN OOS STREAMS")

    mr = load_mr_stream()
    s2r = load_s2r_stream()
    orb = load_orb_stream()

    print(f"MR full stream:   {len(mr):,}")
    print(f"S2R full stream:  {len(s2r):,}")
    print(f"ORB full stream:  {len(orb):,}")

    df = pd.concat(
        [
            mr,
            s2r,
            orb,
        ],
        ignore_index=True,
        sort=False,
    )

    df = df[
        (df["entry_timestamp"] >= OOS_START) & (df["entry_timestamp"] <= OOS_END)
    ].copy()

    df = df.sort_values(
        [
            "entry_timestamp",
            "strategy_name",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    counts = df["strategy_name"].value_counts().to_dict()

    print()
    print("FROZEN OOS COUNT AUDIT")
    print("-" * 40)

    for strategy, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(strategy, 0))

        print(f"{strategy:<6} {actual:>5} / {expected}")

        if actual != expected:
            raise AssertionError(f"{strategy}: expected {expected}, got {actual}")

    if len(df) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} total trades, got {len(df)}")

    print("-" * 40)
    print(f"TOTAL  {len(df):>5} / {EXPECTED_TOTAL}")

    return df


# ============================================================================
# PRODUCTION SIZING
# ============================================================================


def calculate_sizing(
    row: pd.Series,
) -> dict[str, object]:

    strategy = str(row["strategy_name"])

    stop_points = float(row["stop_points"])

    if not np.isfinite(stop_points):
        raise ValueError(f"{strategy}: stop_points is not finite.")

    if stop_points <= 0:
        raise ValueError(f"{strategy}: invalid stop distance {stop_points}")

    risk_per_contract = stop_points * POINT_VALUE_MNQ

    risk_budget = XFA_50K_PRODUCTION_POLICY.risk_per_trade

    theoretical_quantity = risk_budget / risk_per_contract

    # ================================================================
    # CRITICAL PRODUCTION RULE
    # ================================================================
    #
    # Contracts are indivisible.
    #
    # 0.75 -> 0 -> reject
    # 1.55 -> 1
    # 2.99 -> 2
    #
    # NEVER round upward.
    #

    executable_quantity = int(np.floor(theoretical_quantity + 1e-12))

    executable_quantity = min(
        executable_quantity,
        XFA_50K_PRODUCTION_POLICY.max_contracts,
    )

    approved = executable_quantity >= 1

    actual_risk = executable_quantity * risk_per_contract

    if not approved:
        rejection_reason = "risk_budget_below_one_contract"

    else:
        if actual_risk > risk_budget + 1e-9:
            raise AssertionError(
                f"{strategy}: actual risk "
                f"${actual_risk:.6f} exceeds "
                f"budget ${risk_budget:.6f}"
            )

        rejection_reason = ""

    result = {
        "strategy_name": strategy,
        "side": row["side"],
        "entry_timestamp": row["entry_timestamp"],
        "entry_price": row.get(
            "entry_price",
            np.nan,
        ),
        "stop_points": stop_points,
        "risk_per_contract": risk_per_contract,
        "risk_budget": risk_budget,
        "theoretical_quantity": theoretical_quantity,
        "executable_quantity": executable_quantity,
        "actual_risk": actual_risk,
        "risk_utilization": (actual_risk / risk_budget if risk_budget > 0 else 0.0),
        "approved": approved,
        "rejection_reason": rejection_reason,
        "historical_r_multiple": float(row["r_multiple"]),
    }

    return result


# ============================================================================
# VALIDATION
# ============================================================================


def validate_sizing(
    sizing: pd.DataFrame,
) -> None:

    banner("VALIDATING SIZING INVARIANTS")

    if sizing.empty:
        raise AssertionError("Sizing output is empty.")

    # Integer contracts only.
    if not np.all(
        np.isclose(
            sizing["executable_quantity"],
            np.floor(sizing["executable_quantity"]),
        )
    ):
        raise AssertionError("Non-integer executable quantity detected.")

    # Never round upward.
    if (sizing["executable_quantity"] > sizing["theoretical_quantity"] + 1e-9).any():
        raise AssertionError("Sizing rounded upward.")

    approved = sizing[sizing["approved"]]

    rejected = sizing[~sizing["approved"]]

    # Approved trades need at least one contract.
    if not approved.empty:
        if (approved["executable_quantity"] < 1).any():
            raise AssertionError("Approved trade has fewer than one contract.")

    # Approved trades cannot exceed $125.
    if not approved.empty:
        if (approved["actual_risk"] > approved["risk_budget"] + 1e-9).any():
            raise AssertionError("Approved trade exceeds production risk budget.")

    # Rejected trades must have zero contracts.
    if not rejected.empty:
        if (rejected["executable_quantity"] != 0).any():
            raise AssertionError("Rejected trade has non-zero quantity.")

    print("PASS — integer contract sizing")

    print("PASS — no upward rounding")

    print("PASS — approved risk <= $125")

    print("PASS — rejected trades = 0 contracts")


# ============================================================================
# SUMMARY
# ============================================================================


def build_summary(
    sizing: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        sizing.groupby(
            "strategy_name",
            sort=False,
        )
        .agg(
            trades=(
                "strategy_name",
                "size",
            ),
            approved=(
                "approved",
                "sum",
            ),
            mean_theoretical_quantity=(
                "theoretical_quantity",
                "mean",
            ),
            median_theoretical_quantity=(
                "theoretical_quantity",
                "median",
            ),
            mean_executable_quantity=(
                "executable_quantity",
                "mean",
            ),
            median_executable_quantity=(
                "executable_quantity",
                "median",
            ),
            max_executable_quantity=(
                "executable_quantity",
                "max",
            ),
            mean_actual_risk=(
                "actual_risk",
                "mean",
            ),
            max_actual_risk=(
                "actual_risk",
                "max",
            ),
            mean_risk_utilization=(
                "risk_utilization",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["rejected"] = summary["trades"] - summary["approved"]

    summary["approval_rate"] = summary["approved"] / summary["trades"]

    summary["rejection_rate"] = summary["rejected"] / summary["trades"]

    return summary


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    banner("FULL SYSTEM PAPER SIZING REPLAY")

    print("System: MRL1 + S2R + MRS2 + ORB")

    print(f"OOS: {OOS_START.date()} -> {OOS_END.date()}")

    print(f"Production risk: {XFA_50K_PRODUCTION_POLICY.risk_fraction:.2%}")

    print(f"Risk budget: ${XFA_50K_PRODUCTION_POLICY.risk_per_trade:.2f}")

    print(f"MNQ point value: ${POINT_VALUE_MNQ:.2f}")

    data = load_frozen_oos()

    banner("CALCULATING PRODUCTION CONTRACT SIZING")

    rows: list[dict[str, object]] = []

    for _, row in data.iterrows():
        rows.append(calculate_sizing(row))

    sizing = pd.DataFrame(rows)

    validate_sizing(sizing)

    summary = build_summary(sizing)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sizing.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    banner("RESULT")

    print()

    print(summary.to_string(index=False))

    print()

    total_approved = int(sizing["approved"].sum())

    total_rejected = int((~sizing["approved"]).sum())

    print(f"Total trades: {len(sizing):,}")

    print(f"Approved: {total_approved:,}")

    print(f"Rejected: {total_rejected:,}")

    print()

    print("Quantity distribution:")

    print(sizing["executable_quantity"].value_counts().sort_index().to_string())

    print()

    print("Theoretical quantity distribution:")

    print(sizing["theoretical_quantity"].describe().to_string())

    print()

    print(f"Trades written to:\n{OUTPUT_TRADES}")

    print(f"Summary written to:\n{OUTPUT_SUMMARY}")

    print()

    print("PAPER SIZING REPLAY: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
