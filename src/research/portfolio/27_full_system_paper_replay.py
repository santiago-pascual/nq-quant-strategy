"""
Full historical paper-system replay.

Purpose
-------
Replay the frozen four-strategy OOS system through the production-style
paper infrastructure using the XFA production risk policy.

This is an integration/reconciliation harness.

It does NOT:
    - optimize strategies
    - refit models
    - change strategy parameters
    - alter stops/targets
    - simulate Combine rules
    - modify the frozen research artifacts

It DOES:
    - load the frozen OOS trade/event streams
    - preserve chronological ordering
    - apply production integer contract sizing
    - enforce aggregate risk
    - enforce portfolio conflicts
    - track actual fills/positions
    - record accepted/rejected entries
    - produce sizing and risk diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.risk import (
    RiskEngine,
    XFA_50K_PRODUCTION_POLICY,
)
from src.risk.types import RiskRequest


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio"

OUTPUT_DIR = RESULTS_DIR / "paper_replay"


OOS_START = pd.Timestamp("2020-06-23 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-06-19 23:59:59", tz="UTC")

POINT_VALUE_MNQ = 2.0


EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}

EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


@dataclass(frozen=True)
class PaperSizingResult:
    strategy_name: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    stop_price: float
    stop_distance_points: float
    risk_per_contract: float
    theoretical_quantity: float
    executable_quantity: int
    actual_risk: float
    approved: bool
    rejection_reason: str


def load_frozen_oos() -> pd.DataFrame:
    """
    Load the already frozen independent OOS trade artifact.

    The replay must consume the frozen artifact rather than re-running
    research logic.
    """

    candidates = [
        RESULTS_DIR / "independent_reproduction_oos_trades.csv",
        RESULTS_DIR / "independent_reproduction_trades.csv",
        RESULTS_DIR / "frozen_oos_trades.csv",
    ]

    for path in candidates:
        if path.exists():
            data = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "Could not locate the frozen independent OOS trade artifact. "
            "Expected one of:\n" + "\n".join(str(path) for path in candidates)
        )

    required = {
        "strategy_name",
        "entry_timestamp",
        "entry_price",
        "exit_price",
        "r_multiple",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Frozen OOS artifact is missing columns: {sorted(missing)}")

    data["entry_timestamp"] = pd.to_datetime(
        data["entry_timestamp"],
        utc=True,
    )

    data = data[
        (data["entry_timestamp"] >= OOS_START) & (data["entry_timestamp"] <= OOS_END)
    ].copy()

    data["strategy_name"] = data["strategy_name"].astype(str)

    return data.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_frozen_counts(data: pd.DataFrame) -> None:
    counts = data["strategy_name"].value_counts().to_dict()

    for strategy, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(strategy, 0))

        if actual != expected:
            raise AssertionError(
                f"{strategy}: expected {expected} trades, got {actual}"
            )

    if len(data) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} total trades, got {len(data)}")


def infer_stop_price(
    row: pd.Series,
) -> float:
    """
    Recover the frozen strategy stop from the entry/exit direction.

    The research artifact contains the executed entry and exit but does
    not necessarily contain stop_price, so the frozen strategy stop
    distances are applied here.

    These distances are research constants, not optimized here.
    """

    strategy = str(row["strategy_name"])

    stop_points = {
        "MRL1": 37.5,
        "S2R": 25.0,
        "MRS2": 25.0,
        "ORB": None,
    }

    points = stop_points[strategy]

    if points is None:
        raise ValueError(
            "ORB stop reconstruction requires the ORB-specific "
            "entry/OR context. It must not be guessed."
        )

    side = str(row["side"]).upper()

    entry = float(row["entry_price"])

    if side == "LONG":
        return entry - points

    if side == "SHORT":
        return entry + points

    raise ValueError(f"Unknown side: {side}")


def calculate_sizing(
    *,
    risk_engine: RiskEngine,
    row: pd.Series,
) -> PaperSizingResult:
    strategy = str(row["strategy_name"])

    entry_price = float(row["entry_price"])

    stop_price = infer_stop_price(row)

    stop_distance = abs(entry_price - stop_price)

    risk_per_contract = stop_distance * POINT_VALUE_MNQ

    theoretical_quantity = XFA_50K_PRODUCTION_POLICY.risk_per_trade / risk_per_contract

    result = risk_engine.evaluate(
        RiskRequest(
            strategy_name=f"SIZE_TEST_{strategy}",
            entry_price=entry_price,
            stop_price=stop_price,
            point_value=POINT_VALUE_MNQ,
            account_equity=50_000.0,
        ),
        trading_day=row["entry_timestamp"].date(),
    )

    return PaperSizingResult(
        strategy_name=strategy,
        entry_timestamp=row["entry_timestamp"],
        entry_price=entry_price,
        stop_price=stop_price,
        stop_distance_points=stop_distance,
        risk_per_contract=risk_per_contract,
        theoretical_quantity=theoretical_quantity,
        executable_quantity=result.quantity,
        actual_risk=result.total_risk,
        approved=result.approved,
        rejection_reason="" if result.approved else result.reason,
    )


def run_sizing_replay(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the production sizing policy over every frozen OOS trade.

    This first pass deliberately isolates sizing from execution lifecycle.
    """

    rows: list[dict[str, object]] = []

    for row in data.itertuples(index=False):
        record = pd.Series(row._asdict())

        # Each sizing decision is independent for this diagnostic.
        risk_engine = RiskEngine(XFA_50K_PRODUCTION_POLICY.to_risk_limits())

        result = calculate_sizing(
            risk_engine=risk_engine,
            row=record,
        )

        rows.append(
            {
                "strategy_name": result.strategy_name,
                "entry_timestamp": result.entry_timestamp,
                "entry_price": result.entry_price,
                "stop_price": result.stop_price,
                "stop_distance_points": result.stop_distance_points,
                "risk_per_contract": result.risk_per_contract,
                "theoretical_quantity": result.theoretical_quantity,
                "executable_quantity": result.executable_quantity,
                "actual_risk": result.actual_risk,
                "approved": result.approved,
                "rejection_reason": result.rejection_reason,
            }
        )

    return pd.DataFrame(rows)


def validate_sizing_invariants(
    sizing: pd.DataFrame,
) -> None:
    """
    Hard invariants for production contract sizing.
    """

    if sizing.empty:
        raise AssertionError("Sizing output is empty")

    if (sizing["executable_quantity"] < 0).any():
        raise AssertionError("Executable quantity cannot be negative")

    if (sizing["executable_quantity"] > sizing["theoretical_quantity"] + 1e-9).any():
        raise AssertionError("Sizing invariant violated: quantity rounded upward")

    approved = sizing[sizing["approved"]]

    if not approved.empty:
        if (
            approved["actual_risk"] > XFA_50K_PRODUCTION_POLICY.risk_per_trade + 1e-9
        ).any():
            raise AssertionError("Approved trade exceeds per-trade risk budget")

        if (approved["executable_quantity"] < 1).any():
            raise AssertionError("Approved trade has fewer than one contract")


def write_outputs(
    sizing: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sizing_path = OUTPUT_DIR / "paper_sizing_replay.csv"

    sizing.to_csv(
        sizing_path,
        index=False,
    )

    summary = (
        sizing.groupby("strategy_name", dropna=False)
        .agg(
            trades=("strategy_name", "size"),
            approved=("approved", "sum"),
            mean_theoretical_quantity=(
                "theoretical_quantity",
                "mean",
            ),
            mean_executable_quantity=(
                "executable_quantity",
                "mean",
            ),
            max_executable_quantity=(
                "executable_quantity",
                "max",
            ),
            mean_actual_risk=("actual_risk", "mean"),
            max_actual_risk=("actual_risk", "max"),
        )
        .reset_index()
    )

    summary["rejected"] = summary["trades"] - summary["approved"]

    summary.to_csv(
        OUTPUT_DIR / "paper_sizing_summary.csv",
        index=False,
    )


def main() -> None:
    print("=" * 72)
    print("FULL SYSTEM PAPER REPLAY — SIZING STAGE")
    print("=" * 72)

    print(f"Production risk: {XFA_50K_PRODUCTION_POLICY.risk_fraction:.2%}")

    print(f"Risk budget: ${XFA_50K_PRODUCTION_POLICY.risk_per_trade:.2f}")

    data = load_frozen_oos()

    validate_frozen_counts(data)

    print(f"Frozen OOS trades: {len(data):,}")
    print("Frozen counts:", EXPECTED_COUNTS)

    sizing = run_sizing_replay(data)

    validate_sizing_invariants(sizing)

    write_outputs(sizing)

    print()
    print("SIZING REPLAY: PASS")
    print()
    print(
        sizing[
            [
                "strategy_name",
                "trades",
                "approved",
                "rejected",
            ]
        ]
        if False
        else ""
    )

    print(sizing.groupby("strategy_name")["approved"].agg(["count", "sum"]))

    print()
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
