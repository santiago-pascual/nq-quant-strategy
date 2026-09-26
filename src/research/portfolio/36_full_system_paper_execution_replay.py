from __future__ import annotations

"""
36_full_system_paper_execution_replay.py

FULL-SYSTEM PAPER EXECUTION REPLAY
==================================

Purpose
-------
Replay the already-frozen economic trade streams as a deterministic
paper-execution account.

IMPORTANT:
- Does NOT modify scripts 32 or 35.
- Does NOT regenerate strategy signals.
- Does NOT rerun XFA Monte Carlo.
- Does NOT implement the XFA state machine yet.
- Uses the corrected economic streams produced by script 32.
- Uses the execution assumptions already validated by script 35.

Baseline execution scenario:
    - 2 MNQ ticks slippage per side
    - 0% missed fills
    - $1.22 round-trip commission per contract
    - MNQ tick value = $0.50
    - XFA economic base risk = $125

The replay is intentionally deterministic. Historical trade order is
preserved exactly after stable timestamp/strategy sorting.

Outputs:
    full_system_paper_replay_summary.csv
    full_system_paper_replay_trades.csv
    full_system_paper_replay_daily.csv
"""

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "funded"

PAPER_RESULTS_DIR = (
    PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "paper"
)

PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRICT_INPUT = RESULTS_DIR / "adaptive_orb_xfa_strict_corrected.csv"
ADAPTIVE_INPUT = RESULTS_DIR / "adaptive_orb_xfa_adaptive_corrected.csv"

ECONOMICS_SUMMARY = RESULTS_DIR / "full_system_execution_economics.csv"

OUTPUT_SUMMARY = PAPER_RESULTS_DIR / "full_system_paper_replay_summary.csv"
OUTPUT_TRADES = PAPER_RESULTS_DIR / "full_system_paper_replay_trades.csv"
OUTPUT_DAILY = PAPER_RESULTS_DIR / "full_system_paper_replay_daily.csv"


# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

STARTING_BALANCE = 50_000.0
XFA_FIXED_LOSS_FLOOR = 48_000.0
XFA_PROFIT_TARGET = 53_000.0

BASE_XFA_RISK = 125.0

MNQ_TICK_VALUE = 0.50
ROUND_TRIP_COMMISSION = 1.22

SLIPPAGE_TICKS_PER_SIDE = 2
MISSED_FILL_RATE = 0.0

EXPECTED_STRICT_TRADES = 2_075
EXPECTED_ADAPTIVE_TRADES = 2_959
EXPECTED_RECOVERED_ORB = 884

EXPECTED_ADAPTIVE_FINAL_PNL = None
EXPECTED_STRICT_FINAL_PNL = None

ATOL = 1e-8


# =============================================================================
# HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise AssertionError(
            "Could not find required column.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    x = values.astype(float).to_numpy()
    peak = np.maximum.accumulate(np.r_[STARTING_BALANCE, x])
    dd = x - peak[1:]
    return float(dd.min())


def profit_factor(pnl: pd.Series) -> float:
    x = pnl.astype(float).to_numpy()

    gross_profit = float(x[x > 0].sum())
    gross_loss = float(-x[x < 0].sum())

    if gross_loss <= 0:
        return math.inf

    return gross_profit / gross_loss


def sharpe_from_daily(daily_pnl: pd.Series) -> float:
    x = daily_pnl.astype(float).to_numpy()

    if len(x) < 2:
        return np.nan

    std = float(np.std(x, ddof=1))

    if std <= 0:
        return np.nan

    return float(np.sqrt(252.0) * np.mean(x) / std)


def sortino_from_daily(daily_pnl: pd.Series) -> float:
    x = daily_pnl.astype(float).to_numpy()

    if len(x) < 2:
        return np.nan

    downside = x[x < 0]

    if len(downside) == 0:
        return math.inf

    downside_rms = float(np.sqrt(np.mean(downside**2)))

    if downside_rms <= 0:
        return np.nan

    return float(np.sqrt(252.0) * np.mean(x) / downside_rms)


def normalize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    strategy_col = find_column(
        df,
        ["strategy_name", "strategy", "Strategy", "name"],
    )

    entry_col = find_column(
        df,
        ["entry_timestamp", "timestamp", "entry_time"],
    )

    exit_col = find_column(
        df,
        ["exit_timestamp", "exit_time"],
        required=False,
    )

    xfa_r_col = find_column(
        df,
        ["xfa_r_multiple", "xfa_R_multiple", "xfa_r"],
    )

    quantity_col = find_column(
        df,
        [
            "adaptive_quantity",
            "strict_quantity",
            "quantity",
            "scenario_quantity",
            "executable_quantity",
            "contracts",
        ],
        required=False,
    )

    actual_risk_col = find_column(
        df,
        [
            "adaptive_actual_risk",
            "strict_actual_risk",
            "actual_risk",
            "scenario_actual_risk",
        ],
        required=False,
    )

    risk_per_contract_col = find_column(
        df,
        [
            "risk_per_contract",
            "adaptive_risk_per_contract",
        ],
        required=False,
    )

    entry_price_col = find_column(
        df,
        ["entry_price", "entry", "entry_fill_price"],
        required=False,
    )

    exit_price_col = find_column(
        df,
        ["exit_price", "exit", "exit_fill_price"],
        required=False,
    )

    out = df.copy()

    out["_strategy_name"] = out[strategy_col].astype(str).str.strip().str.upper()

    out["_entry_timestamp"] = pd.to_datetime(
        out[entry_col],
        utc=True,
        errors="coerce",
    )

    if exit_col is not None:
        out["_exit_timestamp"] = pd.to_datetime(
            out[exit_col],
            utc=True,
            errors="coerce",
        )
    else:
        out["_exit_timestamp"] = pd.NaT

    out["_xfa_r_multiple"] = pd.to_numeric(
        out[xfa_r_col],
        errors="coerce",
    )

    if quantity_col is not None:
        quantity = pd.to_numeric(
            out[quantity_col],
            errors="coerce",
        )
    elif actual_risk_col is not None and risk_per_contract_col is not None:
        actual_risk = pd.to_numeric(
            out[actual_risk_col],
            errors="coerce",
        )
        risk_per_contract = pd.to_numeric(
            out[risk_per_contract_col],
            errors="coerce",
        )

        quantity = actual_risk / risk_per_contract
    else:
        raise AssertionError(
            "Could not determine contract quantity from corrected stream."
        )

    out["_contracts"] = quantity.astype(float)

    if entry_price_col is not None:
        out["_entry_price"] = pd.to_numeric(
            out[entry_price_col],
            errors="coerce",
        )
    else:
        out["_entry_price"] = np.nan

    if exit_price_col is not None:
        out["_exit_price"] = pd.to_numeric(
            out[exit_price_col],
            errors="coerce",
        )
    else:
        out["_exit_price"] = np.nan

    required_mask = (
        out["_entry_timestamp"].notna()
        & out["_xfa_r_multiple"].notna()
        & out["_contracts"].notna()
    )

    if not required_mask.all():
        bad = int((~required_mask).sum())
        raise AssertionError(f"Corrected stream contains {bad} invalid economic rows.")

    out = out.sort_values(
        ["_entry_timestamp", "_strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


def load_stream(
    path: Path,
    expected_rows: int,
    scenario: str,
) -> pd.DataFrame:
    banner(f"LOAD {scenario}")

    if not path.exists():
        raise FileNotFoundError(f"Missing input file:\n{path}")

    df = pd.read_csv(path)

    print(f"Input: {path}")
    print(f"Raw rows: {len(df):,}")

    out = normalize_strategy(df)

    if len(out) != expected_rows:
        raise AssertionError(
            f"{scenario}: expected {expected_rows:,} rows, got {len(out):,}."
        )

    if (out["_contracts"] <= 0).any():
        raise AssertionError(f"{scenario}: non-positive contract quantity detected.")

    if not np.all(np.isfinite(out["_contracts"].to_numpy(dtype=float))):
        raise AssertionError(f"{scenario}: non-finite contract quantity detected.")

    print(f"Validated rows: {len(out):,}")
    print(
        f"Contracts: mean={out['_contracts'].mean():.4f}, "
        f"max={out['_contracts'].max():.0f}"
    )

    return out


# =============================================================================
# EXECUTION MODEL
# =============================================================================


def execute_trade(
    row: pd.Series,
    *,
    scenario: str,
    equity_before: float,
) -> dict:
    """
    Convert a frozen XFA R result into paper execution economics.

    Historical strategy outcome:
        xfa_r_multiple * $125

    Execution friction:
        commission = contracts * $1.22
        slippage  = contracts * ticks * $0.50 * 2

    No random missed fills are applied here because this is the deterministic
    baseline paper replay. Script 35 already stress-tested missed fills.
    """

    contracts = float(row["_contracts"])
    xfa_r = float(row["_xfa_r_multiple"])

    gross_pnl = xfa_r * BASE_XFA_RISK

    commission = contracts * ROUND_TRIP_COMMISSION

    slippage = contracts * SLIPPAGE_TICKS_PER_SIDE * MNQ_TICK_VALUE * 2.0

    net_pnl = gross_pnl - commission - slippage

    equity_after = equity_before + net_pnl

    return {
        "scenario": scenario,
        "strategy_name": row["_strategy_name"],
        "entry_timestamp": row["_entry_timestamp"],
        "exit_timestamp": row["_exit_timestamp"],
        "contracts": contracts,
        "xfa_r_multiple": xfa_r,
        "gross_pnl": gross_pnl,
        "commission": commission,
        "slippage": slippage,
        "net_pnl": net_pnl,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "loss_floor_breached": equity_after <= XFA_FIXED_LOSS_FLOOR,
        "profit_target_reached": equity_after >= XFA_PROFIT_TARGET,
        "entry_price": row["_entry_price"],
        "exit_price": row["_exit_price"],
    }


# =============================================================================
# REPLAY
# =============================================================================


def replay_stream(
    stream: pd.DataFrame,
    *,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    banner(f"REPLAY {scenario}")

    equity = STARTING_BALANCE

    records: list[dict] = []

    for _, row in stream.iterrows():
        record = execute_trade(
            row,
            scenario=scenario,
            equity_before=equity,
        )

        records.append(record)

        equity = float(record["equity_after"])

    trades = pd.DataFrame(records)

    if trades.empty:
        raise AssertionError(f"{scenario}: replay produced zero trades.")

    # -------------------------------------------------------------------------
    # Daily aggregation
    # -------------------------------------------------------------------------

    trades["_date_ny"] = (
        trades["entry_timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    )

    daily = (
        trades.groupby("_date_ny", sort=True)
        .agg(
            trades=("net_pnl", "size"),
            gross_pnl=("gross_pnl", "sum"),
            commission=("commission", "sum"),
            slippage=("slippage", "sum"),
            net_pnl=("net_pnl", "sum"),
        )
        .reset_index()
    )

    daily["equity"] = STARTING_BALANCE + daily["net_pnl"].cumsum()

    daily["running_peak"] = (STARTING_BALANCE + daily["net_pnl"].cumsum()).cummax()

    daily["drawdown"] = daily["equity"] - daily["running_peak"]

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    net = trades["net_pnl"].astype(float)

    final_equity = float(trades["equity_after"].iloc[-1])
    total_net = float(net.sum())

    gross_profit = float(net[net > 0].sum())
    gross_loss = float(-net[net < 0].sum())

    summary = {
        "scenario": scenario,
        "start_date": str(
            trades["entry_timestamp"].min().tz_convert("America/New_York").date()
        ),
        "end_date": str(
            trades["entry_timestamp"].max().tz_convert("America/New_York").date()
        ),
        "historical_trades": int(len(trades)),
        "historical_days": int(len(daily)),
        "starting_balance": STARTING_BALANCE,
        "final_equity": final_equity,
        "net_pnl": total_net,
        "return_pct": 100.0 * total_net / STARTING_BALANCE,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (gross_profit / gross_loss if gross_loss > 0 else math.inf),
        "expectancy_per_trade": float(net.mean()),
        "win_rate": float((net > 0).mean()),
        "max_drawdown": float(daily["drawdown"].min()),
        "daily_sharpe": sharpe_from_daily(daily["net_pnl"]),
        "daily_sortino": sortino_from_daily(daily["net_pnl"]),
        "total_commission": float(trades["commission"].sum()),
        "total_slippage": float(trades["slippage"].sum()),
        "total_execution_cost": float(
            trades["commission"].sum() + trades["slippage"].sum()
        ),
        "min_equity": float(trades["equity_after"].min()),
        "loss_floor_breached": bool(
            (trades["equity_after"] <= XFA_FIXED_LOSS_FLOOR).any()
        ),
        "profit_target_reached": bool(
            (trades["equity_after"] >= XFA_PROFIT_TARGET).any()
        ),
        "loss_floor": XFA_FIXED_LOSS_FLOOR,
        "profit_target": XFA_PROFIT_TARGET,
        "slippage_ticks_per_side": SLIPPAGE_TICKS_PER_SIDE,
        "missed_fill_rate": MISSED_FILL_RATE,
        "commission_per_contract_round_trip": ROUND_TRIP_COMMISSION,
    }

    return trades, daily, summary


# =============================================================================
# PARITY WITH SCRIPT 35
# =============================================================================


def load_35_reference() -> pd.DataFrame | None:
    if not ECONOMICS_SUMMARY.exists():
        print()
        print("WARNING: script 35 economics summary not found.")
        print(f"Expected: {ECONOMICS_SUMMARY}")
        return None

    df = pd.read_csv(ECONOMICS_SUMMARY)

    required = [
        "scenario",
        "slippage_ticks_per_side",
        "missed_fill_rate",
        "net_pnl",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(
            "WARNING: script 35 output is missing columns:",
            missing,
        )
        return None

    return df


def audit_against_35(
    summaries: pd.DataFrame,
) -> None:
    banner("SCRIPT 35 PARITY AUDIT")

    reference = load_35_reference()

    if reference is None:
        print("Parity audit: SKIPPED — reference unavailable.")
        return

    reference = reference.loc[
        reference["missed_fill_rate"].astype(float) == MISSED_FILL_RATE
    ].copy()

    reference = reference.loc[
        reference["slippage_ticks_per_side"].astype(int) == SLIPPAGE_TICKS_PER_SIDE
    ].copy()

    if reference.empty:
        raise AssertionError(
            "Script 35 does not contain the baseline 2-tick / 0%-missed scenario."
        )

    checks = []

    for scenario in summaries["scenario"]:
        actual = summaries.loc[
            summaries["scenario"] == scenario,
            "net_pnl",
        ].iloc[0]

        ref = reference.loc[
            reference["scenario"] == scenario,
            "net_pnl",
        ]

        if ref.empty:
            raise AssertionError(f"Script 35 reference missing scenario {scenario}.")

        expected = float(ref.iloc[0])
        diff = float(actual - expected)

        passed = abs(diff) <= 1e-6

        checks.append(
            {
                "scenario": scenario,
                "paper_replay_net_pnl": actual,
                "script35_net_pnl": expected,
                "difference": diff,
                "status": "PASS" if passed else "FAIL",
            }
        )

    audit = pd.DataFrame(checks)

    print(audit.to_string(index=False))

    if not (audit["status"] == "PASS").all():
        raise AssertionError("Script 35 parity audit FAILED.")

    print()
    print("Script 35 parity: PASS")


# =============================================================================
# STRUCTURAL AUDITS
# =============================================================================


def audit_streams(
    strict: pd.DataFrame,
    adaptive: pd.DataFrame,
) -> None:
    banner("FROZEN STREAM AUDIT")

    if len(strict) != EXPECTED_STRICT_TRADES:
        raise AssertionError("Strict trade count mismatch.")

    if len(adaptive) != EXPECTED_ADAPTIVE_TRADES:
        raise AssertionError("Adaptive trade count mismatch.")

    recovered = 0

    if "adaptive_quantity" in adaptive.columns:
        aq = pd.to_numeric(
            adaptive["adaptive_quantity"],
            errors="coerce",
        )

        eq = pd.to_numeric(
            adaptive.get(
                "executable_quantity",
                pd.Series(np.nan, index=adaptive.index),
            ),
            errors="coerce",
        )

        recovered = int(
            ((adaptive["_strategy_name"] == "ORB") & (eq == 0) & (aq >= 1)).sum()
        )

    print(f"Strict trades:   {len(strict):,}")
    print(f"Adaptive trades: {len(adaptive):,}")
    print(f"Recovered ORB:   {recovered:,}")

    if recovered != EXPECTED_RECOVERED_ORB:
        raise AssertionError(
            f"Expected {EXPECTED_RECOVERED_ORB:,} recovered ORBs, got {recovered:,}."
        )

    # Exact historical ordering audit.
    for name, df in [
        ("STRICT", strict),
        ("ADAPTIVE", adaptive),
    ]:
        timestamps = df["_entry_timestamp"].astype("int64").to_numpy()

        if np.any(np.diff(timestamps) < 0):
            raise AssertionError(
                f"{name}: entry timestamps are not monotonically ordered."
            )

    print("Frozen stream counts: PASS")
    print("Historical order: PASS")
    print("Adaptive ORB recovery: PASS")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print("=" * 100)
    print("FULL-SYSTEM PAPER EXECUTION REPLAY")
    print("=" * 100)
    print()
    print("This replay uses the already validated 32/35 economic artifacts.")
    print("No strategy parameters are regenerated.")
    print("No XFA Monte Carlo is rerun.")
    print()
    print(f"Starting balance:        ${STARTING_BALANCE:,.2f}")
    print(f"XFA loss floor:          ${XFA_FIXED_LOSS_FLOOR:,.2f}")
    print(f"XFA profit target:       ${XFA_PROFIT_TARGET:,.2f}")
    print(f"Slippage:                {SLIPPAGE_TICKS_PER_SIDE} ticks/side")
    print(f"Missed fills:             {MISSED_FILL_RATE:.0%}")
    print(f"Commission:               ${ROUND_TRIP_COMMISSION:.2f}/contract RT")
    print()

    strict = load_stream(
        STRICT_INPUT,
        EXPECTED_STRICT_TRADES,
        "STRICT_025",
    )

    adaptive = load_stream(
        ADAPTIVE_INPUT,
        EXPECTED_ADAPTIVE_TRADES,
        "ADAPTIVE_ORB_060",
    )

    audit_streams(strict, adaptive)

    strict_trades, strict_daily, strict_summary = replay_stream(
        strict,
        scenario="STRICT_025",
    )

    adaptive_trades, adaptive_daily, adaptive_summary = replay_stream(
        adaptive,
        scenario="ADAPTIVE_ORB_060",
    )

    summaries = pd.DataFrame([strict_summary, adaptive_summary])

    audit_against_35(summaries)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    all_trades = pd.concat(
        [strict_trades, adaptive_trades],
        ignore_index=True,
    )

    all_daily = pd.concat(
        [
            strict_daily.assign(scenario="STRICT_025"),
            adaptive_daily.assign(scenario="ADAPTIVE_ORB_060"),
        ],
        ignore_index=True,
    )

    summaries.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    all_trades.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    all_daily.to_csv(
        OUTPUT_DAILY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    banner("PAPER REPLAY RESULTS")

    print(
        summaries[
            [
                "scenario",
                "historical_trades",
                "historical_days",
                "final_equity",
                "net_pnl",
                "return_pct",
                "expectancy_per_trade",
                "profit_factor",
                "max_drawdown",
                "daily_sharpe",
                "daily_sortino",
                "total_execution_cost",
                "min_equity",
                "loss_floor_breached",
                "profit_target_reached",
            ]
        ].to_string(index=False)
    )

    banner("OUTPUTS")

    print(OUTPUT_SUMMARY)
    print(OUTPUT_TRADES)
    print(OUTPUT_DAILY)

    banner("FULL-SYSTEM PAPER EXECUTION REPLAY — PASS")


if __name__ == "__main__":
    main()
