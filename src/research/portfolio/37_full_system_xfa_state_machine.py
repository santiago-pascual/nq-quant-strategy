"""
37_full_system_xfa_state_machine.py

Deterministic XFA state-machine replay over the already validated economic
trade streams produced by scripts 32/35/36.

IMPORTANT:
- Does NOT regenerate strategy signals.
- Does NOT rerun Monte Carlo.
- Does NOT modify scripts 32/35/36.
- Preserves the historical trade order inside each corrected stream.
- Applies the XFA account state machine sequentially.

XFA policy copied from script 22:
    starting balance       = $50,000
    fixed loss floor       = $48,000
    payout intervals       = 20 / 21 / 22 trades
    payout amounts         = $500 ... $2,000
    minimum winning days   = 5
    winning day threshold  = $150

The stream already contains xfa_r_multiple, which is the economic return
normalized to the base $125 XFA risk. Therefore:

    trade P&L = xfa_r_multiple * $125

This script evaluates every supported payout policy on the COMPLETE historical
stream. It is a state-machine audit, not a strategy backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "paper"

FUNDED_RESULTS_DIR = (
    PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "funded"
)

STRICT_INPUT = FUNDED_RESULTS_DIR / "adaptive_orb_xfa_strict_corrected.csv"
ADAPTIVE_INPUT = FUNDED_RESULTS_DIR / "adaptive_orb_xfa_adaptive_corrected.csv"

SUMMARY_OUTPUT = RESULTS_DIR / "full_system_xfa_state_machine_summary.csv"
TRADES_OUTPUT = RESULTS_DIR / "full_system_xfa_state_machine_trades.csv"
DAILY_OUTPUT = RESULTS_DIR / "full_system_xfa_state_machine_daily.csv"
EVENTS_OUTPUT = RESULTS_DIR / "full_system_xfa_state_machine_events.csv"


# =============================================================================
# XFA POLICY — EXACTLY AS VALIDATED IN SCRIPT 22
# =============================================================================

STARTING_BALANCE = 50_000.0
PROFIT_TARGET = 3_000.0
MAX_LOSS_LIMIT = 2_000.0
LOSS_FLOOR = 48_000.0

BASE_XFA_RISK = 125.0

PAYOUT_INTERVALS = [20, 21, 22]
PAYOUT_AMOUNTS = [500.0, 750.0, 1_000.0, 1_250.0, 1_500.0, 1_750.0, 2_000.0]

MIN_WINNING_DAYS = 5
MIN_WINNING_DAY_PROFIT = 150.0

MAX_XFA_TRADES = 2_000

EXPECTED_STRICT = 2_075
EXPECTED_ADAPTIVE = 2_959
EXPECTED_RECOVERED_ORB = 884


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class XFAState:
    balance: float = STARTING_BALANCE
    total_withdrawn: float = 0.0

    trades: int = 0
    payout_count: int = 0

    winning_days: int = 0
    qualified_day_token: Any = None
    current_day: Any = None
    day_pnl: float = 0.0
    day_won: bool = False

    failed: bool = False
    failure_reason: str | None = None
    passed: bool = False

    max_balance: float = STARTING_BALANCE
    min_balance: float = STARTING_BALANCE

    @property
    def equity(self) -> float:
        return self.balance

    @property
    def max_drawdown(self) -> float:
        return self.min_balance - self.max_balance


# =============================================================================
# HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def load_stream(path: Path, expected_rows: int, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} input missing:\n{path}")

    df = pd.read_csv(path)

    required = {
        "strategy_name",
        "entry_timestamp",
        "xfa_r_multiple",
    }

    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"{name}: missing required columns: {sorted(missing)}")

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    df["xfa_r_multiple"] = pd.to_numeric(
        df["xfa_r_multiple"],
        errors="coerce",
    )

    if df["entry_timestamp"].isna().any():
        raise AssertionError(f"{name}: invalid entry_timestamp values.")

    if df["xfa_r_multiple"].isna().any():
        raise AssertionError(f"{name}: invalid xfa_r_multiple values.")

    if len(df) != expected_rows:
        raise AssertionError(
            f"{name}: expected {expected_rows:,} rows, got {len(df):,}"
        )

    # Stable ordering.  This is only a canonicalization of the already
    # historical stream; no trade is shuffled.
    df = df.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def detect_recovered_orb(df: pd.DataFrame) -> int:
    if "strategy_name" not in df.columns:
        return 0

    return int((df["strategy_name"].astype(str).str.upper().eq("ORB")).sum())


def run_policy(
    df: pd.DataFrame,
    payout_interval: int,
    payout_amount: float,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    state = XFAState()

    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    current_day = None

    for i, row in df.iterrows():
        if state.failed or state.passed:
            break

        ts = row["entry_timestamp"]
        day = ts.tz_convert("America/New_York").date()

        if day != current_day:
            if current_day is not None:
                daily_rows.append(
                    {
                        "policy": label,
                        "date_ny": str(current_day),
                        "day_pnl": state.day_pnl,
                        "winning_day": int(state.day_won),
                        "winning_days_cumulative": state.winning_days,
                        "balance_end": state.balance,
                        "payout_count": state.payout_count,
                        "failed": state.failed,
                    }
                )

            current_day = day
            state.current_day = day
            state.day_pnl = 0.0
            state.day_won = False

        r = float(row["xfa_r_multiple"])
        pnl = r * BASE_XFA_RISK

        balance_before = state.balance
        withdrawn_before = state.total_withdrawn
        winning_days_before = state.winning_days

        state.balance += pnl
        state.trades += 1
        state.day_pnl += pnl

        state.max_balance = max(state.max_balance, state.balance)
        state.min_balance = min(state.min_balance, state.balance)

        winning_day_qualified = False

        if not state.day_won and state.day_pnl >= MIN_WINNING_DAY_PROFIT:
            state.day_won = True

            if state.qualified_day_token != day:
                state.winning_days += 1
                state.qualified_day_token = day
                winning_day_qualified = True

        payout = 0.0
        payout_triggered = False

        # Script 22 checks MLL BEFORE payout.
        if state.balance <= LOSS_FLOOR:
            state.failed = True
            state.failure_reason = "MLL"

            event_rows.append(
                {
                    "policy": label,
                    "event": "FAILURE",
                    "trade_index": state.trades,
                    "timestamp": ts,
                    "reason": "MLL",
                    "balance_before": balance_before,
                    "balance_after": state.balance,
                    "payout": 0.0,
                    "total_withdrawn": state.total_withdrawn,
                }
            )

        # Payout is checked after the MLL check.
        elif state.trades % payout_interval == 0:
            if state.winning_days >= MIN_WINNING_DAYS:
                profit_above_start = max(
                    0.0,
                    state.balance - STARTING_BALANCE,
                )

                payout = min(
                    payout_amount,
                    profit_above_start,
                )

                if payout > 0:
                    state.balance -= payout
                    state.total_withdrawn += payout
                    state.payout_count += 1
                    state.winning_days = 0
                    payout_triggered = True

                    event_rows.append(
                        {
                            "policy": label,
                            "event": "PAYOUT",
                            "trade_index": state.trades,
                            "timestamp": ts,
                            "reason": "QUALIFIED_PAYOUT",
                            "balance_before": balance_before,
                            "balance_after": state.balance,
                            "payout": payout,
                            "total_withdrawn": state.total_withdrawn,
                        }
                    )

        # Profit target is informational here.  The XFA policy in script 22
        # is payout-driven rather than a permanent "pass and stop" state.
        profit_target_reached = state.balance >= (STARTING_BALANCE + PROFIT_TARGET)

        if profit_target_reached and not state.passed:
            state.passed = True

            event_rows.append(
                {
                    "policy": label,
                    "event": "PROFIT_TARGET",
                    "trade_index": state.trades,
                    "timestamp": ts,
                    "reason": "BALANCE_REACHED_TARGET",
                    "balance_before": balance_before,
                    "balance_after": state.balance,
                    "payout": 0.0,
                    "total_withdrawn": state.total_withdrawn,
                }
            )

        trade_rows.append(
            {
                "policy": label,
                "trade_index": state.trades,
                "timestamp": ts,
                "date_ny": str(day),
                "strategy_name": row["strategy_name"],
                "xfa_r_multiple": r,
                "pnl": pnl,
                "balance_before": balance_before,
                "balance_after_trade": state.balance + payout,
                "payout": payout,
                "balance_after": state.balance,
                "total_withdrawn": state.total_withdrawn,
                "winning_day_qualified": int(winning_day_qualified),
                "winning_days_before": winning_days_before,
                "winning_days_after": state.winning_days,
                "payout_triggered": int(payout_triggered),
                "failed": int(state.failed),
                "failure_reason": state.failure_reason,
                "profit_target_reached": int(profit_target_reached),
            }
        )

    # Close final day.
    if current_day is not None:
        daily_rows.append(
            {
                "policy": label,
                "date_ny": str(current_day),
                "day_pnl": state.day_pnl,
                "winning_day": int(state.day_won),
                "winning_days_cumulative": state.winning_days,
                "balance_end": state.balance,
                "payout_count": state.payout_count,
                "failed": state.failed,
            }
        )

    trade_df = pd.DataFrame(trade_rows)
    daily_df = pd.DataFrame(daily_rows)

    final_balance = state.balance
    net_value = final_balance + state.total_withdrawn

    summary = {
        "policy": label,
        "payout_interval": payout_interval,
        "payout_amount": payout_amount,
        "historical_rows_available": len(df),
        "trades_executed": state.trades,
        "historical_days_executed": (
            int(trade_df["date_ny"].nunique()) if len(trade_df) else 0
        ),
        "failed": int(state.failed),
        "failure_reason": state.failure_reason,
        "passed": int(state.passed),
        "final_balance": final_balance,
        "total_withdrawn": state.total_withdrawn,
        "net_value": net_value,
        "payout_count": state.payout_count,
        "min_balance": state.min_balance,
        "max_balance": state.max_balance,
        "max_drawdown": state.max_drawdown,
        "remaining_buffer_to_loss_floor": (final_balance - LOSS_FLOOR),
        "profit_vs_start": final_balance - STARTING_BALANCE,
    }

    return summary, trade_df, daily_df


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    banner("FULL-SYSTEM XFA STATE MACHINE — SCRIPT 37")

    print(f"Starting balance : ${STARTING_BALANCE:,.2f}")
    print(f"Loss floor       : ${LOSS_FLOOR:,.2f}")
    print(f"Base XFA risk    : ${BASE_XFA_RISK:,.2f}")
    print(f"Payout intervals : {PAYOUT_INTERVALS}")
    print(f"Payout amounts   : {PAYOUT_AMOUNTS}")
    print(f"Winning days     : {MIN_WINNING_DAYS}")
    print(f"Winning day P&L  : ${MIN_WINNING_DAY_PROFIT:,.2f}")
    print(f"Max XFA trades   : {MAX_XFA_TRADES:,}")

    banner("LOAD CORRECTED ECONOMIC STREAMS")

    strict = load_stream(
        STRICT_INPUT,
        EXPECTED_STRICT,
        "STRICT_025",
    )

    adaptive = load_stream(
        ADAPTIVE_INPUT,
        EXPECTED_ADAPTIVE,
        "ADAPTIVE_ORB_060",
    )

    print(f"STRICT rows   : {len(strict):,}")
    print(f"ADAPTIVE rows : {len(adaptive):,}")

    # Structural audit.
    strict_orb = detect_recovered_orb(strict)
    adaptive_orb = detect_recovered_orb(adaptive)

    print(f"STRICT ORB rows   : {strict_orb:,}")
    print(f"ADAPTIVE ORB rows : {adaptive_orb:,}")

    if adaptive_orb != EXPECTED_RECOVERED_ORB:
        raise AssertionError(
            f"Adaptive ORB recovery expected {EXPECTED_RECOVERED_ORB}, "
            f"got {adaptive_orb}"
        )

    if strict_orb != 0:
        raise AssertionError(
            f"Strict stream unexpectedly contains {strict_orb} ORB rows."
        )

    print("STRUCTURAL STREAM AUDIT: PASS")

    all_summaries: list[dict[str, Any]] = []
    all_trades: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []

    for stream_name, df in [
        ("STRICT_025", strict),
        ("ADAPTIVE_ORB_060", adaptive),
    ]:
        banner(f"RUN XFA STATE MACHINE — {stream_name}")

        for interval in PAYOUT_INTERVALS:
            for amount in PAYOUT_AMOUNTS:
                label = f"{stream_name}|P{interval}|${amount:.0f}"

                summary, trades, daily = run_policy(
                    df=df,
                    payout_interval=interval,
                    payout_amount=amount,
                    label=label,
                )

                all_summaries.append(summary)

                if not trades.empty:
                    all_trades.append(trades)

                if not daily.empty:
                    all_daily.append(daily)

                # Extract events from the trade-level state transitions.
                events = []

                for _, t in trades.iterrows():
                    if int(t["payout_triggered"]):
                        events.append(
                            {
                                "policy": label,
                                "event": "PAYOUT",
                                "trade_index": int(t["trade_index"]),
                                "timestamp": t["timestamp"],
                                "reason": "QUALIFIED_PAYOUT",
                                "balance_after": t["balance_after"],
                                "payout": t["payout"],
                                "total_withdrawn": t["total_withdrawn"],
                            }
                        )

                    if int(t["failed"]):
                        events.append(
                            {
                                "policy": label,
                                "event": "FAILURE",
                                "trade_index": int(t["trade_index"]),
                                "timestamp": t["timestamp"],
                                "reason": t["failure_reason"],
                                "balance_after": t["balance_after"],
                                "payout": t["payout"],
                                "total_withdrawn": t["total_withdrawn"],
                            }
                        )

                    if int(t["profit_target_reached"]):
                        events.append(
                            {
                                "policy": label,
                                "event": "PROFIT_TARGET",
                                "trade_index": int(t["trade_index"]),
                                "timestamp": t["timestamp"],
                                "reason": "BALANCE_REACHED_TARGET",
                                "balance_after": t["balance_after"],
                                "payout": 0.0,
                                "total_withdrawn": t["total_withdrawn"],
                            }
                        )

                if events:
                    all_events.append(pd.DataFrame(events))

    summary_df = pd.DataFrame(all_summaries)

    trades_df = (
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    )

    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()

    events_df = (
        pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    )

    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    trades_df.to_csv(TRADES_OUTPUT, index=False)
    daily_df.to_csv(DAILY_OUTPUT, index=False)
    events_df.to_csv(EVENTS_OUTPUT, index=False)

    banner("STATE MACHINE AUDIT")

    expected_policies = 2 * len(PAYOUT_INTERVALS) * len(PAYOUT_AMOUNTS)

    if len(summary_df) != expected_policies:
        raise AssertionError(
            f"Expected {expected_policies} policy rows, got {len(summary_df)}"
        )

    print(f"Policy rows : {len(summary_df)}")
    print(f"Trade rows  : {len(trades_df):,}")
    print(f"Daily rows  : {len(daily_df):,}")
    print(f"Event rows  : {len(events_df):,}")

    print()
    print("POLICY RESULTS")
    print(
        summary_df[
            [
                "policy",
                "trades_executed",
                "failed",
                "failure_reason",
                "passed",
                "final_balance",
                "total_withdrawn",
                "net_value",
                "payout_count",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Summary : {SUMMARY_OUTPUT}")
    print(f"Trades  : {TRADES_OUTPUT}")
    print(f"Daily   : {DAILY_OUTPUT}")
    print(f"Events  : {EVENTS_OUTPUT}")

    print()
    print("FULL-SYSTEM XFA STATE MACHINE — PASS")


if __name__ == "__main__":
    main()
