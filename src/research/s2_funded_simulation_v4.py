from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# S2 FUNDED ACCOUNT SIMULATION V4
# ============================================================
#
# PURPOSE
# -------
# Test the FROZEN S2 strategy under funded-account rules.
#
# PART A:
#   $50K Trading Combine
#
# PART B:
#   XFA / Funded
#
# IMPORTANT:
#   Strategy parameters are completely frozen.
#   Only account-management policy is tested.
#
# V4 FIX:
#   Combine simulation has a MAXIMUM OF 60 CALENDAR DAYS.
#
#   Every simulation ends as:
#
#       PASS
#       FAIL
#       TIMEOUT_60D
#
#   No infinite simulations.
#
# ============================================================


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = RESULTS_DIR / "s2_selective_execution_B_trades.csv"


# ============================================================
# STRATEGY — FROZEN
# ============================================================

STRATEGY_NAME = "S2 B-SELECTIVE"

QUALITY_THRESHOLD = 0.75

RR = 1.30

TAIL_PERCENT = 17.5

STOP_POINTS = 20

HORIZON_BARS = 15


# ============================================================
# COMBINE
# ============================================================

STARTING_BALANCE = 50_000.0

PROFIT_TARGET = 3_000.0

MAX_LOSS = 2_000.0

INITIAL_MLL_FLOOR = STARTING_BALANCE - MAX_LOSS


# ============================================================
# CONSISTENCY
# ============================================================

CONSISTENCY_PERCENT = 0.50


# ============================================================
# RISK SCENARIOS
# ============================================================

RISK_SCENARIOS = {
    "0.25%": {
        "initial": 0.0025,
        "after_threshold": 0.0025,
        "threshold": None,
    },
    "0.50%": {
        "initial": 0.0050,
        "after_threshold": 0.0050,
        "threshold": None,
    },
    "0.75%": {
        "initial": 0.0075,
        "after_threshold": 0.0075,
        "threshold": None,
    },
    "1.00%": {
        "initial": 0.0100,
        "after_threshold": 0.0100,
        "threshold": None,
    },
    "0.50_to_1.00": {
        "initial": 0.0050,
        "after_threshold": 0.0100,
        "threshold": 1_000.0,
    },
}


# ============================================================
# SUBSCRIPTION
# ============================================================

SUBSCRIPTION_PRICE = 49.0

SUBSCRIPTION_CYCLE_DAYS = 30


# ============================================================
# COMBINE TIME LIMIT
# ============================================================

MAX_CALENDAR_DAYS = 60


# ============================================================
# MONTE CARLO
# ============================================================

N_SIMULATIONS = 50_000

RANDOM_SEED = 42


# ============================================================
# XFA
# ============================================================

XFA_STARTING_BALANCE = 50_000.0

XFA_MLL = 2_000.0

MIN_WINNING_DAYS = 5

MIN_WINNING_DAY_PROFIT = 150.0

XFA_MONTH_DAYS = 30

XFA_PAYOUT_AMOUNTS = [
    500,
    750,
    1_000,
    1_250,
    1_500,
    1_750,
    2_000,
]


# ============================================================
# LOAD DATA
# ============================================================


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"\nInput file not found:\n{INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "entry_timestamp",
        "net_R",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Handle mixed timezone data.
    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], utc=True)

    # Convert to New York time for trading date.
    df["entry_timestamp_ny"] = df["entry_timestamp"].dt.tz_convert("America/New_York")

    df["trading_date"] = df["entry_timestamp_ny"].dt.date

    df["net_R"] = pd.to_numeric(df["net_R"], errors="coerce")

    df = df.dropna(subset=["net_R"])

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    return df


# ============================================================
# BUILD HISTORICAL RTH DAY BLOCKS
# ============================================================


def build_day_blocks(df):

    blocks = []

    grouped = df.groupby("trading_date", sort=True)

    for trading_date, day_df in grouped:
        returns = day_df["net_R"].to_numpy(dtype=float)

        if len(returns) == 0:
            continue

        blocks.append(
            {
                "date": trading_date,
                "returns": returns,
                "trades": len(returns),
            }
        )

    return blocks


# ============================================================
# NEXT WEEKDAY
# ============================================================


def next_weekday(date):

    date = date + timedelta(days=1)

    while date.weekday() >= 5:
        date += timedelta(days=1)

    return date


# ============================================================
# RISK
# ============================================================


def get_risk_pct(
    balance,
    scenario,
):

    threshold = scenario["threshold"]

    if threshold is not None:
        profit_from_start = balance - STARTING_BALANCE

        if profit_from_start >= threshold:
            return scenario["after_threshold"]

    return scenario["initial"]


# ============================================================
# COMBINE SIMULATION
# ============================================================


def simulate_combine(
    day_blocks,
    scenario,
    rng,
):

    balance = STARTING_BALANCE

    # Trailing MLL.
    mll_floor = INITIAL_MLL_FLOOR

    best_day_profit = 0.0

    total_trades = 0

    trading_days = 0

    calendar_days = 0

    daily_pnl = []

    equity_curve = [balance]

    passed = False

    failed = False

    timeout = False

    failure_reason = None

    # --------------------------------------------------------
    # We use a synthetic calendar.
    #
    # Start on Monday.
    #
    # The strategy can only generate trades on weekdays.
    #
    # Maximum = 60 calendar days.
    # --------------------------------------------------------

    current_date = pd.Timestamp("2026-01-05").date()

    while calendar_days < MAX_CALENDAR_DAYS and not passed and not failed:
        calendar_days += 1

        # Weekends = no trading.
        if current_date.weekday() >= 5:
            current_date = current_date + timedelta(days=1)

            continue

        # ----------------------------------------------------
        # Sample one complete historical RTH day.
        #
        # This preserves the actual number and sequence of
        # S2 trades that happened on that historical day.
        # ----------------------------------------------------

        block = rng.choice(day_blocks)

        trading_days += 1

        current_day_pnl = 0.0

        # ----------------------------------------------------
        # Execute the historical S2 trades.
        # ----------------------------------------------------

        for r in block["returns"]:
            risk_pct = get_risk_pct(balance, scenario)

            risk_dollars = STARTING_BALANCE * risk_pct

            pnl = r * risk_dollars

            balance += pnl

            current_day_pnl += pnl

            total_trades += 1

            equity_curve.append(balance)

            # ------------------------------------------------
            # MLL
            # ------------------------------------------------

            if balance <= mll_floor:
                failed = True

                failure_reason = "MLL"

                break

        if failed:
            break

        # ----------------------------------------------------
        # Daily result
        # ----------------------------------------------------

        daily_pnl.append(current_day_pnl)

        best_day_profit = max(best_day_profit, current_day_pnl)

        # ----------------------------------------------------
        # Update trailing MLL.
        #
        # It cannot move above starting balance.
        # ----------------------------------------------------

        mll_floor = min(STARTING_BALANCE, max(mll_floor, balance - MAX_LOSS))

        # ----------------------------------------------------
        # PROFIT TARGET + CONSISTENCY
        #
        # Target = +$3,000.
        #
        # Best day cannot represent >=50% of total target.
        #
        # Therefore best day must be < $1,500.
        # ----------------------------------------------------

        profit = balance - STARTING_BALANCE

        if profit >= PROFIT_TARGET:
            consistency_limit = PROFIT_TARGET * CONSISTENCY_PERCENT

            if best_day_profit < consistency_limit:
                passed = True

        # Next calendar day.
        current_date = current_date + timedelta(days=1)

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if not passed and not failed:
        timeout = True

        failure_reason = "TIMEOUT_60D"

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    equity = np.asarray(equity_curve, dtype=float)

    running_high = np.maximum.accumulate(equity)

    drawdowns = equity - running_high

    max_drawdown = float(drawdowns.min())

    # --------------------------------------------------------
    # Subscription cycles.
    #
    # A simulation that times out after 60 days has used
    # two subscription cycles.
    #
    # A simulation that passes on day 17 used one cycle.
    # --------------------------------------------------------

    if passed:
        subscription_cycles = int(np.ceil(calendar_days / SUBSCRIPTION_CYCLE_DAYS))

        subscription_cost = subscription_cycles * SUBSCRIPTION_PRICE

    elif timeout:
        subscription_cycles = int(np.ceil(MAX_CALENDAR_DAYS / SUBSCRIPTION_CYCLE_DAYS))

        subscription_cost = subscription_cycles * SUBSCRIPTION_PRICE

    else:
        subscription_cycles = int(np.ceil(calendar_days / SUBSCRIPTION_CYCLE_DAYS))

        subscription_cost = subscription_cycles * SUBSCRIPTION_PRICE

    return {
        "passed": passed,
        "failed": failed,
        "timeout": timeout,
        "failure_reason": failure_reason,
        "trades": total_trades,
        "trading_days": trading_days,
        "calendar_days": calendar_days,
        "subscription_cycles": subscription_cycles,
        "subscription_cost": subscription_cost,
        "final_balance": balance,
        "profit": balance - STARTING_BALANCE,
        "best_day_profit": best_day_profit,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# COMBINE SCENARIO
# ============================================================


def run_combine_scenario(
    risk_name,
    scenario,
    day_blocks,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        results.append(simulate_combine(day_blocks, scenario, rng))

    result_df = pd.DataFrame(results)

    passed = result_df[result_df["passed"]]

    failed = result_df[result_df["failed"]]

    timeout = result_df[result_df["timeout"]]

    # --------------------------------------------------------
    # PASSING SIMULATIONS
    # --------------------------------------------------------

    if len(passed) > 0:
        trading_days = passed["trading_days"]

        calendar_days = passed["calendar_days"]

        median_trading_days = trading_days.median()

        mean_trading_days = trading_days.mean()

        p25_trading_days = np.percentile(trading_days, 25)

        p75_trading_days = np.percentile(trading_days, 75)

        p95_trading_days = np.percentile(trading_days, 95)

        median_calendar_days = calendar_days.median()

        mean_calendar_days = calendar_days.mean()

    else:
        median_trading_days = np.nan
        mean_trading_days = np.nan
        p25_trading_days = np.nan
        p75_trading_days = np.nan
        p95_trading_days = np.nan
        median_calendar_days = np.nan
        mean_calendar_days = np.nan

    # --------------------------------------------------------
    # PASS WITHIN TIME WINDOWS
    #
    # These are unconditional probabilities among ALL
    # simulations.
    # --------------------------------------------------------

    pass_30 = (result_df["passed"] & (result_df["calendar_days"] <= 30)).mean()

    pass_60 = (result_df["passed"] & (result_df["calendar_days"] <= 60)).mean()

    # --------------------------------------------------------
    # Subscription
    # --------------------------------------------------------

    median_subscription_cycles = result_df["subscription_cycles"].median()

    mean_subscription_cost = result_df["subscription_cost"].mean()

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    max_dd = result_df["max_drawdown"]

    # --------------------------------------------------------
    # PASSING RESULTS ONLY
    # --------------------------------------------------------

    if len(passed) > 0:
        median_best_day = passed["best_day_profit"].median()

    else:
        median_best_day = np.nan

    return {
        "risk": risk_name,
        "simulations": N_SIMULATIONS,
        "pass_rate": result_df["passed"].mean(),
        "failure_rate": result_df["failed"].mean(),
        "timeout_rate": result_df["timeout"].mean(),
        "pass_plus_failure_plus_timeout": (
            result_df["passed"].mean()
            + result_df["failed"].mean()
            + result_df["timeout"].mean()
        ),
        "median_trading_days": median_trading_days,
        "mean_trading_days": mean_trading_days,
        "p25_trading_days": p25_trading_days,
        "p75_trading_days": p75_trading_days,
        "p95_trading_days": p95_trading_days,
        "median_calendar_days": median_calendar_days,
        "mean_calendar_days": mean_calendar_days,
        "pass_within_30d": pass_30,
        "pass_within_60d": pass_60,
        "median_subscription_cycles": median_subscription_cycles,
        "mean_subscription_cost": mean_subscription_cost,
        "median_best_day": median_best_day,
        "median_max_DD": max_dd.median(),
        "p95_max_DD": np.percentile(max_dd, 5),
        "p99_max_DD": np.percentile(max_dd, 1),
        "median_failed_days": (
            failed["calendar_days"].median() if len(failed) else np.nan
        ),
        "median_timeout_days": (
            timeout["calendar_days"].median() if len(timeout) else np.nan
        ),
    }


# ============================================================
# XFA
# ============================================================


def simulate_xfa(
    day_blocks,
    scenario,
    payout_amount,
    rng,
):

    balance = XFA_STARTING_BALANCE

    mll_floor = XFA_STARTING_BALANCE - XFA_MLL

    total_withdrawn = 0.0

    payout_count = 0

    winning_days = 0

    total_trades = 0

    trading_days = 0

    days_since_payout = 0

    failed = False

    equity = [balance]

    # --------------------------------------------------------
    # Simulate 12 months of funded operation.
    # --------------------------------------------------------

    for _ in range(252):
        block = rng.choice(day_blocks)

        trading_days += 1

        days_since_payout += 1

        day_pnl = 0.0

        for r in block["returns"]:
            risk_pct = get_risk_pct(balance, scenario)

            risk_dollars = STARTING_BALANCE * risk_pct

            pnl = r * risk_dollars

            balance += pnl

            day_pnl += pnl

            total_trades += 1

            equity.append(balance)

            if balance <= mll_floor:
                failed = True

                break

        if failed:
            break

        # ----------------------------------------------------
        # Winning day
        # ----------------------------------------------------

        if day_pnl >= MIN_WINNING_DAY_PROFIT:
            winning_days += 1

        # ----------------------------------------------------
        # Trailing MLL
        # ----------------------------------------------------

        mll_floor = min(XFA_STARTING_BALANCE, max(mll_floor, balance - XFA_MLL))

        # ----------------------------------------------------
        # Monthly payout
        # ----------------------------------------------------

        if days_since_payout >= XFA_MONTH_DAYS:
            if winning_days >= MIN_WINNING_DAYS:
                available_profit = max(0.0, balance - XFA_STARTING_BALANCE)

                actual_payout = min(payout_amount, available_profit)

                if actual_payout > 0:
                    balance -= actual_payout

                    total_withdrawn += actual_payout

                    payout_count += 1

                    winning_days = 0

            days_since_payout = 0

    equity = np.asarray(equity, dtype=float)

    running_high = np.maximum.accumulate(equity)

    drawdown = equity - running_high

    max_dd = float(drawdown.min())

    return {
        "failed": failed,
        "trades": total_trades,
        "trading_days": trading_days,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payouts": payout_count,
        "total_value": balance + total_withdrawn,
        "max_drawdown": max_dd,
    }


# ============================================================
# XFA SCENARIO
# ============================================================


def run_xfa_scenario(
    risk_name,
    scenario,
    day_blocks,
    payout_amount,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        results.append(simulate_xfa(day_blocks, scenario, payout_amount, rng))

    df = pd.DataFrame(results)

    return {
        "risk": risk_name,
        "payout_amount": payout_amount,
        "survival_rate": (1 - df["failed"].mean()),
        "failure_rate": df["failed"].mean(),
        "median_payouts": df["payouts"].median(),
        "median_total_withdrawn": df["total_withdrawn"].median(),
        "p95_total_withdrawn": np.percentile(df["total_withdrawn"], 95),
        "median_final_balance": df["final_balance"].median(),
        "median_total_value": df["total_value"].median(),
        "median_max_DD": df["max_drawdown"].median(),
        "p95_max_DD": np.percentile(df["max_drawdown"], 5),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 FUNDED ACCOUNT SIMULATION V4")

    print("=" * 110)

    print("\nFROZEN S2")

    print(STRATEGY_NAME)

    print(f"Quality >= {QUALITY_THRESHOLD}")

    print(f"RR = {RR}")

    print(f"{TAIL_PERCENT}% lower-tail")

    print(f"{STOP_POINTS}-point stop")

    print(f"{HORIZON_BARS}-bar horizon")

    print("\nNO STRATEGY OPTIMIZATION.")

    print("ONLY ACCOUNT POLICY IS TESTED.")

    # ========================================================
    # DATA
    # ========================================================

    df = load_data()

    day_blocks = build_day_blocks(df)

    print(f"\nHistorical OOS trades: {len(df)}")

    print(f"Historical OOS trading days: {len(day_blocks)}")

    print(f"Monte Carlo simulations: {N_SIMULATIONS:,}")

    print("\nS2 trades remain RTH-only.")

    print(
        "The simulation never creates trades outside "
        "the historical S2 RTH trade structure."
    )

    print(f"\nCombine maximum evaluation horizon: {MAX_CALENDAR_DAYS} calendar days")

    # ========================================================
    # RNG
    # ========================================================

    rng = np.random.default_rng(RANDOM_SEED)

    # ========================================================
    # PART A
    # ========================================================

    print("\n" + "=" * 110)

    print("PART A — $50K TRADING COMBINE")

    print("=" * 110)

    print(f"Starting balance: ${STARTING_BALANCE:,.0f}")

    print(f"Profit target: +${PROFIT_TARGET:,.0f}")

    print(f"Initial MLL: -${MAX_LOSS:,.0f}")

    print(f"Consistency requirement: {CONSISTENCY_PERCENT:.0%}")

    print(f"Maximum time tested: {MAX_CALENDAR_DAYS} calendar days")

    print(f"Subscription: ${SUBSCRIPTION_PRICE:.2f} / {SUBSCRIPTION_CYCLE_DAYS} days")

    combine_results = []

    for name, scenario in RISK_SCENARIOS.items():
        print(f"\nTesting {name}...")

        result = run_combine_scenario(name, scenario, day_blocks, rng)

        combine_results.append(result)

    combine_df = pd.DataFrame(combine_results)

    print("\n" + "-" * 110)

    print("COMBINE RESULTS")

    print("-" * 110)

    print(combine_df.to_string(index=False))

    # ========================================================
    # OUTCOME CHECK
    # ========================================================

    print("\n" + "=" * 110)

    print("OUTCOME CHECK")

    print("=" * 110)

    for _, row in combine_df.iterrows():
        total = row["pass_rate"] + row["failure_rate"] + row["timeout_rate"]

        print(
            f"{row['risk']:15s} "
            f"PASS={row['pass_rate']:.4%} "
            f"FAIL={row['failure_rate']:.4%} "
            f"TIMEOUT={row['timeout_rate']:.4%} "
            f"TOTAL={total:.4%}"
        )

    # ========================================================
    # PART B
    # ========================================================

    print("\n" + "=" * 110)

    print("PART B — XFA / FUNDED")

    print("=" * 110)

    print("One payout per month.")

    print(f"Minimum winning days: {MIN_WINNING_DAYS}")

    print(f"Minimum winning day: ${MIN_WINNING_DAY_PROFIT:.2f}")

    print("Testing payout amounts:")

    print(XFA_PAYOUT_AMOUNTS)

    xfa_results = []

    for risk_name, scenario in RISK_SCENARIOS.items():
        for payout_amount in XFA_PAYOUT_AMOUNTS:
            print(f"Testing XFA {risk_name} payout=${payout_amount}...")

            result = run_xfa_scenario(
                risk_name, scenario, day_blocks, payout_amount, rng
            )

            xfa_results.append(result)

    xfa_df = pd.DataFrame(xfa_results)

    # ========================================================
    # XFA RESULTS
    # ========================================================

    print("\n" + "=" * 110)

    print("XFA RESULTS")

    print("=" * 110)

    print(xfa_df.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    combine_df.to_csv(RESULTS_DIR / "s2_funded_combine_v4.csv", index=False)

    xfa_df.to_csv(RESULTS_DIR / "s2_funded_xfa_v4.csv", index=False)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 110)

    print("S2 FUNDED ACCOUNT SIMULATION V4 COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_funded_combine_v4.csv")

    print("s2_funded_xfa_v4.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
