from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta


# ============================================================
# S2 FUNDED ACCOUNT SIMULATION V3
# ============================================================
#
# PURPOSE
# -------
# Evaluate the FROZEN S2 strategy under funded-account
# constraints.
#
# PART A:
#   $50K Trading Combine
#
# PART B:
#   XFA / Funded account
#
# IMPORTANT:
#   S2 itself is NOT optimized.
#
#   We only test account-management policies.
#
# ============================================================


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = Path("s2_selective_execution_B_trades.csv")


# ============================================================
# COMBINE PARAMETERS
# ============================================================

STARTING_BALANCE = 50_000.0

PROFIT_TARGET = 3_000.0

INITIAL_MLL = 2_000.0


# ============================================================
# CONSISTENCY
# ============================================================

CONSISTENCY_PERCENT = 0.50


# ============================================================
# RISK POLICIES
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
# MONTE CARLO
# ============================================================

N_SIMULATIONS = 50_000

RANDOM_SEED = 42


# Safety limit only.
#
# This is NOT a time limit for the Combine.
# It prevents an infinite simulation if an extremely
# unlikely pathological path occurs.
MAX_TRADING_DAYS = 2_000


# ============================================================
# XFA PARAMETERS
# ============================================================

XFA_STARTING_BALANCE = 50_000.0

XFA_MLL = 2_000.0

MIN_WINNING_DAYS = 5

MIN_WINNING_DAY_PROFIT = 150.0

PAYOUT_INTERVAL_DAYS = [
    20,
    21,
    22,
]

PAYOUT_AMOUNTS = [
    500,
    750,
    1_000,
    1_250,
    1_500,
    1_750,
    2_000,
]


# ============================================================
# HELPERS
# ============================================================


def percentile(values, p):

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, p))


def get_risk(
    balance,
    scenario,
):

    threshold = scenario["threshold"]

    if threshold is not None:
        profit_from_start = balance - STARTING_BALANCE

        if profit_from_start >= threshold:
            return scenario["after_threshold"]

    return scenario["initial"]


def next_weekday(date):

    date = date + timedelta(days=1)

    while date.weekday() >= 5:
        date += timedelta(days=1)

    return date


# ============================================================
# LOAD DATA
# ============================================================


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"\nInput file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "entry_timestamp",
        "net_R",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], utc=True)

    df["entry_timestamp_ny"] = df["entry_timestamp"].dt.tz_convert("America/New_York")

    df["trading_date"] = df["entry_timestamp_ny"].dt.date

    df["net_R"] = pd.to_numeric(df["net_R"], errors="coerce")

    df = df.dropna(subset=["net_R"])

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    return df


# ============================================================
# ACTUAL RTH DAY BLOCKS
# ============================================================


def build_day_blocks(df):

    blocks = []

    for date, day_df in df.groupby("trading_date", sort=True):
        returns = day_df["net_R"].to_numpy(dtype=float)

        if len(returns) == 0:
            continue

        blocks.append(
            {
                "date": date,
                "returns": returns,
                "trades": len(returns),
            }
        )

    return blocks


# ============================================================
# COMBINE SIMULATION
# ============================================================


def simulate_combine(
    day_blocks,
    scenario,
    rng,
):

    balance = STARTING_BALANCE

    # Initial trailing MLL.
    mll_floor = STARTING_BALANCE - INITIAL_MLL

    best_day_profit = 0.0

    total_trades = 0

    trading_days = 0

    calendar_days = 0

    current_date = pd.Timestamp("2026-01-05").date()

    daily_results = []

    equity = [balance]

    passed = False

    failed = False

    failure_reason = None

    # --------------------------------------------------------
    # Continue until PASS or FAIL.
    #
    # There is NO 47-day stopping condition.
    # --------------------------------------------------------

    while not passed and not failed and trading_days < MAX_TRADING_DAYS:
        # ----------------------------------------------------
        # Sample one actual historical RTH day.
        #
        # The entire day's trade structure is preserved.
        # ----------------------------------------------------

        block = rng.choice(day_blocks)

        trading_days += 1

        # Move through actual weekdays.
        if trading_days > 1:
            current_date = next_weekday(current_date)

        calendar_days = (current_date - pd.Timestamp("2026-01-05").date()).days + 1

        day_pnl = 0.0

        # ----------------------------------------------------
        # Execute every S2 trade from the selected day.
        #
        # These are ONLY trades that existed in the RTH
        # strategy sample.
        # ----------------------------------------------------

        for r in block["returns"]:
            risk_pct = get_risk(balance, scenario)

            risk_dollars = STARTING_BALANCE * risk_pct

            pnl = r * risk_dollars

            balance += pnl

            day_pnl += pnl

            total_trades += 1

            equity.append(balance)

            # ------------------------------------------------
            # MLL breach
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

        daily_results.append(day_pnl)

        best_day_profit = max(best_day_profit, day_pnl)

        # ----------------------------------------------------
        # END-OF-DAY MLL UPDATE
        #
        # The floor can move upward as the account reaches
        # higher closing balances, but cannot move above
        # the starting balance.
        # ----------------------------------------------------

        mll_floor = min(STARTING_BALANCE, max(mll_floor, balance - INITIAL_MLL))

        # ----------------------------------------------------
        # PROFIT TARGET
        # ----------------------------------------------------

        current_profit = balance - STARTING_BALANCE

        if current_profit >= PROFIT_TARGET:
            consistency_limit = PROFIT_TARGET * CONSISTENCY_PERCENT

            # Best day must remain below 50% of target.
            if best_day_profit < consistency_limit:
                passed = True

    # --------------------------------------------------------
    # If safety limit reached.
    # --------------------------------------------------------

    if not passed and not failed:
        failure_reason = "SIMULATION_LIMIT"

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    equity_array = np.asarray(equity, dtype=float)

    running_high = np.maximum.accumulate(equity_array)

    drawdowns = equity_array - running_high

    max_drawdown = float(drawdowns.min())

    # --------------------------------------------------------
    # Subscription cycles
    # --------------------------------------------------------

    if passed:
        subscription_cycles = int(np.ceil(calendar_days / SUBSCRIPTION_CYCLE_DAYS))

        subscription_cost = subscription_cycles * SUBSCRIPTION_PRICE

    else:
        subscription_cycles = np.nan

        subscription_cost = np.nan

    return {
        "passed": passed,
        "failed": failed,
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
        "mll_floor": mll_floor,
    }


# ============================================================
# COMBINE SCENARIO
# ============================================================


def run_combine_scenario(
    name,
    scenario,
    day_blocks,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        result = simulate_combine(day_blocks, scenario, rng)

        results.append(result)

    result_df = pd.DataFrame(results)

    passed = result_df[result_df["passed"]]

    failed = result_df[result_df["failed"]]

    # --------------------------------------------------------
    # Pass statistics
    # --------------------------------------------------------

    if len(passed) > 0:
        trading_days = passed["trading_days"]

        calendar_days = passed["calendar_days"]

        subscription_cycles = passed["subscription_cycles"]

        subscription_cost = passed["subscription_cost"]

        median_trading_days = trading_days.median()

        mean_trading_days = trading_days.mean()

        p25_trading_days = percentile(trading_days, 25)

        p75_trading_days = percentile(trading_days, 75)

        p95_trading_days = percentile(trading_days, 95)

        median_calendar_days = calendar_days.median()

        mean_calendar_days = calendar_days.mean()

        median_cycles = subscription_cycles.median()

        mean_subscription_cost = subscription_cost.mean()

    else:
        median_trading_days = np.nan
        mean_trading_days = np.nan
        p25_trading_days = np.nan
        p75_trading_days = np.nan
        p95_trading_days = np.nan
        median_calendar_days = np.nan
        mean_calendar_days = np.nan
        median_cycles = np.nan
        mean_subscription_cost = np.nan

    # --------------------------------------------------------
    # Time-window pass rates
    # --------------------------------------------------------

    if len(passed) > 0:
        pass_30 = (passed["calendar_days"] <= 30).mean()

        pass_60 = (passed["calendar_days"] <= 60).mean()

        pass_90 = (passed["calendar_days"] <= 90).mean()

    else:
        pass_30 = 0.0
        pass_60 = 0.0
        pass_90 = 0.0

    # --------------------------------------------------------
    # Overall max DD
    # --------------------------------------------------------

    all_dd = result_df["max_drawdown"]

    return {
        "risk": name,
        "simulations": N_SIMULATIONS,
        "pass_rate": result_df["passed"].mean(),
        "failure_rate": result_df["failed"].mean(),
        "unresolved_rate": (
            1 - result_df["passed"].mean() - result_df["failed"].mean()
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
        "pass_within_90d": pass_90,
        "median_subscription_cycles": median_cycles,
        "mean_subscription_cost": mean_subscription_cost,
        "median_best_day": (
            passed["best_day_profit"].median() if len(passed) else np.nan
        ),
        "median_max_DD": all_dd.median(),
        "p95_max_DD": percentile(all_dd, 5),
        "p99_max_DD": percentile(all_dd, 1),
    }


# ============================================================
# XFA SIMULATION
# ============================================================


def simulate_xfa(
    day_blocks,
    scenario,
    payout_interval,
    payout_amount,
    rng,
):

    balance = XFA_STARTING_BALANCE

    mll_floor = XFA_STARTING_BALANCE - XFA_MLL

    total_withdrawn = 0.0

    payout_count = 0

    winning_days = 0

    trading_days = 0

    total_trades = 0

    failed = False

    failure_reason = None

    equity = [balance]

    days_since_payout = 0

    max_days = 47 * 12

    for _ in range(max_days):
        block = rng.choice(day_blocks)

        trading_days += 1

        days_since_payout += 1

        day_pnl = 0.0

        for r in block["returns"]:
            risk_pct = get_risk(balance, scenario)

            risk_dollars = STARTING_BALANCE * risk_pct

            pnl = r * risk_dollars

            balance += pnl

            day_pnl += pnl

            total_trades += 1

            equity.append(balance)

            if balance <= mll_floor:
                failed = True

                failure_reason = "MLL"

                break

        if failed:
            break

        # ----------------------------------------------------
        # Winning day
        # ----------------------------------------------------

        if day_pnl >= MIN_WINNING_DAY_PROFIT:
            winning_days += 1

        # ----------------------------------------------------
        # Daily MLL update
        # ----------------------------------------------------

        mll_floor = min(XFA_STARTING_BALANCE, max(mll_floor, balance - XFA_MLL))

        # ----------------------------------------------------
        # Monthly payout
        # ----------------------------------------------------

        if days_since_payout >= payout_interval:
            if winning_days >= MIN_WINNING_DAYS:
                available_profit = max(0.0, balance - XFA_STARTING_BALANCE)

                actual_payout = min(payout_amount, available_profit)

                if actual_payout > 0:
                    balance -= actual_payout

                    total_withdrawn += actual_payout

                    payout_count += 1

                    winning_days = 0

            days_since_payout = 0

    equity_array = np.asarray(equity, dtype=float)

    running_high = np.maximum.accumulate(equity_array)

    max_drawdown = float((equity_array - running_high).min())

    return {
        "failed": failed,
        "failure_reason": failure_reason,
        "trades": total_trades,
        "trading_days": trading_days,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payouts": payout_count,
        "total_value": balance + total_withdrawn,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# XFA SCENARIO
# ============================================================


def run_xfa_scenario(
    risk_name,
    scenario,
    day_blocks,
    payout_interval,
    payout_amount,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        result = simulate_xfa(day_blocks, scenario, payout_interval, payout_amount, rng)

        results.append(result)

    df = pd.DataFrame(results)

    return {
        "risk": risk_name,
        "payout_interval_days": payout_interval,
        "payout_amount": payout_amount,
        "survival_rate": (1 - df["failed"].mean()),
        "failure_rate": df["failed"].mean(),
        "median_payouts": df["payouts"].median(),
        "median_total_withdrawn": df["total_withdrawn"].median(),
        "p95_total_withdrawn": percentile(df["total_withdrawn"], 95),
        "median_final_balance": df["final_balance"].median(),
        "median_total_value": df["total_value"].median(),
        "median_max_DD": df["max_drawdown"].median(),
        "p95_max_DD": percentile(df["max_drawdown"], 5),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 FUNDED ACCOUNT SIMULATION V3")

    print("=" * 110)

    print("\nFROZEN S2")

    print("B-selective")

    print("Quality >= 0.75")

    print("RR = 1.30")

    print("17.5% lower-tail")

    print("20-point stop")

    print("15-bar horizon")

    print("\nNO STRATEGY OPTIMIZATION.")

    print("ONLY ACCOUNT POLICY IS TESTED.")

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    day_blocks = build_day_blocks(df)

    print(f"\nHistorical OOS trades: {len(df)}")

    print(f"Historical OOS trading days: {len(day_blocks)}")

    print(f"Monte Carlo simulations: {N_SIMULATIONS:,}")

    print("\nThe strategy trades ONLY its existing RTH trades.")

    print("Historical day structure is preserved.")

    print("There is NO 47-day Combine stopping point.")

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

    print("Starting balance: $50,000")

    print("Profit target: +$3,000")

    print("Initial MLL: -$2,000")

    print("Consistency target: 50%")

    print("No artificial time limit")

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
    # IMPORTANT CHECK
    # ========================================================

    print("\n" + "=" * 110)

    print("PASS / FAIL / UNRESOLVED CHECK")

    print("=" * 110)

    for _, row in combine_df.iterrows():
        total = row["pass_rate"] + row["failure_rate"] + row["unresolved_rate"]

        print(
            f"{row['risk']:15s} "
            f"PASS={row['pass_rate']:.4%} "
            f"FAIL={row['failure_rate']:.4%} "
            f"UNRESOLVED={row['unresolved_rate']:.4%} "
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

    xfa_results = []

    for risk_name, scenario in RISK_SCENARIOS.items():
        for interval in PAYOUT_INTERVAL_DAYS:
            for amount in PAYOUT_AMOUNTS:
                result = run_xfa_scenario(
                    risk_name, scenario, day_blocks, interval, amount, rng
                )

                xfa_results.append(result)

    xfa_df = pd.DataFrame(xfa_results)

    # ========================================================
    # XFA TOP POLICIES
    # ========================================================

    print("\n" + "=" * 110)

    print("XFA TOP POLICIES")

    print("=" * 110)

    viable = xfa_df[xfa_df["survival_rate"] >= 0.90]

    if len(viable) == 0:
        print("No policy achieved 90% survival.")

    else:
        top = viable.sort_values(
            [
                "median_total_withdrawn",
                "survival_rate",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(20)

        print(top.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    combine_df.to_csv("s2_funded_combine_v3.csv", index=False)

    xfa_df.to_csv("s2_funded_xfa_v3.csv", index=False)

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 110)

    print("S2 FUNDED ACCOUNT SIMULATION V3 COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_funded_combine_v3.csv")

    print("s2_funded_xfa_v3.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
