from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "s2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# S2 FUNDED SIMULATION V2
# ============================================================
#
# PURPOSE
# -------
# Test the frozen S2 strategy under:
#
# PART A — $50K TRADING COMBINE
# PART B — XFA / FUNDED ACCOUNT
#
# IMPORTANT:
# The strategy itself is NOT modified.
#
# S2 remains:
#   Quality >= 0.75
#   RR = 1.30
#   17.5% lower-tail
#   20-point stop
#   15-bar horizon
#   RTH only
#
# This version preserves the ACTUAL OOS TRADING DAYS.
#
# ============================================================


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = RESULTS_DIR / "s2_selective_execution_B_trades.csv"


# ============================================================
# COMBINE — $50K
# ============================================================

STARTING_BALANCE = 50_000.0

PROFIT_TARGET = 3_000.0

# Topstep $50K Combine MLL.
INITIAL_MLL = 2_000.0


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
# CONSISTENCY
# ============================================================

# 50% of $3,000 target.
#
# A single best day cannot represent 50% or more of the
# required profit target.
CONSISTENCY_PERCENT = 0.50


# ============================================================
# SUBSCRIPTION
# ============================================================
#
# Put the ACTUAL price you pay here.
#
# This is intentionally a variable because your actual
# subscription price may differ due to promotions.
# ============================================================

SUBSCRIPTION_PRICE = 49.0

SUBSCRIPTION_DAYS = 30


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


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"\nInput file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "entry_timestamp",
        "net_R",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize timestamps.
    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], utc=True)

    # Convert to New York.
    df["entry_timestamp_ny"] = df["entry_timestamp"].dt.tz_convert("America/New_York")

    df["trading_date"] = df["entry_timestamp_ny"].dt.date

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    return df


# ============================================================
# BUILD ACTUAL OOS DAY BLOCKS
# ============================================================


def build_day_blocks(df):

    blocks = []

    for date, day_df in df.groupby("trading_date", sort=True):
        returns = day_df["net_R"].astype(float).to_numpy()

        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            continue

        blocks.append(
            {
                "date": date,
                "returns": returns,
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

    # --------------------------------------------------------
    # Trailing MLL
    #
    # The floor is updated using DAILY CLOSING BALANCE.
    # It cannot move downward.
    #
    # It stops at the starting balance.
    # --------------------------------------------------------

    mll_floor = STARTING_BALANCE - INITIAL_MLL

    best_day_profit = 0.0

    equity = [balance]

    trades = 0

    calendar_days = 0

    trading_days = 0

    daily_pnl_list = []

    passed = False

    failed = False

    failure_reason = None

    # --------------------------------------------------------
    # Randomize day order.
    #
    # We preserve each actual day's internal trade structure.
    # --------------------------------------------------------

    sampled_days = rng.choice(
        day_blocks,
        size=len(day_blocks),
        replace=True,
    )

    for block in sampled_days:
        trading_days += 1

        calendar_days += 1

        day_pnl = 0.0

        # ----------------------------------------------------
        # Execute only trades that actually existed in that
        # historical RTH session.
        # ----------------------------------------------------

        for r in block["returns"]:
            risk_pct = get_risk(
                balance,
                scenario,
            )

            risk_dollars = STARTING_BALANCE * risk_pct

            pnl = r * risk_dollars

            balance += pnl

            day_pnl += pnl

            trades += 1

            equity.append(balance)

            # Intraday breach.
            if balance <= mll_floor:
                failed = True

                failure_reason = "MLL"

                break

            # Profit target.
            #
            # We don't immediately pass here because
            # consistency must also be checked.
            if balance >= STARTING_BALANCE + PROFIT_TARGET:
                # Continue to end of day so that we can
                # calculate the consistency requirement.
                pass

        if failed:
            break

        daily_pnl_list.append(day_pnl)

        # ----------------------------------------------------
        # Best trading day
        # ----------------------------------------------------

        best_day_profit = max(best_day_profit, day_pnl)

        # ----------------------------------------------------
        # DAILY MLL UPDATE
        # ----------------------------------------------------
        #
        # The MLL follows the highest daily closing balance,
        # but cannot rise above the starting balance.
        # ----------------------------------------------------

        mll_floor = min(STARTING_BALANCE, max(mll_floor, balance - INITIAL_MLL))

        # ----------------------------------------------------
        # Check target + consistency
        # ----------------------------------------------------

        current_profit = balance - STARTING_BALANCE

        if current_profit >= PROFIT_TARGET:
            consistency_limit = PROFIT_TARGET * CONSISTENCY_PERCENT

            # If best day is too large, the consistency
            # requirement is not satisfied yet.
            if best_day_profit < consistency_limit:
                passed = True

                break

    # --------------------------------------------------------
    # Calendar estimate
    #
    # Each sampled block represents one historical trading
    # day. We therefore estimate calendar days using the
    # historical average spacing between OOS trading dates.
    # --------------------------------------------------------

    historical_dates = [b["date"] for b in day_blocks]

    if len(historical_dates) >= 2:
        date_diffs = [
            (historical_dates[i] - historical_dates[i - 1]).days
            for i in range(1, len(historical_dates))
        ]

        avg_calendar_spacing = np.mean(date_diffs)

    else:
        avg_calendar_spacing = 1.4

    estimated_calendar_days = trading_days * avg_calendar_spacing

    subscription_cycles = (
        np.ceil(estimated_calendar_days / SUBSCRIPTION_DAYS) if passed else np.nan
    )

    subscription_cost = subscription_cycles * SUBSCRIPTION_PRICE if passed else np.nan

    return {
        "passed": passed,
        "failed": failed,
        "failure_reason": failure_reason,
        "trades": trades,
        "trading_days": trading_days,
        "estimated_calendar_days": estimated_calendar_days,
        "subscription_cycles": subscription_cycles,
        "subscription_cost": subscription_cost,
        "final_balance": balance,
        "profit": balance - STARTING_BALANCE,
        "best_day_profit": best_day_profit,
        "max_drawdown": min(np.array(equity) - np.maximum.accumulate(np.array(equity))),
        "max_mll_floor": mll_floor,
    }


# ============================================================
# COMBINE MONTE CARLO
# ============================================================


def run_combine_scenario(
    name,
    scenario,
    day_blocks,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        result = simulate_combine(
            day_blocks,
            scenario,
            rng,
        )

        results.append(result)

    df = pd.DataFrame(results)

    passed = df[df["passed"]]

    if len(passed) > 0:
        median_days = passed["trading_days"].median()

        mean_days = passed["trading_days"].mean()

        p25_days = percentile(passed["trading_days"], 25)

        p75_days = percentile(passed["trading_days"], 75)

        p95_days = percentile(passed["trading_days"], 95)

        median_calendar = passed["estimated_calendar_days"].median()

        median_cycles = passed["subscription_cycles"].median()

        mean_subscription_cost = passed["subscription_cost"].mean()

    else:
        median_days = np.nan
        mean_days = np.nan
        p25_days = np.nan
        p75_days = np.nan
        p95_days = np.nan
        median_calendar = np.nan
        median_cycles = np.nan
        mean_subscription_cost = np.nan

    # --------------------------------------------------------
    # Pass within time windows
    # --------------------------------------------------------

    passed_30 = (passed["estimated_calendar_days"] <= 30).mean() if len(passed) else 0.0

    passed_60 = (passed["estimated_calendar_days"] <= 60).mean() if len(passed) else 0.0

    passed_90 = (passed["estimated_calendar_days"] <= 90).mean() if len(passed) else 0.0

    return {
        "risk": name,
        "simulations": N_SIMULATIONS,
        "pass_rate": df["passed"].mean(),
        "failure_rate": df["failed"].mean(),
        "median_trading_days": median_days,
        "mean_trading_days": mean_days,
        "p25_trading_days": p25_days,
        "p75_trading_days": p75_days,
        "p95_trading_days": p95_days,
        "median_calendar_days": median_calendar,
        "pass_within_30d": passed_30,
        "pass_within_60d": passed_60,
        "pass_within_90d": passed_90,
        "median_subscription_cycles": median_cycles,
        "mean_subscription_cost": mean_subscription_cost,
        "median_best_day": (
            passed["best_day_profit"].median() if len(passed) else np.nan
        ),
        "median_max_DD": df["max_drawdown"].median(),
        "p95_max_DD": percentile(df["max_drawdown"], 5),
        "p99_max_DD": percentile(df["max_drawdown"], 1),
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

    payouts = 0

    winning_days = 0

    total_trading_days = 0

    total_trades = 0

    equity = [balance]

    failed = False

    failure_reason = None

    # We simulate approximately 12 months.
    #
    # 47 OOS trading days × 12 months.
    max_days = 47 * 12

    sampled_days = rng.choice(
        day_blocks,
        size=max_days,
        replace=True,
    )

    days_since_payout = 0

    current_day_pnl = 0.0

    for block in sampled_days:
        total_trading_days += 1

        days_since_payout += 1

        day_pnl = 0.0

        for r in block["returns"]:
            risk_pct = get_risk(
                balance,
                scenario,
            )

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

        if day_pnl >= MIN_WINNING_DAY_PROFIT:
            winning_days += 1

        # ----------------------------------------------------
        # MLL follows daily closing high, capped at start.
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

                    payouts += 1

                    winning_days = 0

            days_since_payout = 0

    return {
        "failed": failed,
        "failure_reason": failure_reason,
        "trades": total_trades,
        "trading_days": total_trading_days,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payouts": payouts,
        "total_value": balance + total_withdrawn,
        "max_drawdown": min(np.array(equity) - np.maximum.accumulate(np.array(equity))),
    }


# ============================================================
# XFA MONTE CARLO
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
        result = simulate_xfa(
            day_blocks,
            scenario,
            payout_interval,
            payout_amount,
            rng,
        )

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

    print("S2 FUNDED ACCOUNT SIMULATION V2")

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

    print("Profit target: +$3,000")

    print("Initial MLL: -$2,000")

    print("Consistency target: 50%")

    print("No artificial 30-day deadline")

    print(f"Subscription cycle: {SUBSCRIPTION_DAYS} days")

    print(f"Subscription price: ${SUBSCRIPTION_PRICE:.2f}")

    combine_results = []

    for name, scenario in RISK_SCENARIOS.items():
        print(f"\nTesting {name}...")

        result = run_combine_scenario(
            name,
            scenario,
            day_blocks,
            rng,
        )

        combine_results.append(result)

    combine_df = pd.DataFrame(combine_results)

    print("\n" + "-" * 110)

    print("COMBINE RESULTS")

    print("-" * 110)

    print(combine_df.to_string(index=False))

    # ========================================================
    # PART B
    # ========================================================

    print("\n" + "=" * 110)

    print("PART B — XFA / FUNDED")

    print("=" * 110)

    print("One payout per month.")

    print(f"Minimum winning days: {MIN_WINNING_DAYS}")

    print(f"Minimum winning day: ${MIN_WINNING_DAY_PROFIT}")

    xfa_results = []

    for risk_name, scenario in RISK_SCENARIOS.items():
        for interval in PAYOUT_INTERVAL_DAYS:
            for amount in PAYOUT_AMOUNTS:
                result = run_xfa_scenario(
                    risk_name,
                    scenario,
                    day_blocks,
                    interval,
                    amount,
                    rng,
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

    combine_df.to_csv(
        RESULTS_DIR / "s2_funded_combine_v2.csv",
        index=False,
    )

    xfa_df.to_csv(
        RESULTS_DIR / "s2_funded_xfa_v2.csv",
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 110)

    print("S2 FUNDED SIMULATION V2 COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_funded_combine_v2.csv")

    print("s2_funded_xfa_v2.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
