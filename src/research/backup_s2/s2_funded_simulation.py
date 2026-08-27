from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# S2 FUNDED ACCOUNT SIMULATION
# ============================================================
#
# PURPOSE
# -------
# Test the FROZEN S2 strategy under realistic funded-account
# constraints.
#
# PART A:
#   $50K Trading Combine
#
# PART B:
#   XFA / funded account
#
# NO STRATEGY OPTIMIZATION.
#
# We are optimizing the ACCOUNT POLICY, not S2 itself:
#
#   - risk per trade
#   - risk transition
#   - payout timing
#   - payout amount
#
# ============================================================


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = Path("s2_selective_execution_B_trades.csv")


# ============================================================
# ACCOUNT
# ============================================================

STARTING_BALANCE = 50_000.0

PROFIT_TARGET = 3_000.0

MAX_LOSS_LIMIT = 2_000.0


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
    # Conservative start, then increase risk after
    # building a $1,000 buffer.
    "0.50_to_1.00": {
        "initial": 0.0050,
        "after_threshold": 0.0100,
        "threshold": 1_000.0,
    },
}


# ============================================================
# MONTE CARLO
# ============================================================

N_SIMULATIONS = 50_000

RANDOM_SEED = 42


# ============================================================
# TIME
# ============================================================

# We use the actual OOS trading-day sequence.
#
# The historical S2 sample contains 47 trading days.
#
# For the funded simulation we allow repeated randomized
# paths so that we can estimate probabilities rather than
# relying on one historical path.
#
MAX_COMBINE_TRADES = 500

MAX_XFA_TRADES = 2_000


# ============================================================
# XFA PAYOUT POLICY
# ============================================================

# One payout per simulated month.
#
# These are POLICY scenarios.
# They do not modify S2.
#
PAYOUT_INTERVALS = [
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
# XFA PARAMETERS
# ============================================================
#
# IMPORTANT:
# Keep these isolated because Topstep account rules can
# differ depending on the XFA route.
#
# The simulator therefore does not hard-code the rule into
# the strategy itself.
# ============================================================

XFA_STARTING_BALANCE = STARTING_BALANCE

XFA_INITIAL_BUFFER = 0.0

# The account's loss limit is represented as a buffer.
#
# We initially model it as $2,000.
#
# The exact post-payout mechanics must match the specific
# Topstep XFA route we use.
XFA_MAX_LOSS_LIMIT = 2_000.0

# Minimum winning days for payout.
MIN_WINNING_DAYS = 5

MIN_WINNING_DAY_PROFIT = 150.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def percentile(values, p):

    return float(np.percentile(values, p))


def max_drawdown_dollars(equity):

    equity = np.asarray(equity, dtype=float)

    running_max = np.maximum.accumulate(equity)

    drawdown = equity - running_max

    return float(drawdown.min())


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


# ============================================================
# LOAD TRADES
# ============================================================


def load_returns():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nMissing input file:\n"
            f"{INPUT_FILE}\n\n"
            "Run the S2 selective execution "
            "test first."
        )

    df = pd.read_csv(INPUT_FILE)

    if "net_R" not in df.columns:
        raise ValueError("CSV must contain net_R.")

    if "entry_timestamp" not in df.columns:
        raise ValueError("CSV must contain entry_timestamp.")

    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], utc=True)

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    df["date_ny"] = df["entry_timestamp"].dt.tz_convert("America/New_York").dt.date

    return df


# ============================================================
# PART A
# TRADING COMBINE
# ============================================================


def simulate_combine(
    returns,
    risk_scenario,
    rng,
):

    balance = STARTING_BALANCE

    equity = [balance]

    trades = 0

    winning_days = set()

    daily_pnl = {}

    passed = False

    failed = False

    fail_reason = None

    for _ in range(MAX_COMBINE_TRADES):
        # Randomly sample an OOS trade.
        #
        # This creates a distribution of possible future
        # paths while preserving the empirical S2 return
        # distribution.
        r = rng.choice(returns)

        risk_pct = get_risk(balance, risk_scenario)

        risk_dollars = STARTING_BALANCE * risk_pct

        pnl = r * risk_dollars

        balance += pnl

        trades += 1

        equity.append(balance)

        # ----------------------------------------------------
        # Daily bookkeeping
        # ----------------------------------------------------

        # We do not have a random timestamp here, so each
        # simulated trade is treated as sequential.
        #
        # This keeps the Combine test focused on account
        # survival rather than historical calendar placement.
        day = trades // 4

        daily_pnl.setdefault(day, 0.0)

        daily_pnl[day] += pnl

        if daily_pnl[day] >= MIN_WINNING_DAY_PROFIT:
            winning_days.add(day)

        # ----------------------------------------------------
        # Failure condition
        # ----------------------------------------------------

        if balance <= STARTING_BALANCE - MAX_LOSS_LIMIT:
            failed = True

            fail_reason = "MLL"

            break

        # ----------------------------------------------------
        # Profit target
        # ----------------------------------------------------

        if balance >= STARTING_BALANCE + PROFIT_TARGET:
            passed = True

            break

    return {
        "passed": passed,
        "failed": failed,
        "fail_reason": fail_reason,
        "trades": trades,
        "final_balance": balance,
        "profit": (balance - STARTING_BALANCE),
        "max_drawdown": (max_drawdown_dollars(equity)),
        "winning_days": len(winning_days),
    }


# ============================================================
# RUN COMBINE SCENARIO
# ============================================================


def run_combine_scenario(
    name,
    scenario,
    returns,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        result = simulate_combine(
            returns,
            scenario,
            rng,
        )

        results.append(result)

    df = pd.DataFrame(results)

    pass_rate = df["passed"].mean()

    fail_rate = df["failed"].mean()

    return {
        "scenario": name,
        "simulations": N_SIMULATIONS,
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,
        "median_trades": df["trades"].median(),
        "median_profit": df["profit"].median(),
        "median_max_DD": df["max_drawdown"].median(),
        "p95_max_DD": percentile(df["max_drawdown"], 5),
        "p99_max_DD": percentile(df["max_drawdown"], 1),
        "median_winning_days": df["winning_days"].median(),
    }


# ============================================================
# PART B
# XFA SIMULATION
# ============================================================


def simulate_xfa(
    returns,
    risk_scenario,
    payout_interval,
    payout_amount,
    rng,
):

    balance = XFA_STARTING_BALANCE

    loss_floor = XFA_STARTING_BALANCE - XFA_MAX_LOSS_LIMIT

    trades = 0

    winning_days = set()

    current_day = 0

    day_pnl = 0.0

    payout_count = 0

    total_withdrawn = 0.0

    failed = False

    failure_reason = None

    equity = [balance]

    next_payout_trade = payout_interval

    while trades < MAX_XFA_TRADES:
        r = rng.choice(returns)

        risk_pct = get_risk(balance, risk_scenario)

        risk_dollars = STARTING_BALANCE * risk_pct

        pnl = r * risk_dollars

        balance += pnl

        trades += 1

        equity.append(balance)

        # ----------------------------------------------------
        # Approximate day structure
        # ----------------------------------------------------

        current_day = trades // 4

        day_pnl += pnl

        if day_pnl >= (MIN_WINNING_DAY_PROFIT):
            winning_days.add(current_day)

        # Reset approximate day
        if trades % 4 == 0:
            day_pnl = 0.0

        # ----------------------------------------------------
        # Loss limit
        # ----------------------------------------------------

        if balance <= loss_floor:
            failed = True

            failure_reason = "MLL"

            break

        # ----------------------------------------------------
        # Monthly payout
        # ----------------------------------------------------

        if trades >= next_payout_trade:
            # Require minimum winning days.
            if len(winning_days) >= MIN_WINNING_DAYS:
                # Never withdraw more than available profit.
                profit_above_start = max(0.0, balance - XFA_STARTING_BALANCE)

                actual_payout = min(payout_amount, profit_above_start)

                if actual_payout > 0:
                    balance -= actual_payout

                    total_withdrawn += actual_payout

                    payout_count += 1

                    # ------------------------------------------------
                    # IMPORTANT
                    #
                    # We intentionally DO NOT automatically
                    # move the loss floor here.
                    #
                    # This is the exact area we want to investigate
                    # against the current Topstep XFA rules.
                    #
                    # ------------------------------------------------

                    winning_days.clear()

            next_payout_trade += payout_interval

    return {
        "failed": failed,
        "failure_reason": failure_reason,
        "trades": trades,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payouts": payout_count,
        "net_value": (balance + total_withdrawn),
        "max_drawdown": max_drawdown_dollars(equity),
        "winning_days": len(winning_days),
    }


# ============================================================
# RUN XFA SCENARIO
# ============================================================


def run_xfa_scenario(
    name,
    scenario,
    returns,
    payout_interval,
    payout_amount,
    rng,
):

    results = []

    for _ in range(N_SIMULATIONS):
        result = simulate_xfa(
            returns,
            scenario,
            payout_interval,
            payout_amount,
            rng,
        )

        results.append(result)

    df = pd.DataFrame(results)

    survival_rate = 1 - df["failed"].mean()

    return {
        "risk": name,
        "payout_interval": payout_interval,
        "payout_amount": payout_amount,
        "simulations": N_SIMULATIONS,
        "survival_rate": survival_rate,
        "failure_rate": df["failed"].mean(),
        "median_payouts": df["payouts"].median(),
        "median_total_withdrawn": df["total_withdrawn"].median(),
        "p95_total_withdrawn": percentile(df["total_withdrawn"], 95),
        "median_final_balance": df["final_balance"].median(),
        "median_net_value": df["net_value"].median(),
        "median_max_DD": df["max_drawdown"].median(),
        "p95_DD": percentile(df["max_drawdown"], 5),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 FUNDED ACCOUNT SIMULATION")

    print("=" * 110)

    print("\nFROZEN S2:")

    print("B-selective")

    print("Quality >= 0.75")

    print("RR = 1.30")

    print("17.5% lower-tail")

    print("20-point stop")

    print("15-bar horizon")

    print("\nNO STRATEGY OPTIMIZATION.")

    print("ONLY ACCOUNT POLICY IS BEING TESTED.")

    # ========================================================
    # LOAD
    # ========================================================

    df = load_returns()

    returns = df["net_R"].astype(float).to_numpy()

    returns = returns[np.isfinite(returns)]

    print(f"\nHistorical OOS trades: {len(returns)}")

    print(f"Historical OOS days: {df['date_ny'].nunique()}")

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

    print(f"Starting balance: ${STARTING_BALANCE:,.0f}")

    print(f"Profit target: +${PROFIT_TARGET:,.0f}")

    print(f"Maximum loss: -${MAX_LOSS_LIMIT:,.0f}")

    combine_results = []

    for name, scenario in RISK_SCENARIOS.items():
        print(f"\nTesting {name}")

        result = run_combine_scenario(
            name,
            scenario,
            returns,
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

    print("PART B — XFA / FUNDED ACCOUNT")

    print("=" * 110)

    print("ONE PAYOUT PER MONTH.")

    print(f"Minimum winning days: {MIN_WINNING_DAYS}")

    print(f"Minimum winning day: ${MIN_WINNING_DAY_PROFIT}")

    xfa_results = []

    for name, scenario in RISK_SCENARIOS.items():
        for interval in PAYOUT_INTERVALS:
            for amount in PAYOUT_AMOUNTS:
                result = run_xfa_scenario(
                    name,
                    scenario,
                    returns,
                    interval,
                    amount,
                    rng,
                )

                xfa_results.append(result)

    xfa_df = pd.DataFrame(xfa_results)

    # ========================================================
    # BEST POLICIES
    # ========================================================

    print("\n" + "=" * 110)

    print("XFA RESULTS — TOP POLICIES")

    print("=" * 110)

    # Require at least 90% survival.
    viable = xfa_df[xfa_df["survival_rate"] >= 0.90].copy()

    if len(viable) == 0:
        print("No policy achieved 90% survival.")

    else:
        best = viable.sort_values(
            [
                "median_total_withdrawn",
                "survival_rate",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(20)

        print(best.to_string(index=False))

    # ========================================================
    # HIGHEST SURVIVAL
    # ========================================================

    print("\n" + "=" * 110)

    print("HIGHEST-SURVIVAL POLICIES")

    print("=" * 110)

    safest = xfa_df.sort_values(
        [
            "survival_rate",
            "median_total_withdrawn",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(20)

    print(safest.to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    combine_df.to_csv(
        "s2_funded_combine_results.csv",
        index=False,
    )

    xfa_df.to_csv(
        "s2_funded_xfa_results.csv",
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 110)

    print("FUNDED SIMULATION COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_funded_combine_results.csv")

    print("s2_funded_xfa_results.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
