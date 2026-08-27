from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# S2 STATISTICAL VALIDATION
# ============================================================
#
# PURPOSE
# -------
# Statistically validate the already-frozen S2 selective
# strategy using ONLY its OOS trades.
#
# NO PARAMETER OPTIMIZATION.
# NO REGIME OPTIMIZATION.
# NO NEW STRATEGY.
#
#
# INPUT
# -----
# s2_selective_execution_B_trades.csv
#
#
# TESTS
# -----
# 1. Basic OOS statistics
# 2. Trade-level bootstrap
# 3. Block bootstrap by trading day
# 4. Monte Carlo sequence simulation
# 5. Maximum drawdown distribution
# 6. Probability of negative expectancy
# 7. Probability of different drawdown levels
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("s2_selective_execution_B_trades.csv")

N_BOOTSTRAP = 20_000
N_BLOCK_BOOTSTRAP = 20_000
N_MONTE_CARLO = 50_000

RANDOM_SEED = 42

# Drawdown levels expressed in R.
DD_LEVELS = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def percentile_interval(values, confidence=0.95):
    """
    Return a two-sided percentile confidence interval.
    """

    alpha = 1.0 - confidence

    lower = np.percentile(values, 100 * alpha / 2)

    upper = np.percentile(values, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def probability_less_equal_zero(values):
    """
    Fraction of simulated expectancy estimates <= 0.
    """

    return float(np.mean(values <= 0))


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown of an R-return sequence.
    """

    equity = np.cumsum(returns)

    running_max = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]

    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_profit_factor(returns):
    """
    Profit factor = gross profits / gross losses.
    """

    profits = returns[returns > 0].sum()

    losses = -returns[returns < 0].sum()

    if losses == 0:
        return np.inf

    return float(profits / losses)


def longest_losing_streak(returns):
    """
    Longest consecutive sequence of losing trades.
    """

    max_streak = 0
    current = 0

    for r in returns:
        if r < 0:
            current += 1

            max_streak = max(max_streak, current)

        else:
            current = 0

    return max_streak


# ============================================================
# BASIC STATISTICS
# ============================================================


def basic_statistics(returns):

    n = len(returns)

    wins = returns[returns > 0]

    losses = returns[returns < 0]

    win_rate = len(wins) / n if n > 0 else np.nan

    mean_R = returns.mean() if n > 0 else np.nan

    median_R = np.median(returns) if n > 0 else np.nan

    total_R = returns.sum()

    pf = calculate_profit_factor(returns)

    max_dd = calculate_max_drawdown(returns)

    streak = longest_losing_streak(returns)

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "mean_R": mean_R,
        "median_R": median_R,
        "total_R": total_R,
        "profit_factor": pf,
        "max_drawdown_R": max_dd,
        "longest_losing_streak": streak,
    }


# ============================================================
# TRADE-LEVEL BOOTSTRAP
# ============================================================


def trade_bootstrap(
    returns,
    n_iterations,
    rng,
):
    """
    Resample individual trades WITH replacement.

    Each simulation contains the same number of trades
    as the original sample.
    """

    n = len(returns)

    mean_results = np.empty(n_iterations)

    total_results = np.empty(n_iterations)

    pf_results = np.empty(n_iterations)

    dd_results = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = rng.choice(
            returns,
            size=n,
            replace=True,
        )

        mean_results[i] = sample.mean()

        total_results[i] = sample.sum()

        pf_results[i] = calculate_profit_factor(sample)

        dd_results[i] = calculate_max_drawdown(sample)

    return {
        "mean_R": mean_results,
        "total_R": total_results,
        "profit_factor": pf_results,
        "max_drawdown_R": dd_results,
    }


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================


def prepare_daily_blocks(
    df,
):

    x = df.copy()

    x["date"] = x["entry_timestamp_ny"].dt.date

    blocks = []

    for _, day in x.groupby(
        "date",
        sort=True,
    ):
        blocks.append(day["net_R"].to_numpy())

    return blocks


def block_bootstrap(
    blocks,
    n_iterations,
    rng,
):
    """
    Resample complete trading days WITH replacement.

    This preserves the internal sequence of trades inside
    each sampled day and therefore introduces more realistic
    dependence than individual-trade bootstrap.
    """

    n_blocks = len(blocks)

    mean_results = np.empty(n_iterations)

    total_results = np.empty(n_iterations)

    dd_results = np.empty(n_iterations)

    for i in range(n_iterations):
        selected = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        sample_blocks = [blocks[j] for j in selected]

        sample = np.concatenate(sample_blocks)

        mean_results[i] = sample.mean()

        total_results[i] = sample.sum()

        dd_results[i] = calculate_max_drawdown(sample)

    return {
        "mean_R": mean_results,
        "total_R": total_results,
        "max_drawdown_R": dd_results,
    }


# ============================================================
# MONTE CARLO
# ============================================================


def monte_carlo(
    returns,
    n_iterations,
    rng,
):
    """
    Randomly reorder the actual OOS trade returns.

    IMPORTANT:
    This does NOT create new returns.

    It asks:

    "Given exactly this return distribution,
     how bad could the path become simply because
     the sequence happens differently?"
    """

    n = len(returns)

    max_dd = np.empty(n_iterations)

    worst_streak = np.empty(n_iterations)

    final_R = np.empty(n_iterations)

    for i in range(n_iterations):
        sequence = rng.permutation(returns)

        max_dd[i] = calculate_max_drawdown(sequence)

        worst_streak[i] = longest_losing_streak(sequence)

        final_R[i] = sequence.sum()

    return {
        "max_drawdown_R": max_dd,
        "worst_streak": worst_streak,
        "final_R": final_R,
    }


# ============================================================
# PRINT DISTRIBUTION
# ============================================================


def print_distribution(
    name,
    values,
):

    q025 = np.percentile(values, 2.5)

    q50 = np.percentile(values, 50)

    q975 = np.percentile(values, 97.5)

    print(f"{name}")

    print(f"  2.5%  : {q025:.4f}")

    print(f"  50%   : {q50:.4f}")

    print(f"  97.5% : {q975:.4f}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 110)

    print("S2 STATISTICAL VALIDATION")

    print("=" * 110)

    print("\nFROZEN STRATEGY:")

    print("S2 B-SELECTIVE")

    print("Quality >= 0.75")

    print("RR = 1.30")

    print("17.5% lower-tail")

    print("20-point stop")

    print("15-bar horizon")

    print("\nNO PARAMETER OPTIMIZATION.")

    print("NO REGIME OPTIMIZATION.")

    print("NO XGBOOST.")

    print("\nInput:")

    print(INPUT_FILE)

    # ========================================================
    # LOAD
    # ========================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{INPUT_FILE}\n\nRun s2_selective_execution first."
        )

    df = pd.read_csv(INPUT_FILE)

    if "net_R" not in df.columns:
        raise ValueError("CSV does not contain 'net_R'.")

    if "entry_timestamp" not in df.columns:
        raise ValueError("CSV does not contain 'entry_timestamp'.")

    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], utc=True)

    df["entry_timestamp_ny"] = df["entry_timestamp"].dt.tz_convert("America/New_York")

    df = df.sort_values("entry_timestamp").reset_index(drop=True)

    returns = df["net_R"].astype(float).to_numpy()

    returns = returns[np.isfinite(returns)]

    # ========================================================
    # BASIC OOS STATISTICS
    # ========================================================

    stats = basic_statistics(returns)

    print("\n" + "=" * 110)

    print("RAW OOS STATISTICS")

    print("=" * 110)

    for key, value in stats.items():
        if isinstance(
            value,
            float,
        ):
            print(f"{key:30s}: {value:.6f}")

        else:
            print(f"{key:30s}: {value}")

    # ========================================================
    # STANDARD ERROR
    # ========================================================

    standard_error = returns.std(ddof=1) / np.sqrt(len(returns))

    t_stat = returns.mean() / standard_error

    print("\n" + "=" * 110)

    print("BASIC MEAN TEST")

    print("=" * 110)

    print(f"Mean R: {returns.mean():.6f}")

    print(f"Standard error: {standard_error:.6f}")

    print(f"t-statistic: {t_stat:.4f}")

    print("\nThis is only a classical reference test.")

    print(
        "The bootstrap/block-bootstrap results below "
        "are more important for our analysis."
    )

    # ========================================================
    # RNG
    # ========================================================

    rng = np.random.default_rng(RANDOM_SEED)

    # ========================================================
    # TRADE BOOTSTRAP
    # ========================================================

    print("\n" + "=" * 110)

    print("TRADE-LEVEL BOOTSTRAP")

    print("=" * 110)

    print(f"Iterations: {N_BOOTSTRAP:,}")

    bootstrap = trade_bootstrap(
        returns,
        N_BOOTSTRAP,
        rng,
    )

    print_distribution(
        "Mean R",
        bootstrap["mean_R"],
    )

    print_distribution(
        "Total R",
        bootstrap["total_R"],
    )

    print_distribution(
        "Profit Factor",
        bootstrap["profit_factor"],
    )

    print_distribution(
        "Maximum Drawdown R",
        bootstrap["max_drawdown_R"],
    )

    probability_negative = probability_less_equal_zero(bootstrap["mean_R"])

    print("\nProbability bootstrap mean R <= 0:")

    print(f"{probability_negative:.4%}")

    # ========================================================
    # BLOCK BOOTSTRAP
    # ========================================================

    print("\n" + "=" * 110)

    print("BLOCK BOOTSTRAP — BY TRADING DAY")

    print("=" * 110)

    blocks = prepare_daily_blocks(df)

    print(f"Trading-day blocks: {len(blocks)}")

    print(f"Iterations: {N_BLOCK_BOOTSTRAP:,}")

    block_results = block_bootstrap(
        blocks,
        N_BLOCK_BOOTSTRAP,
        rng,
    )

    print_distribution(
        "Mean R",
        block_results["mean_R"],
    )

    print_distribution(
        "Total R",
        block_results["total_R"],
    )

    print_distribution(
        "Maximum Drawdown R",
        block_results["max_drawdown_R"],
    )

    block_negative = probability_less_equal_zero(block_results["mean_R"])

    print("\nProbability block-bootstrap mean R <= 0:")

    print(f"{block_negative:.4%}")

    # ========================================================
    # MONTE CARLO
    # ========================================================

    print("\n" + "=" * 110)

    print("MONTE CARLO PATH SIMULATION")

    print("=" * 110)

    print(f"Iterations: {N_MONTE_CARLO:,}")

    mc = monte_carlo(
        returns,
        N_MONTE_CARLO,
        rng,
    )

    print_distribution(
        "Maximum Drawdown R",
        mc["max_drawdown_R"],
    )

    print_distribution(
        "Worst Losing Streak",
        mc["worst_streak"],
    )

    print_distribution(
        "Final R",
        mc["final_R"],
    )

    # ========================================================
    # DRAWDOWN PROBABILITIES
    # ========================================================

    print("\n" + "=" * 110)

    print("MONTE CARLO DRAWDOWN PROBABILITIES")

    print("=" * 110)

    for level in DD_LEVELS:
        probability = np.mean(mc["max_drawdown_R"] <= -level)

        print(f"Probability DD >= {level:>2}R: {probability:.4%}")

    # ========================================================
    # BLOCK DD PROBABILITIES
    # ========================================================

    print("\n" + "=" * 110)

    print("BLOCK BOOTSTRAP DRAWDOWN PROBABILITIES")

    print("=" * 110)

    for level in DD_LEVELS:
        probability = np.mean(block_results["max_drawdown_R"] <= -level)

        print(f"Probability DD >= {level:>2}R: {probability:.4%}")

    # ========================================================
    # MONTHLY ANALYSIS
    # ========================================================

    df["month"] = df["entry_timestamp"].dt.to_period("M")

    monthly = df.groupby("month")["net_R"].sum()

    print("\n" + "=" * 110)

    print("OBSERVED MONTHLY OOS RESULTS")

    print("=" * 110)

    print(monthly.to_string())

    print("\nMonthly statistics:")

    print(f"Months: {len(monthly)}")

    print(f"Profitable months: {(monthly > 0).sum()}")

    print(f"Losing months: {(monthly < 0).sum()}")

    print(f"Mean monthly R: {monthly.mean():.4f}")

    print(f"Median monthly R: {monthly.median():.4f}")

    print(f"Worst month: {monthly.min():.4f}")

    print(f"Best month: {monthly.max():.4f}")

    # ========================================================
    # SAVE RAW DISTRIBUTIONS
    # ========================================================

    bootstrap_df = pd.DataFrame(
        {
            "bootstrap_mean_R": bootstrap["mean_R"],
            "bootstrap_total_R": bootstrap["total_R"],
            "bootstrap_profit_factor": bootstrap["profit_factor"],
            "bootstrap_max_drawdown_R": bootstrap["max_drawdown_R"],
        }
    )

    bootstrap_df.to_csv(
        "s2_statistical_bootstrap.csv",
        index=False,
    )

    block_df = pd.DataFrame(
        {
            "block_mean_R": block_results["mean_R"],
            "block_total_R": block_results["total_R"],
            "block_max_drawdown_R": block_results["max_drawdown_R"],
        }
    )

    block_df.to_csv(
        "s2_statistical_block_bootstrap.csv",
        index=False,
    )

    mc_df = pd.DataFrame(
        {
            "mc_max_drawdown_R": mc["max_drawdown_R"],
            "mc_worst_streak": mc["worst_streak"],
            "mc_final_R": mc["final_R"],
        }
    )

    mc_df.to_csv(
        "s2_statistical_monte_carlo.csv",
        index=False,
    )

    monthly.to_csv("s2_statistical_monthly.csv")

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    bootstrap_ci = percentile_interval(bootstrap["mean_R"])

    block_ci = percentile_interval(block_results["mean_R"])

    mc_dd_95 = np.percentile(mc["max_drawdown_R"], 5)

    mc_dd_99 = np.percentile(mc["max_drawdown_R"], 1)

    print("\n" + "=" * 110)

    print("VALIDATION SUMMARY")

    print("=" * 110)

    print(f"Observed expectancy: {returns.mean():.4f}R")

    print(f"Trade bootstrap 95% CI: [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]")

    print(f"Block bootstrap 95% CI: [{block_ci[0]:.4f}, {block_ci[1]:.4f}]")

    print(f"Probability mean <= 0 (trade bootstrap): {probability_negative:.4%}")

    print(f"Probability mean <= 0 (block bootstrap): {block_negative:.4%}")

    print(f"Monte Carlo 95% DD: {mc_dd_95:.4f}R")

    print(f"Monte Carlo 99% DD: {mc_dd_99:.4f}R")

    print("\nIMPORTANT:")

    print("This is statistical validation, NOT a guarantee of future performance.")

    print("The next stage is the funded-account simulation.")

    print("\n" + "=" * 110)

    print("S2 STATISTICAL VALIDATION COMPLETE")

    print("=" * 110)

    print("Saved:")

    print("s2_statistical_bootstrap.csv")

    print("s2_statistical_block_bootstrap.csv")

    print("s2_statistical_monte_carlo.csv")

    print("s2_statistical_monthly.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()
