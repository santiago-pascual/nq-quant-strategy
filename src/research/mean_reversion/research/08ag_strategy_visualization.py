from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 08AG — MEAN REVERSION STRATEGY VISUALIZATION
# =============================================================================
#
# Purpose:
#   Publication/research-ready visualization of the two frozen new
#   Mean Reversion strategies.
#
# Strategies:
#   MRS2_NEW
#   MRL1_NEW
#
# Source:
#   08AA frozen trade population
#   08AD Monte Carlo methodology
#
# No parameter optimization.
# No event re-selection.
# No strategy modification.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

TRADES_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

OUTPUT_DIR = RESULTS_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

STRATEGIES = [
    "MRS2_NEW",
    "MRL1_NEW",
]

N_SIMULATIONS = 20_000
RANDOM_SEED = 20260909

BLOCK_LENGTHS = [5, 10, 20, 40]


# =============================================================================
# HELPERS
# =============================================================================


def max_drawdown(equity):
    """
    Calculate maximum drawdown from an equity curve.
    """
    equity = np.asarray(equity, dtype=float)

    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_metrics(r_values):
    """
    Calculate basic trade-level metrics.
    """
    r_values = np.asarray(r_values, dtype=float)

    total_r = float(r_values.sum())
    expectancy = float(r_values.mean())

    wins = r_values[r_values > 0]
    losses = r_values[r_values < 0]

    win_rate = float((r_values > 0).mean())

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    equity = np.concatenate([[0.0], np.cumsum(r_values)])
    dd = max_drawdown(equity)

    return {
        "trades": len(r_values),
        "total_r": total_r,
        "expectancy_r": expectancy,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_r": dd,
    }


def iid_bootstrap(r_values, rng, n_simulations):
    """
    IID bootstrap.

    Each simulation resamples the original trade population
    with replacement.
    """
    n = len(r_values)

    sampled = rng.choice(
        r_values,
        size=(n_simulations, n),
        replace=True,
    )

    terminal_r = sampled.sum(axis=1)
    expectancy = sampled.mean(axis=1)

    cumulative = np.cumsum(sampled, axis=1)
    running_max = np.maximum.accumulate(
        np.maximum(cumulative, 0.0),
        axis=1,
    )

    drawdowns = cumulative - running_max
    max_dd = drawdowns.min(axis=1)

    return terminal_r, expectancy, max_dd


def block_bootstrap(r_values, rng, n_simulations, block_length):
    """
    Time-aware block bootstrap.

    Trades are sampled in contiguous blocks to preserve
    local serial dependence.
    """
    n = len(r_values)

    n_blocks = int(np.ceil(n / block_length))

    simulations = np.empty(
        (n_simulations, n),
        dtype=float,
    )

    for sim in range(n_simulations):
        starts = rng.integers(
            0,
            n - block_length + 1,
            size=n_blocks,
        )

        sampled = np.concatenate(
            [r_values[start : start + block_length] for start in starts]
        )

        simulations[sim] = sampled[:n]

    terminal_r = simulations.sum(axis=1)
    expectancy = simulations.mean(axis=1)

    cumulative = np.cumsum(simulations, axis=1)

    running_max = np.maximum.accumulate(
        np.maximum(cumulative, 0.0),
        axis=1,
    )

    drawdowns = cumulative - running_max
    max_dd = drawdowns.min(axis=1)

    return terminal_r, expectancy, max_dd


def percentile_summary(values):
    """
    Return P05/P50/P95.
    """
    return {
        "P05": float(np.percentile(values, 5)),
        "P50": float(np.percentile(values, 50)),
        "P95": float(np.percentile(values, 95)),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("08AG — MEAN REVERSION VISUALIZATION")
print("=" * 80)

print()
print("Loading frozen 08AA trade population...")
print(TRADES_FILE)

trades = pd.read_csv(TRADES_FILE)

trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)

print(f"Total 08AA trades: {len(trades):,}")


# =============================================================================
# VALIDATION
# =============================================================================

required_columns = {
    "candidate_id",
    "timestamp",
    "r",
    "result",
    "window",
}

missing = required_columns - set(trades.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


for strategy in STRATEGIES:
    count = int((trades["candidate_id"] == strategy).sum())

    if count == 0:
        raise ValueError(f"No trades found for {strategy}")

    print(f"{strategy}: {count:,} trades")


# =============================================================================
# GLOBAL COMBINED EQUITY
# =============================================================================

selected = trades[trades["candidate_id"].isin(STRATEGIES)].copy()

selected = selected.sort_values("timestamp").reset_index(drop=True)

selected["cumulative_r"] = selected["r"].cumsum()

plt.figure(figsize=(14, 7))

plt.plot(
    selected["timestamp"],
    selected["cumulative_r"],
    linewidth=1.5,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title("Mean Reversion — Combined Equity Curve")

plt.xlabel("Date")
plt.ylabel("Cumulative R")

plt.grid(True, alpha=0.25)

plt.tight_layout()

combined_path = OUTPUT_DIR / "mean_reversion_combined_equity_curve.png"

plt.savefig(
    combined_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {combined_path}")


# =============================================================================
# INDIVIDUAL STRATEGY ANALYSIS
# =============================================================================

all_metrics = []
all_mc_summary = []


for strategy in STRATEGIES:
    print()
    print("=" * 80)
    print(strategy)
    print("=" * 80)

    df = trades[trades["candidate_id"] == strategy].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    r_values = df["r"].to_numpy(dtype=float)

    metrics = calculate_metrics(r_values)

    print(f"Trades       : {metrics['trades']:,}")

    print(f"Total R      : {metrics['total_r']:.2f}")

    print(f"Expectancy   : {metrics['expectancy_r']:.6f}R")

    print(f"Win Rate     : {metrics['win_rate']:.4%}")

    print(f"Profit Factor: {metrics['profit_factor']:.4f}")

    print(f"Max DD       : {metrics['max_drawdown_r']:.2f}R")

    # =========================================================================
    # EQUITY CURVE
    # =========================================================================

    equity = np.concatenate([[0.0], np.cumsum(r_values)])

    plt.figure(figsize=(14, 7))

    plt.plot(
        range(len(equity)),
        equity,
        linewidth=1.5,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title(f"{strategy} — Equity Curve")

    plt.xlabel("Trade")
    plt.ylabel("Cumulative R")

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    path = OUTPUT_DIR / f"{strategy.lower()}_equity_curve.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================================
    # DRAWDOWN
    # =========================================================================

    running_max = np.maximum.accumulate(equity)

    drawdown = equity - running_max

    plt.figure(figsize=(14, 5))

    plt.plot(
        range(len(drawdown)),
        drawdown,
        linewidth=1.3,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title(f"{strategy} — Drawdown")

    plt.xlabel("Trade")
    plt.ylabel("Drawdown (R)")

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    path = OUTPUT_DIR / f"{strategy.lower()}_drawdown.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================================
    # YEARLY EQUITY
    # =========================================================================

    df["year"] = df["timestamp"].dt.year

    yearly = df.groupby("year")["r"].sum().sort_index()

    plt.figure(figsize=(12, 6))

    plt.bar(
        yearly.index.astype(str),
        yearly.values,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title(f"{strategy} — Annual R")

    plt.xlabel("Year")
    plt.ylabel("Net R")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    path = OUTPUT_DIR / f"{strategy.lower()}_annual_r.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================================
    # IID MONTE CARLO
    # =========================================================================

    rng = np.random.default_rng(RANDOM_SEED)

    terminal_r, expectancy, max_dd = iid_bootstrap(
        r_values,
        rng,
        N_SIMULATIONS,
    )

    terminal_summary = percentile_summary(terminal_r)

    expectancy_summary = percentile_summary(expectancy)

    dd_summary = percentile_summary(max_dd)

    probability_negative = float((terminal_r < 0).mean())

    print()
    print("IID Monte Carlo")
    print(
        f"Terminal R P05/P50/P95: "
        f"{terminal_summary['P05']:.2f} / "
        f"{terminal_summary['P50']:.2f} / "
        f"{terminal_summary['P95']:.2f}"
    )

    print(
        f"Expectancy P05/P50/P95: "
        f"{expectancy_summary['P05']:.6f} / "
        f"{expectancy_summary['P50']:.6f} / "
        f"{expectancy_summary['P95']:.6f}"
    )

    print(
        f"Max DD P05/P50/P95: "
        f"{dd_summary['P05']:.2f} / "
        f"{dd_summary['P50']:.2f} / "
        f"{dd_summary['P95']:.2f}"
    )

    print(f"P(Terminal R < 0): {probability_negative:.4%}")

    # =========================================================================
    # MONTE CARLO TERMINAL DISTRIBUTION
    # =========================================================================

    plt.figure(figsize=(12, 6))

    plt.hist(
        terminal_r,
        bins=80,
    )

    plt.axvline(
        terminal_summary["P05"],
        linestyle="--",
        linewidth=1.2,
    )

    plt.axvline(
        terminal_summary["P50"],
        linestyle="--",
        linewidth=1.2,
    )

    plt.axvline(
        terminal_summary["P95"],
        linestyle="--",
        linewidth=1.2,
    )

    plt.title(f"{strategy} — Monte Carlo Terminal R")

    plt.xlabel("Terminal R")
    plt.ylabel("Frequency")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    path = OUTPUT_DIR / f"{strategy.lower()}_monte_carlo_terminal.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================================
    # MONTE CARLO MAX DRAWdown DISTRIBUTION
    # =========================================================================

    plt.figure(figsize=(12, 6))

    plt.hist(
        max_dd,
        bins=80,
    )

    plt.axvline(
        dd_summary["P05"],
        linestyle="--",
        linewidth=1.2,
    )

    plt.axvline(
        dd_summary["P50"],
        linestyle="--",
        linewidth=1.2,
    )

    plt.axvline(
        dd_summary["P95"],
        linestyle="--",
        linewidth=1.2,
    )

    plt.title(f"{strategy} — Monte Carlo Maximum Drawdown")

    plt.xlabel("Maximum Drawdown (R)")
    plt.ylabel("Frequency")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    path = OUTPUT_DIR / f"{strategy.lower()}_monte_carlo_drawdown.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================================
    # MONTE CARLO EQUITY FAN
    # =========================================================================

    # Re-run a smaller matrix only for visualization.
    # 2,000 paths is enough for a smooth visual fan and avoids
    # storing unnecessary memory.
    n_paths_for_plot = 2_000

    sampled = rng.choice(
        r_values,
        size=(n_paths_for_plot, len(r_values)),
        replace=True,
    )

    mc_equity = np.cumsum(
        sampled,
        axis=1,
    )

    p05 = np.percentile(
        mc_equity,
        5,
        axis=0,
    )

    p50 = np.percentile(
        mc_equity,
        50,
        axis=0,
    )

    p95 = np.percentile(
        mc_equity,
        95,
        axis=0,
    )

    plt.figure(figsize=(14, 7))

    plt.fill_between(
        range(len(r_values)),
        p05,
        p95,
        alpha=0.20,
    )

    plt.plot(
        p50,
        linewidth=1.5,
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title(f"{strategy} — Monte Carlo Equity Fan")

    plt.xlabel("Trade")
    plt.ylabel("Cumulative R")

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    path = OUTPUT_DIR / f"{strategy.lower()}_monte_carlo_equity_fan.png"

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")

    # =========================================================================
    # BLOCK BOOTSTRAP SUMMARY
    # =========================================================================

    for block_length in BLOCK_LENGTHS:
        (
            block_terminal,
            block_expectancy,
            block_dd,
        ) = block_bootstrap(
            r_values,
            rng,
            N_SIMULATIONS,
            block_length,
        )

        all_mc_summary.append(
            {
                "candidate_id": strategy,
                "method": "block_bootstrap",
                "block_length": block_length,
                "terminal_p05": np.percentile(
                    block_terminal,
                    5,
                ),
                "terminal_p50": np.percentile(
                    block_terminal,
                    50,
                ),
                "terminal_p95": np.percentile(
                    block_terminal,
                    95,
                ),
                "expectancy_p05": np.percentile(
                    block_expectancy,
                    5,
                ),
                "expectancy_p50": np.percentile(
                    block_expectancy,
                    50,
                ),
                "expectancy_p95": np.percentile(
                    block_expectancy,
                    95,
                ),
                "max_dd_p05": np.percentile(
                    block_dd,
                    5,
                ),
                "max_dd_p50": np.percentile(
                    block_dd,
                    50,
                ),
                "max_dd_p95": np.percentile(
                    block_dd,
                    95,
                ),
                "probability_terminal_negative": (block_terminal < 0).mean(),
            }
        )

    all_metrics.append(
        {
            "candidate_id": strategy,
            **metrics,
            "mc_terminal_p05": terminal_summary["P05"],
            "mc_terminal_p50": terminal_summary["P50"],
            "mc_terminal_p95": terminal_summary["P95"],
            "mc_expectancy_p05": expectancy_summary["P05"],
            "mc_expectancy_p50": expectancy_summary["P50"],
            "mc_expectancy_p95": expectancy_summary["P95"],
            "mc_dd_p05": dd_summary["P05"],
            "mc_dd_p50": dd_summary["P50"],
            "mc_dd_p95": dd_summary["P95"],
            "mc_probability_terminal_negative": probability_negative,
        }
    )


# =============================================================================
# COMPARISON EQUITY CURVES
# =============================================================================

plt.figure(figsize=(14, 7))

for strategy in STRATEGIES:
    df = trades[trades["candidate_id"] == strategy].copy()

    df = df.sort_values("timestamp")

    equity = df["r"].cumsum()

    plt.plot(
        df["timestamp"],
        equity,
        linewidth=1.4,
        label=strategy,
    )

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title("Mean Reversion — Strategy Comparison")

plt.xlabel("Date")
plt.ylabel("Cumulative R")

plt.legend()

plt.grid(True, alpha=0.25)

plt.tight_layout()

comparison_path = OUTPUT_DIR / "mean_reversion_strategy_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {comparison_path}")


# =============================================================================
# SAVE METRICS
# =============================================================================

metrics_df = pd.DataFrame(all_metrics)

metrics_path = OUTPUT_DIR / "mean_reversion_visualization_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False,
)

mc_df = pd.DataFrame(all_mc_summary)

mc_path = OUTPUT_DIR / "mean_reversion_visualization_monte_carlo.csv"

mc_df.to_csv(
    mc_path,
    index=False,
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print()
print("=" * 80)
print("08AG COMPLETE")
print("=" * 80)

print(f"Output directory: {OUTPUT_DIR}")

print()
print("Generated:")
print("  - Combined equity curve")
print("  - Individual equity curves")
print("  - Individual drawdowns")
print("  - Annual R")
print("  - Monte Carlo terminal distributions")
print("  - Monte Carlo drawdown distributions")
print("  - Monte Carlo equity fans")
print("  - Strategy comparison")
print("  - Metrics CSV")
print("  - Block bootstrap CSV")

print()
print("No parameter optimization.")
print("No event re-selection.")
print("08AA frozen population preserved.")
print("08AA frozen population preserved.")
