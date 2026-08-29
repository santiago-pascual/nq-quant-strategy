from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# S28.1 — TIME-AWARE BLOCK BOOTSTRAP ROBUSTNESS
# =============================================================================
#
# Frozen strategy:
#
#   S2 + MAE/Recovery
#   MAE >= 0.70R
#   Recovery >= +0.20R
#   Deadline = 6 bars
#
# NO PARAMETER OPTIMIZATION.
#
# Purpose:
#   Test whether the observed strategy performance remains stable when
#   preserving some local temporal clustering through block bootstrap.
#
# Outputs:
#   - summary CSV
#   - simulation CSV
#   - quantiles CSV
#   - Total R chart
#   - Drawdown chart
#   - Equity paths chart
#   - Observed vs bootstrap chart
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

TRADES_FILE = RESULTS_DIR / "s27_full_strategy_trades.csv"

SUMMARY_FILE = RESULTS_DIR / "s28_1_block_bootstrap_summary.csv"

SIMULATION_FILE = RESULTS_DIR / "s28_1_block_bootstrap_simulations.csv"

QUANTILES_FILE = RESULTS_DIR / "s28_1_block_bootstrap_quantiles.csv"

PLOT_TOTAL_R = RESULTS_DIR / "s28_1_total_R_distribution.png"

PLOT_DRAWDOWN = RESULTS_DIR / "s28_1_drawdown_distribution.png"

PLOT_EQUITY = RESULTS_DIR / "s28_1_oos_equity_paths.png"

PLOT_COMPARISON = RESULTS_DIR / "s28_1_oos_observed_vs_bootstrap.png"


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

N_SIMULATIONS = 10_000

SEED = 20260828

BLOCK_LENGTHS = [
    5,
    10,
    20,
]

N_EQUITY_PATHS = 100


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(values: np.ndarray) -> dict:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "trades": 0,
            "total_R": np.nan,
            "mean_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
        }

    wins = values[values > 0]

    losses = values[values <= 0]

    gross_profit = float(wins.sum())

    gross_loss = float(abs(losses.sum()))

    if gross_loss == 0:
        profit_factor = math.inf if gross_profit > 0 else np.nan

    else:
        profit_factor = gross_profit / gross_loss

    equity = np.cumsum(values)

    running_max = np.maximum.accumulate(equity)

    drawdown = equity - running_max

    max_drawdown = float(drawdown.min())

    return {
        "trades": int(len(values)),
        "total_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "win_rate": float((values > 0).mean()),
        "profit_factor": profit_factor,
        "max_drawdown_R": max_drawdown,
    }


# =============================================================================
# BLOCK BOOTSTRAP
# =============================================================================


def block_bootstrap(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:

    n = len(values)

    if block_length > n:
        raise ValueError("Block length exceeds sample size.")

    # Circular moving blocks.
    #
    # This preserves local ordering inside each block while allowing
    # blocks to be sampled from anywhere in the original sequence.

    n_blocks = int(np.ceil(n / block_length))

    starts = rng.integers(
        0,
        n,
        size=n_blocks,
    )

    blocks = []

    for start in starts:
        indices = (start + np.arange(block_length)) % n

        blocks.append(values[indices])

    sample = np.concatenate(blocks)

    return sample[:n]


# =============================================================================
# SINGLE SAMPLE
# =============================================================================


def run_sample(
    sample_name: str,
    values: np.ndarray,
    rng: np.random.Generator,
):

    print()
    print("=" * 110)
    print(f"S28.1 SAMPLE: {sample_name}")
    print("=" * 110)

    observed = calculate_metrics(values)

    print(f"Trades        : {observed['trades']}")

    print(f"Observed R    : {observed['total_R']:.4f}")

    print(f"Observed Mean : {observed['mean_R']:.6f}")

    print(f"Observed WR   : {observed['win_rate']:.4%}")

    print(f"Observed PF   : {observed['profit_factor']}")

    print(f"Observed DD   : {observed['max_drawdown_R']:.4f}")

    summary_rows = []

    simulation_rows = []

    equity_paths = []

    for block_length in BLOCK_LENGTHS:
        print()
        print(f"Block length {block_length}: {N_SIMULATIONS:,} simulations")

        totals = np.empty(N_SIMULATIONS)

        means = np.empty(N_SIMULATIONS)

        win_rates = np.empty(N_SIMULATIONS)

        max_dds = np.empty(N_SIMULATIONS)

        for i in range(N_SIMULATIONS):
            sample = block_bootstrap(
                values,
                block_length,
                rng,
            )

            m = calculate_metrics(sample)

            totals[i] = m["total_R"]

            means[i] = m["mean_R"]

            win_rates[i] = m["win_rate"]

            max_dds[i] = m["max_drawdown_R"]

            simulation_rows.append(
                {
                    "sample": sample_name,
                    "block_length": block_length,
                    "simulation": i + 1,
                    "total_R": totals[i],
                    "mean_R": means[i],
                    "win_rate": win_rates[i],
                    "max_drawdown_R": max_dds[i],
                }
            )

            # Keep a limited number of OOS paths for visualization.
            if sample_name == "HOLDOUT_OOS" and i < N_EQUITY_PATHS:
                equity_paths.append(
                    {
                        "block_length": block_length,
                        "simulation": i + 1,
                        "equity": np.cumsum(sample),
                    }
                )

        summary_rows.append(
            {
                "sample": sample_name,
                "block_length": block_length,
                "trades": observed["trades"],
                "observed_total_R": observed["total_R"],
                "observed_mean_R": observed["mean_R"],
                "observed_win_rate": observed["win_rate"],
                "observed_profit_factor": observed["profit_factor"],
                "observed_max_drawdown_R": observed["max_drawdown_R"],
                "negative_probability": float((totals <= 0).mean()),
                "total_R_p05": float(
                    np.quantile(
                        totals,
                        0.05,
                    )
                ),
                "total_R_p50": float(
                    np.quantile(
                        totals,
                        0.50,
                    )
                ),
                "total_R_p95": float(
                    np.quantile(
                        totals,
                        0.95,
                    )
                ),
                "mean_R_p05": float(
                    np.quantile(
                        means,
                        0.05,
                    )
                ),
                "mean_R_p50": float(
                    np.quantile(
                        means,
                        0.50,
                    )
                ),
                "mean_R_p95": float(
                    np.quantile(
                        means,
                        0.95,
                    )
                ),
                "win_rate_p05": float(
                    np.quantile(
                        win_rates,
                        0.05,
                    )
                ),
                "win_rate_p50": float(
                    np.quantile(
                        win_rates,
                        0.50,
                    )
                ),
                "win_rate_p95": float(
                    np.quantile(
                        win_rates,
                        0.95,
                    )
                ),
                "max_DD_p05": float(
                    np.quantile(
                        max_dds,
                        0.05,
                    )
                ),
                "max_DD_p50": float(
                    np.quantile(
                        max_dds,
                        0.50,
                    )
                ),
                "max_DD_p95": float(
                    np.quantile(
                        max_dds,
                        0.95,
                    )
                ),
                "prob_DD_le_10R": float((max_dds <= -10).mean()),
                "prob_DD_le_15R": float((max_dds <= -15).mean()),
                "prob_DD_le_20R": float((max_dds <= -20).mean()),
                "prob_DD_le_25R": float((max_dds <= -25).mean()),
            }
        )

    return (
        summary_rows,
        simulation_rows,
        equity_paths,
    )


# =============================================================================
# PLOTS
# =============================================================================


def create_plots(
    summary: pd.DataFrame,
    equity_paths: list,
    oos_values: np.ndarray,
):

    oos = summary[summary["sample"] == "HOLDOUT_OOS"].copy()

    observed = calculate_metrics(oos_values)

    # =========================================================================
    # 1. TOTAL R
    # =========================================================================

    fig = plt.figure(figsize=(12, 7))

    x = oos["block_length"].to_numpy()

    plt.plot(
        x,
        oos["total_R_p05"],
        marker="o",
        label="P05",
    )

    plt.plot(
        x,
        oos["total_R_p50"],
        marker="o",
        label="Median",
    )

    plt.plot(
        x,
        oos["total_R_p95"],
        marker="o",
        label="P95",
    )

    plt.axhline(
        observed["total_R"],
        linestyle="--",
        linewidth=2,
        label="Observed OOS",
    )

    plt.xlabel("Block Length")

    plt.ylabel("Total R")

    plt.title("S28.1 — OOS Total R Under Time-Aware Block Bootstrap")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_TOTAL_R,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # 2. DRAWDOWN
    # =========================================================================

    fig = plt.figure(figsize=(12, 7))

    plt.plot(
        x,
        oos["max_DD_p05"],
        marker="o",
        label="P05",
    )

    plt.plot(
        x,
        oos["max_DD_p50"],
        marker="o",
        label="Median",
    )

    plt.plot(
        x,
        oos["max_DD_p95"],
        marker="o",
        label="P95",
    )

    plt.axhline(
        observed["max_drawdown_R"],
        linestyle="--",
        linewidth=2,
        label="Observed OOS",
    )

    plt.xlabel("Block Length")

    plt.ylabel("Maximum Drawdown (R)")

    plt.title("S28.1 — OOS Drawdown Under Time-Aware Block Bootstrap")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DRAWDOWN,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # 3. EQUITY PATHS
    # =========================================================================

    fig = plt.figure(figsize=(14, 8))

    for item in equity_paths:
        plt.plot(
            np.arange(
                1,
                len(item["equity"]) + 1,
            ),
            item["equity"],
            alpha=0.08,
        )

    observed_equity = np.cumsum(oos_values)

    plt.plot(
        np.arange(
            1,
            len(observed_equity) + 1,
        ),
        observed_equity,
        linewidth=3,
        label="Observed OOS",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Trade Number")

    plt.ylabel("Cumulative R")

    plt.title("S28.1 — OOS Equity Paths Under Time-Aware Block Bootstrap")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_EQUITY,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # 4. OBSERVED VS BOOTSTRAP
    # =========================================================================

    fig = plt.figure(figsize=(12, 7))

    labels = [
        "Observed",
        "Block 5",
        "Block 10",
        "Block 20",
    ]

    observed_total = observed["total_R"]

    medians = [
        observed_total,
        float(
            oos.loc[
                oos["block_length"] == 5,
                "total_R_p50",
            ].iloc[0]
        ),
        float(
            oos.loc[
                oos["block_length"] == 10,
                "total_R_p50",
            ].iloc[0]
        ),
        float(
            oos.loc[
                oos["block_length"] == 20,
                "total_R_p50",
            ].iloc[0]
        ),
    ]

    p05 = [
        observed_total,
        float(
            oos.loc[
                oos["block_length"] == 5,
                "total_R_p05",
            ].iloc[0]
        ),
        float(
            oos.loc[
                oos["block_length"] == 10,
                "total_R_p05",
            ].iloc[0]
        ),
        float(
            oos.loc[
                oos["block_length"] == 20,
                "total_R_p05",
            ].iloc[0]
        ),
    ]

    p95 = [
        observed_total,
        float(
            oos.loc[
                oos["block_length"] == 5,
                "total_R_p95",
            ].iloc[0]
        ),
        float(
            oos.loc[
                oos["block_length"] == 10,
                "total_R_p95",
            ].iloc[0]
        ),
        float(
            oos.loc[
                oos["block_length"] == 20,
                "total_R_p95",
            ].iloc[0]
        ),
    ]

    positions = np.arange(len(labels))

    plt.errorbar(
        positions[1:],
        medians[1:],
        yerr=[
            np.array(medians[1:]) - np.array(p05[1:]),
            np.array(p95[1:]) - np.array(medians[1:]),
        ],
        fmt="o",
        markersize=8,
        capsize=6,
        linewidth=2,
        label="Bootstrap P05–P95",
    )

    plt.scatter(
        positions[0],
        observed_total,
        s=120,
        label="Observed OOS",
    )

    plt.xticks(
        positions,
        labels,
    )

    plt.ylabel("Total R")

    plt.title("S28.1 — Observed OOS vs Time-Aware Bootstrap")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_COMPARISON,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S28.1 TIME-AWARE BLOCK BOOTSTRAP ROBUSTNESS")
    print("=" * 110)

    print()
    print("Frozen S27 strategy.")

    print("NO PARAMETER OPTIMIZATION.")

    print()
    print(f"Simulations: {N_SIMULATIONS:,}")

    print(f"Block lengths: {BLOCK_LENGTHS}")

    print(f"Random seed: {SEED}")

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("Loading:")

    print(TRADES_FILE)

    df = pd.read_csv(TRADES_FILE)

    required = [
        "_strategy_R",
        "_window_numeric",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing S27 columns: {missing}")

    df["_strategy_R"] = pd.to_numeric(
        df["_strategy_R"],
        errors="coerce",
    )

    df["_window_numeric"] = pd.to_numeric(
        df["_window_numeric"],
        errors="coerce",
    )

    if df["_strategy_R"].isna().any():
        raise RuntimeError("Invalid strategy R detected.")

    # =========================================================================
    # DATASETS
    # =========================================================================

    full_values = df["_strategy_R"].to_numpy(dtype=float)

    development_values = df.loc[
        df["_window_numeric"].between(
            1,
            11,
        ),
        "_strategy_R",
    ].to_numpy(dtype=float)

    oos_values = df.loc[
        df["_window_numeric"].between(
            12,
            22,
        ),
        "_strategy_R",
    ].to_numpy(dtype=float)

    print()
    print("Samples:")

    print(f"  Full dataset : {len(full_values)}")

    print(f"  Development  : {len(development_values)}")

    print(f"  Holdout OOS  : {len(oos_values)}")

    # =========================================================================
    # RUN
    # =========================================================================

    all_summary = []

    all_simulations = []

    all_equity_paths = []

    samples = [
        (
            "FULL_DATASET",
            full_values,
        ),
        (
            "DEVELOPMENT",
            development_values,
        ),
        (
            "HOLDOUT_OOS",
            oos_values,
        ),
    ]

    for sample_name, values in samples:
        rng = np.random.default_rng(
            SEED + len(all_summary) + 1000,
        )

        (
            summary_rows,
            simulation_rows,
            equity_paths,
        ) = run_sample(
            sample_name,
            values,
            rng,
        )

        all_summary.extend(summary_rows)

        all_simulations.extend(simulation_rows)

        all_equity_paths.extend(equity_paths)

    summary_df = pd.DataFrame(all_summary)

    simulations_df = pd.DataFrame(all_simulations)

    # =========================================================================
    # QUANTILE TABLE
    # =========================================================================

    quantile_rows = []

    for _, row in summary_df.iterrows():
        for metric in [
            "total_R",
            "mean_R",
            "win_rate",
            "max_DD",
        ]:
            for quantile_name, column in [
                (
                    "P05",
                    f"{metric}_p05",
                ),
                (
                    "P50",
                    f"{metric}_p50",
                ),
                (
                    "P95",
                    f"{metric}_p95",
                ),
            ]:
                quantile_rows.append(
                    {
                        "sample": row["sample"],
                        "block_length": row["block_length"],
                        "metric": metric,
                        "quantile": quantile_name,
                        "value": row[column],
                    }
                )

    quantiles_df = pd.DataFrame(quantile_rows)

    # =========================================================================
    # SAVE TABLES
    # =========================================================================

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    simulations_df.to_csv(
        SIMULATION_FILE,
        index=False,
    )

    quantiles_df.to_csv(
        QUANTILES_FILE,
        index=False,
    )

    # =========================================================================
    # PLOTS
    # =========================================================================

    create_plots(
        summary_df,
        all_equity_paths,
        oos_values,
    )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print()
    print("=" * 110)
    print("S28.1 SUMMARY")
    print("=" * 110)

    print(summary_df.to_string(index=False))

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    for path in [
        SUMMARY_FILE,
        SIMULATION_FILE,
        QUANTILES_FILE,
        PLOT_TOTAL_R,
        PLOT_DRAWDOWN,
        PLOT_EQUITY,
        PLOT_COMPARISON,
    ]:
        print(path)

    print()
    print("=" * 110)
    print("S28.1 TIME-AWARE ROBUSTNESS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
