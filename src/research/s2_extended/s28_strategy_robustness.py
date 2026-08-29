from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


# =============================================================================
# S28 — STRATEGY ROBUSTNESS
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
# Tests:
#   1. Trade reshuffling Monte Carlo
#   2. Bootstrap
#   3. Probability of negative total R
#   4. Probability of large drawdown
#   5. Percentile distributions
#   6. Worst-case sequences
#   7. OOS-only robustness
#   8. Full-dataset robustness
#   9. S2 vs S2+Recovery comparison
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

TRADES_FILE = RESULTS_DIR / "s27_full_strategy_trades.csv"

SUMMARY_FILE = RESULTS_DIR / "s28_robustness_summary.csv"

MONTE_CARLO_FILE = RESULTS_DIR / "s28_monte_carlo_results.csv"

BOOTSTRAP_FILE = RESULTS_DIR / "s28_bootstrap_results.csv"

QUANTILES_FILE = RESULTS_DIR / "s28_distribution_quantiles.csv"

OOS_FILE = RESULTS_DIR / "s28_oos_robustness.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

N_SIMULATIONS = 10_000

SEED = 20260828

NEGATIVE_R_THRESHOLD = 0.0

# Drawdown thresholds are expressed in R.
DD_THRESHOLDS = [
    -10.0,
    -15.0,
    -20.0,
    -25.0,
    -30.0,
]

# Worst sequence lengths.
WORST_SEQUENCE_LENGTHS = [
    10,
    20,
    50,
    100,
]


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(
    values: np.ndarray,
) -> dict:

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
            "gross_profit_R": np.nan,
            "gross_loss_R": np.nan,
        }

    wins = values[values > 0]

    losses = values[values <= 0]

    gross_profit = float(wins.sum())

    gross_loss = float(abs(losses.sum()))

    if gross_loss == 0:
        pf = math.inf if gross_profit > 0 else np.nan

    else:
        pf = gross_profit / gross_loss

    equity = np.cumsum(values)

    running_max = np.maximum.accumulate(equity)

    drawdown = equity - running_max

    max_dd = float(drawdown.min())

    return {
        "trades": int(len(values)),
        "total_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "win_rate": float((values > 0).mean()),
        "profit_factor": pf,
        "max_drawdown_R": max_dd,
        "gross_profit_R": gross_profit,
        "gross_loss_R": gross_loss,
    }


# =============================================================================
# LOAD DATA
# =============================================================================


def load_data():

    print()
    print("=" * 110)
    print("LOADING FROZEN S27 STRATEGY")
    print("=" * 110)

    df = pd.read_csv(TRADES_FILE)

    print(TRADES_FILE)
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    required = [
        "_strategy_R",
        "_s2_R",
        "_window_numeric",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing S27 columns: {missing}")

    strategy = pd.to_numeric(
        df["_strategy_R"],
        errors="coerce",
    )

    benchmark = pd.to_numeric(
        df["_s2_R"],
        errors="coerce",
    )

    if strategy.isna().any():
        raise RuntimeError("S27 contains invalid strategy R.")

    if benchmark.isna().any():
        raise RuntimeError("S27 contains invalid benchmark R.")

    if not np.isfinite(strategy.to_numpy()).all():
        raise RuntimeError("S27 strategy R contains non-finite values.")

    print("Strategy values: PASS")

    print("No parameter optimization.")

    return df


# =============================================================================
# MONTE CARLO
# =============================================================================


def monte_carlo(
    values: np.ndarray,
    rng: np.random.Generator,
):

    n = len(values)

    totals = np.empty(N_SIMULATIONS)

    max_dds = np.empty(N_SIMULATIONS)

    terminal_values = np.empty(N_SIMULATIONS)

    for i in range(N_SIMULATIONS):
        shuffled = rng.permutation(values)

        equity = np.cumsum(shuffled)

        running_max = np.maximum.accumulate(equity)

        dd = equity - running_max

        totals[i] = equity[-1]

        terminal_values[i] = equity[-1]

        max_dds[i] = dd.min()

    return (
        totals,
        max_dds,
        terminal_values,
    )


# =============================================================================
# BOOTSTRAP
# =============================================================================


def bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
):

    n = len(values)

    totals = np.empty(N_SIMULATIONS)

    means = np.empty(N_SIMULATIONS)

    win_rates = np.empty(N_SIMULATIONS)

    max_dds = np.empty(N_SIMULATIONS)

    for i in range(N_SIMULATIONS):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        equity = np.cumsum(sample)

        running_max = np.maximum.accumulate(equity)

        dd = equity - running_max

        totals[i] = sample.sum()

        means[i] = sample.mean()

        win_rates[i] = (sample > 0).mean()

        max_dds[i] = dd.min()

    return (
        totals,
        means,
        win_rates,
        max_dds,
    )


# =============================================================================
# QUANTILES
# =============================================================================


def quantile_table(
    name: str,
    values: np.ndarray,
    metric_name: str,
):

    qs = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    rows = []

    for q in qs:
        rows.append(
            {
                "sample": name,
                "metric": metric_name,
                "quantile": q,
                "value": float(
                    np.quantile(
                        values,
                        q,
                    )
                ),
            }
        )

    return rows


# =============================================================================
# WORST CASE SEQUENCES
# =============================================================================


def worst_sequence_losses(
    values: np.ndarray,
):

    rows = []

    for length in WORST_SEQUENCE_LENGTHS:
        if len(values) < length:
            continue

        rolling = pd.Series(values).rolling(length).sum()

        worst = float(rolling.min())

        best = float(rolling.max())

        rows.append(
            {
                "sequence_length": length,
                "worst_R": worst,
                "best_R": best,
            }
        )

    return rows


# =============================================================================
# RUN ROBUSTNESS FOR ONE SAMPLE
# =============================================================================


def run_sample(
    name: str,
    values: np.ndarray,
    rng: np.random.Generator,
):

    print()
    print("=" * 110)
    print(f"ROBUSTNESS SAMPLE: {name}")
    print("=" * 110)

    base = calculate_metrics(values)

    print()
    print("Observed metrics:")

    for key, value in base.items():
        print(f"  {key:20s}: {value}")

    # -------------------------------------------------------------------------
    # Monte Carlo
    # -------------------------------------------------------------------------

    print()
    print(f"Running Monte Carlo: {N_SIMULATIONS:,} simulations")

    (
        mc_totals,
        mc_dds,
        mc_terminal,
    ) = monte_carlo(
        values,
        rng,
    )

    # -------------------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------------------

    print(f"Running Bootstrap: {N_SIMULATIONS:,} simulations")

    (
        boot_totals,
        boot_means,
        boot_wr,
        boot_dds,
    ) = bootstrap(
        values,
        rng,
    )

    # -------------------------------------------------------------------------
    # Probabilities
    # -------------------------------------------------------------------------

    probability_negative = float((mc_totals <= NEGATIVE_R_THRESHOLD).mean())

    probability_negative_bootstrap = float((boot_totals <= NEGATIVE_R_THRESHOLD).mean())

    rows = {
        "sample": name,
        "trades": base["trades"],
        "observed_total_R": base["total_R"],
        "observed_mean_R": base["mean_R"],
        "observed_win_rate": base["win_rate"],
        "observed_profit_factor": base["profit_factor"],
        "observed_max_drawdown_R": base["max_drawdown_R"],
        "mc_negative_probability": probability_negative,
        "bootstrap_negative_probability": probability_negative_bootstrap,
        "mc_total_R_p05": np.quantile(
            mc_totals,
            0.05,
        ),
        "mc_total_R_p50": np.quantile(
            mc_totals,
            0.50,
        ),
        "mc_total_R_p95": np.quantile(
            mc_totals,
            0.95,
        ),
        "mc_max_DD_p05": np.quantile(
            mc_dds,
            0.05,
        ),
        "mc_max_DD_p50": np.quantile(
            mc_dds,
            0.50,
        ),
        "mc_max_DD_p95": np.quantile(
            mc_dds,
            0.95,
        ),
        "bootstrap_total_R_p05": np.quantile(
            boot_totals,
            0.05,
        ),
        "bootstrap_total_R_p50": np.quantile(
            boot_totals,
            0.50,
        ),
        "bootstrap_total_R_p95": np.quantile(
            boot_totals,
            0.95,
        ),
        "bootstrap_mean_R_p05": np.quantile(
            boot_means,
            0.05,
        ),
        "bootstrap_mean_R_p50": np.quantile(
            boot_means,
            0.50,
        ),
        "bootstrap_mean_R_p95": np.quantile(
            boot_means,
            0.95,
        ),
        "bootstrap_win_rate_p05": np.quantile(
            boot_wr,
            0.05,
        ),
        "bootstrap_win_rate_p50": np.quantile(
            boot_wr,
            0.50,
        ),
        "bootstrap_win_rate_p95": np.quantile(
            boot_wr,
            0.95,
        ),
        "bootstrap_max_DD_p05": np.quantile(
            boot_dds,
            0.05,
        ),
        "bootstrap_max_DD_p50": np.quantile(
            boot_dds,
            0.50,
        ),
        "bootstrap_max_DD_p95": np.quantile(
            boot_dds,
            0.95,
        ),
    }

    # -------------------------------------------------------------------------
    # DD probabilities
    # -------------------------------------------------------------------------

    for threshold in DD_THRESHOLDS:
        key = f"mc_probability_DD_le_{abs(threshold):g}R"

        rows[key] = float((mc_dds <= threshold).mean())

    # -------------------------------------------------------------------------
    # Worst sequences
    # -------------------------------------------------------------------------

    worst_rows = []

    for row in worst_sequence_losses(values):
        row["sample"] = name

        worst_rows.append(row)

    return (
        rows,
        mc_totals,
        mc_dds,
        boot_totals,
        boot_means,
        boot_wr,
        boot_dds,
        worst_rows,
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S28 STRATEGY ROBUSTNESS")
    print("=" * 110)

    print()
    print("Frozen S27 strategy.")

    print("NO PARAMETER OPTIMIZATION.")

    print()
    print(f"Monte Carlo simulations: {N_SIMULATIONS:,}")

    print(f"Random seed: {SEED}")

    df = load_data()

    rng = np.random.default_rng(SEED)

    # =========================================================================
    # SAMPLES
    # =========================================================================

    full_values = df["_strategy_R"].to_numpy(dtype=float)

    oos_df = df[
        df["_window_numeric"].between(
            12,
            22,
        )
    ]

    oos_values = oos_df["_strategy_R"].to_numpy(dtype=float)

    development_df = df[
        df["_window_numeric"].between(
            1,
            11,
        )
    ]

    development_values = development_df["_strategy_R"].to_numpy(dtype=float)

    # =========================================================================
    # RUN
    # =========================================================================

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

    summary_rows = []
    all_quantiles = []
    all_worst = []

    mc_result_rows = []
    bootstrap_result_rows = []

    for name, values in samples:
        (
            summary,
            mc_totals,
            mc_dds,
            boot_totals,
            boot_means,
            boot_wr,
            boot_dds,
            worst_rows,
        ) = run_sample(
            name,
            values,
            rng,
        )

        summary_rows.append(summary)

        all_worst.extend(worst_rows)

        # ---------------------------------------------------------------------
        # Distribution quantiles
        # ---------------------------------------------------------------------

        all_quantiles.extend(
            quantile_table(
                name,
                mc_totals,
                "MC_TOTAL_R",
            )
        )

        all_quantiles.extend(
            quantile_table(
                name,
                mc_dds,
                "MC_MAX_DD_R",
            )
        )

        all_quantiles.extend(
            quantile_table(
                name,
                boot_totals,
                "BOOTSTRAP_TOTAL_R",
            )
        )

        all_quantiles.extend(
            quantile_table(
                name,
                boot_means,
                "BOOTSTRAP_MEAN_R",
            )
        )

        all_quantiles.extend(
            quantile_table(
                name,
                boot_wr,
                "BOOTSTRAP_WIN_RATE",
            )
        )

        all_quantiles.extend(
            quantile_table(
                name,
                boot_dds,
                "BOOTSTRAP_MAX_DD_R",
            )
        )

        # ---------------------------------------------------------------------
        # Raw simulation results
        # ---------------------------------------------------------------------

        for i in range(N_SIMULATIONS):
            mc_result_rows.append(
                {
                    "sample": name,
                    "simulation": i + 1,
                    "total_R": mc_totals[i],
                    "max_drawdown_R": mc_dds[i],
                }
            )

            bootstrap_result_rows.append(
                {
                    "sample": name,
                    "simulation": i + 1,
                    "total_R": boot_totals[i],
                    "mean_R": boot_means[i],
                    "win_rate": boot_wr[i],
                    "max_drawdown_R": boot_dds[i],
                }
            )

    # =========================================================================
    # SAVE
    # =========================================================================

    summary_df = pd.DataFrame(summary_rows)

    quantiles_df = pd.DataFrame(all_quantiles)

    mc_df = pd.DataFrame(mc_result_rows)

    bootstrap_df = pd.DataFrame(bootstrap_result_rows)

    worst_df = pd.DataFrame(all_worst)

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    quantiles_df.to_csv(
        QUANTILES_FILE,
        index=False,
    )

    mc_df.to_csv(
        MONTE_CARLO_FILE,
        index=False,
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_FILE,
        index=False,
    )

    worst_df.to_csv(
        OOS_FILE,
        index=False,
    )

    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print("S28 ROBUSTNESS SUMMARY")
    print("=" * 110)

    print(summary_df.to_string(index=False))

    print()
    print("=" * 110)
    print("WORST OBSERVED SEQUENCES")
    print("=" * 110)

    print(worst_df.to_string(index=False))

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(SUMMARY_FILE)
    print(MONTE_CARLO_FILE)
    print(BOOTSTRAP_FILE)
    print(QUANTILES_FILE)
    print(OOS_FILE)

    print()
    print("=" * 110)
    print("S28 ROBUSTNESS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
