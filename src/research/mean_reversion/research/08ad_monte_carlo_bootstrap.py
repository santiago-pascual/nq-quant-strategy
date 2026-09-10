from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

TRADES_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

OUTPUT_FILE = RESULTS_DIR / "research_08ad_monte_carlo_summary.csv"

BLOCK_OUTPUT_FILE = RESULTS_DIR / "research_08ad_block_bootstrap_summary.csv"


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATES = (
    "MRS2_NEW",
    "MRL1_NEW",
)

N_SIMULATIONS = 20_000

# Block lengths in trades.
# Multiple lengths prevent the result from depending on
# a single arbitrary temporal block size.
BLOCK_LENGTHS = (
    5,
    10,
    20,
    40,
)

RNG_SEED = 20260909


# ============================================================
# METRICS
# ============================================================


def max_drawdown(r: np.ndarray) -> float:

    if len(r) == 0:
        return np.nan

    equity = np.cumsum(r)

    running_max = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]

    drawdown = equity - running_max

    return float(drawdown.min())


def simulate_iid(
    r: np.ndarray,
    rng: np.random.Generator,
    n_simulations: int,
) -> dict:

    n = len(r)

    if n == 0:
        return {}

    # --------------------------------------------------------
    # IID bootstrap
    # --------------------------------------------------------

    samples = rng.choice(
        r,
        size=(n_simulations, n),
        replace=True,
    )

    terminal = samples.sum(axis=1)

    expectancy = samples.mean(axis=1)

    # --------------------------------------------------------
    # Vectorized maximum drawdown
    # --------------------------------------------------------

    equity = np.cumsum(
        samples,
        axis=1,
    )

    running_max = np.maximum.accumulate(
        np.concatenate(
            [
                np.zeros((n_simulations, 1)),
                equity,
            ],
            axis=1,
        ),
        axis=1,
    )[:, 1:]

    drawdown = equity - running_max

    max_dd = drawdown.min(axis=1)

    return {
        "terminal": terminal,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
    }


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================


def generate_block_sample(
    r: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:

    n = len(r)

    if n == 0:
        return np.array([], dtype=float)

    blocks = []

    while sum(len(block) for block in blocks) < n:
        start = int(
            rng.integers(
                0,
                max(1, n - block_length + 1),
            )
        )

        block = r[
            start : min(
                start + block_length,
                n,
            )
        ]

        blocks.append(block)

    sample = np.concatenate(blocks)[:n]

    return sample


def simulate_block_bootstrap(
    r: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
    n_simulations: int,
) -> dict:

    n = len(r)

    terminal = np.empty(
        n_simulations,
        dtype=float,
    )

    expectancy = np.empty(
        n_simulations,
        dtype=float,
    )

    max_dd = np.empty(
        n_simulations,
        dtype=float,
    )

    for i in range(n_simulations):
        sample = generate_block_sample(
            r,
            block_length,
            rng,
        )

        terminal[i] = sample.sum()

        expectancy[i] = sample.mean()

        max_dd[i] = max_drawdown(sample)

    return {
        "terminal": terminal,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
    }


# ============================================================
# DISTRIBUTION SUMMARY
# ============================================================


def summarize_distribution(
    values: np.ndarray,
) -> dict:

    return {
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08AD — MONTE CARLO + BOOTSTRAP + BLOCK BOOTSTRAP")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    trades = pd.read_csv(TRADES_FILE)

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
    )

    print(f"\nTrades loaded: {len(trades):,}")

    rng = np.random.default_rng(RNG_SEED)

    summary_rows = []

    # ========================================================
    # IID MONTE CARLO
    # ========================================================

    print("\n" + "=" * 72)
    print("IID BOOTSTRAP / MONTE CARLO")
    print("=" * 72)

    for candidate in CANDIDATES:
        data = trades[trades["candidate_id"] == candidate].copy()

        data = data.sort_values("timestamp")

        r = data["r"].to_numpy(dtype=float)

        print(f"\n{candidate}: N={len(r):,}")

        simulation = simulate_iid(
            r,
            rng,
            N_SIMULATIONS,
        )

        terminal = simulation["terminal"]

        expectancy = simulation["expectancy"]

        max_dd = simulation["max_drawdown"]

        terminal_summary = summarize_distribution(terminal)

        expectancy_summary = summarize_distribution(expectancy)

        dd_summary = summarize_distribution(max_dd)

        probability_negative = float(np.mean(terminal < 0))

        probability_expectancy_negative = float(np.mean(expectancy < 0))

        row = {
            "candidate_id": candidate,
            "n_trades": len(r),
            "simulations": N_SIMULATIONS,
            "observed_net_r": float(r.sum()),
            "observed_expectancy_r": float(r.mean()),
            "observed_max_dd_r": max_drawdown(r),
            "terminal_p01": terminal_summary["p01"],
            "terminal_p05": terminal_summary["p05"],
            "terminal_p25": terminal_summary["p25"],
            "terminal_p50": terminal_summary["p50"],
            "terminal_p75": terminal_summary["p75"],
            "terminal_p95": terminal_summary["p95"],
            "terminal_p99": terminal_summary["p99"],
            "expectancy_p01": expectancy_summary["p01"],
            "expectancy_p05": expectancy_summary["p05"],
            "expectancy_p25": expectancy_summary["p25"],
            "expectancy_p50": expectancy_summary["p50"],
            "expectancy_p75": expectancy_summary["p75"],
            "expectancy_p95": expectancy_summary["p95"],
            "expectancy_p99": expectancy_summary["p99"],
            "max_dd_p01": dd_summary["p01"],
            "max_dd_p05": dd_summary["p05"],
            "max_dd_p25": dd_summary["p25"],
            "max_dd_p50": dd_summary["p50"],
            "max_dd_p75": dd_summary["p75"],
            "max_dd_p95": dd_summary["p95"],
            "max_dd_p99": dd_summary["p99"],
            "prob_terminal_negative": probability_negative,
            "prob_expectancy_negative": (probability_expectancy_negative),
        }

        summary_rows.append(row)

        print(
            f"  Terminal P05/P50/P95: "
            f"{terminal_summary['p05']:.2f}R / "
            f"{terminal_summary['p50']:.2f}R / "
            f"{terminal_summary['p95']:.2f}R"
        )

        print(
            f"  Expectancy P05/P50/P95: "
            f"{expectancy_summary['p05']:.6f} / "
            f"{expectancy_summary['p50']:.6f} / "
            f"{expectancy_summary['p95']:.6f}"
        )

        print(
            f"  Max DD P05/P50/P95: "
            f"{dd_summary['p05']:.2f}R / "
            f"{dd_summary['p50']:.2f}R / "
            f"{dd_summary['p95']:.2f}R"
        )

        print(f"  P(terminal < 0): {probability_negative:.4%}")

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # BLOCK BOOTSTRAP
    # ========================================================

    print("\n" + "=" * 72)
    print("TIME-AWARE BLOCK BOOTSTRAP")
    print("=" * 72)

    block_rows = []

    for candidate in CANDIDATES:
        data = trades[trades["candidate_id"] == candidate].copy()

        data = data.sort_values("timestamp")

        r = data["r"].to_numpy(dtype=float)

        print(f"\n{candidate}")

        for block_length in BLOCK_LENGTHS:
            print(f"  Block length: {block_length}")

            simulation = simulate_block_bootstrap(
                r,
                block_length,
                rng,
                N_SIMULATIONS,
            )

            terminal = simulation["terminal"]

            expectancy = simulation["expectancy"]

            max_dd = simulation["max_drawdown"]

            terminal_summary = summarize_distribution(terminal)

            expectancy_summary = summarize_distribution(expectancy)

            dd_summary = summarize_distribution(max_dd)

            probability_negative = float(np.mean(terminal < 0))

            probability_expectancy_negative = float(np.mean(expectancy < 0))

            row = {
                "candidate_id": candidate,
                "block_length": block_length,
                "n_trades": len(r),
                "simulations": N_SIMULATIONS,
                "observed_net_r": float(r.sum()),
                "observed_expectancy_r": float(r.mean()),
                "observed_max_dd_r": max_drawdown(r),
                "terminal_p01": terminal_summary["p01"],
                "terminal_p05": terminal_summary["p05"],
                "terminal_p25": terminal_summary["p25"],
                "terminal_p50": terminal_summary["p50"],
                "terminal_p75": terminal_summary["p75"],
                "terminal_p95": terminal_summary["p95"],
                "terminal_p99": terminal_summary["p99"],
                "expectancy_p01": expectancy_summary["p01"],
                "expectancy_p05": expectancy_summary["p05"],
                "expectancy_p25": expectancy_summary["p25"],
                "expectancy_p50": expectancy_summary["p50"],
                "expectancy_p75": expectancy_summary["p75"],
                "expectancy_p95": expectancy_summary["p95"],
                "expectancy_p99": expectancy_summary["p99"],
                "max_dd_p01": dd_summary["p01"],
                "max_dd_p05": dd_summary["p05"],
                "max_dd_p25": dd_summary["p25"],
                "max_dd_p50": dd_summary["p50"],
                "max_dd_p75": dd_summary["p75"],
                "max_dd_p95": dd_summary["p95"],
                "max_dd_p99": dd_summary["p99"],
                "prob_terminal_negative": probability_negative,
                "prob_expectancy_negative": (probability_expectancy_negative),
            }

            block_rows.append(row)

            print(
                f"    Terminal P05/P50/P95: "
                f"{terminal_summary['p05']:.2f} / "
                f"{terminal_summary['p50']:.2f} / "
                f"{terminal_summary['p95']:.2f}R"
            )

            print(f"    P(terminal < 0): {probability_negative:.4%}")

    block_summary = pd.DataFrame(block_rows)

    block_summary.to_csv(
        BLOCK_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 72)
    print("08AD COMPLETE")
    print("=" * 72)

    print(f"IID results: {OUTPUT_FILE}")

    print(f"Block results: {BLOCK_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
