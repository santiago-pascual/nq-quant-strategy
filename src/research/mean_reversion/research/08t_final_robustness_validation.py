from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# RESEARCH 08T
# FINAL ROBUSTNESS VALIDATION
# =============================================================================
#
# Frozen candidates
#
# MRS2 = SHORT | HMM 2 | VOL 80-100 | Z 2.0 | TP 5 | SL 2 | H 5
# MRL1 = LONG  | HMM 1 | VOL 20-40  | Z 2.5 | TP 5 | SL 2 | H 20
# MRL2 = LONG  | HMM 2 | VOL 60-80  | Z 3.5 | TP 5 | SL 2 | H 2
#
# NO DISCOVERY
# NO PARAMETER OPTIMIZATION
# NO HMM RETRAINING
# NO VOLATILITY OPTIMIZATION
# NO FAILURE FILTER
#
# PURPOSE:
#
#   1. Validate all 22 temporal windows.
#   2. Produce full-history equity curves.
#   3. Produce OOS equity curves from the frozen final OOS period.
#   4. Perform one-window-out temporal validation.
#   5. Perform bootstrap Monte Carlo.
#   6. Perform trade-order Monte Carlo.
#   7. Estimate confidence intervals.
#   8. Study drawdown uncertainty.
#   9. Study parameter neighborhood robustness WITHOUT selecting a new optimum.
#
# IMPORTANT:
#
#   This script is an AUDIT.
#   It does not create a new strategy.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

TRADES_PATH = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

OOS_TRADES_PATH = RESULTS_DIR / "research_08o_oos_trades.csv"

OUT_DIR = RESULTS_DIR / "research_08t_final_validation"

PLOTS_DIR = OUT_DIR / "plots"

TABLES_DIR = OUT_DIR / "tables"

REPORT_PATH = OUT_DIR / "research_08t_final_report.txt"


# =============================================================================
# CANDIDATES
# =============================================================================

CANDIDATES = {
    "MRS2": {
        "candidate_id": "C01",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 5,
    },
    "MRL1": {
        "candidate_id": "C02",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 20,
    },
    "MRL2": {
        "candidate_id": "C06",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 2,
    },
}


# =============================================================================
# CONFIG
# =============================================================================

RANDOM_SEED = 20260901

MC_ITERATIONS = 20_000

BOOTSTRAP_ITERATIONS = 20_000

MIN_TRADES_FOR_STATISTICS = 30

OOS_WINDOWS = 4

ALL_WINDOWS = 22


# =============================================================================
# HELPERS
# =============================================================================


def section(title):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_mean(x):

    x = pd.Series(x).dropna()

    return float(x.mean()) if len(x) else np.nan


def safe_std(x):

    x = pd.Series(x).dropna()

    return float(x.std(ddof=1)) if len(x) > 1 else np.nan


def safe_median(x):

    x = pd.Series(x).dropna()

    return float(x.median()) if len(x) else np.nan


def percentile(x, q):

    x = pd.Series(x).dropna()

    return float(np.percentile(x, q)) if len(x) else np.nan


# =============================================================================
# LOAD TRADES
# =============================================================================


def load_trades():

    section("LOADING FULL-HISTORY TRADES")

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{TRADES_PATH}\n\nRun Research 08P first.")

    trades = pd.read_csv(TRADES_PATH)

    required = [
        "strategy_name",
        "candidate_id",
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "tp",
        "sl",
        "rr",
        "horizon",
        "event_id",
        "window",
        "timestamp",
        "result",
        "r",
    ]

    missing = [c for c in required if c not in trades.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
        errors="coerce",
    )

    trades["window"] = pd.to_numeric(
        trades["window"],
        errors="coerce",
    )

    trades["r"] = pd.to_numeric(
        trades["r"],
        errors="coerce",
    )

    trades["hmm_state"] = pd.to_numeric(
        trades["hmm_state"],
        errors="coerce",
    )

    trades["zscore"] = pd.to_numeric(
        trades["zscore"],
        errors="coerce",
    )

    trades = trades[trades["strategy_name"].isin(CANDIDATES.keys())].copy()

    trades = trades.sort_values("timestamp").reset_index(drop=True)

    print(f"Trades: {len(trades):,}")

    return trades


# =============================================================================
# LOAD INDEPENDENT OOS
# =============================================================================


def load_oos_trades():

    section("LOADING INDEPENDENT OOS TRADES")

    if not OOS_TRADES_PATH.exists():
        print("Independent OOS trade file not found.")

        return pd.DataFrame()

    oos = pd.read_csv(OOS_TRADES_PATH)

    if "timestamp" in oos.columns:
        oos["timestamp"] = pd.to_datetime(
            oos["timestamp"],
            utc=True,
            errors="coerce",
        )

    if "r" in oos.columns:
        oos["r"] = pd.to_numeric(
            oos["r"],
            errors="coerce",
        )

    if "window" in oos.columns:
        oos["window"] = pd.to_numeric(
            oos["window"],
            errors="coerce",
        )

    oos = oos[oos["strategy_name"].isin(CANDIDATES.keys())].copy()

    print(f"OOS trades: {len(oos):,}")

    return oos


# =============================================================================
# BASIC METRICS
# =============================================================================


def calculate_metrics(group):

    group = group.copy()

    r = pd.to_numeric(
        group["r"],
        errors="coerce",
    ).dropna()

    if len(r) == 0:
        return {
            "observations": 0,
            "wins": 0,
            "losses": 0,
            "wr": np.nan,
            "net_r": np.nan,
            "expectancy": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": np.nan,
            "sharpe_trade": np.nan,
        }

    wins = r[r > 0]
    losses = r[r < 0]

    equity = r.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_dd = float(drawdown.min())

    gross_profit = wins.sum() if len(wins) else 0.0

    gross_loss = abs(losses.sum()) if len(losses) else 0.0

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    mean_r = float(r.mean())

    std_r = float(r.std(ddof=1)) if len(r) > 1 else np.nan

    sharpe = (
        mean_r / std_r * np.sqrt(len(r))
        if (np.isfinite(std_r) and std_r > 0)
        else np.nan
    )

    return {
        "observations": len(r),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "wr": float((r > 0).mean()),
        "net_r": float(r.sum()),
        "expectancy": mean_r,
        "profit_factor": pf,
        "max_drawdown_r": max_dd,
        "sharpe_trade": sharpe,
    }


# =============================================================================
# WINDOW ANALYSIS
# =============================================================================


def build_window_table(trades):

    section("22-WINDOW TEMPORAL VALIDATION")

    rows = []

    for strategy in CANDIDATES:
        strategy_trades = trades[trades["strategy_name"] == strategy]

        for window in sorted(strategy_trades["window"].dropna().unique()):
            group = strategy_trades[strategy_trades["window"] == window]

            metrics = calculate_metrics(group)

            rows.append(
                {
                    "strategy_name": strategy,
                    "window": int(window),
                    **metrics,
                }
            )

    result = pd.DataFrame(rows)

    return result


# =============================================================================
# EQUITY CURVE
# =============================================================================


def build_equity(trades):

    work = trades.copy()

    work = work.sort_values(
        [
            "strategy_name",
            "timestamp",
        ]
    )

    work["equity_r"] = work.groupby("strategy_name")["r"].cumsum()

    work["running_max"] = work.groupby("strategy_name")["equity_r"].cummax()

    work["drawdown_r"] = work["equity_r"] - work["running_max"]

    return work


# =============================================================================
# PLOT EQUITY
# =============================================================================


def plot_equity(
    trades,
    filename,
    title,
):

    plt.figure(figsize=(14, 7))

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy]

        if group.empty:
            continue

        group = group.sort_values("timestamp")

        equity = group["r"].cumsum()

        plt.plot(
            group["timestamp"],
            equity,
            label=strategy,
            linewidth=1.4,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title(title)

    plt.xlabel("Time")

    plt.ylabel("Cumulative R")

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=220,
    )

    plt.close()


# =============================================================================
# PLOT INDIVIDUAL EQUITY
# =============================================================================


def plot_individual_equity(trades):

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy].sort_values("timestamp")

        if group.empty:
            continue

        equity = group["r"].cumsum()

        plt.figure(figsize=(14, 7))

        plt.plot(
            equity.values,
            linewidth=1.3,
        )

        plt.axhline(
            0,
            linewidth=0.8,
        )

        plt.title(f"{strategy} — Full History Equity Curve")

        plt.xlabel("Trade")

        plt.ylabel("Cumulative R")

        plt.grid(alpha=0.25)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_full_equity.png",
            dpi=220,
        )

        plt.close()


# =============================================================================
# DRAW DOWN
# =============================================================================


def plot_drawdown(trades):

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy].sort_values("timestamp")

        if group.empty:
            continue

        equity = group["r"].cumsum()

        running_max = equity.cummax()

        drawdown = equity - running_max

        plt.figure(figsize=(14, 6))

        plt.plot(
            drawdown.values,
            linewidth=1.2,
        )

        plt.axhline(
            0,
            linewidth=0.8,
        )

        plt.title(f"{strategy} — Drawdown")

        plt.xlabel("Trade")

        plt.ylabel("Drawdown (R)")

        plt.grid(alpha=0.25)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_drawdown.png",
            dpi=220,
        )

        plt.close()


# =============================================================================
# RETURN DISTRIBUTION
# =============================================================================


def plot_return_distribution(trades):

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy]

        r = group["r"].dropna()

        if len(r) == 0:
            continue

        plt.figure(figsize=(12, 7))

        plt.hist(
            r,
            bins=30,
            alpha=0.75,
        )

        plt.axvline(
            0,
            linewidth=1.0,
        )

        plt.title(f"{strategy} — Trade R Distribution")

        plt.xlabel("R")

        plt.ylabel("Frequency")

        plt.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_r_distribution.png",
            dpi=220,
        )

        plt.close()


# =============================================================================
# ROLLING PERFORMANCE
# =============================================================================


def plot_rolling_performance(
    trades,
    window=100,
):

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy].sort_values("timestamp")

        if len(group) < window:
            continue

        r = group["r"]

        rolling_wr = r.gt(0).rolling(window).mean()

        rolling_exp = r.rolling(window).mean()

        fig = plt.figure(figsize=(14, 7))

        ax = fig.add_subplot(111)

        ax.plot(
            rolling_wr.values,
            label="Rolling WR",
        )

        ax.plot(
            rolling_exp.values,
            label="Rolling Expectancy (R)",
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.axhline(
            0.50,
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_title(f"{strategy} — Rolling {window}-Trade Performance")

        ax.set_xlabel("Trade")

        ax.legend()

        ax.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_rolling_performance.png",
            dpi=220,
        )

        plt.close()


# =============================================================================
# MONTE CARLO — TRADE ORDER
# =============================================================================


def monte_carlo_trade_order(
    r,
    iterations=MC_ITERATIONS,
    seed=RANDOM_SEED,
):

    r = np.asarray(
        r,
        dtype=float,
    )

    r = r[np.isfinite(r)]

    if len(r) < MIN_TRADES_FOR_STATISTICS:
        return None

    rng = np.random.default_rng(seed)

    max_dd = np.empty(iterations)

    terminal = np.empty(iterations)

    min_equity = np.empty(iterations)

    for i in range(iterations):
        shuffled = rng.permutation(r)

        equity = np.cumsum(shuffled)

        running_max = np.maximum.accumulate(equity)

        dd = equity - running_max

        max_dd[i] = dd.min()

        terminal[i] = equity[-1]

        min_equity[i] = equity.min()

    return {
        "max_drawdown": max_dd,
        "terminal": terminal,
        "min_equity": min_equity,
    }


# =============================================================================
# MONTE CARLO — BOOTSTRAP
# =============================================================================


def monte_carlo_bootstrap(
    r,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=RANDOM_SEED + 1,
):

    r = np.asarray(
        r,
        dtype=float,
    )

    r = r[np.isfinite(r)]

    if len(r) < MIN_TRADES_FOR_STATISTICS:
        return None

    rng = np.random.default_rng(seed)

    n = len(r)

    terminal = np.empty(iterations)

    expectancy = np.empty(iterations)

    wr = np.empty(iterations)

    max_dd = np.empty(iterations)

    for i in range(iterations):
        sample = rng.choice(
            r,
            size=n,
            replace=True,
        )

        equity = np.cumsum(sample)

        running_max = np.maximum.accumulate(equity)

        dd = equity - running_max

        terminal[i] = equity[-1]

        expectancy[i] = sample.mean()

        wr[i] = (sample > 0).mean()

        max_dd[i] = dd.min()

    return {
        "terminal": terminal,
        "expectancy": expectancy,
        "wr": wr,
        "max_drawdown": max_dd,
    }


# =============================================================================
# MONTE CARLO SUMMARY
# =============================================================================


def summarize_monte_carlo(
    strategy,
    mc_order,
    mc_bootstrap,
):

    rows = []

    if mc_order is not None:
        dd = mc_order["max_drawdown"]

        terminal = mc_order["terminal"]

        rows.append(
            {
                "strategy_name": strategy,
                "method": "trade_order",
                "iterations": len(dd),
                "terminal_p05": percentile(
                    terminal,
                    5,
                ),
                "terminal_p50": percentile(
                    terminal,
                    50,
                ),
                "terminal_p95": percentile(
                    terminal,
                    95,
                ),
                "max_dd_p05": percentile(
                    dd,
                    5,
                ),
                "max_dd_p50": percentile(
                    dd,
                    50,
                ),
                "max_dd_p95": percentile(
                    dd,
                    95,
                ),
                "prob_terminal_negative": float((terminal < 0).mean()),
            }
        )

    if mc_bootstrap is not None:
        terminal = mc_bootstrap["terminal"]

        expectancy = mc_bootstrap["expectancy"]

        wr = mc_bootstrap["wr"]

        dd = mc_bootstrap["max_drawdown"]

        rows.append(
            {
                "strategy_name": strategy,
                "method": "bootstrap",
                "iterations": len(terminal),
                "terminal_p05": percentile(
                    terminal,
                    5,
                ),
                "terminal_p50": percentile(
                    terminal,
                    50,
                ),
                "terminal_p95": percentile(
                    terminal,
                    95,
                ),
                "max_dd_p05": percentile(
                    dd,
                    5,
                ),
                "max_dd_p50": percentile(
                    dd,
                    50,
                ),
                "max_dd_p95": percentile(
                    dd,
                    95,
                ),
                "prob_terminal_negative": float((terminal < 0).mean()),
                "expectancy_p05": percentile(
                    expectancy,
                    5,
                ),
                "expectancy_p50": percentile(
                    expectancy,
                    50,
                ),
                "expectancy_p95": percentile(
                    expectancy,
                    95,
                ),
                "wr_p05": percentile(
                    wr,
                    5,
                ),
                "wr_p50": percentile(
                    wr,
                    50,
                ),
                "wr_p95": percentile(
                    wr,
                    95,
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MONTE CARLO PLOTS
# =============================================================================


def plot_monte_carlo(
    strategy,
    mc_order,
    mc_bootstrap,
):

    if mc_order is not None:
        terminal = mc_order["terminal"]

        dd = mc_order["max_drawdown"]

        plt.figure(figsize=(12, 7))

        plt.hist(
            terminal,
            bins=60,
            alpha=0.75,
        )

        plt.axvline(
            np.median(terminal),
            linewidth=1.2,
            linestyle="--",
        )

        plt.axvline(
            0,
            linewidth=1.0,
        )

        plt.title(f"{strategy} — Monte Carlo Trade-Order Terminal Equity")

        plt.xlabel("Terminal Cumulative R")

        plt.ylabel("Frequency")

        plt.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_mc_terminal_equity.png",
            dpi=220,
        )

        plt.close()

        plt.figure(figsize=(12, 7))

        plt.hist(
            dd,
            bins=60,
            alpha=0.75,
        )

        plt.axvline(
            np.median(dd),
            linewidth=1.2,
            linestyle="--",
        )

        plt.title(f"{strategy} — Monte Carlo Maximum Drawdown")

        plt.xlabel("Maximum Drawdown (R)")

        plt.ylabel("Frequency")

        plt.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_mc_drawdown.png",
            dpi=220,
        )

        plt.close()

    if mc_bootstrap is not None:
        expectancy = mc_bootstrap["expectancy"]

        wr = mc_bootstrap["wr"]

        plt.figure(figsize=(12, 7))

        plt.hist(
            expectancy,
            bins=60,
            alpha=0.75,
        )

        plt.axvline(
            0,
            linewidth=1.0,
        )

        plt.axvline(
            np.median(expectancy),
            linewidth=1.2,
            linestyle="--",
        )

        plt.title(f"{strategy} — Bootstrap Expectancy Distribution")

        plt.xlabel("Expectancy (R)")

        plt.ylabel("Frequency")

        plt.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_bootstrap_expectancy.png",
            dpi=220,
        )

        plt.close()

        plt.figure(figsize=(12, 7))

        plt.hist(
            wr,
            bins=60,
            alpha=0.75,
        )

        plt.axvline(
            0.50,
            linewidth=1.0,
            linestyle="--",
        )

        plt.axvline(
            np.median(wr),
            linewidth=1.2,
        )

        plt.title(f"{strategy} — Bootstrap Win Rate Distribution")

        plt.xlabel("Win Rate")

        plt.ylabel("Frequency")

        plt.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_bootstrap_wr.png",
            dpi=220,
        )

        plt.close()


# =============================================================================
# ONE WINDOW OUT
# =============================================================================


def one_window_out(trades):

    section("ONE-WINDOW-OUT TEMPORAL VALIDATION")

    rows = []

    windows = sorted(trades["window"].dropna().unique())

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy].copy()

        for held_out in windows:
            test = group[group["window"] == held_out]

            if test.empty:
                continue

            train = group[group["window"] != held_out]

            train_metrics = calculate_metrics(train)

            test_metrics = calculate_metrics(test)

            rows.append(
                {
                    "strategy_name": strategy,
                    "held_out_window": int(held_out),
                    "train_observations": train_metrics["observations"],
                    "train_wr": train_metrics["wr"],
                    "train_expectancy": train_metrics["expectancy"],
                    "train_pf": train_metrics["profit_factor"],
                    "test_observations": test_metrics["observations"],
                    "test_wr": test_metrics["wr"],
                    "test_expectancy": test_metrics["expectancy"],
                    "test_pf": test_metrics["profit_factor"],
                    "test_net_r": test_metrics["net_r"],
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# ONE WINDOW OUT PLOT
# =============================================================================


def plot_one_window_out(data):

    for strategy in CANDIDATES:
        group = data[data["strategy_name"] == strategy]

        if group.empty:
            continue

        plt.figure(figsize=(14, 7))

        plt.plot(
            group["held_out_window"],
            group["test_expectancy"],
            marker="o",
            label="Held-out expectancy",
        )

        plt.axhline(
            0,
            linewidth=1.0,
        )

        plt.title(f"{strategy} — One-Window-Out Expectancy")

        plt.xlabel("Held-out Window")

        plt.ylabel("Test Expectancy (R)")

        plt.grid(alpha=0.20)

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / f"{strategy}_one_window_out.png",
            dpi=220,
        )

        plt.close()


# =============================================================================
# PARAMETER NEIGHBORHOOD
# =============================================================================


def build_parameter_neighborhood():

    section("PARAMETER NEIGHBORHOOD AUDIT")

    #
    # This is NOT a search for a better parameter.
    #
    # We simply ask:
    #
    #   "Does the result collapse immediately if parameters move
    #    slightly away from the frozen point?"
    #
    # The actual historical path is not recomputed here.
    # Instead, this table records the frozen center and explicitly
    # marks the neighborhood for a later path-level audit.
    #

    rows = []

    for strategy, config in CANDIDATES.items():
        tp = config["tp"]
        sl = config["sl"]
        horizon = config["horizon"]

        rows.extend(
            [
                {
                    "strategy_name": strategy,
                    "parameter": "TP",
                    "frozen_value": tp,
                    "neighbor": tp - 0.5,
                    "distance": -0.5,
                },
                {
                    "strategy_name": strategy,
                    "parameter": "TP",
                    "frozen_value": tp,
                    "neighbor": tp + 0.5,
                    "distance": +0.5,
                },
                {
                    "strategy_name": strategy,
                    "parameter": "SL",
                    "frozen_value": sl,
                    "neighbor": sl - 0.5,
                    "distance": -0.5,
                },
                {
                    "strategy_name": strategy,
                    "parameter": "SL",
                    "frozen_value": sl,
                    "neighbor": sl + 0.5,
                    "distance": +0.5,
                },
                {
                    "strategy_name": strategy,
                    "parameter": "HORIZON",
                    "frozen_value": horizon,
                    "neighbor": max(
                        1,
                        horizon - 1,
                    ),
                    "distance": -1,
                },
                {
                    "strategy_name": strategy,
                    "parameter": "HORIZON",
                    "frozen_value": horizon,
                    "neighbor": horizon + 1,
                    "distance": +1,
                },
            ]
        )

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY
# =============================================================================


def build_summary(trades):

    rows = []

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy]

        metrics = calculate_metrics(group)

        config = CANDIDATES[strategy]

        rows.append(
            {
                "strategy_name": strategy,
                "side": config["side"],
                "hmm_state": config["hmm_state"],
                "vol_bucket": config["vol_bucket"],
                "zscore": config["zscore"],
                "tp": config["tp"],
                "sl": config["sl"],
                "rr": config["rr"],
                "horizon": config["horizon"],
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================


def write_report(
    summary,
    mc_summary,
    window_table,
    oos,
    one_out,
):

    lines = []

    lines.append("MEAN REVERSION — RESEARCH 08T")

    lines.append("FINAL ROBUSTNESS VALIDATION")

    lines.append("")

    lines.append("All candidates were frozen before this audit.")

    lines.append("No strategy parameters were optimized.")

    lines.append("No HMM retraining was performed.")

    lines.append("No volatility regime optimization was performed.")

    lines.append("")

    lines.append("FROZEN CANDIDATES")

    for name, config in CANDIDATES.items():
        lines.append(
            f"{name}: "
            f"{config['side']} | "
            f"HMM={config['hmm_state']} | "
            f"VOL={config['vol_bucket']} | "
            f"Z={config['zscore']} | "
            f"TP={config['tp']} | "
            f"SL={config['sl']} | "
            f"H={config['horizon']}"
        )

    lines.append("")

    lines.append("FULL HISTORY")

    lines.append(summary.to_string(index=False))

    lines.append("")

    lines.append("MONTE CARLO")

    if not mc_summary.empty:
        lines.append(mc_summary.to_string(index=False))

    lines.append("")

    lines.append("22-WINDOW RESULTS")

    lines.append(window_table.to_string(index=False))

    lines.append("")

    if not one_out.empty:
        lines.append("ONE-WINDOW-OUT")

        lines.append(one_out.to_string(index=False))

    lines.append("")

    if not oos.empty:
        lines.append("INDEPENDENT OOS")

        for strategy in CANDIDATES:
            group = oos[oos["strategy_name"] == strategy]

            if group.empty:
                continue

            metrics = calculate_metrics(group)

            lines.append(
                f"{strategy}: "
                f"N={metrics['observations']} | "
                f"WR={metrics['wr']:.4f} | "
                f"Expectancy={metrics['expectancy']:.4f} | "
                f"PF={metrics['profit_factor']:.4f} | "
                f"NetR={metrics['net_r']:.2f}"
            )

    lines.append("")

    lines.append("IMPORTANT:")

    lines.append("08T is a robustness audit, not a final")

    lines.append("production approval.")

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08T")

    print("FINAL ROBUSTNESS VALIDATION")

    print("-" * 100)

    print("MRS2 / MRL1 / MRL2 are frozen.")

    print("No optimization.")

    print("No filtering.")

    print("No HMM retraining.")

    print("No volatility recalculation.")

    print(f"Monte Carlo iterations: {MC_ITERATIONS:,}")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    trades = load_trades()

    oos = load_oos_trades()

    # -------------------------------------------------------------------------
    # Full history
    # -------------------------------------------------------------------------

    section("FULL-HISTORY SUMMARY")

    summary = build_summary(trades)

    print(summary.to_string(index=False))

    summary.to_csv(
        TABLES_DIR / "research_08t_full_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 22 windows
    # -------------------------------------------------------------------------

    window_table = build_window_table(trades)

    print("\n22-window results:")

    print(window_table.to_string(index=False))

    window_table.to_csv(
        TABLES_DIR / "research_08t_22_window_results.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Equity
    # -------------------------------------------------------------------------

    section("BUILDING EQUITY CURVES")

    equity = build_equity(trades)

    equity.to_csv(
        TABLES_DIR / "research_08t_equity_trades.csv",
        index=False,
    )

    plot_equity(
        trades,
        "all_candidates_full_equity.png",
        "Mean Reversion — Full History Equity Curves",
    )

    plot_individual_equity(trades)

    plot_drawdown(trades)

    plot_return_distribution(trades)

    plot_rolling_performance(trades)

    # -------------------------------------------------------------------------
    # OOS equity
    # -------------------------------------------------------------------------

    if not oos.empty:
        section("INDEPENDENT OOS EQUITY CURVES")

        plot_equity(
            oos,
            "independent_oos_equity.png",
            "Mean Reversion — Independent OOS Equity Curves",
        )

        for strategy in CANDIDATES:
            group = oos[oos["strategy_name"] == strategy].sort_values("timestamp")

            if group.empty:
                continue

            curve = group["r"].cumsum()

            plt.figure(figsize=(14, 7))

            plt.plot(
                curve.values,
                linewidth=1.4,
            )

            plt.axhline(
                0,
                linewidth=0.8,
            )

            plt.title(f"{strategy} — Independent OOS Equity Curve")

            plt.xlabel("OOS Trade")

            plt.ylabel("Cumulative R")

            plt.grid(alpha=0.20)

            plt.tight_layout()

            plt.savefig(
                PLOTS_DIR / f"{strategy}_oos_equity.png",
                dpi=220,
            )

            plt.close()

    # -------------------------------------------------------------------------
    # Monte Carlo
    # -------------------------------------------------------------------------

    section("MONTE CARLO ROBUSTNESS")

    mc_frames = []

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy]

        r = group["r"].dropna()

        print(f"\n{strategy}")

        print(f"Trades: {len(r):,}")

        if len(r) < MIN_TRADES_FOR_STATISTICS:
            print("Insufficient observations.")

            continue

        mc_order = monte_carlo_trade_order(r)

        mc_bootstrap = monte_carlo_bootstrap(r)

        mc_summary = summarize_monte_carlo(
            strategy,
            mc_order,
            mc_bootstrap,
        )

        mc_frames.append(mc_summary)

        plot_monte_carlo(
            strategy,
            mc_order,
            mc_bootstrap,
        )

    if mc_frames:
        mc_summary_all = pd.concat(
            mc_frames,
            ignore_index=True,
        )

    else:
        mc_summary_all = pd.DataFrame()

    mc_summary_all.to_csv(
        TABLES_DIR / "research_08t_monte_carlo_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # One window out
    # -------------------------------------------------------------------------

    one_out = one_window_out(trades)

    one_out.to_csv(
        TABLES_DIR / "research_08t_one_window_out.csv",
        index=False,
    )

    plot_one_window_out(one_out)

    # -------------------------------------------------------------------------
    # Parameter neighborhood
    # -------------------------------------------------------------------------

    neighborhood = build_parameter_neighborhood()

    neighborhood.to_csv(
        TABLES_DIR / "research_08t_parameter_neighborhood.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    write_report(
        summary,
        mc_summary_all,
        window_table,
        oos,
        one_out,
    )

    # -------------------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------------------

    section("RESEARCH 08T COMPLETE")

    print("FINAL ROBUSTNESS AUDIT COMPLETED.")

    print("\nFrozen candidates:")

    print(summary.to_string(index=False))

    print("\nOutput directory:")

    print(OUT_DIR)

    print("\nPublication plots:")

    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {path.name}")

    print("\nTables:")

    for path in sorted(TABLES_DIR.glob("*.csv")):
        print(f"  {path.name}")

    print("\nReport:")

    print(REPORT_PATH)

    print("\nNEXT STEP:")

    print("Review all robustness results.")

    print("If the candidates survive, freeze the final specification.")

    print("Then perform the final untouched OOS evaluation.")

    print("Only after that should the complete-data report be produced.")


if __name__ == "__main__":
    main()
