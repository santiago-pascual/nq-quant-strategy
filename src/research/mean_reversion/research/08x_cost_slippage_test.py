"""
Research 08X
Mean Reversion — Cost & Slippage Test

Purpose
-------
Stress-test the frozen Mean Reversion strategies against:

    1. TopstepX MNQ round-turn commission
    2. Adverse slippage per execution

Frozen strategies:
    - MRL1
    - MRS2

Input
-----
research_08p_full_confirmation_trades.csv

Important
---------
This is a post-trade economic stress test.

The original TP/SL result is NOT changed here.
We only reduce the realized R by:

    commission
    +
    adverse slippage

A later execution-geometry test can model whether slippage
changes TP/SL hit mechanics themselves.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

TRADES_FILE = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08x_cost_slippage_summary.csv"
OUTPUT_TRADES = RESULTS_DIR / "research_08x_trade_level_cost_slippage.csv"


# ============================================================
# MARKET / COST CONSTANTS
# ============================================================

# MNQ
POINT_VALUE_USD = 2.00
TICK_SIZE = 0.25
TICK_VALUE_USD = 0.50

# TopstepX MNQ round-turn cost
COMMISSION_RT_USD = 1.22

# Stress assumptions:
# adverse slippage PER EXECUTION.
#
# Each trade has:
#     entry
#     exit
#
# Therefore:
#     0 ticks -> 0 ticks total
#     1 tick  -> 2 ticks total
#     2 ticks -> 4 ticks total
#     4 ticks -> 8 ticks total
SLIPPAGE_SCENARIOS = [0, 1, 2, 4]


# ============================================================
# STRATEGIES
# ============================================================

FROZEN_STRATEGIES = ["MRL1", "MRS2"]


# ============================================================
# HELPERS
# ============================================================


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "strategy_name",
        "candidate_id",
        "side",
        "tp",
        "sl",
        "rr",
        "event_id",
        "window",
        "timestamp",
        "entry",
        "result",
        "r",
        "bars_to_result",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            "Input file is missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


def load_trades() -> pd.DataFrame:
    section("LOADING FROZEN 08P TRADES")

    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Could not find:\n{TRADES_FILE}")

    df = pd.read_csv(TRADES_FILE)

    print(f"File  : {TRADES_FILE}")
    print(f"Rows  : {len(df):,}")

    validate_input(df)

    # --------------------------------------------------------
    # SHOW IDENTIFIERS BEFORE FILTERING
    # --------------------------------------------------------

    print()
    print("Available strategy identifiers:")

    if "strategy_name" in df.columns:
        print("\nstrategy_name:")
        print(df["strategy_name"].astype(str).value_counts().to_string())

    if "candidate_id" in df.columns:
        print("\ncandidate_id:")
        print(df["candidate_id"].astype(str).value_counts().to_string())

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "tp",
        "sl",
        "rr",
        "event_id",
        "window",
        "entry",
        "r",
        "bars_to_result",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # IDENTIFY FROZEN STRATEGIES
    # --------------------------------------------------------

    # 08P contains both:
    #     strategy_name
    #     candidate_id
    #
    # We do not assume which one contains "MRL1"/"MRS2".
    #
    # Build a combined identifier for robust filtering.

    strategy_name = df["strategy_name"].astype(str).str.strip().str.upper()

    candidate_id = df["candidate_id"].astype(str).str.strip().str.upper()

    frozen_mask = strategy_name.isin(FROZEN_STRATEGIES) | candidate_id.isin(
        FROZEN_STRATEGIES
    )

    df = df[frozen_mask].copy()

    if df.empty:
        raise RuntimeError(
            "No frozen MRL1/MRS2 trades found.\n\n"
            "The file contains the following identifiers:\n"
            f"strategy_name:\n"
            f"{strategy_name.value_counts().to_string()}\n\n"
            f"candidate_id:\n"
            f"{candidate_id.value_counts().to_string()}"
        )

    # --------------------------------------------------------
    # CREATE CANONICAL STRATEGY COLUMN
    # --------------------------------------------------------

    # Prefer candidate_id when it contains the frozen ID.
    # Otherwise use strategy_name.

    df["strategy"] = np.where(
        candidate_id.loc[df.index].isin(FROZEN_STRATEGIES),
        candidate_id.loc[df.index],
        strategy_name.loc[df.index],
    )

    # --------------------------------------------------------
    # REMOVE INVALID R VALUES
    # --------------------------------------------------------

    before = len(df)

    df = df[np.isfinite(df["r"])].copy()

    removed = before - len(df)

    if removed:
        print(f"\nRemoved invalid R rows: {removed:,}")

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    strategies_found = sorted(df["strategy"].unique())

    print()
    print("Frozen strategies found:")

    print(df["strategy"].value_counts().sort_index().to_string())

    print()
    print(f"Usable frozen trades: {len(df):,}")

    missing_strategies = set(FROZEN_STRATEGIES) - set(strategies_found)

    if missing_strategies:
        raise RuntimeError(
            f"Some frozen strategies are missing:\n{sorted(missing_strategies)}"
        )

    return df.reset_index(drop=True)


# ============================================================
# COST MODEL
# ============================================================


def calculate_costs(
    slippage_ticks_per_execution: int,
) -> dict:
    """
    Calculate fixed commission and adverse slippage.

    Each trade has:
        1 entry execution
        1 exit execution

    Slippage is charged on both.
    """

    executions_per_trade = 2

    # Commission:
    # already round-turn, therefore one RT commission per trade.
    commission_usd = COMMISSION_RT_USD

    # Slippage:
    # ticks per execution × 2 executions
    total_slippage_ticks = slippage_ticks_per_execution * executions_per_trade

    total_slippage_points = total_slippage_ticks * TICK_SIZE

    slippage_usd = total_slippage_points * POINT_VALUE_USD

    total_cost_usd = commission_usd + slippage_usd

    # Convert to R.
    #
    # Frozen strategies use:
    #     SL = 2 points
    # therefore:
    #
    #     1R = 2 points = $4 / contract
    #
    # We keep this generic below and convert per trade
    # using its actual SL.
    return {
        "commission_usd": commission_usd,
        "total_slippage_ticks": total_slippage_ticks,
        "total_slippage_points": total_slippage_points,
        "slippage_usd": slippage_usd,
        "total_cost_usd": total_cost_usd,
    }


def apply_cost_model(
    df: pd.DataFrame,
    slippage_ticks_per_execution: int,
) -> pd.DataFrame:
    """
    Apply commission + slippage to each trade.

    Cost is converted into R using the trade's SL distance.

    This is important because R is strategy/trade normalized.
    """

    result = df.copy()

    cost = calculate_costs(slippage_ticks_per_execution)

    # --------------------------------------------------------
    # 1R in dollars for each trade
    # --------------------------------------------------------

    # Example:
    #
    # SL = 2 points
    # MNQ = $2 / point
    #
    # 1R = $4
    #
    result["risk_usd"] = result["sl"].abs() * POINT_VALUE_USD

    # Safety check
    if (result["risk_usd"] <= 0).any():
        raise RuntimeError("Found trades with non-positive SL.")

    # --------------------------------------------------------
    # Commission in R
    # --------------------------------------------------------

    result["commission_r"] = cost["commission_usd"] / result["risk_usd"]

    # --------------------------------------------------------
    # Slippage in R
    # --------------------------------------------------------

    result["slippage_r"] = cost["slippage_usd"] / result["risk_usd"]

    # --------------------------------------------------------
    # Total execution cost
    # --------------------------------------------------------

    result["total_cost_usd"] = cost["total_cost_usd"]

    result["total_cost_r"] = result["commission_r"] + result["slippage_r"]

    # --------------------------------------------------------
    # Net R
    # --------------------------------------------------------

    result["gross_r"] = result["r"]

    result["net_r"] = result["gross_r"] - result["total_cost_r"]

    # Useful diagnostics
    result["slippage_ticks_per_execution"] = slippage_ticks_per_execution

    result["total_slippage_ticks"] = cost["total_slippage_ticks"]

    result["total_slippage_points"] = cost["total_slippage_points"]

    return result


# ============================================================
# PERFORMANCE METRICS
# ============================================================


def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Calculate max drawdown in R.
    """

    running_max = equity.cummax()

    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_profit_factor(r_values: pd.Series) -> float:
    """
    Profit Factor =
        gross profits / gross losses
    """

    gross_profit = r_values[r_values > 0].sum()

    gross_loss = -r_values[r_values < 0].sum()

    if gross_loss == 0:
        return np.inf

    return float(gross_profit / gross_loss)


def calculate_metrics(
    trades: pd.DataFrame,
) -> dict:
    """
    Calculate performance metrics from net R.
    """

    r = trades["net_r"]

    n = len(r)

    wins = (r > 0).sum()
    losses = (r < 0).sum()

    win_rate = wins / n if n > 0 else np.nan

    net_r = r.sum()

    expectancy = r.mean()

    profit_factor = calculate_profit_factor(r)

    equity = r.cumsum()

    max_dd = calculate_max_drawdown(equity)

    mean = r.mean()
    std = r.std(ddof=1)

    if std > 0:
        trade_sharpe = (mean / std) * np.sqrt(n)
    else:
        trade_sharpe = np.nan

    return {
        "trades": n,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": win_rate,
        "gross_r": trades["gross_r"].sum(),
        "net_r": net_r,
        "expectancy_r": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_dd,
        "trade_sharpe": trade_sharpe,
        "commission_r": trades["commission_r"].sum(),
        "slippage_r": trades["slippage_r"].sum(),
        "total_cost_r": trades["total_cost_r"].sum(),
        "total_cost_usd": trades["total_cost_usd"].sum(),
    }


# ============================================================
# WINDOW ANALYSIS
# ============================================================


def calculate_window_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate net performance by frozen OOS window.
    """

    rows = []

    for window, group in trades.groupby("window", sort=True):
        r = group["net_r"]

        rows.append(
            {
                "window": int(window),
                "trades": len(group),
                "net_r": r.sum(),
                "expectancy_r": r.mean(),
                "win_rate": ((r > 0).mean()),
                "profit_factor": (calculate_profit_factor(r)),
                "max_drawdown_r": (calculate_max_drawdown(r.cumsum())),
                "positive": (r.sum() > 0),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN TEST
# ============================================================


def main() -> None:

    section("RESEARCH 08X — COST & SLIPPAGE TEST")

    print("Instrument           : MNQ")
    print(f"Point value          : ${POINT_VALUE_USD:.2f}")
    print(f"Tick size            : {TICK_SIZE:.2f} points")
    print(f"Tick value           : ${TICK_VALUE_USD:.2f}")
    print(f"Topstep RT commission: ${COMMISSION_RT_USD:.2f}")
    print(f"Slippage scenarios   : {SLIPPAGE_SCENARIOS}")

    df = load_trades()

    all_summary = []
    all_trade_results = []

    # ========================================================
    # RUN EACH STRATEGY
    # ========================================================

    for strategy in FROZEN_STRATEGIES:
        strategy_df = df[df["strategy"] == strategy].copy()

        if strategy_df.empty:
            print()
            print(f"WARNING: no trades for {strategy}")
            continue

        section(f"{strategy} — COST & SLIPPAGE")

        print(f"Trades: {len(strategy_df):,}")

        for slippage_ticks in SLIPPAGE_SCENARIOS:
            scenario_df = apply_cost_model(
                strategy_df,
                slippage_ticks,
            )

            metrics = calculate_metrics(scenario_df)

            metrics.update(
                {
                    "strategy": strategy,
                    "slippage_ticks_per_execution": (slippage_ticks),
                    "slippage_ticks_total": (slippage_ticks * 2),
                    "slippage_points_total": (slippage_ticks * 2 * TICK_SIZE),
                    "commission_usd_per_trade": (COMMISSION_RT_USD),
                    "slippage_usd_per_trade": (
                        calculate_costs(slippage_ticks)["slippage_usd"]
                    ),
                    "total_cost_usd_per_trade": (
                        calculate_costs(slippage_ticks)["total_cost_usd"]
                    ),
                }
            )

            all_summary.append(metrics)

            # Add scenario information
            scenario_df["strategy"] = strategy

            all_trade_results.append(scenario_df)

            print()
            print(f"--- {slippage_ticks} tick(s) per execution ---")

            print(f"Gross R       : {metrics['gross_r']:+.2f}")

            print(f"Commission R  : -{metrics['commission_r']:.2f}")

            print(f"Slippage R    : -{metrics['slippage_r']:.2f}")

            print(f"Net R         : {metrics['net_r']:+.2f}")

            print(f"Expectancy    : {metrics['expectancy_r']:+.4f} R")

            print(f"Win rate      : {metrics['win_rate']:.2%}")

            print(f"Profit Factor : {metrics['profit_factor']:.4f}")

            print(f"Max DD        : {metrics['max_drawdown_r']:+.2f} R")

        # ====================================================
        # WINDOW ROBUSTNESS
        # ====================================================

        print()
        print(f"{strategy} — WINDOW ROBUSTNESS")

        for slippage_ticks in SLIPPAGE_SCENARIOS:
            scenario_df = apply_cost_model(
                strategy_df,
                slippage_ticks,
            )

            windows = calculate_window_metrics(scenario_df)

            positive_windows = int(windows["positive"].sum())

            total_windows = len(windows)

            print(
                f"{slippage_ticks}T: "
                f"{positive_windows}/{total_windows} "
                f"positive windows"
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(all_summary)

    summary_df = summary_df.sort_values(
        [
            "strategy",
            "slippage_ticks_per_execution",
        ]
    ).reset_index(drop=True)

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # SAVE TRADE-LEVEL RESULTS
    # ========================================================

    trade_results_df = pd.concat(
        all_trade_results,
        ignore_index=True,
    )

    trade_results_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    section("FINAL COST & SLIPPAGE COMPARISON")

    display_columns = [
        "strategy",
        "slippage_ticks_per_execution",
        "trades",
        "gross_r",
        "commission_r",
        "slippage_r",
        "net_r",
        "expectancy_r",
        "win_rate",
        "profit_factor",
        "max_drawdown_r",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            formatters={
                "gross_r": "{:+.2f}".format,
                "commission_r": "{:.2f}".format,
                "slippage_r": "{:.2f}".format,
                "net_r": "{:+.2f}".format,
                "expectancy_r": "{:+.4f}".format,
                "win_rate": "{:.2%}".format,
                "profit_factor": "{:.4f}".format,
                "max_drawdown_r": "{:+.2f}".format,
            },
        )
    )

    # ========================================================
    # BREAK-EVEN ANALYSIS
    # ========================================================

    section("BREAK-EVEN SLIPPAGE ANALYSIS")

    for strategy in FROZEN_STRATEGIES:
        strategy_summary = summary_df[summary_df["strategy"] == strategy].copy()

        if strategy_summary.empty:
            continue

        print()
        print(strategy)

        # We calculate the maximum adverse slippage
        # per execution before expectancy reaches zero.
        #
        # For these frozen strategies:
        #
        #   1R = SL points × $2
        #
        # Each additional tick per execution creates:
        #
        #   2 executions × 0.25 points × $2
        #
        # of cost.
        #
        # The exact break-even is interpolated from
        # gross expectancy.

        gross_expectancy = strategy_df[df["candidate_id"].astype(str) == strategy][
            "r"
        ].mean()

        average_risk_usd = (
            strategy_df[df["candidate_id"].astype(str) == strategy]["sl"].abs().mean()
            * POINT_VALUE_USD
        )

        commission_r = COMMISSION_RT_USD / average_risk_usd

        remaining_expectancy = gross_expectancy - commission_r

        r_cost_per_tick = 2 * TICK_SIZE * POINT_VALUE_USD / average_risk_usd

        if r_cost_per_tick > 0:
            breakeven_ticks = remaining_expectancy / r_cost_per_tick
        else:
            breakeven_ticks = np.nan

        print(f"Gross expectancy       : {gross_expectancy:+.4f} R")

        print(f"Commission impact      : -{commission_r:.4f} R/trade")

        print(f"Expectancy after fees  : {remaining_expectancy:+.4f} R")

        print(f"Break-even slippage    : {breakeven_ticks:.2f} ticks per execution")

    # ========================================================
    # OUTPUT
    # ========================================================

    section("OUTPUT FILES")

    print(f"Summary:\n{OUTPUT_SUMMARY}")

    print(f"\nTrade-level:\n{OUTPUT_TRADES}")

    print()
    print("Cost & slippage test completed.")


if __name__ == "__main__":
    main()
