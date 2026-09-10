"""
Research 08Y
Mean Reversion — Execution Geometry Test

Purpose
-------
Model execution more realistically than Research 08X.

Order model:

    ENTRY
        Market order
        -> adverse slippage possible

    TP
        Limit order
        -> no adverse slippage assumed

    SL
        Stop-market order
        -> adverse slippage possible

Costs:

    TopstepX MNQ round-turn commission = $1.22 / contract

Slippage scenarios:

    0, 1, 2, 4 ticks

Important
---------
This script does NOT optimize the strategies.

Frozen strategies:
    MRL1
    MRS2

The purpose is to determine how sensitive the frozen
strategies are to execution assumptions.

This is still a bar/path-cache execution model.
It does not claim to reproduce actual exchange queue
position or tick-by-tick bid/ask execution.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

TRADES_FILE = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

PATH_CACHE_FILE = RESULTS_DIR / "cache" / "research_07_future_path_cache.npz"

METADATA_FILE = RESULTS_DIR / "cache" / "research_07_event_metadata.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08y_execution_geometry_summary.csv"

OUTPUT_TRADES = RESULTS_DIR / "research_08y_execution_geometry_trades.csv"


# ============================================================
# MARKET CONSTANTS
# ============================================================

POINT_VALUE_USD = 2.00
TICK_SIZE = 0.25
TICK_VALUE_USD = 0.50

COMMISSION_RT_USD = 1.22

SLIPPAGE_SCENARIOS = [0, 1, 2, 4]

FROZEN_STRATEGIES = ["MRL1", "MRS2"]


# ============================================================
# DISPLAY
# ============================================================


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# LOAD 08P
# ============================================================


def load_trades() -> pd.DataFrame:

    section("LOADING FROZEN 08P TRADES")

    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"Could not find:\n{TRADES_FILE}")

    df = pd.read_csv(TRADES_FILE)

    print(f"File : {TRADES_FILE}")
    print(f"Rows : {len(df):,}")

    required = {
        "strategy_name",
        "candidate_id",
        "side",
        "tp",
        "sl",
        "rr",
        "horizon",
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
            "Missing columns:\n" + "\n".join(f"  - {x}" for x in missing)
        )

    # --------------------------------------------------------
    # Canonical strategy identifier
    # --------------------------------------------------------

    strategy_name = df["strategy_name"].astype(str).str.strip().str.upper()

    candidate_id = df["candidate_id"].astype(str).str.strip().str.upper()

    mask = strategy_name.isin(FROZEN_STRATEGIES) | candidate_id.isin(FROZEN_STRATEGIES)

    df = df[mask].copy()

    if df.empty:
        raise RuntimeError("No MRL1/MRS2 trades found.")

    df["strategy"] = np.where(
        strategy_name.loc[df.index].isin(FROZEN_STRATEGIES),
        strategy_name.loc[df.index],
        candidate_id.loc[df.index],
    )

    numeric_columns = [
        "tp",
        "sl",
        "rr",
        "horizon",
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

    df = df[
        np.isfinite(df["event_id"]) & np.isfinite(df["entry"]) & np.isfinite(df["r"])
    ].copy()

    df["event_id"] = df["event_id"].astype(int)

    df["horizon"] = df["horizon"].astype(int)

    df["window"] = df["window"].astype(int)

    print()
    print("Frozen strategies:")

    print(df["strategy"].value_counts().sort_index().to_string())

    print(f"\nFrozen trades loaded: {len(df):,}")

    return df.reset_index(drop=True)


# ============================================================
# LOAD PATH CACHE
# ============================================================


def load_path_cache():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_FILE.exists():
        raise FileNotFoundError(f"Could not find:\n{PATH_CACHE_FILE}")

    cache = np.load(
        PATH_CACHE_FILE,
        allow_pickle=False,
    )

    required = [
        "future_close",
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    for key in required:
        if key not in cache:
            raise RuntimeError(f"Path cache missing array: {key}")

    arrays = {key: cache[key] for key in required}

    print(f"future_close shape   : {arrays['future_close'].shape}")

    print(f"long_favorable      : {arrays['long_favorable'].shape}")

    print(f"long_adverse        : {arrays['long_adverse'].shape}")

    print(f"short_favorable     : {arrays['short_favorable'].shape}")

    print(f"short_adverse       : {arrays['short_adverse'].shape}")

    return arrays


# ============================================================
# PATH RESOLUTION
# ============================================================


def resolve_trade_geometry(
    side: str,
    entry: float,
    tp: float,
    sl: float,
    horizon: int,
    event_id: int,
    arrays: dict,
):
    """
    Resolve the original TP/SL path using the Research 07
    favorable/adverse excursion arrays.

    Conservative same-bar rule:

        TP and SL hit on same future bar
        -> STOP wins

    Returns:
        result
        bars_to_result
        target_bar
        stop_bar
        collision
    """

    if side == "LONG":
        favorable = arrays["long_favorable"][event_id]

        adverse = arrays["long_adverse"][event_id]

    elif side == "SHORT":
        favorable = arrays["short_favorable"][event_id]

        adverse = arrays["short_adverse"][event_id]

    else:
        raise ValueError(f"Unknown side: {side}")

    # --------------------------------------------------------
    # Only evaluate the strategy horizon.
    # --------------------------------------------------------

    horizon = min(
        int(horizon),
        len(favorable),
    )

    favorable = favorable[:horizon]

    adverse = adverse[:horizon]

    # --------------------------------------------------------
    # Excursion thresholds
    #
    # These arrays represent favorable/adverse excursion
    # relative to the event entry.
    # --------------------------------------------------------

    target_hit = favorable >= float(tp)

    stop_hit = adverse >= float(sl)

    target_indices = np.flatnonzero(target_hit)

    stop_indices = np.flatnonzero(stop_hit)

    target_bar = int(target_indices[0] + 1) if len(target_indices) else None

    stop_bar = int(stop_indices[0] + 1) if len(stop_indices) else None

    # --------------------------------------------------------
    # No exit
    # --------------------------------------------------------

    if target_bar is None and stop_bar is None:
        return {
            "result": "UNRESOLVED",
            "bars_to_result": horizon,
            "target_bar": None,
            "stop_bar": None,
            "same_bar_collision": False,
        }

    # --------------------------------------------------------
    # Target only
    # --------------------------------------------------------

    if target_bar is not None and stop_bar is None:
        return {
            "result": "TARGET",
            "bars_to_result": target_bar,
            "target_bar": target_bar,
            "stop_bar": None,
            "same_bar_collision": False,
        }

    # --------------------------------------------------------
    # Stop only
    # --------------------------------------------------------

    if stop_bar is not None and target_bar is None:
        return {
            "result": "STOP",
            "bars_to_result": stop_bar,
            "target_bar": None,
            "stop_bar": stop_bar,
            "same_bar_collision": False,
        }

    # --------------------------------------------------------
    # BOTH HIT
    #
    # Conservative rule:
    # STOP wins ties.
    # --------------------------------------------------------

    collision = target_bar == stop_bar

    if stop_bar <= target_bar:
        return {
            "result": "STOP",
            "bars_to_result": stop_bar,
            "target_bar": target_bar,
            "stop_bar": stop_bar,
            "same_bar_collision": collision,
        }

    return {
        "result": "TARGET",
        "bars_to_result": target_bar,
        "target_bar": target_bar,
        "stop_bar": stop_bar,
        "same_bar_collision": False,
    }


# ============================================================
# EXECUTION MODEL
# ============================================================


def execute_trade(
    row: pd.Series,
    slippage_ticks: int,
    arrays: dict,
):
    """
    Model realistic execution.

    ENTRY:
        market order
        -> adverse slippage

    TP:
        limit order
        -> no adverse slippage

    SL:
        stop-market
        -> adverse slippage

    Important:
    The original path resolution is first performed against
    the frozen strategy geometry.

    Slippage then modifies the realized entry/exit economics.

    This allows us to distinguish:
        - strategy path
        - execution cost
    """

    side = str(row["side"]).upper()

    entry = float(row["entry"])

    tp = float(row["tp"])

    sl = float(row["sl"])

    horizon = int(row["horizon"])

    event_id = int(row["event_id"])

    path = resolve_trade_geometry(
        side=side,
        entry=entry,
        tp=tp,
        sl=sl,
        horizon=horizon,
        event_id=event_id,
        arrays=arrays,
    )

    result = path["result"]

    # --------------------------------------------------------
    # Entry execution
    #
    # Adverse market execution:
    #
    # LONG  -> pay higher
    # SHORT -> sell lower
    # --------------------------------------------------------

    entry_slippage_points = slippage_ticks * TICK_SIZE

    if side == "LONG":
        executed_entry = entry + entry_slippage_points

    else:
        executed_entry = entry - entry_slippage_points

    # --------------------------------------------------------
    # Exit execution
    # --------------------------------------------------------

    if result == "TARGET":
        # Limit order:
        # no adverse slippage assumption.

        executed_exit = entry + tp if side == "LONG" else entry - tp

        exit_slippage_points = 0.0

    elif result == "STOP":
        # Stop-market:
        # adverse slippage.

        if side == "LONG":
            executed_exit = entry - sl - entry_slippage_points

        else:
            executed_exit = entry + sl + entry_slippage_points

        exit_slippage_points = entry_slippage_points

    else:
        # Unresolved trade.
        # No realized result.

        executed_exit = np.nan
        exit_slippage_points = 0.0

    # --------------------------------------------------------
    # Gross price P&L after execution
    # --------------------------------------------------------

    if result == "TARGET":
        if side == "LONG":
            gross_points = executed_exit - executed_entry

        else:
            gross_points = executed_entry - executed_exit

    elif result == "STOP":
        if side == "LONG":
            gross_points = executed_exit - executed_entry

        else:
            gross_points = executed_entry - executed_exit

    else:
        gross_points = 0.0

    # --------------------------------------------------------
    # Convert to USD
    # --------------------------------------------------------

    gross_usd = gross_points * POINT_VALUE_USD

    # --------------------------------------------------------
    # Commission
    # --------------------------------------------------------

    commission_usd = COMMISSION_RT_USD

    net_usd = gross_usd - commission_usd

    # --------------------------------------------------------
    # R normalization
    #
    # 1R = SL points × $2
    # --------------------------------------------------------

    risk_usd = sl * POINT_VALUE_USD

    net_r = net_usd / risk_usd

    gross_r = gross_usd / risk_usd

    return {
        "execution_result": result,
        "bars_to_result_reconstructed": (path["bars_to_result"]),
        "target_bar": path["target_bar"],
        "stop_bar": path["stop_bar"],
        "same_bar_collision": (path["same_bar_collision"]),
        "executed_entry": executed_entry,
        "executed_exit": executed_exit,
        "entry_slippage_points": (entry_slippage_points),
        "exit_slippage_points": (exit_slippage_points),
        "gross_points_after_execution": (gross_points),
        "gross_usd_after_execution": (gross_usd),
        "commission_usd": (commission_usd),
        "net_usd": (net_usd),
        "gross_r_after_execution": (gross_r),
        "net_r_after_execution": (net_r),
    }


# ============================================================
# METRICS
# ============================================================


def max_drawdown(r_values):

    equity = r_values.cumsum()

    running_max = equity.cummax()

    dd = equity - running_max

    return float(dd.min())


def profit_factor(r_values):

    profits = r_values[r_values > 0].sum()

    losses = -r_values[r_values < 0].sum()

    if losses == 0:
        return np.inf

    return float(profits / losses)


def calculate_metrics(
    df: pd.DataFrame,
):

    realized = df[df["execution_result"] != "UNRESOLVED"].copy()

    r = realized["net_r_after_execution"]

    if len(r) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "gross_r": np.nan,
            "net_r": np.nan,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": np.nan,
            "commission_usd": np.nan,
        }

    wins = (r > 0).sum()

    losses = (r < 0).sum()

    return {
        "trades": len(r),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": (wins / len(r)),
        "gross_r": (realized["gross_r_after_execution"].sum()),
        "net_r": (r.sum()),
        "expectancy_r": (r.mean()),
        "profit_factor": (profit_factor(r)),
        "max_drawdown_r": (max_drawdown(r)),
        "commission_usd": (realized["commission_usd"].sum()),
    }


# ============================================================
# WINDOW METRICS
# ============================================================


def window_metrics(
    df: pd.DataFrame,
):

    rows = []

    for window, group in df.groupby(
        "window",
        sort=True,
    ):
        realized = group[group["execution_result"] != "UNRESOLVED"]

        if realized.empty:
            continue

        r = realized["net_r_after_execution"]

        rows.append(
            {
                "window": int(window),
                "trades": len(r),
                "net_r": r.sum(),
                "expectancy_r": r.mean(),
                "win_rate": ((r > 0).mean()),
                "profit_factor": (profit_factor(r)),
                "max_drawdown_r": (max_drawdown(r)),
                "positive": (r.sum() > 0),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main():

    section("RESEARCH 08Y — EXECUTION GEOMETRY TEST")

    print("Execution model:")

    print("  Entry : MARKET -> adverse slippage")

    print("  TP    : LIMIT  -> no adverse slippage")

    print("  SL    : STOP-MARKET -> adverse slippage")

    print()
    print(f"MNQ point value       : ${POINT_VALUE_USD:.2f}")

    print(f"MNQ tick size         : {TICK_SIZE:.2f}")

    print(f"MNQ tick value        : ${TICK_VALUE_USD:.2f}")

    print(f"Commission RT         : ${COMMISSION_RT_USD:.2f}")

    print(f"Slippage scenarios    : {SLIPPAGE_SCENARIOS}")

    trades = load_trades()

    arrays = load_path_cache()

    summary_rows = []

    all_trade_rows = []

    # ========================================================
    # STRATEGIES
    # ========================================================

    for strategy in FROZEN_STRATEGIES:
        strategy_df = trades[trades["strategy"] == strategy].copy()

        section(f"{strategy} — EXECUTION GEOMETRY")

        print(f"Trades: {len(strategy_df):,}")

        for slippage_ticks in SLIPPAGE_SCENARIOS:
            print()
            print(f"--- {slippage_ticks} tick(s) per execution ---")

            rows = []

            for _, row in strategy_df.iterrows():
                execution = execute_trade(
                    row=row,
                    slippage_ticks=(slippage_ticks),
                    arrays=arrays,
                )

                output = row.to_dict()

                output.update(
                    {
                        "strategy": strategy,
                        "slippage_ticks_per_execution": (slippage_ticks),
                        **execution,
                    }
                )

                rows.append(output)

            scenario_df = pd.DataFrame(rows)

            metrics = calculate_metrics(scenario_df)

            metrics.update(
                {
                    "strategy": strategy,
                    "slippage_ticks_per_execution": (slippage_ticks),
                    "total_slippage_ticks_per_trade": (slippage_ticks * 2),
                    "positive_windows": np.nan,
                    "total_windows": np.nan,
                }
            )

            windows = window_metrics(scenario_df)

            if not windows.empty:
                metrics["positive_windows"] = int(windows["positive"].sum())

                metrics["total_windows"] = len(windows)

            summary_rows.append(metrics)

            scenario_df["slippage_ticks_per_execution"] = slippage_ticks

            all_trade_rows.append(scenario_df)

            print(f"Gross R       : {metrics['gross_r']:+.2f}")

            print(f"Net R         : {metrics['net_r']:+.2f}")

            print(f"Expectancy    : {metrics['expectancy_r']:+.4f} R")

            print(f"Win rate      : {metrics['win_rate']:.2%}")

            print(f"Profit Factor : {metrics['profit_factor']:.4f}")

            print(f"Max DD        : {metrics['max_drawdown_r']:+.2f} R")

            print(
                f"Positive OOS  : "
                f"{int(metrics['positive_windows'])}/"
                f"{int(metrics['total_windows'])}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        [
            "strategy",
            "slippage_ticks_per_execution",
        ]
    )

    trade_df = pd.concat(
        all_trade_rows,
        ignore_index=True,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    trade_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    section("FINAL EXECUTION GEOMETRY COMPARISON")

    columns = [
        "strategy",
        "slippage_ticks_per_execution",
        "trades",
        "gross_r",
        "net_r",
        "expectancy_r",
        "win_rate",
        "profit_factor",
        "max_drawdown_r",
        "positive_windows",
        "total_windows",
    ]

    print(
        summary_df[columns].to_string(
            index=False,
            formatters={
                "gross_r": "{:+.2f}".format,
                "net_r": "{:+.2f}".format,
                "expectancy_r": "{:+.4f}".format,
                "win_rate": "{:.2%}".format,
                "profit_factor": "{:.4f}".format,
                "max_drawdown_r": "{:+.2f}".format,
            },
        )
    )

    # ========================================================
    # EXECUTION DIAGNOSTICS
    # ========================================================

    section("EXECUTION DIAGNOSTICS")

    for strategy in FROZEN_STRATEGIES:
        strategy_trades = trade_df[trade_df["strategy"] == strategy]

        print()
        print(strategy)

        # Use 0-slippage scenario for path diagnostics.
        base = strategy_trades[strategy_trades["slippage_ticks_per_execution"] == 0]

        collisions = int(base["same_bar_collision"].sum())

        unresolved = int((base["execution_result"] == "UNRESOLVED").sum())

        print(f"Same-bar TP/SL collisions: {collisions:,}")

        print(f"Unresolved trades: {unresolved:,}")

    # ========================================================
    # OUTPUT
    # ========================================================

    section("OUTPUT FILES")

    print(f"Summary:\n{OUTPUT_SUMMARY}")

    print(f"\nTrade-level:\n{OUTPUT_TRADES}")

    print()
    print("Execution geometry test completed.")


if __name__ == "__main__":
    main()
