from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# =============================================================================
# S13 — MAE RECOVERY EXIT LADDER — TEMPORAL OOS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

# -------------------------------------------------------------------------
# FROZEN BENCHMARK
# -------------------------------------------------------------------------

STOP_R = 1.0
TARGET_R = 1.75
HORIZON = 20

# -------------------------------------------------------------------------
# DISCOVERY GRID
# -------------------------------------------------------------------------

MAE_THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
]

RECOVERY_LEVELS = [
    -0.50,
    -0.25,
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.75,
]

RECOVERY_HORIZONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
]

# Temporal split
DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# HELPERS
# =============================================================================


def normalize_window(value):
    if pd.isna(value):
        return np.nan

    match = re.search(r"(\d+)", str(value))

    if match:
        return int(match.group(1))

    try:
        return int(float(value))
    except Exception:
        return np.nan


def detect_path_columns(df, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)R$")

    found = []

    for col in df.columns:
        match = pattern.match(str(col))

        if match:
            found.append((int(match.group(1)), col))

    found.sort()

    return found


def safe_profit_factor(values):
    values = np.asarray(values, dtype=float)

    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()

    if losses == 0:
        if gains > 0:
            return np.inf
        return 0.0

    return gains / losses


def max_drawdown(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    equity = np.cumsum(values)
    running_max = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]

    drawdown = equity - running_max

    return float(drawdown.min())


def window_stats(df, r_col="strategy_R"):
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
        }

    values = df[r_col].astype(float).to_numpy()

    wins = int((values > 0).sum())
    losses = int((values <= 0).sum())

    return {
        "trades": len(values),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(values),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": safe_profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
    }


# =============================================================================
# PATH LOGIC
# =============================================================================


def build_crossing_table(df, mae_cols):
    """
    For every trade and every MAE threshold:

        1. Find first bar where adverse MAE reaches threshold.
        2. After that crossing, inspect future close path.
        3. Determine whether recovery level was reached.
        4. Record first recovery bar.

    IMPORTANT:
    Close path is expressed in R from the original entry.
    """

    observations = []

    for idx, row in df.iterrows():
        for mae_threshold in MAE_THRESHOLDS:
            crossing_bar = None

            # -------------------------------------------------------------
            # Find first MAE crossing
            # -------------------------------------------------------------

            for bar, mae_col in mae_cols:
                value = row[mae_col]

                if pd.isna(value):
                    continue

                if float(value) >= mae_threshold:
                    crossing_bar = bar
                    break

            if crossing_bar is None:
                continue

            observations.append(
                {
                    "row_index": idx,
                    "mae_threshold": mae_threshold,
                    "crossing_bar": crossing_bar,
                }
            )

    return pd.DataFrame(observations)


def evaluate_recovery(
    df,
    crossing_df,
    close_cols,
    recovery_level,
    recovery_horizon,
):
    """
    Evaluate whether a trade recovers to recovery_level within N bars
    after the MAE threshold crossing.

    The recovery level is measured using the close path.

    Example:

        MAE >= 0.80R
        recovery_level = -0.20R
        horizon = 3

    means:

        after the adverse excursion,
        did the close recover back to -0.20R
        within the next 3 bars?
    """

    records = []

    close_lookup = dict(close_cols)

    for _, obs in crossing_df.iterrows():
        row_idx = obs["row_index"]
        threshold = obs["mae_threshold"]
        crossing_bar = int(obs["crossing_bar"])

        row = df.loc[row_idx]

        recovery_bar = None
        recovery_value = np.nan

        # -------------------------------------------------------------
        # Search future bars
        # -------------------------------------------------------------

        start_bar = crossing_bar + 1
        end_bar = min(
            crossing_bar + recovery_horizon,
            max(close_lookup.keys()),
        )

        for bar in range(start_bar, end_bar + 1):
            col = close_lookup.get(bar)

            if col is None:
                continue

            value = row[col]

            if pd.isna(value):
                continue

            value = float(value)

            if value >= recovery_level:
                recovery_bar = bar
                recovery_value = value
                break

        recovered = recovery_bar is not None

        records.append(
            {
                "row_index": row_idx,
                "mae_threshold": threshold,
                "crossing_bar": crossing_bar,
                "recovery_level": recovery_level,
                "recovery_horizon": recovery_horizon,
                "recovered": recovered,
                "recovery_bar": recovery_bar,
                "recovery_value": recovery_value,
                "final_R": float(row["final_close_R"]),
                "window": row["_window_numeric"],
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# EXIT SIMULATION
# =============================================================================


def simulate_exit_strategy(
    df,
    recovery_obs,
):
    """
    Strategy:

        If MAE threshold is reached,
        wait for recovery to the selected level.

        If recovery occurs:
            exit at recovery_level.

        If recovery does not occur:
            keep original benchmark result.

    This is deliberately conservative and transparent.

    It allows us to answer:

        "Would taking the recovery exit have improved the trade?"

    without inventing a new stop or target.
    """

    strategy_R = df["final_close_R"].astype(float).copy()

    recovery_mask = recovery_obs["recovered"]

    for _, obs in recovery_obs.loc[recovery_mask].iterrows():
        idx = obs["row_index"]
        recovery_level = float(obs["recovery_level"])

        strategy_R.loc[idx] = recovery_level

    return strategy_R


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def main():

    print("=" * 110)
    print("S13 MAE RECOVERY EXIT LADDER — TEMPORAL OOS")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop       = 25.0 points")
    print(f"  RR         = {TARGET_R}")
    print(f"  Horizon    = {HORIZON}")

    print()
    print("MAE thresholds:")
    print(" ", ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))

    print()
    print("Recovery exit levels:")
    print(" ", ", ".join(f"{x:+.2f}R" for x in RECOVERY_LEVELS))

    print()
    print("Recovery horizons:")
    print(" ", RECOVERY_HORIZONS)

    print()
    print("Development windows:", DEVELOPMENT_WINDOWS)
    print("Holdout windows    :", HOLDOUT_WINDOWS)

    # ---------------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------------

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    required = [
        "final_close_R",
        "window",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    df["_window_numeric"] = df["window"].apply(normalize_window)

    # ---------------------------------------------------------------------
    # PATH DETECTION
    # ---------------------------------------------------------------------

    mae_cols = detect_path_columns(df, "mae")
    close_cols = detect_path_columns(df, "close")

    if not mae_cols:
        raise RuntimeError("No MAE path columns found.")

    if not close_cols:
        raise RuntimeError("No close path columns found.")

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_cols)}")
    print(f"  MAE range  : {mae_cols[0][0]} -> {mae_cols[-1][0]}")

    print(f"  Close bars : {len(close_cols)}")
    print(f"  Close range: {close_cols[0][0]} -> {close_cols[-1][0]}")

    # ---------------------------------------------------------------------
    # CROSSINGS
    # ---------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BUILDING MAE CROSSINGS")
    print("=" * 110)

    crossing_df = build_crossing_table(
        df,
        mae_cols,
    )

    print(f"MAE crossing observations: {len(crossing_df)}")

    # ---------------------------------------------------------------------
    # DISCOVERY MATRIX
    # ---------------------------------------------------------------------

    print()
    print("=" * 110)
    print("1. RECOVERY EXIT LADDER DISCOVERY")
    print("=" * 110)

    discovery_rows = []

    for mae_threshold in MAE_THRESHOLDS:
        threshold_crossings = crossing_df[crossing_df["mae_threshold"] == mae_threshold]

        triggered = threshold_crossings["row_index"].nunique()

        for recovery_level in RECOVERY_LEVELS:
            for horizon in RECOVERY_HORIZONS:
                obs = evaluate_recovery(
                    df,
                    threshold_crossings,
                    close_cols,
                    recovery_level,
                    horizon,
                )

                recovered = obs[obs["recovered"]]

                recovered_count = len(recovered)

                recovery_pct = recovered_count / triggered if triggered else np.nan

                if recovered_count:
                    recovered_final = recovered["final_R"]

                    recovered_wr = (recovered_final > 0).mean()

                    recovered_mean = recovered_final.mean()

                    recovered_total = recovered_final.sum()

                else:
                    recovered_wr = np.nan
                    recovered_mean = np.nan
                    recovered_total = 0.0

                # ---------------------------------------------------------
                # Counterfactual exit R
                # ---------------------------------------------------------

                counterfactual_R = df["final_close_R"].astype(float).copy()

                for _, r in recovered.iterrows():
                    counterfactual_R.loc[r["row_index"]] = recovery_level

                values = counterfactual_R.to_numpy()

                discovery_rows.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "recovery_horizon": horizon,
                        "triggered_trades": triggered,
                        "recovered_trades": recovered_count,
                        "recovery_pct": recovery_pct,
                        "recovered_win_rate": recovered_wr,
                        "recovered_mean_final_R": recovered_mean,
                        "recovered_total_final_R": recovered_total,
                        "strategy_mean_R": values.mean(),
                        "strategy_total_R": values.sum(),
                        "strategy_win_rate": (values > 0).mean(),
                        "strategy_profit_factor": safe_profit_factor(values),
                    }
                )

    discovery_df = pd.DataFrame(discovery_rows)

    # ---------------------------------------------------------------------
    # PRINT MOST IMPORTANT DISCOVERY
    # ---------------------------------------------------------------------

    print()
    print("TOP RECOVERY LEVELS BY TOTAL R")

    top = discovery_df.sort_values(
        [
            "strategy_total_final_R"
            if "strategy_total_final_R" in discovery_df.columns
            else "strategy_total_R"
        ],
        ascending=False,
    ).head(25)

    print(top.to_string(index=False))

    # ---------------------------------------------------------------------
    # DEVELOPMENT SEARCH
    # ---------------------------------------------------------------------

    print()
    print("=" * 110)
    print("2. DEVELOPMENT SEARCH")
    print("=" * 110)

    development_df = df[df["_window_numeric"].isin(DEVELOPMENT_WINDOWS)].copy()

    benchmark_dev = development_df["final_close_R"].astype(float)

    benchmark_dev_R = benchmark_dev.sum()

    search_rows = []

    for mae_threshold in MAE_THRESHOLDS:
        threshold_crossings = crossing_df[crossing_df["mae_threshold"] == mae_threshold]

        threshold_crossings = threshold_crossings[
            threshold_crossings["window"].isin(DEVELOPMENT_WINDOWS)
        ]

        triggered = threshold_crossings["row_index"].nunique()

        if triggered == 0:
            continue

        for recovery_level in RECOVERY_LEVELS:
            for horizon in RECOVERY_HORIZONS:
                obs = evaluate_recovery(
                    df,
                    threshold_crossings,
                    close_cols,
                    recovery_level,
                    horizon,
                )

                recovered = obs[obs["recovered"]]

                strategy_R = development_df["final_close_R"].astype(float).copy()

                # map original index -> counterfactual R
                for _, r in recovered.iterrows():
                    idx = r["row_index"]

                    if idx in strategy_R.index:
                        strategy_R.loc[idx] = recovery_level

                values = strategy_R.to_numpy()

                total_R = values.sum()

                delta_R = total_R - benchmark_dev_R

                search_rows.append(
                    {
                        "mae_threshold": mae_threshold,
                        "recovery_level": recovery_level,
                        "recovery_horizon": horizon,
                        "trades": len(development_df),
                        "triggered_trades": triggered,
                        "recovered_trades": len(recovered),
                        "recovery_pct": (len(recovered) / triggered),
                        "benchmark_R": benchmark_dev_R,
                        "strategy_R": total_R,
                        "delta_R": delta_R,
                        "mean_R": values.mean(),
                        "win_rate": (values > 0).mean(),
                        "profit_factor": safe_profit_factor(values),
                    }
                )

    development_results = pd.DataFrame(search_rows)

    development_results = development_results.sort_values(
        [
            "delta_R",
            "strategy_R",
        ],
        ascending=False,
    )

    print(development_results.head(30).to_string(index=False))

    # ---------------------------------------------------------------------
    # SELECT RULE
    # ---------------------------------------------------------------------

    if development_results.empty:
        raise RuntimeError("No valid development rules found.")

    selected = development_results.iloc[0]

    selected_mae = float(selected["mae_threshold"])

    selected_level = float(selected["recovery_level"])

    selected_horizon = int(selected["recovery_horizon"])

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(f"MAE threshold    : {selected_mae:.2f}R")

    print(f"Recovery exit    : {selected_level:+.2f}R")

    print(f"Recovery horizon : {selected_horizon} bars")

    print(f"Development ΔR   : {selected['delta_R']:.4f}")

    # ---------------------------------------------------------------------
    # HOLDOUT OOS
    # ---------------------------------------------------------------------

    print()
    print("=" * 110)
    print("3. HOLDOUT OOS")
    print("=" * 110)

    holdout_df = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    benchmark_holdout = holdout_df["final_close_R"].astype(float)

    benchmark_total = benchmark_holdout.sum()

    selected_crossings = crossing_df[crossing_df["mae_threshold"] == selected_mae]

    selected_crossings = selected_crossings[
        selected_crossings["window"].isin(HOLDOUT_WINDOWS)
    ]

    selected_obs = evaluate_recovery(
        df,
        selected_crossings,
        close_cols,
        selected_level,
        selected_horizon,
    )

    selected_recovered = selected_obs[selected_obs["recovered"]]

    strategy_R = holdout_df["final_close_R"].astype(float).copy()

    for _, r in selected_recovered.iterrows():
        idx = r["row_index"]

        if idx in strategy_R.index:
            strategy_R.loc[idx] = selected_level

    strategy_values = strategy_R.to_numpy()

    benchmark_stats = window_stats(
        pd.DataFrame({"strategy_R": benchmark_holdout.values})
    )

    strategy_stats = window_stats(pd.DataFrame({"strategy_R": strategy_values}))

    print()
    print("FROZEN RULE")

    print(f"  MAE >= {selected_mae:.2f}R")

    print(f"  Recovery >= {selected_level:+.2f}R")

    print(f"  Horizon = {selected_horizon} bars")

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades    : {benchmark_stats['trades']}")

    print(f"  Win rate  : {benchmark_stats['win_rate']:.4f}")

    print(f"  Mean R    : {benchmark_stats['mean_R']:.4f}")

    print(f"  Total R   : {benchmark_stats['total_R']:.4f}")

    print(f"  PF        : {benchmark_stats['profit_factor']:.4f}")

    print()
    print("RECOVERY EXIT HOLDOUT")

    print(f"  Trades    : {strategy_stats['trades']}")

    print(f"  Win rate  : {strategy_stats['win_rate']:.4f}")

    print(f"  Mean R    : {strategy_stats['mean_R']:.4f}")

    print(f"  Total R   : {strategy_stats['total_R']:.4f}")

    print(f"  PF        : {strategy_stats['profit_factor']}")

    print()
    print("IMPROVEMENT")

    print(f"  Delta R   : {strategy_stats['total_R'] - benchmark_stats['total_R']:.4f}")

    print(
        f"  Delta WR  : {strategy_stats['win_rate'] - benchmark_stats['win_rate']:.4f}"
    )

    print(
        f"  Delta PF  : "
        f"{strategy_stats['profit_factor'] - benchmark_stats['profit_factor']}"
    )

    # ---------------------------------------------------------------------
    # WINDOW BY WINDOW
    # ---------------------------------------------------------------------

    print()
    print("=" * 110)
    print("4. WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    window_rows = []

    for window in HOLDOUT_WINDOWS:
        wdf = holdout_df[holdout_df["_window_numeric"] == window]

        if wdf.empty:
            continue

        benchmark_values = wdf["final_close_R"].astype(float).to_numpy()

        strategy_values_window = wdf["final_close_R"].astype(float).copy()

        recovered_window = selected_recovered[selected_recovered["window"] == window]

        for _, r in recovered_window.iterrows():
            idx = r["row_index"]

            if idx in strategy_values_window.index:
                strategy_values_window.loc[idx] = selected_level

        strategy_values_window = strategy_values_window.to_numpy()

        window_rows.append(
            {
                "window": window,
                "trades": len(wdf),
                "recovered_trades": len(recovered_window),
                "benchmark_R": benchmark_values.sum(),
                "strategy_R": strategy_values_window.sum(),
                "delta_R": (strategy_values_window.sum() - benchmark_values.sum()),
                "benchmark_WR": (benchmark_values > 0).mean(),
                "strategy_WR": (strategy_values_window > 0).mean(),
                "benchmark_PF": safe_profit_factor(benchmark_values),
                "strategy_PF": safe_profit_factor(strategy_values_window),
            }
        )

    window_df = pd.DataFrame(window_rows)

    print(window_df.to_string(index=False))

    # ---------------------------------------------------------------------
    # TRADE-LEVEL OUTPUT
    # ---------------------------------------------------------------------

    trade_output = holdout_df[
        [
            c
            for c in [
                "entry_timestamp",
                "exit_timestamp",
                "window",
                "entry_price",
                "final_close_R",
                "outcome",
            ]
            if c in holdout_df.columns
        ]
    ].copy()

    trade_output["strategy_R"] = strategy_values

    trade_output["delta_R"] = trade_output["strategy_R"] - trade_output["final_close_R"]

    trade_output["recovery_exit"] = False

    for _, r in selected_recovered.iterrows():
        idx = r["row_index"]

        if idx in trade_output.index:
            trade_output.loc[idx, "recovery_exit"] = True

    # ---------------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery_path = OUTPUT_DIR / "s13_mae_recovery_exit_ladder_discovery.csv"

    development_path = OUTPUT_DIR / "s13_mae_recovery_exit_ladder_development.csv"

    oos_path = OUTPUT_DIR / "s13_mae_recovery_exit_ladder_oos_trades.csv"

    window_path = OUTPUT_DIR / "s13_mae_recovery_exit_ladder_oos_by_window.csv"

    discovery_df.to_csv(
        discovery_path,
        index=False,
    )

    development_results.to_csv(
        development_path,
        index=False,
    )

    trade_output.to_csv(
        oos_path,
        index=False,
    )

    window_df.to_csv(
        window_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(discovery_path)
    print(development_path)
    print(oos_path)
    print(window_path)

    print()
    print("=" * 110)
    print("S13 MAE RECOVERY EXIT LADDER COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
