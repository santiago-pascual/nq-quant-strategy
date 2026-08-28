"""
S11 — MAE FILTER TRUE OOS IMPLEMENTATION

Purpose
-------
Test whether an EARLY adverse-movement filter can be converted into
an executable trading rule and improve the frozen benchmark OOS.

Frozen benchmark:
    Stop       = 25.0 points
    RR         = 1.75
    Horizon    = 20 bars

Core hypothesis:
    If the trade reaches a sufficiently large MAE early enough,
    the probability of eventual failure becomes extremely high.

IMPORTANT
---------
This is the first TRUE IMPLEMENTATION test.

We do NOT use future outcome information to decide the exit.

For every candidate rule:

    if MAE threshold is crossed by decision_bar:
        EXIT
    else:
        continue benchmark

The exit is executed using SAME_BAR_CLOSE because the dataset contains
close paths but no open paths.

Temporal protocol:
    Development windows = 1..11
    Holdout windows     = 12..22

Candidate thresholds:
    0.70R, 0.75R, 0.80R, 0.85R, 0.90R, 0.95R, 1.00R

Candidate decision bars:
    2, 3, 4, 5, 6, 8

No optimization is performed on holdout.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = Path("src/research/results/s2_extended/s4_adverse_recovery_enriched.csv")

OUTPUT_DIR = Path("src/research/results/s2_extended")

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

MAE_THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

DECISION_BARS = [
    2,
    3,
    4,
    5,
    6,
    8,
]

DEV_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# HELPERS
# =============================================================================


def detect_path_columns(df: pd.DataFrame, prefix: str) -> dict[int, str]:
    """
    Detect columns such as:
        mae_1, mae_2, ...
        close_1, close_2, ...

    Also accepts variants containing the prefix followed by a number.
    """
    pattern = re.compile(
        rf"^{re.escape(prefix)}[_]?(\d+)$",
        re.IGNORECASE,
    )

    result: dict[int, str] = {}

    for col in df.columns:
        match = pattern.match(str(col))
        if match:
            bar = int(match.group(1))
            result[bar] = col

    return dict(sorted(result.items()))


def find_window_column(df: pd.DataFrame) -> str:
    candidates = [
        "window",
        "walk_forward_window",
        "oos_window",
        "test_window",
        "period",
        "sample_window",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Could not identify temporal window column.\n"
        f"Available columns include:\n{list(df.columns)}"
    )


def find_final_r_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "final_R",
        "final_r",
        "final_close_R",
        "final_close_r",
        "result_R",
        "result_r",
        "R",
        "r_multiple",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def find_outcome_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "outcome",
        "final_outcome",
        "result",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def normalise_window_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Convert window labels to numeric values.
    """
    values = df[col].astype(str).str.extract(r"(\d+)", expand=False)

    return pd.to_numeric(values, errors="coerce")


def calculate_metrics(results: pd.DataFrame) -> dict:
    if results.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
        }

    r = pd.to_numeric(results["strategy_R"], errors="coerce").fillna(0.0)

    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())

    if gross_loss == 0:
        pf = np.inf if gross_profit > 0 else np.nan
    else:
        pf = gross_profit / gross_loss

    equity = r.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min())

    return {
        "trades": int(len(r)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(r)),
        "mean_R": float(r.mean()),
        "total_R": float(r.sum()),
        "profit_factor": float(pf),
        "max_drawdown_R": max_dd,
    }


def compute_benchmark_r(
    row: pd.Series,
    final_r_col: str | None,
) -> float:
    """
    Benchmark is the original frozen trade result.

    Prefer final R already present in the enriched dataset.
    """
    if final_r_col is not None:
        value = pd.to_numeric(
            pd.Series([row[final_r_col]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            return float(value)

    raise RuntimeError(
        "No usable final R column found. S11 requires the original benchmark final R."
    )


def get_close_value(row: pd.Series, close_cols: dict[int, str], bar: int):
    col = close_cols.get(bar)

    if col is None:
        return np.nan

    return pd.to_numeric(
        pd.Series([row[col]]),
        errors="coerce",
    ).iloc[0]


def get_mae_value(row: pd.Series, mae_cols: dict[int, str], bar: int):
    col = mae_cols.get(bar)

    if col is None:
        return np.nan

    return pd.to_numeric(
        pd.Series([row[col]]),
        errors="coerce",
    ).iloc[0]


def execute_candidate(
    row: pd.Series,
    threshold: float,
    decision_bar: int,
    mae_cols: dict[int, str],
    close_cols: dict[int, str],
    final_r_col: str,
) -> tuple[float, bool, int | None]:
    """
    Execute one candidate rule.

    Rule:
        If MAE >= threshold by decision_bar:
            exit at close of FIRST bar that crosses threshold
            occurring on or before decision_bar.

        Otherwise:
            preserve benchmark final R.

    Returns:
        strategy_R
        triggered
        crossing_bar
    """

    crossing_bar = None

    # We must find the FIRST actual crossing.
    for bar in sorted(mae_cols):
        if bar > decision_bar:
            break

        mae = get_mae_value(row, mae_cols, bar)

        if pd.notna(mae) and mae >= threshold:
            crossing_bar = bar
            break

    if crossing_bar is None:
        benchmark_r = compute_benchmark_r(row, final_r_col)
        return benchmark_r, False, None

    # Same-bar close execution.
    exit_r = get_close_value(
        row,
        close_cols,
        crossing_bar,
    )

    if pd.isna(exit_r):
        raise RuntimeError(f"Missing close path value for crossing bar {crossing_bar}.")

    return float(exit_r), True, crossing_bar


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def main():

    print("=" * 110)
    print("S11 MAE FILTER — TRUE OOS IMPLEMENTATION")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop            = {STOP_POINTS} points")
    print(f"  RR              = {RR}")
    print(f"  Horizon         = {HORIZON} bars")
    print()
    print("Candidate MAE thresholds:")
    print(" ", ", ".join(f"{x:.2f}R" for x in MAE_THRESHOLDS))
    print()
    print("Candidate decision bars:")
    print(" ", DECISION_BARS)
    print()
    print("Execution:")
    print("  FIRST crossing <= decision bar")
    print("  SAME_BAR_CLOSE")
    print()
    print("Development windows:", DEV_WINDOWS)
    print("Holdout windows    :", HOLDOUT_WINDOWS)

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found:\n{INPUT_PATH.resolve()}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Trades loaded: {len(df)}")

    # -------------------------------------------------------------------------
    # DETECT COLUMNS
    # -------------------------------------------------------------------------

    mae_cols = detect_path_columns(df, "mae")
    close_cols = detect_path_columns(df, "close")

    if not mae_cols:
        raise RuntimeError(
            "No MAE path columns found.\nExpected columns such as mae_1, mae_2, ..."
        )

    if not close_cols:
        raise RuntimeError(
            "No close path columns found.\n"
            "Expected columns such as close_1, close_2, ..."
        )

    window_col = find_window_column(df)
    final_r_col = find_final_r_column(df)

    if final_r_col is None:
        raise RuntimeError("Could not find the original benchmark final R column.")

    df["_window_numeric"] = normalise_window_column(
        df,
        window_col,
    )

    if df["_window_numeric"].isna().all():
        raise RuntimeError(
            f"Window column '{window_col}' could not be converted to numeric."
        )

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_cols)}")
    print(f"  MAE range  : {min(mae_cols)} -> {max(mae_cols)}")
    print(f"  Close bars : {len(close_cols)}")
    print(f"  Close range: {min(close_cols)} -> {max(close_cols)}")
    print(f"  Window col : {window_col}")
    print(f"  Final R col: {final_r_col}")

    # -------------------------------------------------------------------------
    # BUILD ALL CANDIDATES
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BUILDING CANDIDATE RULES")
    print("=" * 110)

    all_trade_results = []
    all_window_results = []

    for threshold in MAE_THRESHOLDS:
        for decision_bar in DECISION_BARS:
            print(f"Testing MAE >= {threshold:.2f}R by bar {decision_bar}...")

            rows = []

            for idx, row in df.iterrows():
                benchmark_r = compute_benchmark_r(
                    row,
                    final_r_col,
                )

                strategy_r, triggered, crossing_bar = execute_candidate(
                    row=row,
                    threshold=threshold,
                    decision_bar=decision_bar,
                    mae_cols=mae_cols,
                    close_cols=close_cols,
                    final_r_col=final_r_col,
                )

                rows.append(
                    {
                        "index": idx,
                        "window": row["_window_numeric"],
                        "mae_threshold": threshold,
                        "decision_bar": decision_bar,
                        "benchmark_R": benchmark_r,
                        "strategy_R": strategy_r,
                        "triggered": triggered,
                        "crossing_bar": crossing_bar,
                    }
                )

            result_df = pd.DataFrame(rows)

            # -----------------------------------------------------------------
            # DEVELOPMENT / HOLDOUT
            # -----------------------------------------------------------------

            for sample_name, windows in [
                ("DEVELOPMENT", DEV_WINDOWS),
                ("HOLDOUT", HOLDOUT_WINDOWS),
            ]:
                subset = result_df[result_df["window"].isin(windows)].copy()

                if subset.empty:
                    continue

                benchmark_metrics = calculate_metrics(
                    subset.assign(strategy_R=subset["benchmark_R"])
                )

                strategy_metrics = calculate_metrics(subset)

                triggered_count = int(subset["triggered"].sum())

                positive_windows = 0
                total_windows = 0
                window_totals = []

                for window, window_df in subset.groupby("window"):
                    if window_df.empty:
                        continue

                    total_windows += 1

                    window_r = float(window_df["strategy_R"].sum())

                    window_totals.append(
                        {
                            "window": int(window),
                            "strategy_R": window_r,
                            "benchmark_R": float(window_df["benchmark_R"].sum()),
                        }
                    )

                    if window_r > 0:
                        positive_windows += 1

                if total_windows > 0:
                    positive_window_pct = positive_windows / total_windows
                else:
                    positive_window_pct = np.nan

                all_trade_results.append(
                    {
                        "sample": sample_name,
                        "mae_threshold": threshold,
                        "decision_bar": decision_bar,
                        "trades": strategy_metrics["trades"],
                        "triggered_trades": triggered_count,
                        "benchmark_R": benchmark_metrics["total_R"],
                        "strategy_R": strategy_metrics["total_R"],
                        "delta_R": (
                            strategy_metrics["total_R"] - benchmark_metrics["total_R"]
                        ),
                        "benchmark_mean_R": benchmark_metrics["mean_R"],
                        "strategy_mean_R": strategy_metrics["mean_R"],
                        "delta_mean_R": (
                            strategy_metrics["mean_R"] - benchmark_metrics["mean_R"]
                        ),
                        "benchmark_WR": benchmark_metrics["win_rate"],
                        "strategy_WR": strategy_metrics["win_rate"],
                        "delta_WR": (
                            strategy_metrics["win_rate"] - benchmark_metrics["win_rate"]
                        ),
                        "benchmark_PF": benchmark_metrics["profit_factor"],
                        "strategy_PF": strategy_metrics["profit_factor"],
                        "benchmark_max_DD_R": benchmark_metrics["max_drawdown_R"],
                        "strategy_max_DD_R": strategy_metrics["max_drawdown_R"],
                        "delta_max_DD_R": (
                            strategy_metrics["max_drawdown_R"]
                            - benchmark_metrics["max_drawdown_R"]
                        ),
                        "positive_windows": positive_windows,
                        "total_windows": total_windows,
                        "positive_window_pct": positive_window_pct,
                    }
                )

                for item in window_totals:
                    all_window_results.append(
                        {
                            "sample": sample_name,
                            "mae_threshold": threshold,
                            "decision_bar": decision_bar,
                            **item,
                        }
                    )

            # Save every trade result internally for final output.
            result_df["sample"] = np.where(
                result_df["window"].isin(DEV_WINDOWS),
                "DEVELOPMENT",
                np.where(
                    result_df["window"].isin(HOLDOUT_WINDOWS),
                    "HOLDOUT",
                    "OTHER",
                ),
            )

            # Keep a copy for final trade-level audit.
            all_trade_results.extend(result_df.to_dict("records"))

    trade_df = pd.DataFrame(all_trade_results)

    # The previous loop contains both summary rows and trade rows.
    # Separate them explicitly.
    #
    # Trade rows have "index" and "strategy_R".
    # Summary rows have "sample" and "triggered_trades".
    #
    # We rebuild clean summary data below from the trade-level dataframe.

    raw_trade_df = trade_df[
        trade_df["index"].notna() & trade_df["strategy_R"].notna()
    ].copy()

    raw_trade_df["index"] = raw_trade_df["index"].astype(int)

    # Remove accidental duplicate trade rows caused by sample processing.
    raw_trade_df = raw_trade_df.drop_duplicates(
        subset=[
            "index",
            "mae_threshold",
            "decision_bar",
        ],
        keep="first",
    )

    # -------------------------------------------------------------------------
    # CLEAN SUMMARY
    # -------------------------------------------------------------------------

    summary_rows = []
    window_rows = []

    for threshold in MAE_THRESHOLDS:
        for decision_bar in DECISION_BARS:
            candidate = raw_trade_df[
                (raw_trade_df["mae_threshold"] == threshold)
                & (raw_trade_df["decision_bar"] == decision_bar)
            ].copy()

            if candidate.empty:
                continue

            for sample_name, windows in [
                ("DEVELOPMENT", DEV_WINDOWS),
                ("HOLDOUT", HOLDOUT_WINDOWS),
            ]:
                subset = candidate[candidate["window"].isin(windows)].copy()

                if subset.empty:
                    continue

                benchmark = subset["benchmark_R"].astype(float)
                strategy = subset["strategy_R"].astype(float)

                benchmark_win_rate = float((benchmark > 0).mean())

                strategy_win_rate = float((strategy > 0).mean())

                def pf(series):
                    gp = series[series > 0].sum()
                    gl = -series[series < 0].sum()

                    if gl == 0:
                        return np.inf if gp > 0 else np.nan

                    return float(gp / gl)

                def max_dd(series):
                    equity = series.cumsum()
                    dd = equity - equity.cummax()
                    return float(dd.min())

                positive_windows = 0
                total_windows = 0

                for window, window_df in subset.groupby("window"):
                    total_windows += 1

                    if window_df["strategy_R"].sum() > 0:
                        positive_windows += 1

                    window_rows.append(
                        {
                            "sample": sample_name,
                            "mae_threshold": threshold,
                            "decision_bar": decision_bar,
                            "window": int(window),
                            "trades": len(window_df),
                            "triggered_trades": int(window_df["triggered"].sum()),
                            "benchmark_R": float(window_df["benchmark_R"].sum()),
                            "strategy_R": float(window_df["strategy_R"].sum()),
                            "delta_R": float(
                                window_df["strategy_R"].sum()
                                - window_df["benchmark_R"].sum()
                            ),
                            "benchmark_WR": float(
                                (window_df["benchmark_R"] > 0).mean()
                            ),
                            "strategy_WR": float((window_df["strategy_R"] > 0).mean()),
                        }
                    )

                summary_rows.append(
                    {
                        "sample": sample_name,
                        "mae_threshold": threshold,
                        "decision_bar": decision_bar,
                        "trades": len(subset),
                        "triggered_trades": int(subset["triggered"].sum()),
                        "benchmark_R": float(benchmark.sum()),
                        "strategy_R": float(strategy.sum()),
                        "delta_R": float(strategy.sum() - benchmark.sum()),
                        "benchmark_mean_R": float(benchmark.mean()),
                        "strategy_mean_R": float(strategy.mean()),
                        "delta_mean_R": float(strategy.mean() - benchmark.mean()),
                        "benchmark_WR": benchmark_win_rate,
                        "strategy_WR": strategy_win_rate,
                        "delta_WR": (strategy_win_rate - benchmark_win_rate),
                        "benchmark_PF": pf(benchmark),
                        "strategy_PF": pf(strategy),
                        "benchmark_max_DD_R": max_dd(benchmark),
                        "strategy_max_DD_R": max_dd(strategy),
                        "delta_max_DD_R": (max_dd(strategy) - max_dd(benchmark)),
                        "positive_windows": positive_windows,
                        "total_windows": total_windows,
                        "positive_window_pct": (
                            positive_windows / total_windows
                            if total_windows
                            else np.nan
                        ),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    window_df = pd.DataFrame(window_rows)

    # -------------------------------------------------------------------------
    # DEVELOPMENT SEARCH
    # -------------------------------------------------------------------------

    dev = summary_df[summary_df["sample"] == "DEVELOPMENT"].copy()

    dev = dev.sort_values(
        [
            "delta_R",
            "positive_window_pct",
            "strategy_WR",
        ],
        ascending=False,
    )

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    print(
        dev[
            [
                "mae_threshold",
                "decision_bar",
                "trades",
                "triggered_trades",
                "benchmark_R",
                "strategy_R",
                "delta_R",
                "benchmark_WR",
                "strategy_WR",
                "positive_windows",
                "total_windows",
                "positive_window_pct",
            ]
        ].to_string(index=False)
    )

    if dev.empty:
        raise RuntimeError("No development results were generated.")

    selected = dev.iloc[0]

    selected_threshold = float(selected["mae_threshold"])

    selected_bar = int(selected["decision_bar"])

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(f"MAE threshold : {selected_threshold:.2f}R")
    print(f"Decision bar  : {selected_bar}")
    print(f"Development delta R : {selected['delta_R']:.4f}")
    print(f"Positive windows    : {selected['positive_window_pct']:.4f}")

    # -------------------------------------------------------------------------
    # HOLDOUT — FROZEN RULE
    # -------------------------------------------------------------------------

    holdout = summary_df[
        (summary_df["sample"] == "HOLDOUT")
        & (summary_df["mae_threshold"] == selected_threshold)
        & (summary_df["decision_bar"] == selected_bar)
    ].copy()

    if holdout.empty:
        raise RuntimeError("Selected development rule has no holdout result.")

    h = holdout.iloc[0]

    print()
    print("=" * 110)
    print("HOLDOUT OOS RESULT")
    print("=" * 110)

    print()
    print("Frozen rule:")
    print(f"  MAE >= {selected_threshold:.2f}R by bar {selected_bar}")
    print("  Execution = SAME_BAR_CLOSE")

    print()
    print("BENCHMARK HOLDOUT")
    print(f"  Trades          : {int(h['trades'])}")
    print(f"  Win rate        : {h['benchmark_WR']:.4f}")
    print(f"  Mean R          : {h['benchmark_mean_R']:.4f}")
    print(f"  Total R         : {h['benchmark_R']:.4f}")
    print(f"  PF              : {h['benchmark_PF']}")
    print(f"  Max DD           : {h['benchmark_max_DD_R']:.4f}")

    print()
    print("MAE FILTER HOLDOUT")
    print(f"  Trades          : {int(h['trades'])}")
    print(f"  Triggered       : {int(h['triggered_trades'])}")
    print(f"  Win rate        : {h['strategy_WR']:.4f}")
    print(f"  Mean R          : {h['strategy_mean_R']:.4f}")
    print(f"  Total R         : {h['strategy_R']:.4f}")
    print(f"  PF              : {h['strategy_PF']}")
    print(f"  Max DD           : {h['strategy_max_DD_R']:.4f}")

    print()
    print("IMPROVEMENT")
    print(f"  Delta R         : {h['delta_R']:.4f}")
    print(f"  Delta Mean R    : {h['delta_mean_R']:.4f}")
    print(f"  Delta Win Rate  : {h['delta_WR']:.4f}")
    print(f"  Delta Max DD    : {h['delta_max_DD_R']:.4f}")

    # -------------------------------------------------------------------------
    # HOLDOUT WINDOWS
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW HOLDOUT")
    print("=" * 110)

    selected_window_df = window_df[
        (window_df["sample"] == "HOLDOUT")
        & (window_df["mae_threshold"] == selected_threshold)
        & (window_df["decision_bar"] == selected_bar)
    ].copy()

    print(
        selected_window_df[
            [
                "window",
                "trades",
                "triggered_trades",
                "benchmark_R",
                "strategy_R",
                "delta_R",
                "benchmark_WR",
                "strategy_WR",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # CROSSING DISTRIBUTION
    # -------------------------------------------------------------------------

    selected_trades = raw_trade_df[
        (raw_trade_df["mae_threshold"] == selected_threshold)
        & (raw_trade_df["decision_bar"] == selected_bar)
        & raw_trade_df["window"].isin(HOLDOUT_WINDOWS)
    ].copy()

    crossing_distribution = (
        selected_trades[selected_trades["triggered"]]
        .groupby("crossing_bar")
        .size()
        .reset_index(name="trades")
        .sort_values("crossing_bar")
    )

    print()
    print("=" * 110)
    print("SELECTED RULE — HOLDOUT CROSSING DISTRIBUTION")
    print("=" * 110)

    if crossing_distribution.empty:
        print("No triggered holdout trades.")
    else:
        print(crossing_distribution.to_string(index=False))

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = OUTPUT_DIR / "s11_mae_filter_true_oos_summary.csv"

    window_path = OUTPUT_DIR / "s11_mae_filter_true_oos_by_window.csv"

    trades_path = OUTPUT_DIR / "s11_mae_filter_true_oos_trades.csv"

    crossing_path = OUTPUT_DIR / "s11_mae_filter_true_oos_crossing_distribution.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    window_df.to_csv(
        window_path,
        index=False,
    )

    raw_trade_df.to_csv(
        trades_path,
        index=False,
    )

    crossing_distribution.to_csv(
        crossing_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(summary_path.resolve())
    print(window_path.resolve())
    print(trades_path.resolve())
    print(crossing_path.resolve())

    print()
    print("=" * 110)
    print("S11 MAE FILTER TRUE OOS IMPLEMENTATION COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
