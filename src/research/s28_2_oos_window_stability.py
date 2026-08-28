"""
S28.2 — OOS WINDOW STABILITY / LEAVE-ONE-WINDOW-OUT

Frozen S2R strategy.
NO PARAMETER OPTIMIZATION.

Research question:
Does S2R's OOS performance depend excessively on any single OOS window?

Method:
1. Load the frozen S27 S2R trade-level results.
2. Restrict analysis to holdout OOS windows 12-22.
3. Calculate observed OOS performance.
4. Remove each individual OOS window one at a time.
5. Recalculate performance using the remaining OOS windows.
6. Measure concentration and stability.
7. Generate publication-ready charts.

This is a robustness test only.
The S2R parameters remain completely frozen.

Frozen S2R:
    MAE >= 0.70R
    Recovery >= +0.20R
    Deadline = 6 bars
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s27_full_strategy_trades.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

OOS_WINDOWS = list(range(12, 23))

FROZEN_MAE = 0.70
FROZEN_RECOVERY = 0.20
FROZEN_DEADLINE = 6


# ======================================================================
# UTILITIES
# ======================================================================


def print_header(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def detect_strategy_r(df: pd.DataFrame) -> str:
    candidates = [
        "_strategy_R",
        "strategy_R",
        "net_R_strategy",
        "s2r_R",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Unable to identify S2R return column.\n\n"
        f"Available columns:\n  " + "\n  ".join(df.columns.astype(str))
    )


def detect_window(df: pd.DataFrame) -> str:
    candidates = [
        "_window_numeric",
        "window_numeric",
        "window",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError("Unable to identify OOS window column.")


def safe_profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def max_drawdown(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)

    equity = values.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_metrics(
    df: pd.DataFrame,
    r_col: str,
) -> dict:

    values = pd.to_numeric(
        df[r_col],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return {
            "trades": 0,
            "total_R": 0.0,
            "mean_R": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }

    return {
        "trades": int(len(values)),
        "total_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "win_rate": float((values > 0).mean()),
        "profit_factor": safe_profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
    }


# ======================================================================
# LOAD DATA
# ======================================================================


def load_data() -> tuple[pd.DataFrame, str, str]:

    print_header("LOADING FROZEN S2R")

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    r_col = detect_strategy_r(df)
    window_col = detect_window(df)

    print()
    print("Detected:")
    print(f"  Strategy R : {r_col}")
    print(f"  Window     : {window_col}")

    df["_window_numeric_internal"] = pd.to_numeric(
        df[window_col],
        errors="coerce",
    )

    df["_strategy_R_internal"] = pd.to_numeric(
        df[r_col],
        errors="coerce",
    )

    if df["_window_numeric_internal"].isna().any():
        raise RuntimeError("Some rows have invalid OOS window values.")

    if df["_strategy_R_internal"].isna().any():
        raise RuntimeError("Some rows have invalid S2R R values.")

    return df, "_strategy_R_internal", "_window_numeric_internal"


# ======================================================================
# OOS DATASET
# ======================================================================


def build_oos_dataset(
    df: pd.DataFrame,
    window_col: str,
) -> pd.DataFrame:

    oos = df[df[window_col].isin(OOS_WINDOWS)].copy()

    if len(oos) == 0:
        raise RuntimeError("No OOS trades found in windows 12-22.")

    return oos


# ======================================================================
# BASELINE
# ======================================================================


def calculate_baseline(
    oos: pd.DataFrame,
    r_col: str,
    window_col: str,
) -> dict:

    metrics = calculate_metrics(oos, r_col)

    metrics["sample"] = "FULL_OOS"
    metrics["removed_window"] = None
    metrics["remaining_windows"] = len(sorted(oos[window_col].unique()))

    return metrics


# ======================================================================
# LEAVE-ONE-WINDOW-OUT
# ======================================================================


def calculate_leave_one_out(
    oos: pd.DataFrame,
    r_col: str,
    window_col: str,
) -> pd.DataFrame:

    rows = []

    available_windows = sorted(oos[window_col].unique())

    for removed_window in available_windows:
        subset = oos[oos[window_col] != removed_window].copy()

        metrics = calculate_metrics(
            subset,
            r_col,
        )

        metrics["sample"] = "LEAVE_ONE_OUT"
        metrics["removed_window"] = int(removed_window)
        metrics["remaining_windows"] = len(available_windows) - 1

        rows.append(metrics)

    return pd.DataFrame(rows)


# ======================================================================
# WINDOW CONTRIBUTION
# ======================================================================


def calculate_window_contributions(
    oos: pd.DataFrame,
    r_col: str,
    window_col: str,
) -> pd.DataFrame:

    rows = []

    for window in sorted(oos[window_col].unique()):
        subset = oos[oos[window_col] == window]

        metrics = calculate_metrics(
            subset,
            r_col,
        )

        rows.append(
            {
                "window": int(window),
                **metrics,
            }
        )

    result = pd.DataFrame(rows)

    total_R = result["total_R"].sum()

    if total_R != 0:
        result["share_of_total_R"] = result["total_R"] / total_R
    else:
        result["share_of_total_R"] = np.nan

    return result


# ======================================================================
# CONCENTRATION METRICS
# ======================================================================


def calculate_concentration(
    window_df: pd.DataFrame,
) -> dict:

    positive = window_df[window_df["total_R"] > 0].copy()

    negative = window_df[window_df["total_R"] < 0].copy()

    total_R = window_df["total_R"].sum()

    if len(positive) > 0:
        largest_positive = positive["total_R"].max()
        positive_sum = positive["total_R"].sum()
    else:
        largest_positive = 0.0
        positive_sum = 0.0

    if len(negative) > 0:
        largest_negative = negative["total_R"].min()
        negative_sum = negative["total_R"].sum()
    else:
        largest_negative = 0.0
        negative_sum = 0.0

    sorted_positive = positive.sort_values(
        "total_R",
        ascending=False,
    )

    top_1_R = (
        sorted_positive["total_R"].head(1).sum() if len(sorted_positive) > 0 else 0.0
    )

    top_2_R = (
        sorted_positive["total_R"].head(2).sum() if len(sorted_positive) > 0 else 0.0
    )

    top_3_R = (
        sorted_positive["total_R"].head(3).sum() if len(sorted_positive) > 0 else 0.0
    )

    return {
        "oos_windows": len(window_df),
        "positive_windows": int((window_df["total_R"] > 0).sum()),
        "negative_windows": int((window_df["total_R"] < 0).sum()),
        "largest_positive_window_R": float(largest_positive),
        "largest_negative_window_R": float(largest_negative),
        "top_1_positive_R": float(top_1_R),
        "top_2_positive_R": float(top_2_R),
        "top_3_positive_R": float(top_3_R),
        "positive_window_R": float(positive_sum),
        "negative_window_R": float(negative_sum),
        "total_R": float(total_R),
    }


# ======================================================================
# AUDIT
# ======================================================================


def run_audit(
    oos: pd.DataFrame,
    baseline: dict,
    leave_one_out: pd.DataFrame,
    window_df: pd.DataFrame,
) -> None:

    print_header("S28.2 AUDIT")

    assert len(oos) > 0

    assert baseline["trades"] == len(oos)

    assert len(leave_one_out) == len(oos["_window_numeric_internal"].unique())

    assert len(window_df) == len(oos["_window_numeric_internal"].unique())

    assert np.isfinite(oos["_strategy_R_internal"]).all()

    print("OOS trades present              : PASS")

    print("Baseline trade count preserved  : PASS")

    print("All OOS windows represented     : PASS")

    print("Leave-one-window-out complete   : PASS")

    print("Finite S2R R values             : PASS")


# ======================================================================
# CHART 1 — WINDOW PERFORMANCE
# ======================================================================


def plot_window_performance(
    window_df: pd.DataFrame,
) -> Path:

    path = OUTPUT_DIR / "s28_2_oos_window_performance.png"

    plt.figure(figsize=(14, 8))

    plt.bar(
        window_df["window"],
        window_df["total_R"],
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("OOS Window")
    plt.ylabel("Total R")
    plt.title("S28.2 — S2R OOS Performance by Window")

    plt.xticks(window_df["window"])

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return path


# ======================================================================
# CHART 2 — LEAVE-ONE-OUT TOTAL R
# ======================================================================


def plot_leave_one_out(
    loo_df: pd.DataFrame,
    baseline: dict,
) -> Path:

    path = OUTPUT_DIR / "s28_2_leave_one_window_out.png"

    plt.figure(figsize=(14, 8))

    plt.plot(
        loo_df["removed_window"],
        loo_df["total_R"],
        marker="o",
        linewidth=2,
        label="Remaining OOS Total R",
    )

    plt.axhline(
        baseline["total_R"],
        linestyle="--",
        label="Full OOS",
    )

    plt.axhline(
        0,
        linestyle=":",
    )

    plt.xlabel("Excluded OOS Window")
    plt.ylabel("Total R After Exclusion")

    plt.title("S28.2 — Leave-One-OOS-Window-Out Stability")

    plt.xticks(loo_df["removed_window"])

    plt.legend()

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return path


# ======================================================================
# CHART 3 — LEAVE-ONE-OUT MAX DD
# ======================================================================


def plot_leave_one_out_dd(
    loo_df: pd.DataFrame,
) -> Path:

    path = OUTPUT_DIR / "s28_2_leave_one_window_out_drawdown.png"

    plt.figure(figsize=(14, 8))

    plt.plot(
        loo_df["removed_window"],
        loo_df["max_drawdown_R"],
        marker="o",
        linewidth=2,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Excluded OOS Window")
    plt.ylabel("Maximum Drawdown (R)")

    plt.title("S28.2 — Drawdown After Removing Each OOS Window")

    plt.xticks(loo_df["removed_window"])

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return path


# ======================================================================
# CHART 4 — TOTAL R VS WIN RATE
# ======================================================================


def plot_window_scatter(
    window_df: pd.DataFrame,
) -> Path:

    path = OUTPUT_DIR / "s28_2_window_R_vs_winrate.png"

    plt.figure(figsize=(12, 8))

    plt.scatter(
        window_df["win_rate"] * 100,
        window_df["total_R"],
        s=100,
    )

    for _, row in window_df.iterrows():
        plt.annotate(
            str(int(row["window"])),
            (
                row["win_rate"] * 100,
                row["total_R"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Win Rate (%)")
    plt.ylabel("Total R")

    plt.title("S28.2 — OOS Window Total R vs Win Rate")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return path


# ======================================================================
# SAVE RESULTS
# ======================================================================


def save_results(
    baseline: dict,
    loo_df: pd.DataFrame,
    window_df: pd.DataFrame,
    concentration: dict,
    chart_paths: list[Path],
) -> None:

    baseline_df = pd.DataFrame([baseline])

    concentration_df = pd.DataFrame([concentration])

    baseline_df.to_csv(
        OUTPUT_DIR / "s28_2_oos_baseline.csv",
        index=False,
    )

    loo_df.to_csv(
        OUTPUT_DIR / "s28_2_leave_one_window_out.csv",
        index=False,
    )

    window_df.to_csv(
        OUTPUT_DIR / "s28_2_oos_window_contributions.csv",
        index=False,
    )

    concentration_df.to_csv(
        OUTPUT_DIR / "s28_2_oos_concentration.csv",
        index=False,
    )


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:

    print_header("S28.2 OOS WINDOW STABILITY")

    print(
        """
Frozen S2R strategy.
NO PARAMETER OPTIMIZATION.

Research question:
Does S2R's OOS performance depend excessively on any single OOS window?

Frozen model:
  MAE >= 0.70R
  Recovery >= +0.20R
  Deadline = 6 bars

OOS windows:
  12 -> 22
"""
    )

    df, r_col, window_col = load_data()

    oos = build_oos_dataset(
        df,
        window_col,
    )

    print_header("OOS SAMPLE")

    print(f"Trades          : {len(oos)}")

    print(f"Windows         : {sorted(oos[window_col].unique().astype(int).tolist())}")

    baseline = calculate_baseline(
        oos,
        r_col,
        window_col,
    )

    print()
    print("FULL OOS BASELINE")
    print(f"  Total R       : {baseline['total_R']:.4f}")
    print(f"  Mean R        : {baseline['mean_R']:.6f}")
    print(f"  Win rate      : {baseline['win_rate']:.4%}")
    print(f"  Profit Factor : {baseline['profit_factor']}")
    print(f"  Max DD        : {baseline['max_drawdown_R']:.4f}")

    # --------------------------------------------------------------
    # WINDOW CONTRIBUTIONS
    # --------------------------------------------------------------

    print_header("WINDOW-BY-WINDOW PERFORMANCE")

    window_df = calculate_window_contributions(
        oos,
        r_col,
        window_col,
    )

    display_window = window_df[
        [
            "window",
            "trades",
            "total_R",
            "mean_R",
            "win_rate",
            "profit_factor",
            "max_drawdown_R",
        ]
    ].copy()

    print(display_window.to_string(index=False))

    # --------------------------------------------------------------
    # LEAVE ONE OUT
    # --------------------------------------------------------------

    print_header("LEAVE-ONE-OOS-WINDOW-OUT")

    loo_df = calculate_leave_one_out(
        oos,
        r_col,
        window_col,
    )

    print(
        loo_df[
            [
                "removed_window",
                "trades",
                "total_R",
                "mean_R",
                "win_rate",
                "profit_factor",
                "max_drawdown_R",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------------
    # CONCENTRATION
    # --------------------------------------------------------------

    print_header("OOS CONCENTRATION")

    concentration = calculate_concentration(window_df)

    for key, value in concentration.items():
        label = key.replace(
            "_",
            " ",
        )

        if isinstance(value, float):
            print(f"{label:35s}: {value:.4f}")
        else:
            print(f"{label:35s}: {value}")

    # --------------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------------

    run_audit(
        oos,
        baseline,
        loo_df,
        window_df,
    )

    # --------------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------------

    print_header("GENERATING CHARTS")

    chart_paths = []

    chart_paths.append(plot_window_performance(window_df))

    chart_paths.append(
        plot_leave_one_out(
            loo_df,
            baseline,
        )
    )

    chart_paths.append(plot_leave_one_out_dd(loo_df))

    chart_paths.append(plot_window_scatter(window_df))

    for path in chart_paths:
        print(path)

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    save_results(
        baseline,
        loo_df,
        window_df,
        concentration,
        chart_paths,
    )

    print_header("S28.2 FINAL STATUS")

    min_loo_R = loo_df["total_R"].min()
    negative_loo = int((loo_df["total_R"] < 0).sum())

    print(f"Full OOS Total R        : {baseline['total_R']:.4f}")

    print(f"Worst leave-one-out R   : {min_loo_R:.4f}")

    print(f"Negative LOO samples    : {negative_loo}/{len(loo_df)}")

    if min_loo_R > 0:
        print()
        print("PASS — OOS remains positive after removing every individual window.")
    else:
        print()
        print(
            "CAUTION — at least one individual "
            "window materially supports the aggregate OOS result."
        )

    print()
    print("S28.2 OOS WINDOW STABILITY COMPLETE")


if __name__ == "__main__":
    main()
