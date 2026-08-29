from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# =============================================================================
# S13 — STOP BOUNDARY OOS TEST
#
# Objective:
# Test whether replacing the frozen 1.00R stop with a tighter stop improves
# the ORIGINAL strategy across the complete 537-trade benchmark.
#
# This is NOT a recovery test.
# This is NOT an optimization of entry conditions.
#
# We simply ask:
#
#   "If the trade reaches a MAE boundary before the original 1R stop,
#    would exiting there improve the strategy?"
#
# Temporal OOS:
#   Development = windows 1-11
#   Holdout     = windows 12-22
#
# IMPORTANT:
# We use the per-bar MAE path to determine whether a candidate stop was hit.
# If a candidate stop is hit, the trade exits at that stop level.
# Otherwise the original benchmark final R is retained.
#
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
)


# =============================================================================
# FROZEN BENCHMARK
# =============================================================================

ORIGINAL_STOP_R = 1.00
ORIGINAL_STOP_POINTS = 25.0
TARGET_R = 1.75
HORIZON = 20


# =============================================================================
# CANDIDATE STOPS
# =============================================================================
#
# Stop is represented as positive R distance.
#
# Example:
#   0.80R = stop at -0.80R
#
# =============================================================================

CANDIDATE_STOPS = [
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]


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


def detect_mae_columns(df):
    """
    Detect columns such as:

        mae_1R
        mae_2R
        ...
        mae_20R

    Returns:
        [(1, "mae_1R"), (2, "mae_2R"), ...]
    """

    pattern = re.compile(r"^mae_(\d+)R$")

    result = []

    for column in df.columns:

        match = pattern.match(str(column))

        if match:
            result.append(
                (
                    int(match.group(1)),
                    column,
                )
            )

    result.sort(key=lambda x: x[0])

    return result


def safe_profit_factor(values):
    values = np.asarray(values, dtype=float)

    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return 0.0

    return gross_profit / gross_loss


def calculate_max_drawdown(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    equity = np.cumsum(values)

    running_peak = np.maximum.accumulate(
        np.insert(equity, 0, 0.0)
    )[1:]

    drawdown = equity - running_peak

    return float(drawdown.min())


def calculate_stats(values):

    values = np.asarray(values, dtype=float)

    trades = len(values)

    if trades == 0:

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

    wins = int((values > 0).sum())
    losses = int((values <= 0).sum())

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades,
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": safe_profit_factor(values),
        "max_drawdown_R": calculate_max_drawdown(values),
    }


def positive_window_pct(
    df,
    r_column,
):

    results = []

    for window in sorted(
        df["_window_numeric"].dropna().unique()
    ):

        window_df = df[
            df["_window_numeric"] == window
        ]

        if window_df.empty:
            continue

        total_R = (
            window_df[r_column]
            .astype(float)
            .sum()
        )

        results.append(
            total_R > 0
        )

    if not results:
        return np.nan

    return float(np.mean(results))


def worst_window_R(
    df,
    r_column,
):

    totals = (
        df.groupby("_window_numeric")[r_column]
        .sum()
    )

    if totals.empty:
        return np.nan

    return float(totals.min())


def best_window_R(
    df,
    r_column,
):

    totals = (
        df.groupby("_window_numeric")[r_column]
        .sum()
    )

    if totals.empty:
        return np.nan

    return float(totals.max())


def longest_losing_streak(values):

    values = np.asarray(values, dtype=float)

    longest = 0
    current = 0

    for value in values:

        if value <= 0:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


# =============================================================================
# STOP SIMULATION
# =============================================================================

def simulate_candidate_stop(
    df,
    mae_columns,
    stop_R,
):
    """
    Simulate a tighter stop.

    MAE path is positive magnitude.

    Example:

        mae_1R = 0.30
        mae_2R = 0.65
        mae_3R = 0.82

    Candidate stop = 0.80R

    => stop is triggered at bar 3.

    Resulting trade R = -0.80R.

    If MAE never reaches the candidate stop:
        keep original final_close_R.

    This is deliberately simple and conservative.
    """

    strategy_R = []
    stop_triggered = []
    stop_bar = []

    for _, row in df.iterrows():

        triggered = False
        trigger_bar = np.nan

        for bar, column in mae_columns:

            value = row[column]

            if pd.isna(value):
                continue

            value = float(value)

            if value >= stop_R:

                triggered = True
                trigger_bar = bar

                break

        if triggered:

            result_R = -float(stop_R)

        else:

            result_R = float(
                row["final_close_R"]
            )

        strategy_R.append(
            result_R
        )

        stop_triggered.append(
            triggered
        )

        stop_bar.append(
            trigger_bar
        )

    result = pd.DataFrame(
        {
            "strategy_R": strategy_R,
            "stop_triggered": stop_triggered,
            "stop_bar": stop_bar,
        },
        index=df.index,
    )

    return result


# =============================================================================
# EVALUATE ONE STOP
# =============================================================================

def evaluate_stop(
    df,
    simulation,
):

    working = df.copy()

    working["strategy_R"] = (
        simulation["strategy_R"]
    )

    working["stop_triggered"] = (
        simulation["stop_triggered"]
    )

    working["stop_bar"] = (
        simulation["stop_bar"]
    )

    stats = calculate_stats(
        working["strategy_R"].values
    )

    return {
        **stats,
        "triggered_trades": int(
            working["stop_triggered"].sum()
        ),
        "positive_window_pct": positive_window_pct(
            working,
            "strategy_R",
        ),
        "worst_window_R": worst_window_R(
            working,
            "strategy_R",
        ),
        "best_window_R": best_window_R(
            working,
            "strategy_R",
        ),
        "longest_losing_streak": longest_losing_streak(
            working["strategy_R"].values
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print("S13 STOP BOUNDARY — TEMPORAL OOS TEST")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop       = {ORIGINAL_STOP_POINTS} points")
    print(f"  Stop R     = {ORIGINAL_STOP_R:.2f}R")
    print(f"  RR         = {TARGET_R}")
    print(f"  Horizon    = {HORIZON} bars")

    print()
    print("Candidate stops:")

    print(
        "  "
        + ", ".join(
            f"{x:.2f}R"
            for x in CANDIDATE_STOPS
        )
    )

    print()
    print(
        "Development windows:",
        DEVELOPMENT_WINDOWS,
    )

    print(
        "Holdout windows    :",
        HOLDOUT_WINDOWS,
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print()
    print("=" * 110)
    print("LOADING ORIGINAL ENRICHED BENCHMARK")
    print("=" * 110)

    print(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Trades loaded: {len(df)}"
    )

    required_columns = [
        "final_close_R",
        "window",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    df["_window_numeric"] = (
        df["window"]
        .apply(normalize_window)
    )

    # =========================================================================
    # PATH DETECTION
    # =========================================================================

    mae_columns = detect_mae_columns(
        df
    )

    if not mae_columns:

        raise RuntimeError(
            "No MAE path columns found."
        )

    print()
    print("Detected MAE path:")

    print(
        f"  Bars  : {len(mae_columns)}"
    )

    print(
        f"  Range : "
        f"{mae_columns[0][0]} -> "
        f"{mae_columns[-1][0]}"
    )

    # =========================================================================
    # BENCHMARK
    # =========================================================================

    benchmark_stats = calculate_stats(
        df["final_close_R"].values
    )

    print()
    print("=" * 110)
    print("BASELINE BENCHMARK")
    print("=" * 110)

    print(
        f"Trades           : "
        f"{benchmark_stats['trades']}"
    )

    print(
        f"Win rate         : "
        f"{benchmark_stats['win_rate']:.4f}"
    )

    print(
        f"Mean R           : "
        f"{benchmark_stats['mean_R']:.4f}"
    )

    print(
        f"Total R          : "
        f"{benchmark_stats['total_R']:.4f}"
    )

    print(
        f"Profit Factor    : "
        f"{benchmark_stats['profit_factor']:.4f}"
    )

    print(
        f"Max DD           : "
        f"{benchmark_stats['max_drawdown_R']:.4f}"
    )

    # =========================================================================
    # DEVELOPMENT SEARCH
    # =========================================================================

    print()
    print("=" * 110)
    print("DEVELOPMENT SEARCH")
    print("=" * 110)

    development_df = df[
        df["_window_numeric"].isin(
            DEVELOPMENT_WINDOWS
        )
    ].copy()

    benchmark_dev_values = (
        development_df[
            "final_close_R"
        ]
        .astype(float)
        .values
    )

    benchmark_dev_stats = (
        calculate_stats(
            benchmark_dev_values
        )
    )

    development_rows = []

    development_simulations = {}

    for stop_R in CANDIDATE_STOPS:

        print(
            f"Testing stop = "
            f"{stop_R:.2f}R..."
        )

        simulation = simulate_candidate_stop(
            development_df,
            mae_columns,
            stop_R,
        )

        development_simulations[
            stop_R
        ] = simulation

        stats = evaluate_stop(
            development_df,
            simulation,
        )

        development_rows.append(
            {
                "stop_R": stop_R,
                "trades": stats["trades"],
                "triggered_trades": stats[
                    "triggered_trades"
                ],
                "win_rate": stats[
                    "win_rate"
                ],
                "mean_R": stats[
                    "mean_R"
                ],
                "total_R": stats[
                    "total_R"
                ],
                "profit_factor": stats[
                    "profit_factor"
                ],
                "max_drawdown_R": stats[
                    "max_drawdown_R"
                ],
                "positive_window_pct": stats[
                    "positive_window_pct"
                ],
                "worst_window_R": stats[
                    "worst_window_R"
                ],
                "best_window_R": stats[
                    "best_window_R"
                ],
                "longest_losing_streak": stats[
                    "longest_losing_streak"
                ],
                "benchmark_R": benchmark_dev_stats[
                    "total_R"
                ],
                "delta_R": (
                    stats["total_R"]
                    - benchmark_dev_stats[
                        "total_R"
                    ]
                ),
                "delta_win_rate": (
                    stats["win_rate"]
                    - benchmark_dev_stats[
                        "win_rate"
                    ]
                ),
                "delta_max_drawdown_R": (
                    stats["max_drawdown_R"]
                    - benchmark_dev_stats[
                        "max_drawdown_R"
                    ]
                ),
            }
        )

    development_results = pd.DataFrame(
        development_rows
    )

    development_results = (
        development_results
        .sort_values(
            [
                "delta_R",
                "profit_factor",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print(
        development_results.to_string(
            index=False
        )
    )

    # =========================================================================
    # SELECT DEVELOPMENT RULE
    # =========================================================================

    selected = development_results.iloc[0]

    selected_stop_R = float(
        selected["stop_R"]
    )

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT STOP")
    print("=" * 110)

    print(
        f"Stop R             : "
        f"{selected_stop_R:.2f}R"
    )

    print(
        f"Triggered trades   : "
        f"{int(selected['triggered_trades'])}"
    )

    print(
        f"Development Total R: "
        f"{selected['total_R']:.4f}"
    )

    print(
        f"Development ΔR     : "
        f"{selected['delta_R']:.4f}"
    )

    print(
        f"Development PF     : "
        f"{selected['profit_factor']}"
    )

    print(
        f"Development Max DD  : "
        f"{selected['max_drawdown_R']:.4f}"
    )

    # =========================================================================
    # HOLDOUT OOS
    # =========================================================================

    print()
    print("=" * 110)
    print("HOLDOUT OOS TEST")
    print("=" * 110)

    holdout_df = df[
        df["_window_numeric"].isin(
            HOLDOUT_WINDOWS
        )
    ].copy()

    benchmark_holdout_values = (
        holdout_df[
            "final_close_R"
        ]
        .astype(float)
        .values
    )

    benchmark_holdout_stats = (
        calculate_stats(
            benchmark_holdout_values
        )
    )

    selected_holdout_simulation = (
        simulate_candidate_stop(
            holdout_df,
            mae_columns,
            selected_stop_R,
        )
    )

    selected_holdout_stats = (
        evaluate_stop(
            holdout_df,
            selected_holdout_simulation,
        )
    )

    print()
    print(
        f"Frozen stop = "
        f"{selected_stop_R:.2f}R"
    )

    print()
    print("BENCHMARK HOLDOUT")

    print(
        f"  Trades       : "
        f"{benchmark_holdout_stats['trades']}"
    )

    print(
        f"  Win rate     : "
        f"{benchmark_holdout_stats['win_rate']:.4f}"
    )

    print(
        f"  Mean R       : "
        f"{benchmark_holdout_stats['mean_R']:.4f}"
    )

    print(
        f"  Total R      : "
        f"{benchmark_holdout_stats['total_R']:.4f}"
    )

    print(
        f"  PF           : "
        f"{benchmark_holdout_stats['profit_factor']}"
    )

    print(
        f"  Max DD       : "
        f"{benchmark_holdout_stats['max_drawdown_R']:.4f}"
    )

    print()
    print(
        f"STOP {selected_stop_R:.2f}R HOLDOUT"
    )

    print(
        f"  Trades       : "
        f"{selected_holdout_stats['trades']}"
    )

    print(
        f"  Triggered    : "
        f"{selected_holdout_stats['triggered_trades']}"
    )

    print(
        f"  Win rate     : "
        f"{selected_holdout_stats['win_rate']:.4f}"
    )

    print(
        f"  Mean R       : "
        f"{selected_holdout_stats['mean_R']:.4f}"
    )

    print(
        f"  Total R      : "
        f"{selected_holdout_stats['total_R']:.4f}"
    )

    print(
        f"  PF           : "
        f"{selected_holdout_stats['profit_factor']}"
    )

    print(
        f"  Max DD       : "
        f"{selected_holdout_stats['max_drawdown_R']:.4f}"
    )

    print()
    print("OOS IMPROVEMENT")

    delta_total_R = (
        selected_holdout_stats[
            "total_R"
        ]
        - benchmark_holdout_stats[
            "total_R"
        ]
    )

    delta_wr = (
        selected_holdout_stats[
            "win_rate"
        ]
        - benchmark_holdout_stats[
            "win_rate"
        ]
    )

    delta_dd = (
        selected_holdout_stats[
            "max_drawdown_R"
        ]
        - benchmark_holdout_stats[
            "max_drawdown_R"
        ]
    )

    print(
        f"  Delta R      : "
        f"{delta_total_R:.4f}"
    )

    print(
        f"  Delta WR     : "
        f"{delta_wr:.4f}"
    )

    print(
        f"  Delta Max DD : "
        f"{delta_dd:.4f}"
    )

    # =========================================================================
    # ALL STOPS ON HOLDOUT
    # =========================================================================

    print()
    print("=" * 110)
    print("ALL CANDIDATE STOPS — HOLDOUT")
    print("=" * 110)

    all_holdout_rows = []

    for stop_R in CANDIDATE_STOPS:

        simulation = simulate_candidate_stop(
            holdout_df,
            mae_columns,
            stop_R,
        )

        stats = evaluate_stop(
            holdout_df,
            simulation,
        )

        all_holdout_rows.append(
            {
                "stop_R": stop_R,
                "trades": stats["trades"],
                "triggered_trades": stats[
                    "triggered_trades"
                ],
                "win_rate": stats[
                    "win_rate"
                ],
                "mean_R": stats[
                    "mean_R"
                ],
                "total_R": stats[
                    "total_R"
                ],
                "profit_factor": stats[
                    "profit_factor"
                ],
                "max_drawdown_R": stats[
                    "max_drawdown_R"
                ],
                "positive_window_pct": stats[
                    "positive_window_pct"
                ],
                "worst_window_R": stats[
                    "worst_window_R"
                ],
                "best_window_R": stats[
                    "best_window_R"
                ],
                "longest_losing_streak": stats[
                    "longest_losing_streak"
                ],
                "delta_R_vs_benchmark": (
                    stats["total_R"]
                    - benchmark_holdout_stats[
                        "total_R"
                    ]
                ),
                "delta_WR_vs_benchmark": (
                    stats["win_rate"]
                    - benchmark_holdout_stats[
                        "win_rate"
                    ]
                ),
            }
        )

    all_holdout_results = pd.DataFrame(
        all_holdout_rows
    )

    print(
        all_holdout_results.to_string(
            index=False
        )
    )

    # =========================================================================
    # WINDOW-BY-WINDOW SELECTED RULE
    # =========================================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW OOS — SELECTED STOP")
    print("=" * 110)

    window_rows = []

    selected_simulation = (
        selected_holdout_simulation
    )

    for window in HOLDOUT_WINDOWS:

        window_df = holdout_df[
            holdout_df["_window_numeric"]
            == window
        ]

        if window_df.empty:
            continue

        benchmark_values = (
            window_df[
                "final_close_R"
            ]
            .astype(float)
            .values
        )

        strategy_values = (
            selected_simulation.loc[
                window_df.index,
                "strategy_R",
            ]
            .astype(float)
            .values
        )

        benchmark_window_stats = (
            calculate_stats(
                benchmark_values
            )
        )

        strategy_window_stats = (
            calculate_stats(
                strategy_values
            )
        )

        triggered_window = int(
            selected_simulation.loc[
                window_df.index,
                "stop_triggered",
            ].sum()
        )

        window_rows.append(
            {
                "window": window,
                "trades": len(window_df),
                "triggered_trades": (
                    triggered_window
                ),
                "benchmark_R": (
                    benchmark_window_stats[
                        "total_R"
                    ]
                ),
                "strategy_R": (
                    strategy_window_stats[
                        "total_R"
                    ]
                ),
                "delta_R": (
                    strategy_window_stats[
                        "total_R"
                    ]
                    - benchmark_window_stats[
                        "total_R"
                    ]
                ),
                "benchmark_WR": (
                    benchmark_window_stats[
                        "win_rate"
                    ]
                ),
                "strategy_WR": (
                    strategy_window_stats[
                        "win_rate"
                    ]
                ),
                "benchmark_PF": (
                    benchmark_window_stats[
                        "profit_factor"
                    ]
                ),
                "strategy_PF": (
                    strategy_window_stats[
                        "profit_factor"
                    ]
                ),
                "benchmark_DD": (
                    benchmark_window_stats[
                        "max_drawdown_R"
                    ]
                ),
                "strategy_DD": (
                    strategy_window_stats[
                        "max_drawdown_R"
                    ]
                ),
            }
        )

    window_results = pd.DataFrame(
        window_rows
    )

    print(
        window_results.to_string(
            index=False
        )
    )

    # =========================================================================
    # TRADE-LEVEL OUTPUT
    # =========================================================================

    print()
    print("=" * 110)
    print("BUILDING TRADE-LEVEL AUDIT")
    print("=" * 110)

    trade_output = holdout_df.copy()

    trade_output[
        "benchmark_R"
    ] = trade_output[
        "final_close_R"
    ].astype(float)

    trade_output[
        "strategy_R"
    ] = selected_simulation[
        "strategy_R"
    ]

    trade_output[
        "delta_R"
    ] = (
        trade_output[
            "strategy_R"
        ]
        - trade_output[
            "benchmark_R"
        ]
    )

    trade_output[
        "stop_triggered"
    ] = selected_simulation[
        "stop_triggered"
    ]

    trade_output[
        "stop_bar"
    ] = selected_simulation[
        "stop_bar"
    ]

    trade_output[
        "stop_distance_R"
    ] = selected_stop_R

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_path = (
        OUTPUT_DIR
        / "s13_stop_boundary_development.csv"
    )

    holdout_path = (
        OUTPUT_DIR
        / "s13_stop_boundary_holdout.csv"
    )

    window_path = (
        OUTPUT_DIR
        / "s13_stop_boundary_by_window.csv"
    )

    trade_path = (
        OUTPUT_DIR
        / "s13_stop_boundary_trades.csv"
    )

    development_results.to_csv(
        development_path,
        index=False,
    )

    all_holdout_results.to_csv(
        holdout_path,
        index=False,
    )

    window_results.to_csv(
        window_path,
        index=False,
    )

    trade_output.to_csv(
        trade_path,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(development_path)
    print(holdout_path)
    print(window_path)
    print(trade_path)

    print()
    print("=" * 110)
    print("S13 STOP BOUNDARY OOS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()