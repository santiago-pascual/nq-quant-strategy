"""
S25 — FINAL MAE + RECOVERY OOS VALIDATION

Purpose
-------
Final validation of the MAE/recovery branch.

NO parameter optimization.

We compare only three rules selected from the robust S24 region:

A:
    MAE >= 0.70R
    Recovery >= +0.20R
    Deadline = 6 bars

B:
    MAE >= 0.80R
    Recovery >= +0.20R
    Deadline = 6 bars

C:
    MAE >= 0.80R
    Recovery >= +0.10R
    Deadline = 6 bars

The benchmark is the original frozen strategy.

The purpose is NOT to find another optimum.

The purpose is to determine which already-discovered rule is the
most defensible final implementation based on:

    - OOS total R
    - OOS mean R
    - OOS win rate
    - OOS drawdown
    - PF
    - temporal block consistency
    - number of positive OOS windows
    - recovery/deadline behavior

If the evidence is sufficiently close, preference is given to the
simpler / more central rule rather than chasing the highest result.

After S25, the MAE/recovery branch should be considered frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s4_adverse_recovery_enriched.csv"
)

OUTPUT_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"


# =============================================================================
# FROZEN BENCHMARK
# =============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20


# =============================================================================
# FINAL CANDIDATES
# =============================================================================

FINAL_RULES = {
    "RULE_A": {
        "mae_threshold": 0.70,
        "recovery_level": 0.20,
        "deadline": 6,
    },
    "RULE_B": {
        "mae_threshold": 0.80,
        "recovery_level": 0.20,
        "deadline": 6,
    },
    "RULE_C": {
        "mae_threshold": 0.80,
        "recovery_level": 0.10,
        "deadline": 6,
    },
}


# =============================================================================
# TEMPORAL WINDOWS
# =============================================================================

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))

HOLDOUT_BLOCKS = {
    "EARLY_OOS": [12, 13, 14, 15],
    "MID_OOS": [16, 18, 19],
    "LATE_OOS": [20, 21, 22],
}


# =============================================================================
# HELPERS
# =============================================================================


def normalize_window(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)

    text = str(value).strip()

    digits = "".join(c for c in text if c.isdigit())

    if digits:
        return int(digits)

    return np.nan


def detect_path_columns(
    df: pd.DataFrame,
) -> Tuple[List[int], List[int]]:

    close_bars = []
    mae_bars = []

    for column in df.columns:
        if column.startswith("close_") and column.endswith("R"):
            try:
                bar = int(column.split("_")[1][:-1])
                close_bars.append(bar)
            except ValueError:
                pass

        if column.startswith("mae_") and column.endswith("R"):
            try:
                bar = int(column.split("_")[1][:-1])
                mae_bars.append(bar)
            except ValueError:
                pass

    close_bars = sorted(set(close_bars))

    mae_bars = sorted(set(mae_bars))

    if not close_bars:
        raise RuntimeError("No close_N_R columns found.")

    if not mae_bars:
        raise RuntimeError("No mae_N_R columns found.")

    return close_bars, mae_bars


def profit_factor(
    values: pd.Series,
) -> float:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return 0.0

    gross_profit = values[values > 0].sum()

    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return float(gross_profit / gross_loss)


def max_drawdown(
    values: pd.Series,
) -> float:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:
        return 0.0

    equity = values.cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    return float(drawdown.min())


def metrics(
    values: pd.Series,
) -> Dict[str, float]:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }

    wins = int((values > 0).sum())

    losses = int((values <= 0).sum())

    return {
        "trades": len(values),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(values),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
    }


# =============================================================================
# MAE CROSSING
# =============================================================================


def find_mae_crossing(
    row: pd.Series,
    mae_bars: List[int],
    threshold: float,
) -> Optional[int]:

    for bar in mae_bars:
        value = row.get(
            f"mae_{bar}R",
            np.nan,
        )

        if pd.isna(value):
            continue

        if float(value) >= threshold:
            return bar

    return None


# =============================================================================
# RECOVERY CROSSING
# =============================================================================


def find_recovery_crossing(
    row: pd.Series,
    close_bars: List[int],
    mae_bar: int,
    recovery_level: float,
    deadline: int,
) -> Optional[int]:

    start_bar = mae_bar + 1

    end_bar = min(
        mae_bar + deadline,
        max(close_bars),
    )

    for bar in close_bars:
        if bar < start_bar:
            continue

        if bar > end_bar:
            break

        value = row.get(
            f"close_{bar}R",
            np.nan,
        )

        if pd.isna(value):
            continue

        if float(value) >= recovery_level:
            return bar

    return None


# =============================================================================
# EXECUTION
# =============================================================================


def execute_rule(
    row: pd.Series,
    mae_threshold: float,
    recovery_level: float,
    deadline: int,
    close_bars: List[int],
    mae_bars: List[int],
) -> Dict[str, object]:

    benchmark_R = float(row["_benchmark_R"])

    mae_bar = find_mae_crossing(
        row,
        mae_bars,
        mae_threshold,
    )

    # No MAE event:
    # preserve original trade outcome.

    if mae_bar is None:
        return {
            "strategy_R": benchmark_R,
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "exit_type": "BENCHMARK",
        }

    recovery_bar = find_recovery_crossing(
        row,
        close_bars,
        mae_bar,
        recovery_level,
        deadline,
    )

    # Recovery reached.

    if recovery_bar is not None:
        recovery_R = float(row[f"close_{recovery_bar}R"])

        return {
            "strategy_R": recovery_R,
            "mae_bar": mae_bar,
            "recovery_bar": recovery_bar,
            "exit_type": "RECOVERY_EXIT",
        }

    # Recovery not reached.
    # Exit at deadline.

    deadline_bar = min(
        mae_bar + deadline,
        max(close_bars),
    )

    deadline_R = row.get(
        f"close_{deadline_bar}R",
        np.nan,
    )

    if pd.isna(deadline_R):
        strategy_R = benchmark_R
    else:
        strategy_R = float(deadline_R)

    return {
        "strategy_R": strategy_R,
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "exit_type": "DEADLINE_EXIT",
    }


# =============================================================================
# APPLY RULE
# =============================================================================


def apply_rule(
    df: pd.DataFrame,
    rule_name: str,
    rule: Dict[str, float],
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:

    records = []

    for index, row in df.iterrows():
        result = execute_rule(
            row,
            rule["mae_threshold"],
            rule["recovery_level"],
            int(rule["deadline"]),
            close_bars,
            mae_bars,
        )

        records.append(
            {
                "original_index": index,
                "window": row["_window_numeric"],
                "rule": rule_name,
                "mae_threshold": rule["mae_threshold"],
                "recovery_level": rule["recovery_level"],
                "deadline": rule["deadline"],
                "benchmark_R": float(row["_benchmark_R"]),
                "strategy_R": float(result["strategy_R"]),
                "mae_bar": result["mae_bar"],
                "recovery_bar": result["recovery_bar"],
                "exit_type": result["exit_type"],
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(
    trades: pd.DataFrame,
) -> Dict[str, float]:

    strategy = metrics(trades["strategy_R"])

    benchmark = metrics(trades["benchmark_R"])

    triggered = trades[trades["exit_type"] != "BENCHMARK"]

    recovery = trades[trades["exit_type"] == "RECOVERY_EXIT"]

    deadline = trades[trades["exit_type"] == "DEADLINE_EXIT"]

    windows = sorted(trades["window"].dropna().unique())

    window_results = []

    for window in windows:
        w = trades[trades["window"] == window]

        if len(w) == 0:
            continue

        window_results.append(float(w["strategy_R"].sum() - w["benchmark_R"].sum()))

    positive_windows = int(sum(x > 0 for x in window_results))

    total_windows = len(window_results)

    return {
        "trades": strategy["trades"],
        "wins": strategy["wins"],
        "losses": strategy["losses"],
        "win_rate": strategy["win_rate"],
        "mean_R": strategy["mean_R"],
        "total_R": strategy["total_R"],
        "profit_factor": strategy["profit_factor"],
        "max_drawdown_R": strategy["max_drawdown_R"],
        "benchmark_total_R": benchmark["total_R"],
        "benchmark_mean_R": benchmark["mean_R"],
        "benchmark_win_rate": benchmark["win_rate"],
        "benchmark_max_drawdown_R": benchmark["max_drawdown_R"],
        "delta_R": strategy["total_R"] - benchmark["total_R"],
        "delta_mean_R": strategy["mean_R"] - benchmark["mean_R"],
        "delta_win_rate": strategy["win_rate"] - benchmark["win_rate"],
        "delta_DD": strategy["max_drawdown_R"] - benchmark["max_drawdown_R"],
        "mae_triggered": len(triggered),
        "recovery_exits": len(recovery),
        "deadline_exits": len(deadline),
        "recovery_exit_pct": (
            len(recovery) / len(triggered) if len(triggered) > 0 else np.nan
        ),
        "positive_windows": positive_windows,
        "total_windows": total_windows,
        "positive_window_pct": (
            positive_windows / total_windows if total_windows > 0 else np.nan
        ),
    }


# =============================================================================
# WINDOW ANALYSIS
# =============================================================================


def window_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    windows = sorted(trades["window"].dropna().unique())

    for window in windows:
        subset = trades[trades["window"] == window]

        benchmark = metrics(subset["benchmark_R"])

        strategy = metrics(subset["strategy_R"])

        rows.append(
            {
                "window": int(window),
                "trades": len(subset),
                "benchmark_R": benchmark["total_R"],
                "strategy_R": strategy["total_R"],
                "delta_R": strategy["total_R"] - benchmark["total_R"],
                "benchmark_WR": benchmark["win_rate"],
                "strategy_WR": strategy["win_rate"],
                "benchmark_DD": benchmark["max_drawdown_R"],
                "strategy_DD": strategy["max_drawdown_R"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TEMPORAL BLOCK ANALYSIS
# =============================================================================


def block_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for block_name, windows in HOLDOUT_BLOCKS.items():
        subset = trades[trades["window"].isin(windows)]

        if len(subset) == 0:
            continue

        benchmark = metrics(subset["benchmark_R"])

        strategy = metrics(subset["strategy_R"])

        rows.append(
            {
                "block": block_name,
                "windows": ",".join(str(x) for x in windows),
                "trades": len(subset),
                "benchmark_total_R": benchmark["total_R"],
                "strategy_total_R": strategy["total_R"],
                "delta_R": strategy["total_R"] - benchmark["total_R"],
                "benchmark_mean_R": benchmark["mean_R"],
                "strategy_mean_R": strategy["mean_R"],
                "benchmark_WR": benchmark["win_rate"],
                "strategy_WR": strategy["win_rate"],
                "benchmark_DD": benchmark["max_drawdown_R"],
                "strategy_DD": strategy["max_drawdown_R"],
                "delta_DD": strategy["max_drawdown_R"] - benchmark["max_drawdown_R"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# FINAL SCORE
# =============================================================================


def final_score(
    summary: Dict[str, float],
) -> float:
    """
    Ranking score.

    IMPORTANT:
    This is NOT used to optimize parameters.

    It is only used to organize the three already-frozen candidates.

    Priority:
        1. positive OOS R improvement
        2. drawdown improvement
        3. temporal consistency
        4. mean R
    """

    score = 0.0

    score += 1.00 * summary["delta_R"]

    score += 0.50 * summary["delta_DD"]

    score += 20.0 * summary["delta_mean_R"]

    score += 5.0 * summary["positive_window_pct"]

    return score


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S25 FINAL MAE + RECOVERY OOS VALIDATION")
    print("=" * 110)

    print()
    print("NO PARAMETER OPTIMIZATION.")

    print("Only three previously selected candidate rules are being compared.")

    print()
    print("Frozen benchmark:")

    print(f"  Stop       = {STOP_POINTS} points")

    print(f"  RR         = {RR}")

    print(f"  Horizon    = {HORIZON}")

    print()
    print("FINAL CANDIDATES")

    for name, rule in FINAL_RULES.items():
        print(
            f"  {name}: "
            f"MAE >= {rule['mae_threshold']:.2f}R | "
            f"Recovery >= {rule['recovery_level']:+.2f}R | "
            f"Deadline = {rule['deadline']} bars"
        )

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    required = [
        "final_close_R",
        "window",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    close_bars, mae_bars = detect_path_columns(df)

    print()
    print("Detected paths:")

    print(f"  MAE bars   : {len(mae_bars)}")

    print(f"  MAE range  : {min(mae_bars)} -> {max(mae_bars)}")

    print(f"  Close bars : {len(close_bars)}")

    print(f"  Close range: {min(close_bars)} -> {max(close_bars)}")

    # =========================================================================
    # PREP
    # =========================================================================

    df = df.copy()

    df["_window_numeric"] = df["window"].apply(normalize_window)

    df["_benchmark_R"] = pd.to_numeric(
        df["final_close_R"],
        errors="coerce",
    )

    df = df[df["_benchmark_R"].notna()].copy()

    # =========================================================================
    # HOLDOUT ONLY
    # =========================================================================

    holdout_df = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)].copy()

    print()
    print(f"Holdout trades: {len(holdout_df)}")

    # =========================================================================
    # BENCHMARK
    # =========================================================================

    benchmark_metrics = metrics(holdout_df["_benchmark_R"])

    print()
    print("=" * 110)
    print("BENCHMARK OOS")
    print("=" * 110)

    print(f"Trades     : {benchmark_metrics['trades']}")

    print(f"Win rate   : {benchmark_metrics['win_rate']:.4f}")

    print(f"Mean R     : {benchmark_metrics['mean_R']:.4f}")

    print(f"Total R    : {benchmark_metrics['total_R']:.4f}")

    print(f"PF         : {benchmark_metrics['profit_factor']:.4f}")

    print(f"Max DD     : {benchmark_metrics['max_drawdown_R']:.4f}")

    # =========================================================================
    # CANDIDATE TESTS
    # =========================================================================

    print()
    print("=" * 110)
    print("FINAL CANDIDATE OOS COMPARISON")
    print("=" * 110)

    all_trades = {}
    summary_rows = []

    for rule_name, rule in FINAL_RULES.items():
        trades = apply_rule(
            holdout_df,
            rule_name,
            rule,
            close_bars,
            mae_bars,
        )

        all_trades[rule_name] = trades

        summary = evaluate(trades)

        summary["rule"] = rule_name

        summary["mae_threshold"] = rule["mae_threshold"]

        summary["recovery_level"] = rule["recovery_level"]

        summary["deadline"] = rule["deadline"]

        summary["final_score"] = final_score(summary)

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    print()

    print(
        summary_df[
            [
                "rule",
                "mae_threshold",
                "recovery_level",
                "deadline",
                "trades",
                "win_rate",
                "mean_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "delta_R",
                "delta_mean_R",
                "delta_win_rate",
                "delta_DD",
                "mae_triggered",
                "recovery_exits",
                "deadline_exits",
                "positive_windows",
                "total_windows",
                "positive_window_pct",
            ]
        ]
        .sort_values(
            "delta_R",
            ascending=False,
        )
        .to_string(index=False)
    )

    # =========================================================================
    # TEMPORAL BLOCKS
    # =========================================================================

    print()
    print("=" * 110)
    print("TEMPORAL BLOCK COMPARISON")
    print("=" * 110)

    block_rows = []

    for rule_name, trades in all_trades.items():
        blocks = block_analysis(trades)

        blocks.insert(
            0,
            "rule",
            rule_name,
        )

        block_rows.append(blocks)

    block_df = pd.concat(
        block_rows,
        ignore_index=True,
    )

    print(block_df.to_string(index=False))

    # =========================================================================
    # WINDOW-BY-WINDOW
    # =========================================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    window_rows = []

    for rule_name, trades in all_trades.items():
        windows = window_analysis(trades)

        windows.insert(
            0,
            "rule",
            rule_name,
        )

        window_rows.append(windows)

    window_df = pd.concat(
        window_rows,
        ignore_index=True,
    )

    print(window_df.to_string(index=False))

    # =========================================================================
    # CONSISTENCY
    # =========================================================================

    print()
    print("=" * 110)
    print("FINAL CONSISTENCY CHECK")
    print("=" * 110)

    for rule_name in FINAL_RULES:
        trades = all_trades[rule_name]

        blocks = block_df[block_df["rule"] == rule_name]

        positive_blocks = int((blocks["delta_R"] > 0).sum())

        total_blocks = len(blocks)

        positive_windows = int(
            (window_df[window_df["rule"] == rule_name]["delta_R"] > 0).sum()
        )

        total_windows = len(window_df[window_df["rule"] == rule_name])

        summary = summary_df[summary_df["rule"] == rule_name].iloc[0]

        print()
        print(rule_name)

        print(f"  Positive OOS windows : {positive_windows}/{total_windows}")

        print(f"  Positive OOS blocks  : {positive_blocks}/{total_blocks}")

        print(f"  ΔR                   : {summary['delta_R']:.4f}")

        print(f"  ΔDD                  : {summary['delta_DD']:.4f}")

        print(f"  ΔMean R              : {summary['delta_mean_R']:.4f}")

    # =========================================================================
    # FINAL DECISION
    # =========================================================================

    ranked = summary_df.sort_values(
        [
            "delta_R",
            "delta_DD",
            "delta_mean_R",
        ],
        ascending=False,
    ).reset_index(drop=True)

    winner = ranked.iloc[0]

    winner_name = winner["rule"]

    winner_rule = FINAL_RULES[winner_name]

    print()
    print("=" * 110)
    print("FINAL S25 DECISION")
    print("=" * 110)

    print()
    print(f"Selected rule: {winner_name}")

    print(f"MAE threshold : {winner_rule['mae_threshold']:.2f}R")

    print(f"Recovery      : {winner_rule['recovery_level']:+.2f}R")

    print(f"Deadline      : {winner_rule['deadline']} bars")

    print()
    print("OOS performance:")

    print(f"  Total R     : {winner['total_R']:.4f}")

    print(f"  Mean R      : {winner['mean_R']:.4f}")

    print(f"  Win rate    : {winner['win_rate']:.4f}")

    print(f"  PF          : {winner['profit_factor']:.4f}")

    print(f"  Max DD      : {winner['max_drawdown_R']:.4f}")

    print()
    print(f"Improvement vs benchmark:")

    print(f"  ΔR          : {winner['delta_R']:.4f}")

    print(f"  ΔMean R     : {winner['delta_mean_R']:.4f}")

    print(f"  ΔWR         : {winner['delta_win_rate']:.4f}")

    print(f"  ΔDD         : {winner['delta_DD']:.4f}")

    # =========================================================================
    # FINAL FREEZE MESSAGE
    # =========================================================================

    print()
    print("=" * 110)
    print("MAE / RECOVERY BRANCH STATUS")
    print("=" * 110)

    if winner["delta_R"] > 0 and winner["delta_DD"] > 0:
        print("PASS")

        print()
        print("The selected MAE/recovery rule improves OOS total R and drawdown.")

        print()
        print("FREEZE THIS BRANCH.")

        print("Do not perform additional MAE/recovery parameter optimization.")

        print()
        print("NEXT STEP:")

        print(
            "Integrate the frozen rule into the "
            "full strategy and run the final "
            "end-to-end validation."
        )

    else:
        print("INCONCLUSIVE")

        print()
        print("The final candidate rules do not provide sufficient OOS confirmation.")

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = OUTPUT_DIR / "s25_mae_recovery_final_oos_summary.csv"

    blocks_path = OUTPUT_DIR / "s25_mae_recovery_final_oos_blocks.csv"

    windows_path = OUTPUT_DIR / "s25_mae_recovery_final_oos_windows.csv"

    trades_path = OUTPUT_DIR / "s25_mae_recovery_final_oos_trades.csv"

    decision_path = OUTPUT_DIR / "s25_mae_recovery_final_decision.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    block_df.to_csv(
        blocks_path,
        index=False,
    )

    window_df.to_csv(
        windows_path,
        index=False,
    )

    combined_trades = pd.concat(
        all_trades.values(),
        ignore_index=True,
    )

    combined_trades.to_csv(
        trades_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "selected_rule": winner_name,
                "mae_threshold": winner_rule["mae_threshold"],
                "recovery_level": winner_rule["recovery_level"],
                "deadline": winner_rule["deadline"],
                "oos_total_R": winner["total_R"],
                "oos_mean_R": winner["mean_R"],
                "oos_win_rate": winner["win_rate"],
                "oos_profit_factor": winner["profit_factor"],
                "oos_max_drawdown_R": winner["max_drawdown_R"],
                "delta_R": winner["delta_R"],
                "delta_mean_R": winner["delta_mean_R"],
                "delta_win_rate": winner["delta_win_rate"],
                "delta_DD": winner["delta_DD"],
                "positive_windows": winner["positive_windows"],
                "total_windows": winner["total_windows"],
            }
        ]
    ).to_csv(
        decision_path,
        index=False,
    )

    # =========================================================================
    # FILES
    # =========================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(summary_path)
    print(blocks_path)
    print(windows_path)
    print(trades_path)
    print(decision_path)

    print()
    print("=" * 110)
    print("S25 FINAL MAE + RECOVERY OOS VALIDATION COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
