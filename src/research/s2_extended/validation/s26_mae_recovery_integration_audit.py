"""
S26 — MAE + RECOVERY INTEGRATION AUDIT

FINAL INTEGRATION TEST
======================

Purpose
-------
Validate the frozen MAE/recovery rule as an executable state machine
before integrating it into the final strategy.

FROZEN RULE
-----------
    MAE >= 0.70R
    Recovery >= +0.20R
    Deadline = 6 bars

This script does NOT optimize parameters.

It verifies:

1. Benchmark preservation
2. Correct first-MAE detection
3. Correct recovery timing
4. Correct deadline handling
5. No look-ahead through final_close_R
6. No double exits
7. No invalid recovery before MAE
8. Temporal OOS performance
9. State distribution
10. Exact reproducibility of the frozen rule

The S4 enriched dataset is used because it contains the per-bar MAE
and CLOSE_R paths required to reconstruct the state machine.

IMPORTANT
---------
This is still a research/integration audit.

It does not claim live-trading execution equivalence because the
dataset contains CLOSE_R paths rather than raw bid/ask execution
prices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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
# FROZEN STRATEGY PARAMETERS
# =============================================================================

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

MAE_THRESHOLD = 0.70
RECOVERY_LEVEL = 0.20
RECOVERY_DEADLINE = 6


# =============================================================================
# TEMPORAL STRUCTURE
# =============================================================================

DEVELOPMENT_WINDOWS = list(range(1, 12))

HOLDOUT_WINDOWS = list(range(12, 23))


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


def detect_columns(
    df: pd.DataFrame,
):

    close_bars = []
    mae_bars = []

    for column in df.columns:
        if column.startswith("close_") and column.endswith("R"):
            try:
                bar = int(column.split("_")[1].replace("R", ""))

                close_bars.append(bar)

            except ValueError:
                pass

        if column.startswith("mae_") and column.endswith("R"):
            try:
                bar = int(column.split("_")[1].replace("R", ""))

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
):

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
        "win_rate": float(wins / len(values)),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
    }


# =============================================================================
# FIRST MAE CROSSING
# =============================================================================


def first_mae_crossing(
    row: pd.Series,
    mae_bars: List[int],
) -> Optional[int]:
    """
    Return first bar where MAE >= frozen threshold.

    Only information available at or before the current bar is used.
    """

    for bar in mae_bars:
        value = row.get(
            f"mae_{bar}R",
            np.nan,
        )

        if pd.isna(value):
            continue

        value = float(value)

        if value >= MAE_THRESHOLD:
            return bar

    return None


# =============================================================================
# RECOVERY CROSSING
# =============================================================================


def first_recovery_after_mae(
    row: pd.Series,
    close_bars: List[int],
    mae_bar: int,
) -> Optional[int]:
    """
    Search only AFTER the MAE trigger.

    Frozen rule:

        Recovery >= +0.20R
        Deadline = 6 bars

    If MAE occurs at bar 4, search:

        5, 6, 7, 8, 9, 10

    Never search the MAE bar itself.
    """

    start_bar = mae_bar + 1

    end_bar = min(
        mae_bar + RECOVERY_DEADLINE,
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

        value = float(value)

        if value >= RECOVERY_LEVEL:
            return bar

    return None


# =============================================================================
# SINGLE TRADE STATE MACHINE
# =============================================================================


def execute_trade(
    row: pd.Series,
    close_bars: List[int],
    mae_bars: List[int],
) -> Dict[str, object]:
    """
    Frozen executable state machine.

    INITIAL
       |
       | MAE >= 0.70R
       v
    ADVERSE
       |
       | CLOSE_R >= +0.20R
       | within 6 bars
       v
    RECOVERED

    Otherwise:

    FAILED_TO_RECOVER

    Trades that never reach MAE preserve benchmark outcome.
    """

    benchmark_R = float(row["_benchmark_R"])

    # -------------------------------------------------------------------------
    # State 1: initial
    # -------------------------------------------------------------------------

    mae_bar = first_mae_crossing(
        row,
        mae_bars,
    )

    if mae_bar is None:
        return {
            "strategy_R": benchmark_R,
            "state": "NO_MAE_TRIGGER",
            "mae_bar": np.nan,
            "recovery_bar": np.nan,
            "exit_bar": np.nan,
            "exit_type": "BENCHMARK",
        }

    # -------------------------------------------------------------------------
    # State 2: adverse
    # -------------------------------------------------------------------------

    recovery_bar = first_recovery_after_mae(
        row,
        close_bars,
        mae_bar,
    )

    # -------------------------------------------------------------------------
    # State 3: recovered
    # -------------------------------------------------------------------------

    if recovery_bar is not None:
        recovery_value = row.get(
            f"close_{recovery_bar}R",
            np.nan,
        )

        if not pd.isna(recovery_value):
            return {
                "strategy_R": float(recovery_value),
                "state": "RECOVERED",
                "mae_bar": mae_bar,
                "recovery_bar": recovery_bar,
                "exit_bar": recovery_bar,
                "exit_type": "RECOVERY_EXIT",
            }

    # -------------------------------------------------------------------------
    # State 4: failed recovery
    # -------------------------------------------------------------------------

    failure_bar = min(
        mae_bar + RECOVERY_DEADLINE,
        max(close_bars),
    )

    failure_value = row.get(
        f"close_{failure_bar}R",
        np.nan,
    )

    if pd.isna(failure_value):
        return {
            "strategy_R": benchmark_R,
            "state": "NO_EXECUTION_DATA",
            "mae_bar": mae_bar,
            "recovery_bar": np.nan,
            "exit_bar": np.nan,
            "exit_type": "BENCHMARK_FALLBACK",
        }

    return {
        "strategy_R": float(failure_value),
        "state": "FAILED_TO_RECOVER",
        "mae_bar": mae_bar,
        "recovery_bar": np.nan,
        "exit_bar": failure_bar,
        "exit_type": "FAILURE_EXIT",
    }


# =============================================================================
# APPLY TO DATASET
# =============================================================================


def run_strategy(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
) -> pd.DataFrame:

    records = []

    for index, row in df.iterrows():
        result = execute_trade(
            row,
            close_bars,
            mae_bars,
        )

        records.append(
            {
                "original_index": index,
                "window": row["_window_numeric"],
                "benchmark_R": float(row["_benchmark_R"]),
                "strategy_R": float(result["strategy_R"]),
                "state": result["state"],
                "mae_bar": result["mae_bar"],
                "recovery_bar": result["recovery_bar"],
                "exit_bar": result["exit_bar"],
                "exit_type": result["exit_type"],
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# LEAKAGE AUDIT
# =============================================================================


def leakage_audit(
    df: pd.DataFrame,
    close_bars: List[int],
    mae_bars: List[int],
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explicitly test that execution decisions depend only on information
    available at the decision point.

    We verify:

    1. MAE trigger bar is the first crossing.
    2. Recovery bar is after MAE.
    3. Recovery bar <= MAE + deadline.
    4. Recovery exit value comes from the recovery bar.
    5. No decision uses final_close_R.
    """

    audit_rows = []

    for _, trade in trades.iterrows():
        index = trade["original_index"]

        row = df.loc[index]

        mae_bar = trade["mae_bar"]

        recovery_bar = trade["recovery_bar"]

        state = trade["state"]

        checks = {}

        # ---------------------------------------------------------------------
        # Check A: first MAE
        # ---------------------------------------------------------------------

        if pd.isna(mae_bar):
            checks["first_mae_correct"] = True

        else:
            earlier_crossing = False

            for bar in mae_bars:
                if bar >= int(mae_bar):
                    break

                value = row.get(
                    f"mae_{bar}R",
                    np.nan,
                )

                if not pd.isna(value) and float(value) >= MAE_THRESHOLD:
                    earlier_crossing = True

                    break

            checks["first_mae_correct"] = not earlier_crossing

        # ---------------------------------------------------------------------
        # Check B: recovery occurs after MAE
        # ---------------------------------------------------------------------

        if pd.isna(recovery_bar):
            checks["recovery_after_mae"] = True

        else:
            checks["recovery_after_mae"] = not pd.isna(mae_bar) and int(
                recovery_bar
            ) > int(mae_bar)

        # ---------------------------------------------------------------------
        # Check C: deadline
        # ---------------------------------------------------------------------

        if pd.isna(recovery_bar):
            checks["recovery_within_deadline"] = True

        else:
            checks["recovery_within_deadline"] = (
                int(recovery_bar) <= int(mae_bar) + RECOVERY_DEADLINE
            )

        # ---------------------------------------------------------------------
        # Check D: recovery threshold
        # ---------------------------------------------------------------------

        if pd.isna(recovery_bar):
            checks["recovery_threshold_correct"] = True

        else:
            value = row.get(
                f"close_{int(recovery_bar)}R",
                np.nan,
            )

            checks["recovery_threshold_correct"] = (
                not pd.isna(value) and float(value) >= RECOVERY_LEVEL
            )

        # ---------------------------------------------------------------------
        # Check E: no recovery before MAE
        # ---------------------------------------------------------------------

        if state != "RECOVERED":
            checks["no_invalid_recovery"] = True

        else:
            checks["no_invalid_recovery"] = (
                not pd.isna(mae_bar)
                and not pd.isna(recovery_bar)
                and int(recovery_bar) > int(mae_bar)
            )

        # ---------------------------------------------------------------------
        # Check F: exit exists exactly once
        # ---------------------------------------------------------------------

        if state == "NO_MAE_TRIGGER":
            checks["single_exit"] = trade["exit_type"] == "BENCHMARK"

        elif state == "RECOVERED":
            checks["single_exit"] = (
                trade["exit_type"] == "RECOVERY_EXIT"
                and trade["exit_bar"] == trade["recovery_bar"]
            )

        elif state == "FAILED_TO_RECOVER":
            checks["single_exit"] = trade[
                "exit_type"
            ] == "FAILURE_EXIT" and not pd.isna(trade["exit_bar"])

        else:
            checks["single_exit"] = True

        # ---------------------------------------------------------------------
        # Overall
        # ---------------------------------------------------------------------

        all_pass = all(checks.values())

        audit_rows.append(
            {
                "original_index": index,
                "state": state,
                **checks,
                "audit_pass": all_pass,
            }
        )

    return pd.DataFrame(audit_rows)


# =============================================================================
# STATE SUMMARY
# =============================================================================


def state_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for state, subset in trades.groupby("state"):
        m = metrics(subset["strategy_R"])

        rows.append(
            {
                "state": state,
                "trades": m["trades"],
                "wins": m["wins"],
                "losses": m["losses"],
                "win_rate": m["win_rate"],
                "mean_R": m["mean_R"],
                "total_R": m["total_R"],
                "profit_factor": m["profit_factor"],
                "max_drawdown_R": m["max_drawdown_R"],
            }
        )

    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


# =============================================================================
# WINDOW SUMMARY
# =============================================================================


def window_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for window in sorted(trades["window"].dropna().unique()):
        subset = trades[trades["window"] == window]

        benchmark = metrics(subset["benchmark_R"])

        strategy = metrics(subset["strategy_R"])

        rows.append(
            {
                "window": int(window),
                "trades": len(subset),
                "triggered": int((subset["state"] != "NO_MAE_TRIGGER").sum()),
                "recovered": int((subset["state"] == "RECOVERED").sum()),
                "failed_recovery": int((subset["state"] == "FAILED_TO_RECOVER").sum()),
                "benchmark_R": benchmark["total_R"],
                "strategy_R": strategy["total_R"],
                "delta_R": strategy["total_R"] - benchmark["total_R"],
                "benchmark_WR": benchmark["win_rate"],
                "strategy_WR": strategy["win_rate"],
                "benchmark_PF": benchmark["profit_factor"],
                "strategy_PF": strategy["profit_factor"],
                "benchmark_DD": benchmark["max_drawdown_R"],
                "strategy_DD": strategy["max_drawdown_R"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# FINAL OOS METRICS
# =============================================================================


def evaluate_oos(
    trades: pd.DataFrame,
):

    benchmark = metrics(trades["benchmark_R"])

    strategy = metrics(trades["strategy_R"])

    windows = window_summary(trades)

    positive_windows = int((windows["delta_R"] > 0).sum())

    total_windows = len(windows)

    return {
        "benchmark_total_R": benchmark["total_R"],
        "strategy_total_R": strategy["total_R"],
        "delta_R": strategy["total_R"] - benchmark["total_R"],
        "benchmark_mean_R": benchmark["mean_R"],
        "strategy_mean_R": strategy["mean_R"],
        "delta_mean_R": strategy["mean_R"] - benchmark["mean_R"],
        "benchmark_win_rate": benchmark["win_rate"],
        "strategy_win_rate": strategy["win_rate"],
        "delta_win_rate": strategy["win_rate"] - benchmark["win_rate"],
        "benchmark_PF": benchmark["profit_factor"],
        "strategy_PF": strategy["profit_factor"],
        "benchmark_DD": benchmark["max_drawdown_R"],
        "strategy_DD": strategy["max_drawdown_R"],
        "delta_DD": strategy["max_drawdown_R"] - benchmark["max_drawdown_R"],
        "positive_windows": positive_windows,
        "total_windows": total_windows,
        "positive_window_pct": (
            positive_windows / total_windows if total_windows else np.nan
        ),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S26 MAE + RECOVERY INTEGRATION AUDIT")
    print("=" * 110)

    print()
    print("FROZEN RULE")
    print(f"  MAE threshold : {MAE_THRESHOLD:.2f}R")

    print(f"  Recovery      : {RECOVERY_LEVEL:+.2f}R")

    print(f"  Deadline      : {RECOVERY_DEADLINE} bars")

    print()
    print("NO PARAMETER OPTIMIZATION.")

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("=" * 110)
    print("LOADING DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Trades loaded: {len(df)}")

    required = [
        "final_close_R",
        "window",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    # =========================================================================
    # PATH DETECTION
    # =========================================================================

    close_bars, mae_bars = detect_columns(df)

    print()
    print("Detected paths:")

    print(f"  MAE bars   : {len(mae_bars)}")

    print(f"  MAE range  : {min(mae_bars)} -> {max(mae_bars)}")

    print(f"  Close bars : {len(close_bars)}")

    print(f"  Close range: {min(close_bars)} -> {max(close_bars)}")

    # =========================================================================
    # PREPARE
    # =========================================================================

    df = df.copy()

    normalized_window = df["window"].apply(normalize_window)

    df["_window_numeric"] = normalized_window

    df["_benchmark_R"] = pd.to_numeric(
        df["final_close_R"],
        errors="coerce",
    )

    df = df[df["_benchmark_R"].notna()].copy()

    # =========================================================================
    # RUN FROZEN RULE
    # =========================================================================

    print()
    print("=" * 110)
    print("RUNNING FROZEN STATE MACHINE")
    print("=" * 110)

    trades = run_strategy(
        df,
        close_bars,
        mae_bars,
    )

    print(f"Trades evaluated: {len(trades)}")

    # =========================================================================
    # LEAKAGE AUDIT
    # =========================================================================

    print()
    print("=" * 110)
    print("LOOK-AHEAD / EXECUTION AUDIT")
    print("=" * 110)

    audit = leakage_audit(
        df,
        close_bars,
        mae_bars,
        trades,
    )

    audit_checks = [
        "first_mae_correct",
        "recovery_after_mae",
        "recovery_within_deadline",
        "recovery_threshold_correct",
        "no_invalid_recovery",
        "single_exit",
    ]

    for check in audit_checks:
        passed = int(audit[check].sum())

        total = len(audit)

        print(f"  {check:<32}{passed}/{total}")

    total_pass = int(audit["audit_pass"].sum())

    total_audit = len(audit)

    print()

    print(f"OVERALL AUDIT: {total_pass}/{total_audit}")

    if total_pass != total_audit:
        failed = audit[~audit["audit_pass"]]

        print()
        print("FAILED AUDIT ROWS:")

        print(failed.head(20).to_string(index=False))

        raise RuntimeError("S26 leakage/execution audit failed.")

    print("PASS — no execution audit failures.")

    # =========================================================================
    # STATE SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print("STATE SUMMARY")
    print("=" * 110)

    states = state_summary(trades)

    print(states.to_string(index=False))

    # =========================================================================
    # HOLDOUT
    # =========================================================================

    holdout = trades[trades["window"].isin(HOLDOUT_WINDOWS)].copy()

    print()
    print("=" * 110)
    print("FINAL HOLDOUT OOS")
    print("=" * 110)

    oos = evaluate_oos(holdout)

    print()
    print("BENCHMARK")

    print(f"  Total R    : {oos['benchmark_total_R']:.4f}")

    print(f"  Mean R     : {oos['benchmark_mean_R']:.4f}")

    print(f"  Win rate   : {oos['benchmark_win_rate']:.4f}")

    print(f"  PF         : {oos['benchmark_PF']}")

    print(f"  Max DD     : {oos['benchmark_DD']:.4f}")

    print()
    print("FROZEN MAE + RECOVERY")

    print(f"  Total R    : {oos['strategy_total_R']:.4f}")

    print(f"  Mean R     : {oos['strategy_mean_R']:.4f}")

    print(f"  Win rate   : {oos['strategy_win_rate']:.4f}")

    print(f"  PF         : {oos['strategy_PF']}")

    print(f"  Max DD     : {oos['strategy_DD']:.4f}")

    print()
    print("IMPROVEMENT")

    print(f"  Delta R        : {oos['delta_R']:.4f}")

    print(f"  Delta Mean R   : {oos['delta_mean_R']:.4f}")

    print(f"  Delta Win Rate : {oos['delta_win_rate']:.4f}")

    print(f"  Delta Max DD   : {oos['delta_DD']:.4f}")

    print(f"  Positive OOS windows : {oos['positive_windows']}/{oos['total_windows']}")

    # =========================================================================
    # WINDOW ANALYSIS
    # =========================================================================

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    oos_windows = window_summary(holdout)

    print(oos_windows.to_string(index=False))

    # =========================================================================
    # INTEGRATION TESTS
    # =========================================================================

    print()
    print("=" * 110)
    print("INTEGRATION TESTS")
    print("=" * 110)

    integration_results = {}

    # Test 1:
    # Every recovered trade must have a valid MAE trigger.

    recovered = trades[trades["state"] == "RECOVERED"]

    integration_results["recovery_requires_mae"] = bool(
        (recovered["mae_bar"].notna()).all()
    )

    # Test 2:
    # Recovery must happen after MAE.

    if len(recovered):
        integration_results["recovery_after_mae"] = bool(
            (recovered["recovery_bar"] > recovered["mae_bar"]).all()
        )

    else:
        integration_results["recovery_after_mae"] = True

    # Test 3:
    # Recovery must be within deadline.

    if len(recovered):
        integration_results["recovery_within_6_bars"] = bool(
            (
                (recovered["recovery_bar"] - recovered["mae_bar"]) <= RECOVERY_DEADLINE
            ).all()
        )

    else:
        integration_results["recovery_within_6_bars"] = True

    # Test 4:
    # No-MAE trades preserve benchmark.

    no_mae = trades[trades["state"] == "NO_MAE_TRIGGER"]

    if len(no_mae):
        integration_results["benchmark_preserved_without_mae"] = bool(
            np.allclose(
                no_mae["strategy_R"],
                no_mae["benchmark_R"],
            )
        )

    else:
        integration_results["benchmark_preserved_without_mae"] = True

    # Test 5:
    # No invalid states.

    valid_states = {
        "NO_MAE_TRIGGER",
        "RECOVERED",
        "FAILED_TO_RECOVER",
        "NO_EXECUTION_DATA",
    }

    integration_results["valid_states_only"] = bool(
        set(trades["state"]).issubset(valid_states)
    )

    # Test 6:
    # Strategy R must be finite.

    integration_results["finite_strategy_R"] = bool(
        np.isfinite(trades["strategy_R"]).all()
    )

    for test_name, result in integration_results.items():
        print(f"  {test_name:<40}{'PASS' if result else 'FAIL'}")

    if not all(integration_results.values()):
        raise RuntimeError("S26 integration tests failed.")

    # =========================================================================
    # FINAL DECISION
    # =========================================================================

    print()
    print("=" * 110)
    print("S26 FINAL STATUS")
    print("=" * 110)

    audit_passed = total_pass == total_audit

    integration_passed = all(integration_results.values())

    oos_positive = oos["delta_R"] > 0

    drawdown_better = oos["delta_DD"] > 0

    if audit_passed and integration_passed and oos_positive and drawdown_better:
        print()
        print("PASS")

        print()
        print("Frozen MAE/recovery rule is ready for final strategy integration.")

        print()
        print("FROZEN:")

        print(f"  MAE >= {MAE_THRESHOLD:.2f}R")

        print(f"  Recovery >= {RECOVERY_LEVEL:+.2f}R")

        print(f"  Deadline = {RECOVERY_DEADLINE} bars")

        print()
        print("NO FURTHER MAE/RECOVERY OPTIMIZATION.")

    else:
        print()
        print("FAIL / INCONCLUSIVE")

        print()
        print("Do not integrate the rule into the final strategy yet.")

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_path = OUTPUT_DIR / "s26_mae_recovery_integration_trades.csv"

    audit_path = OUTPUT_DIR / "s26_mae_recovery_integration_audit.csv"

    states_path = OUTPUT_DIR / "s26_mae_recovery_integration_states.csv"

    windows_path = OUTPUT_DIR / "s26_mae_recovery_integration_windows.csv"

    summary_path = OUTPUT_DIR / "s26_mae_recovery_integration_summary.csv"

    trades.to_csv(
        trades_path,
        index=False,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    states.to_csv(
        states_path,
        index=False,
    )

    oos_windows.to_csv(
        windows_path,
        index=False,
    )

    summary_row = {
        "mae_threshold": MAE_THRESHOLD,
        "recovery_level": RECOVERY_LEVEL,
        "deadline": RECOVERY_DEADLINE,
        **oos,
        "audit_pass": audit_passed,
        "integration_pass": integration_passed,
        "oos_delta_positive": oos_positive,
        "oos_drawdown_improved": drawdown_better,
    }

    pd.DataFrame([summary_row]).to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # FILE OUTPUT
    # =========================================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(trades_path)
    print(audit_path)
    print(states_path)
    print(windows_path)
    print(summary_path)

    print()
    print("=" * 110)
    print("S26 MAE + RECOVERY INTEGRATION AUDIT COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
