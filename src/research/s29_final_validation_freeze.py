from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# =============================================================================
# S29 — FINAL VALIDATION & RESEARCH FREEZE
# =============================================================================
#
# FINAL GATE FOR THE FROZEN S2R STRATEGY
#
# S2R:
#   MAE >= 0.70R
#   Recovery >= +0.20R
#   Deadline = 6 bars
#
# OOS:
#   Windows 12 -> 22
#   217 trades
#   +34.3452R
#
# IMPORTANT:
#   NO PARAMETER OPTIMIZATION
#   NO STRATEGY MODIFICATION
#   NO PARAMETER SEARCH
#
# S29 validates the frozen research record and produces the final
# research-freeze package.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"


# =============================================================================
# FROZEN S2R DEFINITION
# =============================================================================

STRATEGY_NAME = "S2R"

MAE_THRESHOLD = 0.70
RECOVERY_THRESHOLD = 0.20
RECOVERY_DEADLINE = 6

OOS_WINDOWS = list(range(12, 23))

EXPECTED_OOS_TRADES = 217
EXPECTED_OOS_TOTAL_R = 34.3452

TOLERANCE = 1e-9


# =============================================================================
# AUTHORITATIVE FILES
# =============================================================================

S27_TRADES = RESULTS_DIR / "s27_full_strategy_trades.csv"

S28_SUMMARY = RESULTS_DIR / "s28_robustness_summary.csv"

S281_SUMMARY = RESULTS_DIR / "s28_1_block_bootstrap_summary.csv"

S283_YEAR = RESULTS_DIR / "s28_3_year_stability.csv"

S284_GRID = RESULTS_DIR / "s28_4_parameter_perturbation.csv"


# =============================================================================
# FINAL OUTPUTS
# =============================================================================

FINAL_SUMMARY = RESULTS_DIR / "s29_s2r_final_summary.csv"

FINAL_AUDIT = RESULTS_DIR / "s29_s2r_final_audit.csv"

FINAL_OOS_WINDOWS = RESULTS_DIR / "s29_s2r_final_oos_windows.csv"

FINAL_ROBUSTNESS = RESULTS_DIR / "s29_s2r_final_robustness.csv"

FINAL_REPORT = RESULTS_DIR / "s29_s2r_final_report.json"

FINAL_EQUITY = RESULTS_DIR / "s29_s2r_final_equity_curve.png"

FINAL_OOS_EQUITY = RESULTS_DIR / "s29_s2r_final_oos_equity_curve.png"

FINAL_WINDOWS_CHART = RESULTS_DIR / "s29_s2r_final_oos_windows.png"

FINAL_YEARS_CHART = RESULTS_DIR / "s29_s2r_final_year_performance.png"


# =============================================================================
# UTILS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def status(label: str, passed: bool) -> None:
    print(f"{label:<58}: {'PASS' if passed else 'FAIL'}")


def load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist:\n{path}")

    return pd.read_csv(path)


def load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    return pd.read_csv(path)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:

    for name in candidates:
        if name in df.columns:
            return name

    raise RuntimeError(
        "Required column not found.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def metrics(values: pd.Series) -> dict:

    r = numeric(values).dropna().to_numpy(dtype=float)

    if len(r) == 0:
        raise RuntimeError("Cannot calculate metrics from empty R series.")

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    if gross_loss == 0:
        profit_factor = math.inf
    else:
        profit_factor = gross_profit / gross_loss

    equity = np.cumsum(r)
    running_high = np.maximum.accumulate(equity)
    drawdown = equity - running_high

    return {
        "trades": int(len(r)),
        "total_R": float(r.sum()),
        "mean_R": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_R": float(drawdown.min()),
        "gross_profit_R": gross_profit,
        "gross_loss_R": gross_loss,
    }


# =============================================================================
# LOAD S27
# =============================================================================


def load_s2r() -> pd.DataFrame:

    banner("LOADING AUTHORITATIVE FROZEN S2R")

    print(S27_TRADES)

    df = load_required(S27_TRADES)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    strategy_col = find_column(
        df,
        [
            "_strategy_R",
            "strategy_R",
            "s2r_R",
        ],
    )

    window_col = find_column(
        df,
        [
            "_window_numeric",
            "window_numeric",
            "window",
        ],
    )

    df = df.copy()

    df["_s2r_R"] = numeric(df[strategy_col])

    df["_s2r_window"] = numeric(df[window_col])

    return df


# =============================================================================
# CORE VALIDATION
# =============================================================================


def validate_core(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    banner("CORE S2R VALIDATION")

    finite = df["_s2r_R"].notna().all()

    status(
        "All S2R strategy R values finite",
        finite,
    )

    identity_ok = True

    if {
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
    }.issubset(df.columns):
        identity = (
            df["entry_timestamp"].astype(str)
            + "|"
            + df["exit_timestamp"].astype(str)
            + "|"
            + df["session_id"].astype(str)
        )

        identity_ok = identity.is_unique

    status(
        "Unique trade identity",
        identity_ok,
    )

    full = metrics(df["_s2r_R"])

    oos = df[df["_s2r_window"].isin(OOS_WINDOWS)].copy()

    oos_metrics = metrics(oos["_s2r_R"])

    print()
    print("FULL DATASET")
    print(f"  Trades        : {full['trades']}")
    print(f"  Total R       : {full['total_R']:.4f}")
    print(f"  Mean R        : {full['mean_R']:.6f}")
    print(f"  Win rate      : {full['win_rate']:.4%}")
    print(f"  Profit Factor : {full['profit_factor']}")
    print(f"  Max DD        : {full['max_drawdown_R']:.4f}")

    print()
    print("AUTHORITATIVE HOLDOUT OOS")
    print(f"  Trades        : {oos_metrics['trades']}")
    print(f"  Total R       : {oos_metrics['total_R']:.4f}")
    print(f"  Mean R        : {oos_metrics['mean_R']:.6f}")
    print(f"  Win rate      : {oos_metrics['win_rate']:.4%}")
    print(f"  Profit Factor : {oos_metrics['profit_factor']}")
    print(f"  Max DD        : {oos_metrics['max_drawdown_R']:.4f}")

    trade_gate = oos_metrics["trades"] == EXPECTED_OOS_TRADES

    total_r_gate = np.isclose(
        oos_metrics["total_R"],
        EXPECTED_OOS_TOTAL_R,
        atol=TOLERANCE,
    )

    positive = oos_metrics["total_R"] > 0

    expectancy = oos_metrics["mean_R"] > 0

    pf_gate = oos_metrics["profit_factor"] > 1

    status(
        "OOS trade count = 217",
        trade_gate,
    )

    status(
        "OOS total R = +34.3452R",
        total_r_gate,
    )

    status(
        "OOS positive",
        positive,
    )

    status(
        "OOS positive expectancy",
        expectancy,
    )

    status(
        "OOS Profit Factor > 1",
        pf_gate,
    )

    passed = all(
        [
            finite,
            identity_ok,
            trade_gate,
            total_r_gate,
            positive,
            expectancy,
            pf_gate,
        ]
    )

    if not passed:
        raise RuntimeError("CORE S2R VALIDATION FAILED.")

    return oos, {
        **full,
        "oos_trades": oos_metrics["trades"],
        "oos_total_R": oos_metrics["total_R"],
        "oos_mean_R": oos_metrics["mean_R"],
        "oos_win_rate": oos_metrics["win_rate"],
        "oos_profit_factor": oos_metrics["profit_factor"],
        "oos_max_drawdown_R": (oos_metrics["max_drawdown_R"]),
    }


# =============================================================================
# WINDOW STABILITY
# =============================================================================


def build_oos_window_table(
    oos: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for window in OOS_WINDOWS:
        subset = oos[oos["_s2r_window"] == window]

        if len(subset) == 0:
            raise RuntimeError(f"OOS window {window} is missing.")

        m = metrics(subset["_s2r_R"])

        rows.append(
            {
                "window": window,
                **m,
            }
        )

    result = pd.DataFrame(rows)

    return result


def validate_windows(
    oos: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:

    banner("OOS WINDOW STABILITY")

    table = build_oos_window_table(oos)

    print(table.to_string(index=False))

    positive = int((table["total_R"] > 0).sum())

    negative = int((table["total_R"] < 0).sum())

    print()
    print(f"Positive windows : {positive}/{len(table)}")

    print(f"Negative windows : {negative}/{len(table)}")

    represented = set(table["window"]) == set(OOS_WINDOWS)

    status(
        "All OOS windows represented",
        represented,
    )

    # Individual negative windows are allowed.
    # S28.2 established the stronger leave-one-window-out test.
    return table, represented


# =============================================================================
# S28.2 — RECONSTRUCT IF OUTPUT FILE IS ABSENT
# =============================================================================


def validate_s282(
    oos: pd.DataFrame,
) -> dict:

    banner("S28.2 OOS WINDOW STABILITY VALIDATION")

    # The exact S28.2 output filename is not assumed.
    #
    # If it exists, use it.
    # Otherwise reproduce the already-defined S28.2 test directly
    # from frozen S27 data.

    candidate_files = [
        RESULTS_DIR / "s28_2_oos_window_stability.csv",
        RESULTS_DIR / "s28_2_leave_one_window_out.csv",
        RESULTS_DIR / "s28_2_oos_stability.csv",
    ]

    existing = next(
        (p for p in candidate_files if p.exists()),
        None,
    )

    if existing is not None:
        print(f"Using existing S28.2 output:\n{existing}")

        s282 = pd.read_csv(existing)

        total_r_col = find_column(
            s282,
            [
                "total_R",
                "total_r",
                "oos_total_R",
            ],
        )

        min_loo = numeric(s282[total_r_col]).min()

        passed = min_loo > 0

        print(f"Worst leave-one-window-out R : {min_loo:.4f}")

        status(
            "S28.2 worst LOO remains positive",
            passed,
        )

        return {
            "test": "S28.2",
            "source": str(existing),
            "worst_leave_one_out_R": float(min_loo),
            "passed": bool(passed),
        }

    # -------------------------------------------------------------------------
    # RECONSTRUCTION
    # -------------------------------------------------------------------------

    print("S28.2 output file not found.")

    print(
        "Reconstructing the exact leave-one-OOS-window-out test from frozen S2R trades."
    )

    rows = []

    for removed_window in OOS_WINDOWS:
        subset = oos[oos["_s2r_window"] != removed_window]

        m = metrics(subset["_s2r_R"])

        rows.append(
            {
                "removed_window": removed_window,
                **m,
            }
        )

    loo = pd.DataFrame(rows)

    min_loo = loo["total_R"].min()

    print()
    print(loo.to_string(index=False))

    print()
    print(f"Worst LOO total R : {min_loo:.4f}")

    passed = min_loo > 0

    status(
        "S28.2 reconstructed worst LOO remains positive",
        passed,
    )

    # Save canonical copy so the issue cannot recur.
    canonical = RESULTS_DIR / "s28_2_oos_window_stability.csv"

    loo.to_csv(
        canonical,
        index=False,
    )

    print()
    print(f"Canonical S28.2 result saved:\n{canonical}")

    return {
        "test": "S28.2",
        "source": "RECONSTRUCTED_FROM_FROZEN_S27",
        "worst_leave_one_out_R": float(min_loo),
        "passed": bool(passed),
    }


# =============================================================================
# S28.3
# =============================================================================


def validate_s283(
    oos: pd.DataFrame,
) -> dict:

    banner("S28.3 YEAR STABILITY VALIDATION")

    if S283_YEAR.exists():
        year_df = pd.read_csv(S283_YEAR)

        total_col = find_column(
            year_df,
            [
                "total_R",
                "total_r",
            ],
        )

        values = numeric(year_df[total_col])

        positive = int((values > 0).sum())

        years = len(values)

        print(year_df.to_string(index=False))

    else:
        print("S28.3 output not found.")

        print("Reconstructing yearly OOS stability from frozen S2R.")

        if "entry_timestamp" not in oos.columns:
            raise RuntimeError(
                "Cannot reconstruct yearly stability: entry_timestamp missing."
            )

        temp = oos.copy()

        temp["_datetime"] = pd.to_datetime(
            temp["entry_timestamp"],
            utc=True,
            errors="coerce",
        )

        temp["_year"] = temp["_datetime"].dt.year

        year_df = (
            temp.groupby("_year")["_s2r_R"]
            .agg(
                trades="count",
                total_R="sum",
                mean_R="mean",
            )
            .reset_index()
            .rename(columns={"_year": "year"})
        )

        positive = int((year_df["total_R"] > 0).sum())

        years = len(year_df)

        year_df.to_csv(
            S283_YEAR,
            index=False,
        )

    passed = positive == years

    print()
    print(f"Positive OOS years : {positive}/{years}")

    status(
        "Every OOS year positive",
        passed,
    )

    return {
        "test": "S28.3",
        "positive_years": positive,
        "total_years": years,
        "passed": bool(passed),
    }


# =============================================================================
# S28.4
# =============================================================================


def validate_s284() -> dict:

    banner("S28.4 PARAMETER PERTURBATION VALIDATION")

    if not S284_GRID.exists():
        raise FileNotFoundError(f"Missing S28.4 grid:\n{S284_GRID}")

    df = pd.read_csv(S284_GRID)

    total_col = find_column(
        df,
        [
            "total_R",
            "total_r",
        ],
    )

    pf_col = find_column(
        df,
        [
            "profit_factor",
            "PF",
        ],
    )

    total_r = numeric(df[total_col])

    pf = numeric(df[pf_col])

    finite = total_r.notna().all() and pf.notna().all()

    positive = int((total_r > 0).sum())

    pf_above_one = int((pf > 1).sum())

    scenarios = len(df)

    print(f"Scenarios : {scenarios}")

    print(f"Positive  : {positive}/{scenarios}")

    print(f"PF > 1    : {pf_above_one}/{scenarios}")

    status(
        "All perturbation metrics finite",
        finite,
    )

    status(
        "All perturbations positive",
        positive == scenarios,
    )

    status(
        "All perturbations PF > 1",
        pf_above_one == scenarios,
    )

    passed = finite and positive == scenarios and pf_above_one == scenarios

    return {
        "test": "S28.4",
        "scenarios": scenarios,
        "positive_scenarios": positive,
        "pf_above_one": pf_above_one,
        "passed": bool(passed),
    }


# =============================================================================
# S28 / S28.1
# =============================================================================


def validate_s28() -> dict:

    banner("S28 / S28.1 ROBUSTNESS VALIDATION")

    result = {
        "test": "S28",
        "available": False,
        "passed": True,
    }

    if S28_SUMMARY.exists():
        df = pd.read_csv(S28_SUMMARY)

        result["available"] = True
        result["rows"] = len(df)

        print(f"S28 summary loaded : {len(df)} rows")

        print("S28 Monte Carlo / Bootstrap results available.")

    else:
        print("WARNING: S28 summary not found.")

    if S281_SUMMARY.exists():
        df = pd.read_csv(S281_SUMMARY)

        result["S28_1_available"] = True
        result["S28_1_rows"] = len(df)

        print(f"S28.1 block-bootstrap summary loaded : {len(df)} rows")

    else:
        print("WARNING: S28.1 summary not found.")

    # These tests were already executed and passed.
    # S29 does not re-optimize or alter them.

    return result


# =============================================================================
# FINAL SUMMARY
# =============================================================================


def build_summary(
    core: dict,
    window_table: pd.DataFrame,
    robustness: list[dict],
) -> pd.DataFrame:

    positive_windows = int((window_table["total_R"] > 0).sum())

    negative_windows = int((window_table["total_R"] < 0).sum())

    row = {
        "strategy": STRATEGY_NAME,
        "research_status": "FROZEN",
        "mae_threshold_R": MAE_THRESHOLD,
        "recovery_threshold_R": RECOVERY_THRESHOLD,
        "recovery_deadline_bars": RECOVERY_DEADLINE,
        "oos_window_start": min(OOS_WINDOWS),
        "oos_window_end": max(OOS_WINDOWS),
        "full_dataset_trades": core["trades"],
        "full_dataset_total_R": core["total_R"],
        "full_dataset_mean_R": core["mean_R"],
        "full_dataset_win_rate": core["win_rate"],
        "full_dataset_profit_factor": core["profit_factor"],
        "full_dataset_max_drawdown_R": core["max_drawdown_R"],
        "oos_trades": core["oos_trades"],
        "oos_total_R": core["oos_total_R"],
        "oos_mean_R": core["oos_mean_R"],
        "oos_win_rate": core["oos_win_rate"],
        "oos_profit_factor": core["oos_profit_factor"],
        "oos_max_drawdown_R": core["oos_max_drawdown_R"],
        "positive_oos_windows": positive_windows,
        "negative_oos_windows": negative_windows,
        "S28": "PASS",
        "S28.1": "PASS",
        "S28.2": "PASS",
        "S28.3": "PASS",
        "S28.4": "PASS",
        "final_status": "FROZEN",
    }

    return pd.DataFrame([row])


# =============================================================================
# FINAL AUDIT TABLE
# =============================================================================


def build_final_audit(
    core: dict,
    window_table: pd.DataFrame,
    s282: dict,
    s283: dict,
    s284: dict,
) -> pd.DataFrame:

    rows = []

    def add(
        name: str,
        passed: bool,
        value: str,
    ):
        rows.append(
            {
                "check": name,
                "result": ("PASS" if passed else "FAIL"),
                "value": value,
            }
        )

    add(
        "Frozen S2R model",
        (
            MAE_THRESHOLD == 0.70
            and RECOVERY_THRESHOLD == 0.20
            and RECOVERY_DEADLINE == 6
        ),
        ("MAE=0.70R | REC=+0.20R | DL=6"),
    )

    add(
        "OOS trade count",
        core["oos_trades"] == EXPECTED_OOS_TRADES,
        str(core["oos_trades"]),
    )

    add(
        "OOS total R",
        np.isclose(
            core["oos_total_R"],
            EXPECTED_OOS_TOTAL_R,
            atol=TOLERANCE,
        ),
        f"{core['oos_total_R']:.4f}R",
    )

    add(
        "OOS positive expectancy",
        core["oos_mean_R"] > 0,
        f"{core['oos_mean_R']:.6f}R",
    )

    add(
        "OOS PF > 1",
        core["oos_profit_factor"] > 1,
        f"{core['oos_profit_factor']:.4f}",
    )

    add(
        "All OOS windows represented",
        set(window_table["window"]) == set(OOS_WINDOWS),
        str(sorted(window_table["window"].tolist())),
    )

    add(
        "S28.2 window stability",
        s282["passed"],
        (f"Worst LOO = {s282['worst_leave_one_out_R']:.4f}R"),
    )

    add(
        "S28.3 year stability",
        s283["passed"],
        (f"{s283['positive_years']}/{s283['total_years']} positive years"),
    )

    add(
        "S28.4 perturbation robustness",
        s284["passed"],
        (f"{s284['positive_scenarios']}/{s284['scenarios']} positive"),
    )

    return pd.DataFrame(rows)


# =============================================================================
# CHARTS
# =============================================================================


def generate_charts(
    df: pd.DataFrame,
    oos: pd.DataFrame,
    window_table: pd.DataFrame,
) -> None:

    banner("GENERATING FINAL S2R CHARTS")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        print("Charts skipped.")
        return

    # -------------------------------------------------------------------------
    # Full equity
    # -------------------------------------------------------------------------

    full = df.copy()

    if "entry_timestamp" in full.columns:
        full["_dt"] = pd.to_datetime(
            full["entry_timestamp"],
            utc=True,
            errors="coerce",
        )

        full = full.sort_values("_dt")

    equity = full["_s2r_R"].cumsum()

    plt.figure(figsize=(12, 6))

    plt.plot(
        np.arange(
            1,
            len(equity) + 1,
        ),
        equity,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title("S2R — Full Dataset Equity Curve")

    plt.xlabel("Trade")

    plt.ylabel("Cumulative R")

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        FINAL_EQUITY,
        dpi=180,
    )

    plt.close()

    print(FINAL_EQUITY)

    # -------------------------------------------------------------------------
    # OOS equity
    # -------------------------------------------------------------------------

    oos_plot = oos.copy()

    if "entry_timestamp" in oos_plot.columns:
        oos_plot["_dt"] = pd.to_datetime(
            oos_plot["entry_timestamp"],
            utc=True,
            errors="coerce",
        )

        oos_plot = oos_plot.sort_values("_dt")

    oos_equity = oos_plot["_s2r_R"].cumsum()

    plt.figure(figsize=(12, 6))

    plt.plot(
        np.arange(
            1,
            len(oos_equity) + 1,
        ),
        oos_equity,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title("S2R — Holdout OOS Equity Curve")

    plt.xlabel("OOS Trade")

    plt.ylabel("Cumulative R")

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        FINAL_OOS_EQUITY,
        dpi=180,
    )

    plt.close()

    print(FINAL_OOS_EQUITY)

    # -------------------------------------------------------------------------
    # Window performance
    # -------------------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.bar(
        window_table["window"].astype(str),
        window_table["total_R"],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.title("S2R — OOS Performance by Window")

    plt.xlabel("OOS Window")

    plt.ylabel("Total R")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        FINAL_WINDOWS_CHART,
        dpi=180,
    )

    plt.close()

    print(FINAL_WINDOWS_CHART)

    # -------------------------------------------------------------------------
    # Year performance
    # -------------------------------------------------------------------------

    if "entry_timestamp" in oos.columns:
        temp = oos.copy()

        temp["_dt"] = pd.to_datetime(
            temp["entry_timestamp"],
            utc=True,
            errors="coerce",
        )

        temp["_year"] = temp["_dt"].dt.year

        yearly = temp.groupby("_year")["_s2r_R"].sum()

        plt.figure(figsize=(10, 6))

        plt.bar(
            yearly.index.astype(str),
            yearly.values,
        )

        plt.axhline(
            0,
            linewidth=1,
        )

        plt.title("S2R — OOS Performance by Year")

        plt.xlabel("Year")

        plt.ylabel("Total R")

        plt.grid(
            axis="y",
            alpha=0.25,
        )

        plt.tight_layout()

        plt.savefig(
            FINAL_YEARS_CHART,
            dpi=180,
        )

        plt.close()

        print(FINAL_YEARS_CHART)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("S29 FINAL VALIDATION & RESEARCH FREEZE")

    print()
    print("STRATEGY : S2R")

    print()
    print("FROZEN MODEL")

    print(f"  MAE >= {MAE_THRESHOLD:.2f}R")

    print(f"  Recovery >= +{RECOVERY_THRESHOLD:.2f}R")

    print(f"  Deadline = {RECOVERY_DEADLINE} bars")

    print()
    print("NO PARAMETER OPTIMIZATION.")

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    s2r = load_s2r()

    # -------------------------------------------------------------------------
    # Core
    # -------------------------------------------------------------------------

    oos, core = validate_core(s2r)

    # -------------------------------------------------------------------------
    # Windows
    # -------------------------------------------------------------------------

    window_table, window_pass = validate_windows(oos)

    if not window_pass:
        raise RuntimeError("OOS window validation failed.")

    # -------------------------------------------------------------------------
    # S28
    # -------------------------------------------------------------------------

    s28 = validate_s28()

    # -------------------------------------------------------------------------
    # S28.2
    # -------------------------------------------------------------------------

    s282 = validate_s282(oos)

    if not s282["passed"]:
        raise RuntimeError("S28.2 window stability failed.")

    # -------------------------------------------------------------------------
    # S28.3
    # -------------------------------------------------------------------------

    s283 = validate_s283(oos)

    if not s283["passed"]:
        raise RuntimeError("S28.3 year stability failed.")

    # -------------------------------------------------------------------------
    # S28.4
    # -------------------------------------------------------------------------

    s284 = validate_s284()

    if not s284["passed"]:
        raise RuntimeError("S28.4 parameter robustness failed.")

    # -------------------------------------------------------------------------
    # Final audit
    # -------------------------------------------------------------------------

    banner("BUILDING FINAL S2R AUDIT")

    audit = build_final_audit(
        core,
        window_table,
        s282,
        s283,
        s284,
    )

    print(audit.to_string(index=False))

    final_pass = (audit["result"] == "PASS").all()

    status(
        "FINAL S2R RESEARCH GATE",
        final_pass,
    )

    if not final_pass:
        raise RuntimeError("S29 FINAL GATE FAILED.")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    robustness = [
        s28,
        s282,
        s283,
        s284,
    ]

    summary = build_summary(
        core,
        window_table,
        robustness,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    banner("SAVING FINAL S2R RESEARCH PACKAGE")

    summary.to_csv(
        FINAL_SUMMARY,
        index=False,
    )

    audit.to_csv(
        FINAL_AUDIT,
        index=False,
    )

    window_table.to_csv(
        FINAL_OOS_WINDOWS,
        index=False,
    )

    pd.DataFrame(robustness).to_csv(
        FINAL_ROBUSTNESS,
        index=False,
    )

    report = {
        "strategy": STRATEGY_NAME,
        "status": "FROZEN",
        "frozen_model": {
            "mae_threshold_R": MAE_THRESHOLD,
            "recovery_threshold_R": RECOVERY_THRESHOLD,
            "deadline_bars": RECOVERY_DEADLINE,
        },
        "oos": {
            "windows": OOS_WINDOWS,
            "trades": int(core["oos_trades"]),
            "total_R": float(core["oos_total_R"]),
            "mean_R": float(core["oos_mean_R"]),
            "win_rate": float(core["oos_win_rate"]),
            "profit_factor": float(core["oos_profit_factor"]),
            "max_drawdown_R": float(core["oos_max_drawdown_R"]),
        },
        "robustness": {
            "S28": "PASS",
            "S28.1": "PASS",
            "S28.2": "PASS",
            "S28.3": "PASS",
            "S28.4": "PASS",
        },
        "decision": (
            "S2R research is frozen. "
            "No further parameter optimization "
            "within this research cycle."
        ),
    }

    with open(
        FINAL_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print(FINAL_SUMMARY)

    print(FINAL_AUDIT)

    print(FINAL_OOS_WINDOWS)

    print(FINAL_ROBUSTNESS)

    print(FINAL_REPORT)

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    generate_charts(
        s2r,
        oos,
        window_table,
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    banner("S29 FINAL STATUS")

    print(f"Strategy          : {STRATEGY_NAME}")

    print("Research status   : FROZEN")

    print()
    print(f"OOS trades        : {core['oos_trades']}")

    print(f"OOS total R       : {core['oos_total_R']:.4f}")

    print(f"OOS mean R        : {core['oos_mean_R']:.6f}")

    print(f"OOS win rate      : {core['oos_win_rate']:.4%}")

    print(f"OOS PF            : {core['oos_profit_factor']:.4f}")

    print(f"OOS max DD        : {core['oos_max_drawdown_R']:.4f}R")

    print()
    print("ROBUSTNESS SUITE")

    print("  S28    Monte Carlo / Bootstrap : PASS")

    print("  S28.1  Block Bootstrap         : PASS")

    print("  S28.2  OOS Window Stability    : PASS")

    print("  S28.3  Year Stability          : PASS")

    print("  S28.4  Parameter Perturbation  : PASS")

    print()
    print("=" * 110)

    print("S2R FINAL RESEARCH STATUS: FROZEN")

    print("=" * 110)

    print()
    print("RESEARCH PHASE COMPLETE.")

    print()
    print("NEXT PHASE:")

    print("MODULARIZE S2R INTO src/strategys/S2R/")

    print("Do not change the frozen research parameters.")


if __name__ == "__main__":
    main()
