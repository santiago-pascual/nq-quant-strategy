from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================================================================================================
# PATHS
# ==================================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

S27_PATH = RESULTS_DIR / "s27_full_strategy_trades.csv"
S4_PATH = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"
S26_PATH = RESULTS_DIR / "s26_mae_recovery_integration_trades.csv"

OUTPUT_SCENARIOS = RESULTS_DIR / "s28_4_parameter_perturbation.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "s28_4_parameter_perturbation_summary.csv"
OUTPUT_OOS = RESULTS_DIR / "s28_4_parameter_perturbation_oos.csv"

OUTPUT_HEATMAP = RESULTS_DIR / "s28_4_parameter_perturbation_heatmap.png"
OUTPUT_DEADLINE = RESULTS_DIR / "s28_4_parameter_perturbation_deadline.png"
OUTPUT_SENSITIVITY = RESULTS_DIR / "s28_4_parameter_perturbation_sensitivity.png"


# ==================================================================================================
# FROZEN MODEL
# ==================================================================================================

OOS_WINDOWS = set(range(12, 23))

BASE_MAE = 0.70
BASE_RECOVERY = 0.20
BASE_DEADLINE = 6

# Deliberately small local perturbations.
MAE_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90]
RECOVERY_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]
DEADLINE_VALUES = [4, 5, 6, 7, 8]

ATOL = 1e-9


# ==================================================================================================
# HELPERS
# ==================================================================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def metric_summary(r: pd.Series | np.ndarray) -> dict:
    x = pd.to_numeric(pd.Series(r), errors="coerce").dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return {
            "trades": 0,
            "total_R": 0.0,
            "mean_R": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
            "gross_profit_R": 0.0,
            "gross_loss_R": 0.0,
        }

    equity = np.cumsum(x)
    running_max = np.maximum.accumulate(np.r_[0.0, equity])
    dd = equity - running_max[1:]

    gross_profit = float(x[x > 0].sum())
    gross_loss = float(-x[x < 0].sum())

    pf = math.inf if gross_loss == 0 else gross_profit / gross_loss

    return {
        "trades": int(len(x)),
        "total_R": float(x.sum()),
        "mean_R": float(x.mean()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": float(pf),
        "max_drawdown_R": float(dd.min()) if len(dd) else 0.0,
        "gross_profit_R": gross_profit,
        "gross_loss_R": gross_loss,
    }


def make_identity(df: pd.DataFrame) -> pd.Series:
    entry = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
    ).astype(str)

    exit_ = pd.to_datetime(
        df["exit_timestamp"],
        utc=True,
    ).astype(str)

    session = df["session_id"].astype(str)

    return entry + "|" + exit_ + "|" + session


def detect_window(df: pd.DataFrame) -> pd.Series:
    if "_window_numeric" in df.columns:
        return pd.to_numeric(
            df["_window_numeric"],
            errors="coerce",
        )

    if "window" in df.columns:
        return pd.to_numeric(
            df["window"],
            errors="coerce",
        )

    raise RuntimeError("No window column found.")


def detect_strategy_r(df: pd.DataFrame) -> str:
    for c in ["_strategy_R", "strategy_R"]:
        if c in df.columns:
            return c

    raise RuntimeError("S27 strategy-R column not found.")


# ==================================================================================================
# LOAD DATA
# ==================================================================================================


def load_data():
    banner("LOADING COMPLETE S2R DATA")

    if not S27_PATH.exists():
        raise RuntimeError(f"Missing S27 file: {S27_PATH}")

    s27 = pd.read_csv(S27_PATH)

    print(S27_PATH)
    print(f"Rows    : {len(s27)}")
    print(f"Columns : {len(s27.columns)}")

    s27 = s27.copy()

    s27["_identity"] = make_identity(s27)
    s27["_window_numeric"] = detect_window(s27)

    s27_r_col = detect_strategy_r(s27)

    if not s27["_identity"].is_unique:
        raise RuntimeError("S27 trade identity is not unique.")

    banner("LOADING S4 RECOVERY PATH DATA")

    if not S4_PATH.exists():
        raise RuntimeError(f"Missing S4 file: {S4_PATH}")

    s4 = pd.read_csv(S4_PATH)

    print(S4_PATH)
    print(f"Rows    : {len(s4)}")
    print(f"Columns : {len(s4.columns)}")

    s4 = s4.copy()
    s4["_identity"] = make_identity(s4)

    if not s4["_identity"].is_unique:
        raise RuntimeError("S4 trade identity is not unique.")

    banner("LOADING AUDITED S26 DATA")

    if not S26_PATH.exists():
        raise RuntimeError(f"Missing S26 file: {S26_PATH}")

    s26 = pd.read_csv(S26_PATH)

    print(S26_PATH)
    print(f"Rows    : {len(s26)}")
    print(f"Columns : {len(s26.columns)}")

    required = {
        "original_index",
        "window",
        "benchmark_R",
        "strategy_R",
        "state",
        "mae_bar",
        "recovery_bar",
        "exit_bar",
        "exit_type",
    }

    missing = required - set(s26.columns)

    if missing:
        raise RuntimeError(f"S26 missing columns: {sorted(missing)}")

    return s27, s4, s26, s27_r_col


# ==================================================================================================
# CONNECT S26 -> S4 -> S27
# ==================================================================================================


def connect_data(
    s27: pd.DataFrame,
    s4: pd.DataFrame,
    s26: pd.DataFrame,
    s27_r_col: str,
) -> pd.DataFrame:

    banner("CONNECTING S26 -> S4 -> S27")

    s26 = s26.copy()

    s26["original_index"] = pd.to_numeric(
        s26["original_index"],
        errors="raise",
    ).astype(int)

    if (
        not s26["original_index"]
        .between(
            0,
            len(s4) - 1,
        )
        .all()
    ):
        raise RuntimeError("Invalid S26 original_index.")

    # ----------------------------------------------------------------------------------------------
    # S26 -> S4
    # ----------------------------------------------------------------------------------------------

    s26_s4 = s4.iloc[s26["original_index"].to_numpy()].copy()

    s26["_identity"] = s26_s4["_identity"].to_numpy()

    if not s26["_identity"].is_unique:
        raise RuntimeError("S26 -> S4 identity mapping is not unique.")

    # ----------------------------------------------------------------------------------------------
    # S4 -> S27
    # ----------------------------------------------------------------------------------------------

    s27_identity_set = set(s27["_identity"])

    matched = s26["_identity"].isin(s27_identity_set)

    print(f"S26 trades              : {len(s26)}")

    print(f"Valid S26 -> S4 indices : {len(s26_s4)} / {len(s26)}")

    print(f"S26 -> S27 matched      : {matched.sum()} / {len(matched)}")

    if not matched.all():
        raise RuntimeError("Not all S26 trades map to S27.")

    # ----------------------------------------------------------------------------------------------
    # Authoritative S27 fields
    # ----------------------------------------------------------------------------------------------

    s27_lookup = (
        s27[
            [
                "_identity",
                "_window_numeric",
                s27_r_col,
            ]
        ]
        .drop_duplicates("_identity")
        .set_index("_identity")
    )

    s26["window_s27"] = s26["_identity"].map(s27_lookup["_window_numeric"])

    s26["_authoritative_R"] = s26["_identity"].map(s27_lookup[s27_r_col])

    if s26["window_s27"].isna().any():
        raise RuntimeError("Some S26 trades could not be mapped to an S27 window.")

    if s26["_authoritative_R"].isna().any():
        raise RuntimeError(
            "Some S26 trades could not be mapped to authoritative S27 R."
        )

    print(
        "S26 authoritative S27 mapping : "
        f"{s26['_authoritative_R'].notna().sum()} / "
        f"{len(s26)}"
    )

    print(f"Recovery universe       : {len(s26)} trades")

    print(f"S26 OOS trades          : {s26['window_s27'].isin(OOS_WINDOWS).sum()}")

    print(f"Complete S2R OOS trades : {s27['_window_numeric'].isin(OOS_WINDOWS).sum()}")

    return s26


# ==================================================================================================
# PATH ACCESS
# ==================================================================================================


def get_mae_value(
    row: pd.Series,
    bar: int,
) -> float:

    col = f"mae_{bar}R"

    if col not in row.index:
        return np.nan

    value = pd.to_numeric(
        pd.Series([row[col]]),
        errors="coerce",
    ).iloc[0]

    return float(value)


def get_close_value(
    row: pd.Series,
    bar: int,
) -> float:

    col = f"close_{bar}R"

    if col not in row.index:
        return np.nan

    value = pd.to_numeric(
        pd.Series([row[col]]),
        errors="coerce",
    ).iloc[0]

    return float(value)


# ==================================================================================================
# FROZEN STATE MACHINE
# ==================================================================================================


def frozen_state_machine(
    row: pd.Series,
    mae_threshold: float,
    recovery_threshold: float,
    deadline: int,
) -> tuple[
    float,
    str,
    int | None,
    int | None,
    int,
]:
    """
    Recovery-path state machine.

    IMPORTANT:
    This function is used for parameter perturbation scenarios.

    S27 remains the authoritative frozen baseline.
    """

    benchmark_r = float(row["_authoritative_R"])

    path_length = 20

    # ----------------------------------------------------------------------------------------------
    # First MAE trigger
    # ----------------------------------------------------------------------------------------------

    mae_bar = None

    for bar in range(
        1,
        path_length + 1,
    ):
        value = get_mae_value(
            row,
            bar,
        )

        if not np.isfinite(value):
            continue

        if value >= mae_threshold:
            mae_bar = bar
            break

    # No MAE trigger.
    if mae_bar is None:
        return (
            benchmark_r,
            "NO_MAE_TRIGGER",
            None,
            None,
            0,
        )

    # ----------------------------------------------------------------------------------------------
    # Recovery deadline
    # ----------------------------------------------------------------------------------------------

    deadline_bar = min(
        mae_bar + deadline,
        path_length,
    )

    # ----------------------------------------------------------------------------------------------
    # Recovery search starts strictly AFTER MAE
    # ----------------------------------------------------------------------------------------------

    for bar in range(
        mae_bar + 1,
        deadline_bar + 1,
    ):
        close_r = get_close_value(
            row,
            bar,
        )

        if not np.isfinite(close_r):
            continue

        if close_r >= recovery_threshold:
            return (
                close_r,
                "RECOVERED",
                mae_bar,
                bar,
                bar,
            )

    # ----------------------------------------------------------------------------------------------
    # Failed recovery
    # ----------------------------------------------------------------------------------------------

    final_close = get_close_value(
        row,
        deadline_bar,
    )

    if not np.isfinite(final_close):
        raise RuntimeError(f"Missing close path at deadline {deadline_bar}.")

    return (
        final_close,
        "FAILED_TO_RECOVER",
        mae_bar,
        None,
        deadline_bar,
    )


# ==================================================================================================
# MODEL ROWS
# ==================================================================================================


def prepare_model_rows(
    s26: pd.DataFrame,
    s4: pd.DataFrame,
) -> pd.DataFrame:

    s4_by_identity = s4.set_index("_identity")

    rows = []

    for _, model_row in s26.iterrows():
        identity = model_row["_identity"]

        if identity not in s4_by_identity.index:
            raise RuntimeError(f"S4 identity missing for S26 trade: {identity}")

        source = s4_by_identity.loc[identity]

        row = model_row.copy()

        # Add S4 fields where S26 does not already have them.
        for col in s4.columns:
            if col not in row.index:
                row[col] = source[col]

        rows.append(row)

    result = pd.DataFrame(rows)

    return result


# ==================================================================================================
# COMPLETE STRATEGY
# ==================================================================================================


def reconstruct_complete(
    s27: pd.DataFrame,
    model_rows: pd.DataFrame,
    mae_threshold: float,
    recovery_threshold: float,
    deadline: int,
    s27_r_col: str,
) -> pd.DataFrame:
    """
    Construct a complete strategy scenario.

    342 trades remain exactly equal to S27.

    The 195 recovery-universe trades are recalculated
    using the supplied parameter scenario.
    """

    result = s27[
        [
            "_identity",
            "_window_numeric",
            s27_r_col,
        ]
    ].copy()

    result = result.rename(columns={s27_r_col: "_authoritative_R"})

    result["_strategy_R"] = pd.to_numeric(
        result["_authoritative_R"],
        errors="coerce",
    ).astype(float)

    result["_state"] = "UNTOUCHED_S27"

    result["_mae_bar"] = np.nan
    result["_recovery_bar"] = np.nan
    result["_exit_bar"] = np.nan
    result["_exit_type"] = "UNTOUCHED_S27"

    model_lookup = model_rows.set_index("_identity")

    for identity in model_lookup.index:
        row = model_lookup.loc[identity]

        (
            strategy_r,
            state,
            mae_bar,
            recovery_bar,
            exit_bar,
        ) = frozen_state_machine(
            row,
            mae_threshold=mae_threshold,
            recovery_threshold=recovery_threshold,
            deadline=deadline,
        )

        mask = result["_identity"] == identity

        result.loc[
            mask,
            "_strategy_R",
        ] = strategy_r

        result.loc[
            mask,
            "_state",
        ] = state

        result.loc[
            mask,
            "_mae_bar",
        ] = np.nan if mae_bar is None else mae_bar

        result.loc[
            mask,
            "_recovery_bar",
        ] = np.nan if recovery_bar is None else recovery_bar

        result.loc[
            mask,
            "_exit_bar",
        ] = exit_bar

        result.loc[
            mask,
            "_exit_type",
        ] = state

    return result


# ==================================================================================================
# AUTHORITATIVE BASELINE
# ==================================================================================================


def baseline_gate(
    s27: pd.DataFrame,
    s27_r_col: str,
) -> pd.DataFrame:

    banner("MANDATORY AUTHORITATIVE S2R BASELINE GATE")

    print("Frozen parameters:")
    print(f"  MAE threshold : {BASE_MAE:.2f}R")
    print(f"  Recovery      : +{BASE_RECOVERY:.2f}R")
    print(f"  Deadline      : {BASE_DEADLINE} bars")

    oos = s27[s27["_window_numeric"].isin(OOS_WINDOWS)].copy()

    metrics = metric_summary(oos[s27_r_col])

    print()
    print("AUTHORITATIVE S2R OOS:")
    print(f"  Trades  : {metrics['trades']}")
    print(f"  Total R : {metrics['total_R']:.4f}")
    print(f"  Mean R  : {metrics['mean_R']:.6f}")
    print(f"  WR      : {metrics['win_rate']:.6%}")
    print(f"  PF      : {metrics['profit_factor']}")
    print(f"  Max DD  : {metrics['max_drawdown_R']:.4f}")

    checks = {
        "trade_count": (metrics["trades"] == 217),
        "total_R": np.isclose(
            metrics["total_R"],
            34.3452,
            atol=ATOL,
        ),
    }

    print()
    print("AUTHORITATIVE BASELINE:")

    for name, passed in checks.items():
        print(f"  {name:<20}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError("AUTHORITATIVE S2R BASELINE IS NOT 217 trades / +34.3452R.")

    print()
    print("PASS — authoritative S2R baseline confirmed.")

    return oos


# ==================================================================================================
# PERTURBATION SCENARIO
# ==================================================================================================


def run_scenario(
    s27: pd.DataFrame,
    model_rows: pd.DataFrame,
    s27_r_col: str,
    mae_threshold: float,
    recovery_threshold: float,
    deadline: int,
) -> tuple[dict, pd.DataFrame]:

    # ----------------------------------------------------------------------------------------------
    # Frozen point is authoritative S27.
    # ----------------------------------------------------------------------------------------------

    if (
        np.isclose(
            mae_threshold,
            BASE_MAE,
            atol=ATOL,
        )
        and np.isclose(
            recovery_threshold,
            BASE_RECOVERY,
            atol=ATOL,
        )
        and deadline == BASE_DEADLINE
    ):
        complete = s27[
            [
                "_identity",
                "_window_numeric",
                s27_r_col,
            ]
        ].copy()

        complete = complete.rename(columns={s27_r_col: "_authoritative_R"})

        complete["_strategy_R"] = complete["_authoritative_R"].astype(float)

    else:
        complete = reconstruct_complete(
            s27=s27,
            model_rows=model_rows,
            mae_threshold=mae_threshold,
            recovery_threshold=recovery_threshold,
            deadline=deadline,
            s27_r_col=s27_r_col,
        )

    oos = complete[complete["_window_numeric"].isin(OOS_WINDOWS)].copy()

    metrics = metric_summary(oos["_strategy_R"])

    # Authoritative frozen baseline.
    baseline_oos = s27[s27["_window_numeric"].isin(OOS_WINDOWS)]

    base_metrics = metric_summary(baseline_oos[s27_r_col])

    result = {
        "mae_threshold": mae_threshold,
        "recovery_threshold": recovery_threshold,
        "deadline": deadline,
        **metrics,
        "delta_total_R_vs_frozen": (metrics["total_R"] - base_metrics["total_R"]),
        "delta_mean_R_vs_frozen": (metrics["mean_R"] - base_metrics["mean_R"]),
        "delta_win_rate_vs_frozen": (metrics["win_rate"] - base_metrics["win_rate"]),
        "delta_max_DD_vs_frozen": (
            metrics["max_drawdown_R"] - base_metrics["max_drawdown_R"]
        ),
        "positive_OOS": (metrics["total_R"] > 0),
        "PF_above_1": (metrics["profit_factor"] > 1),
    }

    return result, oos


# ==================================================================================================
# PERTURBATION GRID
# ==================================================================================================


def run_perturbations(
    s27: pd.DataFrame,
    model_rows: pd.DataFrame,
    s27_r_col: str,
) -> pd.DataFrame:

    banner("RUNNING PARAMETER PERTURBATION GRID")

    scenarios = []

    total = len(MAE_VALUES) * len(RECOVERY_VALUES) * len(DEADLINE_VALUES)

    count = 0

    for mae in MAE_VALUES:
        for recovery in RECOVERY_VALUES:
            for deadline in DEADLINE_VALUES:
                count += 1

                print(
                    f"[{count:03d}/{total}] "
                    f"MAE={mae:.2f} "
                    f"REC={recovery:.2f} "
                    f"DL={deadline}"
                )

                result, _ = run_scenario(
                    s27=s27,
                    model_rows=model_rows,
                    s27_r_col=s27_r_col,
                    mae_threshold=mae,
                    recovery_threshold=recovery,
                    deadline=deadline,
                )

                scenarios.append(result)

    return pd.DataFrame(scenarios)


# ==================================================================================================
# AUDIT
# ==================================================================================================


def audit_scenarios(
    df: pd.DataFrame,
) -> None:

    banner("S28.4 AUDIT")

    expected = len(MAE_VALUES) * len(RECOVERY_VALUES) * len(DEADLINE_VALUES)

    print(f"Scenarios expected : {expected}")

    print(f"Scenarios produced : {len(df)}")

    if len(df) != expected:
        raise RuntimeError("Incomplete perturbation grid.")

    metric_columns = [
        "total_R",
        "mean_R",
        "win_rate",
        "max_drawdown_R",
    ]

    if df[metric_columns].isna().any().any():
        raise RuntimeError("NaN metrics detected.")

    print("Scenario count          : PASS")

    print("Finite metrics          : PASS")

    print("All parameter dimensions: PASS")

    frozen = df[
        (df["mae_threshold"] == BASE_MAE)
        & (df["recovery_threshold"] == BASE_RECOVERY)
        & (df["deadline"] == BASE_DEADLINE)
    ]

    if len(frozen) != 1:
        raise RuntimeError("Frozen scenario missing or duplicated.")

    row = frozen.iloc[0]

    if not np.isclose(
        row["total_R"],
        34.3452,
        atol=ATOL,
    ):
        raise RuntimeError("Frozen scenario does not equal authoritative S2R.")

    if int(row["trades"]) != 217:
        raise RuntimeError("Frozen scenario does not contain 217 OOS trades.")

    print("Frozen scenario          : PASS")

    print("Frozen OOS = 217 trades : PASS")

    print("Frozen OOS = +34.3452R  : PASS")


# ==================================================================================================
# CHARTS
# ==================================================================================================


def generate_charts(
    df: pd.DataFrame,
) -> None:

    banner("GENERATING CHARTS")

    # ----------------------------------------------------------------------------------------------
    # Heatmap: MAE vs Recovery at frozen deadline
    # ----------------------------------------------------------------------------------------------

    subset = df[df["deadline"] == BASE_DEADLINE].copy()

    pivot = subset.pivot(
        index="mae_threshold",
        columns="recovery_threshold",
        values="total_R",
    )

    plt.figure(figsize=(10, 7))

    plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    plt.xticks(
        range(len(pivot.columns)),
        [f"{x:.2f}" for x in pivot.columns],
    )

    plt.yticks(
        range(len(pivot.index)),
        [f"{x:.2f}" for x in pivot.index],
    )

    plt.xlabel("Recovery threshold (R)")

    plt.ylabel("MAE threshold (R)")

    plt.title("S2R OOS Total R — MAE / Recovery Perturbation\nDeadline = 6 bars")

    plt.colorbar(label="OOS Total R")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[
                i,
                j,
            ]

            plt.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_HEATMAP,
        dpi=180,
    )

    plt.close()

    # ----------------------------------------------------------------------------------------------
    # Deadline stability
    # ----------------------------------------------------------------------------------------------

    subset = df[
        (df["mae_threshold"] == BASE_MAE) & (df["recovery_threshold"] == BASE_RECOVERY)
    ].sort_values("deadline")

    plt.figure(figsize=(9, 6))

    plt.plot(
        subset["deadline"],
        subset["total_R"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.axvline(
        BASE_DEADLINE,
        linestyle=":",
    )

    plt.xlabel("Recovery deadline (bars)")

    plt.ylabel("OOS Total R")

    plt.title(
        "S2R OOS Total R — Deadline Perturbation\nMAE = 0.70R / Recovery = +0.20R"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DEADLINE,
        dpi=180,
    )

    plt.close()

    # ----------------------------------------------------------------------------------------------
    # Sensitivity around frozen parameters
    # ----------------------------------------------------------------------------------------------

    frozen_deadline = df[df["deadline"] == BASE_DEADLINE].copy()

    plt.figure(figsize=(10, 7))

    for recovery in RECOVERY_VALUES:
        subset = frozen_deadline[
            frozen_deadline["recovery_threshold"] == recovery
        ].sort_values("mae_threshold")

        plt.plot(
            subset["mae_threshold"],
            subset["total_R"],
            marker="o",
            label=f"Recovery {recovery:.2f}R",
        )

    plt.axvline(
        BASE_MAE,
        linestyle="--",
    )

    plt.axhline(
        34.3452,
        linestyle=":",
    )

    plt.xlabel("MAE threshold (R)")

    plt.ylabel("OOS Total R")

    plt.title("S2R Parameter Sensitivity\nDeadline = 6 bars")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_SENSITIVITY,
        dpi=180,
    )

    plt.close()

    print(OUTPUT_HEATMAP)
    print(OUTPUT_DEADLINE)
    print(OUTPUT_SENSITIVITY)


# ==================================================================================================
# SAVE
# ==================================================================================================


def save_results(
    df: pd.DataFrame,
) -> None:

    banner("SAVING RESULTS")

    df.to_csv(
        OUTPUT_SCENARIOS,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "frozen_mae": BASE_MAE,
                "frozen_recovery": BASE_RECOVERY,
                "frozen_deadline": BASE_DEADLINE,
                "scenario_count": len(df),
                "positive_scenarios": int(df["positive_OOS"].sum()),
                "pf_above_1_scenarios": int(df["PF_above_1"].sum()),
                "min_total_R": float(df["total_R"].min()),
                "median_total_R": float(df["total_R"].median()),
                "max_total_R": float(df["total_R"].max()),
                "min_mean_R": float(df["mean_R"].min()),
                "median_mean_R": float(df["mean_R"].median()),
                "max_mean_R": float(df["mean_R"].max()),
                "min_win_rate": float(df["win_rate"].min()),
                "median_win_rate": float(df["win_rate"].median()),
                "max_win_rate": float(df["win_rate"].max()),
                "min_max_DD": float(df["max_drawdown_R"].min()),
                "median_max_DD": float(df["max_drawdown_R"].median()),
                "max_max_DD": float(df["max_drawdown_R"].max()),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    oos_columns = [
        "mae_threshold",
        "recovery_threshold",
        "deadline",
        "trades",
        "total_R",
        "mean_R",
        "win_rate",
        "profit_factor",
        "max_drawdown_R",
        "delta_total_R_vs_frozen",
        "delta_mean_R_vs_frozen",
        "delta_win_rate_vs_frozen",
        "delta_max_DD_vs_frozen",
        "positive_OOS",
        "PF_above_1",
    ]

    df[oos_columns].to_csv(
        OUTPUT_OOS,
        index=False,
    )

    print(OUTPUT_SUMMARY)
    print(OUTPUT_SCENARIOS)
    print(OUTPUT_OOS)


# ==================================================================================================
# MAIN
# ==================================================================================================


def main() -> None:

    banner("S28.4 PARAMETER PERTURBATION ROBUSTNESS")

    print("Frozen S2R strategy.")

    print("NO PARAMETER OPTIMIZATION.")

    print()

    print("Frozen model:")

    print(f"  MAE >= {BASE_MAE:.2f}R")

    print(f"  Recovery >= +{BASE_RECOVERY:.2f}R")

    print(f"  Deadline = {BASE_DEADLINE} bars")

    print()

    print("Complete S2R OOS expected:")

    print("  Trades  = 217")

    print("  Total R = 34.3452")

    # ----------------------------------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------------------------------

    s27, s4, s26, s27_r_col = load_data()

    # ----------------------------------------------------------------------------------------------
    # Connect
    # ----------------------------------------------------------------------------------------------

    s26 = connect_data(
        s27=s27,
        s4=s4,
        s26=s26,
        s27_r_col=s27_r_col,
    )

    # ----------------------------------------------------------------------------------------------
    # Build model rows
    # ----------------------------------------------------------------------------------------------

    model_rows = prepare_model_rows(
        s26=s26,
        s4=s4,
    )

    # ----------------------------------------------------------------------------------------------
    # Authoritative baseline
    # ----------------------------------------------------------------------------------------------

    baseline_oos = baseline_gate(
        s27=s27,
        s27_r_col=s27_r_col,
    )

    print()

    # ----------------------------------------------------------------------------------------------
    # Perturbations
    # ----------------------------------------------------------------------------------------------

    scenarios = run_perturbations(
        s27=s27,
        model_rows=model_rows,
        s27_r_col=s27_r_col,
    )

    # ----------------------------------------------------------------------------------------------
    # Audit
    # ----------------------------------------------------------------------------------------------

    audit_scenarios(scenarios)

    # ----------------------------------------------------------------------------------------------
    # Charts
    # ----------------------------------------------------------------------------------------------

    generate_charts(scenarios)

    # ----------------------------------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------------------------------

    save_results(scenarios)

    # ----------------------------------------------------------------------------------------------
    # Final
    # ----------------------------------------------------------------------------------------------

    banner("S28.4 FINAL STATUS")

    frozen = scenarios[
        (scenarios["mae_threshold"] == BASE_MAE)
        & (scenarios["recovery_threshold"] == BASE_RECOVERY)
        & (scenarios["deadline"] == BASE_DEADLINE)
    ].iloc[0]

    print(f"Frozen S2R OOS Total R : {frozen['total_R']:.4f}")

    print(
        f"Positive perturbations : "
        f"{int(scenarios['positive_OOS'].sum())}"
        f"/{len(scenarios)}"
    )

    print(
        f"PF > 1 perturbations   : "
        f"{int(scenarios['PF_above_1'].sum())}"
        f"/{len(scenarios)}"
    )

    print()

    print(
        "PASS — authoritative S2R baseline "
        "confirmed and parameter perturbation "
        "robustness completed."
    )

    print()

    print("S28.4 PARAMETER PERTURBATION COMPLETE")


if __name__ == "__main__":
    main()
