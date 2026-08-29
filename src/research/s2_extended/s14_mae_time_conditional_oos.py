from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
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

OUTPUT_DIR = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
)

# Original frozen benchmark
STOP_R = 1.00
RR = 1.75
HORIZON = 20

# Candidate MAE boundaries.
#
# Interpretation:
# If MAE reaches X R, the trade is considered to have crossed
# the "edge failure boundary".
#
MAE_BOUNDARIES = [
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

def safe_float(value, default=np.nan) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass

    return default


def find_path_columns(
    df: pd.DataFrame,
    prefix: str,
) -> dict[int, str]:
    """
    Detect columns such as:

        mae_1R
        mae_2R
        ...
        close_1R
        close_2R
        ...

    Returns:
        {1: "mae_1R", 2: "mae_2R", ...}
    """

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)R$",
        re.IGNORECASE,
    )

    result: dict[int, str] = {}

    for col in df.columns:
        match = pattern.match(str(col))

        if match:
            bar = int(match.group(1))
            result[bar] = col

    return dict(sorted(result.items()))


def normalise_window(series: pd.Series) -> pd.Series:
    """
    Convert window identifiers to integers where possible.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        return numeric.astype(int)

    extracted = (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
    )

    return pd.to_numeric(
        extracted,
        errors="coerce",
    ).astype("Int64")


def max_drawdown(values) -> float:
    """
    R-unit maximum drawdown.
    """

    if len(values) == 0:
        return 0.0

    arr = np.asarray(
        values,
        dtype=float,
    )

    cumulative = np.cumsum(arr)

    running_max = np.maximum.accumulate(
        np.insert(
            cumulative,
            0,
            0.0,
        )
    )[1:]

    drawdown = cumulative - running_max

    return float(np.min(drawdown))


def profit_factor(values) -> float:
    """
    Gross profit / gross loss.
    """

    arr = np.asarray(
        values,
        dtype=float,
    )

    gross_profit = arr[arr > 0].sum()

    gross_loss = -arr[arr < 0].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return float("inf")

        return 0.0

    return float(
        gross_profit / gross_loss
    )


def metrics(
    trades: pd.DataFrame,
    column: str,
) -> dict:

    if trades.empty:
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

    values = pd.to_numeric(
        trades[column],
        errors="coerce",
    ).fillna(0.0)

    wins = int(
        (values > 0).sum()
    )

    losses = int(
        (values <= 0).sum()
    )

    return {
        "trades": int(len(values)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(
            (values > 0).mean()
        ),
        "mean_R": float(
            values.mean()
        ),
        "total_R": float(
            values.sum()
        ),
        "profit_factor": profit_factor(
            values
        ),
        "max_drawdown_R": max_drawdown(
            values
        ),
    }


# =============================================================================
# PATH BUILDING
# =============================================================================

def build_trade_paths(
    df: pd.DataFrame,
    mae_cols: dict[int, str],
    close_cols: dict[int, str],
) -> pd.DataFrame:
    """
    Convert wide path columns into per-trade dictionaries.
    """

    records = []

    common_bars = sorted(
        set(mae_cols.keys())
        & set(close_cols.keys())
    )

    if not common_bars:
        raise RuntimeError(
            "No common MAE/close path bars found."
        )

    for idx, row in df.iterrows():

        mae_path = {}
        close_path = {}

        for bar in common_bars:

            mae_path[bar] = safe_float(
                row[mae_cols[bar]]
            )

            close_path[bar] = safe_float(
                row[close_cols[bar]]
            )

        records.append(
            {
                "_row_index": idx,
                "mae_path": mae_path,
                "close_path": close_path,
            }
        )

    paths = pd.DataFrame(
        records
    ).set_index("_row_index")

    return df.join(paths)


# =============================================================================
# MAE CROSSING
# =============================================================================

def first_mae_crossing(
    mae_path: dict[int, float],
    threshold: float,
) -> int | None:
    """
    First bar where MAE reaches the candidate boundary.
    """

    for bar in sorted(mae_path):

        value = mae_path.get(bar)

        if value is None:
            continue

        if not np.isfinite(value):
            continue

        if value >= threshold:
            return bar

    return None


# =============================================================================
# BENCHMARK R
# =============================================================================

def get_original_R(
    row: pd.Series,
) -> float:

    value = row.get(
        "final_close_R",
        row.get(
            "net_R",
            np.nan,
        ),
    )

    return safe_float(value)


# =============================================================================
# BOUNDARY EXIT
# =============================================================================

def evaluate_boundary_rule(
    df: pd.DataFrame,
    boundary: float,
    execution_model: str,
) -> pd.DataFrame:
    """
    Evaluate a shorter MAE stop.

    execution_model:

        THEORETICAL_BOUNDARY
            Exit exactly at -boundary R.

        BAR_CLOSE_BOUNDARY
            Detect the MAE crossing and use the close
            of the crossing bar.

    Important:
        THEORETICAL_BOUNDARY is an analytical upper-bound
        assumption for the stop execution.

        BAR_CLOSE_BOUNDARY is deliberately different:
        it uses observable bar-close information and can
        therefore be worse than the theoretical boundary.
    """

    rows = []

    for idx, row in df.iterrows():

        original_R = get_original_R(row)

        crossing_bar = first_mae_crossing(
            row["mae_path"],
            boundary,
        )

        triggered = crossing_bar is not None

        strategy_R = original_R

        execution_R = np.nan

        if triggered:

            if execution_model == "THEORETICAL_BOUNDARY":

                execution_R = -float(boundary)

                strategy_R = execution_R

            elif execution_model == "BAR_CLOSE_BOUNDARY":

                close_value = row["close_path"].get(
                    crossing_bar,
                    np.nan,
                )

                if np.isfinite(close_value):

                    execution_R = float(
                        close_value
                    )

                    strategy_R = execution_R

                else:

                    strategy_R = original_R

            else:

                raise ValueError(
                    f"Unknown execution model: "
                    f"{execution_model}"
                )

        rows.append(
            {
                "trade_index": idx,
                "window": row["_window_numeric"],
                "boundary_R": boundary,
                "execution_model": execution_model,
                "crossing_bar": crossing_bar,
                "triggered": triggered,
                "original_R": original_R,
                "strategy_R": strategy_R,
                "delta_R": strategy_R - original_R,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# BOUNDARY AUDIT
# =============================================================================

def build_trigger_audit(
    df: pd.DataFrame,
    boundary: float,
) -> pd.DataFrame:
    """
    For every trade that crosses the boundary, determine
    what happened to it under the original strategy.

    This directly answers:

        "If the trade goes X R against us,
         does it usually fail?"

    """

    rows = []

    for idx, row in df.iterrows():

        crossing_bar = first_mae_crossing(
            row["mae_path"],
            boundary,
        )

        if crossing_bar is None:
            continue

        original_R = get_original_R(row)

        rows.append(
            {
                "trade_index": idx,
                "window": row["_window_numeric"],
                "crossing_bar": crossing_bar,
                "original_R": original_R,
                "would_finish_positive": original_R > 0,
                "would_finish_negative": original_R <= 0,
                "theoretical_boundary_R": -boundary,
                "delta_if_boundary_exit": (
                    -boundary - original_R
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# DEVELOPMENT SEARCH
# =============================================================================

def development_search(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the MAE boundary exclusively using development windows.

    We search both execution models.

    Ranking:
        1. development delta R
        2. development PF
        3. positive-window percentage
        4. smaller boundary as tie-breaker

    The smaller-boundary tie-breaker is intentional:
    if two boundaries perform similarly, prefer the
    less aggressive modification.
    """

    development = df[
        df["_window_numeric"].isin(
            DEVELOPMENT_WINDOWS
        )
    ].copy()

    rows = []

    for boundary in MAE_BOUNDARIES:

        for execution_model in [
            "THEORETICAL_BOUNDARY",
            "BAR_CLOSE_BOUNDARY",
        ]:

            trade_results = evaluate_boundary_rule(
                development,
                boundary,
                execution_model,
            )

            strategy_metrics = metrics(
                trade_results,
                "strategy_R",
            )

            benchmark_metrics = metrics(
                trade_results,
                "original_R",
            )

            triggered = int(
                trade_results["triggered"].sum()
            )

            if triggered < 5:
                continue

            positive_windows = 0
            total_windows = 0

            for window, group in trade_results.groupby(
                "window"
            ):

                if pd.isna(window):
                    continue

                total_windows += 1

                if group["strategy_R"].sum() > 0:
                    positive_windows += 1

            positive_window_pct = (
                positive_windows / total_windows
                if total_windows
                else np.nan
            )

            rows.append(
                {
                    "boundary_R": boundary,
                    "execution_model": execution_model,
                    "development_trades": len(
                        trade_results
                    ),
                    "triggered_trades": triggered,
                    "development_WR": strategy_metrics[
                        "win_rate"
                    ],
                    "development_mean_R": strategy_metrics[
                        "mean_R"
                    ],
                    "development_R": strategy_metrics[
                        "total_R"
                    ],
                    "development_PF": strategy_metrics[
                        "profit_factor"
                    ],
                    "development_DD": strategy_metrics[
                        "max_drawdown_R"
                    ],
                    "benchmark_R": benchmark_metrics[
                        "total_R"
                    ],
                    "development_delta_R": (
                        strategy_metrics["total_R"]
                        - benchmark_metrics["total_R"]
                    ),
                    "development_delta_WR": (
                        strategy_metrics["win_rate"]
                        - benchmark_metrics["win_rate"]
                    ),
                    "positive_window_pct": (
                        positive_window_pct
                    ),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        [
            "development_delta_R",
            "development_PF",
            "positive_window_pct",
            "boundary_R",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    return result.reset_index(
        drop=True
    )


# =============================================================================
# OOS
# =============================================================================

def run_oos(
    df: pd.DataFrame,
    boundary: float,
    execution_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    holdout = df[
        df["_window_numeric"].isin(
            HOLDOUT_WINDOWS
        )
    ].copy()

    trades = evaluate_boundary_rule(
        holdout,
        boundary,
        execution_model,
    )

    rows = []

    for window, group in trades.groupby(
        "window"
    ):

        benchmark_R = group[
            "original_R"
        ].sum()

        strategy_R = group[
            "strategy_R"
        ].sum()

        rows.append(
            {
                "window": window,
                "trades": len(group),
                "triggered_trades": int(
                    group["triggered"].sum()
                ),
                "benchmark_R": benchmark_R,
                "strategy_R": strategy_R,
                "delta_R": (
                    strategy_R
                    - benchmark_R
                ),
                "benchmark_WR": float(
                    (
                        group["original_R"] > 0
                    ).mean()
                ),
                "strategy_WR": float(
                    (
                        group["strategy_R"] > 0
                    ).mean()
                ),
                "benchmark_PF": profit_factor(
                    group["original_R"]
                ),
                "strategy_PF": profit_factor(
                    group["strategy_R"]
                ),
                "benchmark_DD": max_drawdown(
                    group["original_R"]
                ),
                "strategy_DD": max_drawdown(
                    group["strategy_R"]
                ),
            }
        )

    return (
        trades,
        pd.DataFrame(rows),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print("MAE BOUNDARY STOP — TEMPORAL OOS TEST")
    print("=" * 110)

    print()

    print("CORE HYPOTHESIS:")
    print(
        "If a trade reaches a sufficiently large MAE, "
        "the original trade edge may already be invalid."
    )

    print()

    print("Frozen benchmark:")
    print(
        f"  Original stop = {STOP_R:.2f}R"
    )
    print(
        f"  RR            = {RR:.2f}"
    )
    print(
        f"  Horizon       = {HORIZON} bars"
    )

    print()

    print("Candidate MAE boundaries:")
    print(
        "  "
        + ", ".join(
            f"{x:.2f}R"
            for x in MAE_BOUNDARIES
        )
    )

    print()

    print(
        f"Development windows : "
        f"{DEVELOPMENT_WINDOWS}"
    )

    print(
        f"Holdout windows     : "
        f"{HOLDOUT_WINDOWS}"
    )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Trades loaded: {len(df)}"
    )

    # -------------------------------------------------------------------------
    # DETECT PATHS
    # -------------------------------------------------------------------------

    mae_cols = find_path_columns(
        df,
        "mae",
    )

    close_cols = find_path_columns(
        df,
        "close",
    )

    if not mae_cols:

        raise RuntimeError(
            "No MAE path columns found."
        )

    if not close_cols:

        raise RuntimeError(
            "No close path columns found."
        )

    common_bars = sorted(
        set(mae_cols.keys())
        & set(close_cols.keys())
    )

    if not common_bars:

        raise RuntimeError(
            "No common MAE/close bars found."
        )

    print()
    print("Detected paths:")
    print(
        f"  MAE bars   : {len(mae_cols)}"
    )
    print(
        f"  MAE range  : "
        f"{min(mae_cols)} -> {max(mae_cols)}"
    )
    print(
        f"  Close bars : {len(close_cols)}"
    )
    print(
        f"  Close range: "
        f"{min(close_cols)} -> {max(close_cols)}"
    )

    # -------------------------------------------------------------------------
    # WINDOW
    # -------------------------------------------------------------------------

    if "window" not in df.columns:

        raise RuntimeError(
            "Required column 'window' not found."
        )

    df["_window_numeric"] = normalise_window(
        df["window"]
    )

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------

    df = build_trade_paths(
        df,
        mae_cols,
        close_cols,
    )

    # -------------------------------------------------------------------------
    # DISCOVERY: HOW STRONGLY DOES MAE PREDICT FAILURE?
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("1. MAE FAILURE EVIDENCE")
    print("=" * 110)

    evidence_rows = []

    for boundary in MAE_BOUNDARIES:

        audit = build_trigger_audit(
            df,
            boundary,
        )

        triggered = len(audit)

        if triggered == 0:
            continue

        positive = int(
            audit[
                "would_finish_positive"
            ].sum()
        )

        negative = int(
            audit[
                "would_finish_negative"
            ].sum()
        )

        failure_rate = (
            negative / triggered
        )

        survival_rate = (
            positive / triggered
        )

        mean_original_R = float(
            audit["original_R"].mean()
        )

        total_original_R = float(
            audit["original_R"].sum()
        )

        mean_crossing_bar = float(
            audit["crossing_bar"].mean()
        )

        evidence_rows.append(
            {
                "mae_boundary_R": boundary,
                "trades_reaching_boundary": triggered,
                "finished_positive": positive,
                "finished_negative": negative,
                "post_boundary_failure_pct": failure_rate,
                "post_boundary_survival_pct": survival_rate,
                "mean_final_R": mean_original_R,
                "total_final_R": total_original_R,
                "mean_crossing_bar": mean_crossing_bar,
            }
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    print(
        evidence.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # DEVELOPMENT SEARCH
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("2. DEVELOPMENT SEARCH — MAE STOP BOUNDARY")
    print("=" * 110)

    development = development_search(
        df
    )

    if development.empty:

        raise RuntimeError(
            "No valid development rules."
        )

    print()

    print(
        development.to_string(
            index=False
        )
    )

    selected = development.iloc[0]

    selected_boundary = float(
        selected["boundary_R"]
    )

    selected_model = str(
        selected["execution_model"]
    )

    # -------------------------------------------------------------------------
    # SELECTED RULE
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(
        f"MAE boundary       : "
        f"{selected_boundary:.2f}R"
    )

    print(
        f"Execution model    : "
        f"{selected_model}"
    )

    print(
        f"Triggered trades   : "
        f"{int(selected['triggered_trades'])}"
    )

    print(
        f"Development WR     : "
        f"{selected['development_WR']:.4f}"
    )

    print(
        f"Development R      : "
        f"{selected['development_R']:.4f}"
    )

    print(
        f"Development ΔR     : "
        f"{selected['development_delta_R']:.4f}"
    )

    print(
        f"Development PF     : "
        f"{selected['development_PF']}"
    )

    print(
        f"Development DD     : "
        f"{selected['development_DD']:.4f}"
    )

    print(
        f"Positive windows   : "
        f"{selected['positive_window_pct']:.4f}"
    )

    # -------------------------------------------------------------------------
    # OOS
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("3. HOLDOUT OOS TEST")
    print("=" * 110)

    oos_trades, oos_windows = run_oos(
        df,
        selected_boundary,
        selected_model,
    )

    benchmark = metrics(
        oos_trades,
        "original_R",
    )

    strategy = metrics(
        oos_trades,
        "strategy_R",
    )

    print()

    print("FROZEN OOS RULE:")
    print(
        f"  MAE boundary    = "
        f"{selected_boundary:.2f}R"
    )
    print(
        f"  Execution model = "
        f"{selected_model}"
    )

    print()

    print("BENCHMARK HOLDOUT")
    print(
        f"  Trades       : "
        f"{benchmark['trades']}"
    )
    print(
        f"  Win rate     : "
        f"{benchmark['win_rate']:.4f}"
    )
    print(
        f"  Mean R       : "
        f"{benchmark['mean_R']:.4f}"
    )
    print(
        f"  Total R      : "
        f"{benchmark['total_R']:.4f}"
    )
    print(
        f"  PF           : "
        f"{benchmark['profit_factor']}"
    )
    print(
        f"  Max DD       : "
        f"{benchmark['max_drawdown_R']:.4f}"
    )

    print()

    print("MAE BOUNDARY HOLDOUT")
    print(
        f"  Trades       : "
        f"{strategy['trades']}"
    )
    print(
        f"  Triggered    : "
        f"{int(oos_trades['triggered'].sum())}"
    )
    print(
        f"  Win rate     : "
        f"{strategy['win_rate']:.4f}"
    )
    print(
        f"  Mean R       : "
        f"{strategy['mean_R']:.4f}"
    )
    print(
        f"  Total R      : "
        f"{strategy['total_R']:.4f}"
    )
    print(
        f"  PF           : "
        f"{strategy['profit_factor']}"
    )
    print(
        f"  Max DD       : "
        f"{strategy['max_drawdown_R']:.4f}"
    )

    print()

    print("OOS IMPROVEMENT")

    print(
        f"  Delta R      : "
        f"{strategy['total_R'] - benchmark['total_R']:.4f}"
    )

    print(
        f"  Delta Mean R : "
        f"{strategy['mean_R'] - benchmark['mean_R']:.4f}"
    )

    print(
        f"  Delta WR     : "
        f"{strategy['win_rate'] - benchmark['win_rate']:.4f}"
    )

    print(
        f"  Delta Max DD : "
        f"{strategy['max_drawdown_R'] - benchmark['max_drawdown_R']:.4f}"
    )

    # -------------------------------------------------------------------------
    # WINDOW BY WINDOW
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("4. WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    print(
        oos_windows.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # TRIGGER AUDIT
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("5. TRIGGER AUDIT")
    print("=" * 110)

    triggered = oos_trades[
        oos_trades["triggered"]
    ].copy()

    if triggered.empty:

        print(
            "No holdout trades crossed the selected boundary."
        )

    else:

        triggered["saved_R"] = (
            triggered["strategy_R"]
            - triggered["original_R"]
        )

        print()

        print(
            f"Triggered trades: "
            f"{len(triggered)}"
        )

        print(
            f"Original winners among triggered: "
            f"{int((triggered['original_R'] > 0).sum())}"
        )

        print(
            f"Original losers among triggered: "
            f"{int((triggered['original_R'] <= 0).sum())}"
        )

        print()

        print(
            triggered[
                [
                    "trade_index",
                    "window",
                    "crossing_bar",
                    "original_R",
                    "strategy_R",
                    "saved_R",
                ]
            ].to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # IMPORTANT COUNTERFACTUAL
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("6. COUNTERFACTUAL CHECK")
    print("=" * 110)

    print(
        "This section asks the critical question:"
    )

    print(
        "Of the trades that crossed the MAE boundary,"
    )

    print(
        "how many would actually have recovered under the original strategy?"
    )

    print()

    if not triggered.empty:

        original_winners = triggered[
            triggered["original_R"] > 0
        ]

        original_losers = triggered[
            triggered["original_R"] <= 0
        ]

        winner_pct = (
            len(original_winners)
            / len(triggered)
        )

        loser_pct = (
            len(original_losers)
            / len(triggered)
        )

        print(
            f"Boundary crossers : "
            f"{len(triggered)}"
        )

        print(
            f"Eventually winners : "
            f"{len(original_winners)} "
            f"({winner_pct:.2%})"
        )

        print(
            f"Eventually losers  : "
            f"{len(original_losers)} "
            f"({loser_pct:.2%})"
        )

        print()

        print(
            "This number is crucial for deciding whether"
        )

        print(
            "the MAE boundary should become a HARD STOP."
        )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_file = (
        OUTPUT_DIR
        / "s15_mae_boundary_evidence.csv"
    )

    development_file = (
        OUTPUT_DIR
        / "s15_mae_boundary_development.csv"
    )

    oos_trades_file = (
        OUTPUT_DIR
        / "s15_mae_boundary_oos_trades.csv"
    )

    oos_windows_file = (
        OUTPUT_DIR
        / "s15_mae_boundary_oos_by_window.csv"
    )

    trigger_file = (
        OUTPUT_DIR
        / "s15_mae_boundary_trigger_audit.csv"
    )

    evidence.to_csv(
        evidence_file,
        index=False,
    )

    development.to_csv(
        development_file,
        index=False,
    )

    oos_trades.to_csv(
        oos_trades_file,
        index=False,
    )

    oos_windows.to_csv(
        oos_windows_file,
        index=False,
    )

    if not triggered.empty:

        triggered.to_csv(
            trigger_file,
            index=False,
        )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(evidence_file)
    print(development_file)
    print(oos_trades_file)
    print(oos_windows_file)

    if not triggered.empty:
        print(trigger_file)

    print()
    print("=" * 110)
    print("S15 MAE BOUNDARY STOP OOS TEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()