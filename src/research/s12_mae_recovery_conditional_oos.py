"""
S12 — MAE RECOVERY CONDITIONAL OOS

Core hypothesis:

    A large adverse excursion is strongly associated with failure,
    BUT some trades recover after reaching that adverse boundary.

Question:

    After MAE >= threshold, can early recovery distinguish
    trades that should remain alive from trades that should be exited?

This is a CONDITIONAL recovery test.

It is NOT an immediate MAE exit test.

Temporal split:
    Development = windows 1..11
    Holdout     = windows 12..22
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

DEV_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))

MAE_THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
]

RECOVERY_LEVELS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

# Maximum possible horizon will be clipped automatically
# to whatever path data actually exists.
RECOVERY_HORIZONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
]


# =============================================================================
# COLUMN DETECTION
# =============================================================================


def detect_path_columns(
    df: pd.DataFrame,
    prefix: str,
) -> dict[int, str]:

    pattern = re.compile(
        rf"^{re.escape(prefix)}[_]?(\d+)$",
        re.IGNORECASE,
    )

    result = {}

    for column in df.columns:
        match = pattern.match(str(column))

        if match:
            result[int(match.group(1))] = column

    return dict(sorted(result.items()))


def detect_window_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "window",
        "walk_forward_window",
        "oos_window",
        "test_window",
        "period",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError("Could not identify temporal window column.")


def detect_final_R_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "final_close_R",
        "final_close_r",
        "final_R",
        "final_r",
        "net_R",
        "net_r",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError("Could not identify final R column.")


# =============================================================================
# HELPERS
# =============================================================================


def numeric_value(
    row: pd.Series,
    column: str,
) -> float:

    value = pd.to_numeric(
        row[column],
        errors="coerce",
    )

    if pd.isna(value):
        return np.nan

    return float(value)


def path_value(
    row: pd.Series,
    columns: dict[int, str],
    bar: int,
) -> float:

    column = columns.get(bar)

    if column is None:
        return np.nan

    return numeric_value(
        row,
        column,
    )


def numeric_window(
    series: pd.Series,
) -> pd.Series:

    extracted = series.astype(str).str.extract(
        r"(\d+)",
        expand=False,
    )

    return pd.to_numeric(
        extracted,
        errors="coerce",
    )


# =============================================================================
# FIRST MAE CROSSING
# =============================================================================


def first_mae_crossing(
    row: pd.Series,
    mae_columns: dict[int, str],
    threshold: float,
) -> int | None:

    for bar in sorted(mae_columns):
        value = path_value(
            row,
            mae_columns,
            bar,
        )

        if pd.isna(value):
            continue

        if value >= threshold:
            return bar

    return None


# =============================================================================
# RECOVERY TEST
# =============================================================================


def test_recovery(
    row: pd.Series,
    close_columns: dict[int, str],
    crossing_bar: int,
    recovery_level: float,
    recovery_horizon: int,
) -> dict:
    """
    After the first MAE threshold crossing:

        recovery = close_R >= recovery_level

    within N bars after the crossing.

    Only information available after the crossing
    and before the decision horizon is used.
    """

    last_available_bar = max(close_columns.keys())

    decision_bar = min(
        crossing_bar + recovery_horizon,
        last_available_bar,
    )

    start_bar = crossing_bar + 1

    if start_bar > decision_bar:
        return {
            "recovered": False,
            "recovery_bar": np.nan,
            "decision_bar": decision_bar,
            "bars_to_recovery": np.nan,
            "decision_close_R": np.nan,
        }

    for bar in range(
        start_bar,
        decision_bar + 1,
    ):
        close_r = path_value(
            row,
            close_columns,
            bar,
        )

        if pd.isna(close_r):
            continue

        if close_r >= recovery_level:
            return {
                "recovered": True,
                "recovery_bar": bar,
                "decision_bar": decision_bar,
                "bars_to_recovery": (bar - crossing_bar),
                "decision_close_R": close_r,
            }

    final_close = path_value(
        row,
        close_columns,
        decision_bar,
    )

    return {
        "recovered": False,
        "recovery_bar": np.nan,
        "decision_bar": decision_bar,
        "bars_to_recovery": np.nan,
        "decision_close_R": final_close,
    }


# =============================================================================
# BUILD OBSERVATIONS
# =============================================================================


def build_observations(
    df: pd.DataFrame,
    mae_columns: dict[int, str],
    close_columns: dict[int, str],
    final_R_column: str,
) -> pd.DataFrame:

    records = []

    max_path_bar = min(
        max(mae_columns.keys()),
        max(close_columns.keys()),
    )

    valid_horizons = [h for h in RECOVERY_HORIZONS if h <= max_path_bar]

    for index, row in df.iterrows():
        benchmark_R = numeric_value(
            row,
            final_R_column,
        )

        window = row["_window_numeric"]

        for threshold in MAE_THRESHOLDS:
            crossing_bar = first_mae_crossing(
                row,
                mae_columns,
                threshold,
            )

            if crossing_bar is None:
                continue

            for recovery_level in RECOVERY_LEVELS:
                for recovery_horizon in valid_horizons:
                    result = test_recovery(
                        row=row,
                        close_columns=close_columns,
                        crossing_bar=crossing_bar,
                        recovery_level=recovery_level,
                        recovery_horizon=recovery_horizon,
                    )

                    records.append(
                        {
                            "index": index,
                            "window": window,
                            "benchmark_R": benchmark_R,
                            "mae_threshold": threshold,
                            "crossing_bar": crossing_bar,
                            "recovery_level": recovery_level,
                            "recovery_horizon": recovery_horizon,
                            "recovered": bool(result["recovered"]),
                            "recovery_bar": result["recovery_bar"],
                            "bars_to_recovery": result["bars_to_recovery"],
                            "decision_bar": result["decision_bar"],
                            "decision_close_R": result["decision_close_R"],
                        }
                    )

    return pd.DataFrame(records)


# =============================================================================
# METRICS
# =============================================================================


def metrics(
    values: pd.Series,
) -> dict:

    r = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if r.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
        }

    wins = int((r > 0).sum())

    losses = int((r <= 0).sum())

    gross_profit = float(r[r > 0].sum())

    gross_loss = float(-r[r < 0].sum())

    if gross_loss == 0:
        pf = np.inf if gross_profit > 0 else np.nan

    else:
        pf = gross_profit / gross_loss

    return {
        "trades": len(r),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(r),
        "mean_R": float(r.mean()),
        "total_R": float(r.sum()),
        "profit_factor": pf,
    }


# =============================================================================
# DISCOVERY
# =============================================================================


def run_discovery(
    observations: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for threshold in MAE_THRESHOLDS:
        base = observations[observations["mae_threshold"] == threshold]

        for level in RECOVERY_LEVELS:
            for horizon in RECOVERY_HORIZONS:
                subset = base[
                    (base["recovery_level"] == level)
                    & (base["recovery_horizon"] == horizon)
                ]

                if subset.empty:
                    continue

                # Explicit bool conversion.
                # This fixes the previous ~True/~False bug.
                recovered_mask = subset["recovered"].astype(bool)

                recovered = subset[recovered_mask]

                not_recovered = subset[~recovered_mask]

                recovered_metrics = metrics(recovered["benchmark_R"])

                failure_metrics = metrics(not_recovered["benchmark_R"])

                rows.append(
                    {
                        "mae_threshold": threshold,
                        "recovery_level": level,
                        "recovery_horizon": horizon,
                        "triggered_trades": len(subset),
                        "recovered_trades": len(recovered),
                        "recovery_pct": (len(recovered) / len(subset)),
                        "recovered_win_rate": recovered_metrics["win_rate"],
                        "recovered_mean_R": recovered_metrics["mean_R"],
                        "recovered_total_R": recovered_metrics["total_R"],
                        "not_recovered_trades": len(not_recovered),
                        "not_recovered_win_rate": failure_metrics["win_rate"],
                        "not_recovered_mean_R": failure_metrics["mean_R"],
                        "not_recovered_total_R": failure_metrics["total_R"],
                        "win_rate_difference": (
                            recovered_metrics["win_rate"] - failure_metrics["win_rate"]
                        ),
                        "mean_R_difference": (
                            recovered_metrics["mean_R"] - failure_metrics["mean_R"]
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# EXECUTABLE CONDITIONAL RULE
# =============================================================================


def execute_rule(
    df: pd.DataFrame,
    mae_columns: dict[int, str],
    close_columns: dict[int, str],
    final_R_column: str,
    threshold: float,
    recovery_level: float,
    recovery_horizon: int,
) -> pd.DataFrame:

    records = []

    for index, row in df.iterrows():
        benchmark_R = numeric_value(
            row,
            final_R_column,
        )

        crossing_bar = first_mae_crossing(
            row,
            mae_columns,
            threshold,
        )

        # -------------------------------------------------------------
        # MAE never reaches boundary.
        # Preserve original trade.
        # -------------------------------------------------------------

        if crossing_bar is None:
            records.append(
                {
                    "index": index,
                    "window": row["_window_numeric"],
                    "benchmark_R": benchmark_R,
                    "strategy_R": benchmark_R,
                    "triggered": False,
                    "recovered": False,
                    "decision": "BENCHMARK",
                    "crossing_bar": np.nan,
                    "decision_bar": np.nan,
                }
            )

            continue

        # -------------------------------------------------------------
        # MAE boundary reached.
        # -------------------------------------------------------------

        recovery = test_recovery(
            row=row,
            close_columns=close_columns,
            crossing_bar=crossing_bar,
            recovery_level=recovery_level,
            recovery_horizon=recovery_horizon,
        )

        recovered = bool(recovery["recovered"])

        if recovered:
            # ---------------------------------------------------------
            # Recovery confirmed.
            #
            # Keep the original benchmark outcome.
            # ---------------------------------------------------------

            strategy_R = benchmark_R

            decision = "RECOVERY_CONTINUE"

        else:
            # ---------------------------------------------------------
            # No recovery.
            #
            # Exit at the last observable close
            # in the recovery horizon.
            # ---------------------------------------------------------

            decision_close = recovery["decision_close_R"]

            if pd.isna(decision_close):
                strategy_R = benchmark_R

                decision = "BENCHMARK_FALLBACK"

            else:
                strategy_R = float(decision_close)

                decision = "EARLY_EXIT"

        records.append(
            {
                "index": index,
                "window": row["_window_numeric"],
                "benchmark_R": benchmark_R,
                "strategy_R": strategy_R,
                "triggered": True,
                "recovered": recovered,
                "decision": decision,
                "crossing_bar": crossing_bar,
                "decision_bar": recovery["decision_bar"],
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S12 MAE RECOVERY CONDITIONAL OOS")
    print("=" * 110)

    print()
    print("Frozen benchmark:")
    print(f"  Stop       = {STOP_POINTS} points")
    print(f"  RR         = {RR}")

    print()
    print(
        "Development windows:",
        DEV_WINDOWS,
    )

    print(
        "Holdout windows    :",
        HOLDOUT_WINDOWS,
    )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("LOADING ENRICHED DATASET")
    print("=" * 110)

    print(INPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH.resolve())

    df = pd.read_csv(INPUT_PATH)

    print(f"Trades loaded: {len(df)}")

    # -------------------------------------------------------------------------
    # DETECT PATHS
    # -------------------------------------------------------------------------

    mae_columns = detect_path_columns(
        df,
        "mae",
    )

    close_columns = detect_path_columns(
        df,
        "close",
    )

    if not mae_columns:
        raise RuntimeError("No MAE path columns found.")

    if not close_columns:
        raise RuntimeError("No close path columns found.")

    window_column = detect_window_column(df)

    final_R_column = detect_final_R_column(df)

    df = df.copy()

    df["_window_numeric"] = numeric_window(df[window_column])

    print()
    print("Detected paths:")
    print(f"  MAE bars   : {len(mae_columns)}")

    print(f"  MAE range  : {min(mae_columns)} -> {max(mae_columns)}")

    print(f"  Close bars : {len(close_columns)}")

    print(f"  Close range: {min(close_columns)} -> {max(close_columns)}")

    # Only test horizons that the actual dataset can support.
    max_path_bar = min(
        max(mae_columns.keys()),
        max(close_columns.keys()),
    )

    valid_horizons = [h for h in RECOVERY_HORIZONS if h <= max_path_bar]

    print()
    print("Valid recovery horizons:")

    print(
        " ",
        valid_horizons,
    )

    # -------------------------------------------------------------------------
    # BUILD OBSERVATIONS
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BUILDING MAE RECOVERY OBSERVATIONS")
    print("=" * 110)

    observations = build_observations(
        df=df,
        mae_columns=mae_columns,
        close_columns=close_columns,
        final_R_column=final_R_column,
    )

    print(f"Recovery observations: {len(observations)}")

    # -------------------------------------------------------------------------
    # DISCOVERY
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("1. RECOVERY VS NON-RECOVERY")
    print("=" * 110)

    discovery = run_discovery(observations)

    display = discovery.sort_values(
        [
            "win_rate_difference",
            "mean_R_difference",
        ],
        ascending=False,
    ).head(30)

    print(display.to_string(index=False))

    # -------------------------------------------------------------------------
    # DEVELOPMENT
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("2. DEVELOPMENT RULE SEARCH")
    print("=" * 110)

    development_rows = []

    for threshold in MAE_THRESHOLDS:
        for level in RECOVERY_LEVELS:
            for horizon in valid_horizons:
                development_df = df[df["_window_numeric"].isin(DEV_WINDOWS)]

                result = execute_rule(
                    df=development_df,
                    mae_columns=mae_columns,
                    close_columns=close_columns,
                    final_R_column=final_R_column,
                    threshold=threshold,
                    recovery_level=level,
                    recovery_horizon=horizon,
                )

                if result.empty:
                    continue

                benchmark = metrics(result["benchmark_R"])

                strategy = metrics(result["strategy_R"])

                triggered = int(result["triggered"].sum())

                recovered = int(result["recovered"].sum())

                development_rows.append(
                    {
                        "mae_threshold": threshold,
                        "recovery_level": level,
                        "recovery_horizon": horizon,
                        "trades": strategy["trades"],
                        "triggered_trades": triggered,
                        "recovered_trades": recovered,
                        "recovery_pct": (
                            recovered / triggered if triggered else np.nan
                        ),
                        "benchmark_R": benchmark["total_R"],
                        "strategy_R": strategy["total_R"],
                        "delta_R": (strategy["total_R"] - benchmark["total_R"]),
                        "benchmark_WR": benchmark["win_rate"],
                        "strategy_WR": strategy["win_rate"],
                        "delta_WR": (strategy["win_rate"] - benchmark["win_rate"]),
                        "benchmark_PF": benchmark["profit_factor"],
                        "strategy_PF": strategy["profit_factor"],
                    }
                )

    development = pd.DataFrame(development_rows)

    development = development.sort_values(
        [
            "delta_R",
            "delta_WR",
        ],
        ascending=False,
    )

    print(development.head(30).to_string(index=False))

    # -------------------------------------------------------------------------
    # SELECT
    # -------------------------------------------------------------------------

    selected = development.iloc[0]

    selected_threshold = float(selected["mae_threshold"])

    selected_level = float(selected["recovery_level"])

    selected_horizon = int(selected["recovery_horizon"])

    print()
    print("=" * 110)
    print("SELECTED DEVELOPMENT RULE")
    print("=" * 110)

    print(f"MAE threshold    : {selected_threshold:.2f}R")

    print(f"Recovery level   : {selected_level:.2f}R")

    print(f"Recovery horizon : {selected_horizon} bars")

    print(f"Development ΔR   : {selected['delta_R']:.4f}")

    # -------------------------------------------------------------------------
    # OOS
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("3. HOLDOUT OOS")
    print("=" * 110)

    holdout_df = df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)]

    holdout = execute_rule(
        df=holdout_df,
        mae_columns=mae_columns,
        close_columns=close_columns,
        final_R_column=final_R_column,
        threshold=selected_threshold,
        recovery_level=selected_level,
        recovery_horizon=selected_horizon,
    )

    benchmark = metrics(holdout["benchmark_R"])

    strategy = metrics(holdout["strategy_R"])

    print()
    print("FROZEN RULE")

    print(f"  MAE >= {selected_threshold:.2f}R")

    print(f"  Recovery >= {selected_level:.2f}R")

    print(f"  Horizon = {selected_horizon} bars")

    print()
    print("BENCHMARK HOLDOUT")

    print(f"  Trades    : {benchmark['trades']}")

    print(f"  Win rate  : {benchmark['win_rate']:.4f}")

    print(f"  Mean R    : {benchmark['mean_R']:.4f}")

    print(f"  Total R   : {benchmark['total_R']:.4f}")

    print(f"  PF        : {benchmark['profit_factor']}")

    print()
    print("RECOVERY-CONDITIONAL STRATEGY")

    print(f"  Trades    : {strategy['trades']}")

    print(f"  Win rate  : {strategy['win_rate']:.4f}")

    print(f"  Mean R    : {strategy['mean_R']:.4f}")

    print(f"  Total R   : {strategy['total_R']:.4f}")

    print(f"  PF        : {strategy['profit_factor']}")

    print()
    print("IMPROVEMENT")

    print(f"  Delta R   : {strategy['total_R'] - benchmark['total_R']:.4f}")

    print(f"  Delta WR  : {strategy['win_rate'] - benchmark['win_rate']:.4f}")

    # -------------------------------------------------------------------------
    # OOS BY WINDOW
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("4. WINDOW-BY-WINDOW OOS")
    print("=" * 110)

    window_rows = []

    for window, group in holdout.groupby("window"):
        bm = metrics(group["benchmark_R"])

        st = metrics(group["strategy_R"])

        window_rows.append(
            {
                "window": int(window),
                "trades": len(group),
                "triggered_trades": int(group["triggered"].sum()),
                "recovered_trades": int(group["recovered"].sum()),
                "benchmark_R": bm["total_R"],
                "strategy_R": st["total_R"],
                "delta_R": (st["total_R"] - bm["total_R"]),
                "benchmark_WR": bm["win_rate"],
                "strategy_WR": st["win_rate"],
                "benchmark_PF": bm["profit_factor"],
                "strategy_PF": st["profit_factor"],
            }
        )

    window_df = pd.DataFrame(window_rows)

    print(window_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery_path = OUTPUT_DIR / "s12_mae_recovery_discovery.csv"

    development_path = OUTPUT_DIR / "s12_mae_recovery_development.csv"

    holdout_path = OUTPUT_DIR / "s12_mae_recovery_holdout_trades.csv"

    window_path = OUTPUT_DIR / "s12_mae_recovery_holdout_by_window.csv"

    observations_path = OUTPUT_DIR / "s12_mae_recovery_observations.csv"

    discovery.to_csv(
        discovery_path,
        index=False,
    )

    development.to_csv(
        development_path,
        index=False,
    )

    holdout.to_csv(
        holdout_path,
        index=False,
    )

    window_df.to_csv(
        window_path,
        index=False,
    )

    observations.to_csv(
        observations_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(discovery_path.resolve())

    print(development_path.resolve())

    print(holdout_path.resolve())

    print(window_path.resolve())

    print(observations_path.resolve())

    print()
    print("=" * 110)
    print("S12 MAE RECOVERY CONDITIONAL OOS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
