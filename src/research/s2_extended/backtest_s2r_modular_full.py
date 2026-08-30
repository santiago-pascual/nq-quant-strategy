from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# S2R MODULAR — AUTHORITATIVE REPRODUCTION
# =============================================================================
#
# IMPORTANT:
#
# The authoritative S2R research implementation does NOT rediscover entries
# from the raw dataset.
#
# It starts from the COMPLETE FROZEN S2 benchmark:
#
#     537 trades
#
# Then it applies the audited S26 recovery model to the subset enriched by S4.
#
# Frozen recovery rule:
#
#     MAE >= 0.70R
#     Recovery >= +0.20R
#     Deadline = 6 bars
#
# S27 explicitly preserves all 537 S2 trades and replaces the strategy R only
# for trades matched to the S26 execution model.
#
# This modular implementation reproduces that architecture without:
#
#     - retraining HMM
#     - rediscovering entries
#     - optimizing parameters
#     - changing the S2 universe
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

S2_PATH = RESULTS_DIR / "s2_benchmark_trades_enriched.csv"

S4_PATH = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"

S26_PATH = RESULTS_DIR / "s26_mae_recovery_integration_trades.csv"

OUTPUT_TRADES = RESULTS_DIR / "s2r_modular_authoritative_reproduction.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "s2r_modular_authoritative_windows.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "s2r_modular_authoritative_summary.csv"


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

MAE_THRESHOLD_R = 0.70
RECOVERY_LEVEL_R = 0.20
RECOVERY_DEADLINE_BARS = 6

EXPECTED_TRADES = 537
EXPECTED_TOTAL_R = 49.2652


# =============================================================================
# TRADE IDENTITY
# =============================================================================


def add_trade_identity(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    required = [
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(f"Missing trade identity columns: {missing}")

    df["_entry_ts"] = pd.to_datetime(
        df["entry_timestamp"],
        errors="coerce",
        utc=True,
    )

    df["_exit_ts"] = pd.to_datetime(
        df["exit_timestamp"],
        errors="coerce",
        utc=True,
    )

    if df["_entry_ts"].isna().any():
        raise RuntimeError("Invalid entry timestamps.")

    if df["_exit_ts"].isna().any():
        raise RuntimeError("Invalid exit timestamps.")

    df["_trade_key"] = (
        df["_entry_ts"].astype(str)
        + "|"
        + df["_exit_ts"].astype(str)
        + "|"
        + df["session_id"].astype(str).str.strip()
    )

    if df["_trade_key"].duplicated().any():
        raise RuntimeError("Duplicate trade identity detected.")

    return df


# =============================================================================
# LOAD S2
# =============================================================================


def load_s2() -> pd.DataFrame:

    print()
    print("=" * 100)
    print("LOADING COMPLETE FROZEN S2 BENCHMARK")
    print("=" * 100)

    if not S2_PATH.exists():
        raise FileNotFoundError(f"S2 benchmark not found:\n{S2_PATH}")

    df = pd.read_csv(S2_PATH)

    print(f"Path  : {S2_PATH}")
    print(f"Rows  : {len(df):,}")

    required = [
        "net_R",
        "window",
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(f"S2 missing columns: {missing}")

    df = add_trade_identity(df)

    df["_s2_R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    if df["_s2_R"].isna().any():
        raise RuntimeError("S2 contains invalid net_R values.")

    return df


# =============================================================================
# LOAD S4
# =============================================================================


def load_s4() -> pd.DataFrame:

    print()
    print("=" * 100)
    print("LOADING S4 ADVERSE / RECOVERY ENRICHMENT")
    print("=" * 100)

    if not S4_PATH.exists():
        raise FileNotFoundError(f"S4 file not found:\n{S4_PATH}")

    df = pd.read_csv(S4_PATH)

    print(f"Path  : {S4_PATH}")
    print(f"Rows  : {len(df):,}")

    required = [
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
        "net_R",
        "window",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(f"S4 missing columns: {missing}")

    df = add_trade_identity(df)

    df["_s4_row_index"] = np.arange(
        len(df),
        dtype=int,
    )

    return df


# =============================================================================
# LOAD S26
# =============================================================================


def load_s26() -> pd.DataFrame:

    print()
    print("=" * 100)
    print("LOADING AUDITED S26 RECOVERY MODEL")
    print("=" * 100)

    if not S26_PATH.exists():
        raise FileNotFoundError(f"S26 file not found:\n{S26_PATH}")

    df = pd.read_csv(S26_PATH)

    print(f"Path  : {S26_PATH}")
    print(f"Rows  : {len(df):,}")

    required = [
        "original_index",
        "window",
        "benchmark_R",
        "strategy_R",
        "state",
        "mae_bar",
        "recovery_bar",
        "exit_bar",
        "exit_type",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(f"S26 missing columns: {missing}")

    df["_s4_row_index"] = pd.to_numeric(
        df["original_index"],
        errors="coerce",
    )

    if df["_s4_row_index"].isna().any():
        raise RuntimeError("Invalid S26 original_index.")

    df["_s4_row_index"] = df["_s4_row_index"].astype(int)

    if df["_s4_row_index"].duplicated().any():
        raise RuntimeError("S26 original_index is not unique.")

    return df


# =============================================================================
# CONNECT S26 -> S4
# =============================================================================


def build_model_mapping(
    s4: pd.DataFrame,
    s26: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 100)
    print("CONNECTING S26 -> S4")
    print("=" * 100)

    if (
        not s26["_s4_row_index"]
        .between(
            0,
            len(s4) - 1,
        )
        .all()
    ):
        raise RuntimeError("S26 contains invalid S4 row indices.")

    # ---------------------------------------------------------
    # S26.original_index points directly to the S4 row.
    # ---------------------------------------------------------

    s26_s4 = s4.iloc[s26["_s4_row_index"].to_numpy()].copy()

    s26_s4 = s26_s4.reset_index(drop=True)

    execution = s26[
        [
            "strategy_R",
            "state",
            "mae_bar",
            "recovery_bar",
            "exit_bar",
            "exit_type",
        ]
    ].reset_index(drop=True)

    model = pd.concat(
        [
            s26_s4,
            execution,
        ],
        axis=1,
    )

    if model["_trade_key"].duplicated().any():
        raise RuntimeError("S26/S4 model contains duplicate trade keys.")

    print(f"S26 rows mapped to S4 : {len(model):,}")

    return model


# =============================================================================
# INTEGRATE INTO COMPLETE S2
# =============================================================================


def integrate(
    s2: pd.DataFrame,
    model: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 100)
    print("INTEGRATING FROZEN RECOVERY MODEL INTO S2")
    print("=" * 100)

    model_columns = [
        "_trade_key",
        "strategy_R",
        "state",
        "mae_bar",
        "recovery_bar",
        "exit_bar",
        "exit_type",
    ]

    model_small = model[model_columns].copy()

    model_small = model_small.rename(
        columns={
            "strategy_R": "_model_strategy_R",
            "state": "_model_state",
            "mae_bar": "_model_mae_bar",
            "recovery_bar": "_model_recovery_bar",
            "exit_bar": "_model_exit_bar",
            "exit_type": "_model_exit_type",
        }
    )

    result = s2.merge(
        model_small,
        on="_trade_key",
        how="left",
        validate="one_to_one",
        indicator="_model_match",
    )

    matched = result["_model_match"] == "both"

    unmatched = result["_model_match"] == "left_only"

    print(f"Complete S2 trades : {len(result):,}")

    print(f"S26/S4 matched     : {int(matched.sum()):,}")

    print(f"Unmatched S2        : {int(unmatched.sum()):,}")

    # ---------------------------------------------------------
    # Critical integrity condition.
    # ---------------------------------------------------------

    if len(result) != EXPECTED_TRADES:
        raise RuntimeError(f"Expected {EXPECTED_TRADES} S2 trades, got {len(result)}.")

    if int(matched.sum()) != len(model):
        raise RuntimeError("Not all S26/S4 trades matched S2.")

    # ---------------------------------------------------------
    # Preserve original S2 for unmatched trades.
    #
    # Use audited S26 strategy R for matched trades.
    # ---------------------------------------------------------

    result["_strategy_R"] = np.where(
        matched,
        pd.to_numeric(
            result["_model_strategy_R"],
            errors="coerce",
        ),
        result["_s2_R"],
    )

    result["_state"] = np.where(
        matched,
        result["_model_state"],
        "NO_RECOVERY_ENRICHMENT",
    )

    result["_mae_bar"] = np.where(
        matched,
        result["_model_mae_bar"],
        np.nan,
    )

    result["_recovery_bar"] = np.where(
        matched,
        result["_model_recovery_bar"],
        np.nan,
    )

    result["_exit_bar"] = np.where(
        matched,
        result["_model_exit_bar"],
        np.nan,
    )

    result["_exit_type"] = np.where(
        matched,
        result["_model_exit_type"],
        "ORIGINAL_S2",
    )

    result["_delta_R"] = result["_strategy_R"] - result["_s2_R"]

    return result


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(
    df: pd.DataFrame,
) -> dict:

    values = pd.to_numeric(
        df["_strategy_R"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "trades": 0,
            "total_R": 0.0,
            "mean_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
        }

    wins = values[values > 0]

    losses = values[values <= 0]

    gross_profit = float(wins.sum())

    gross_loss = float(abs(losses.sum()))

    if gross_loss == 0:
        profit_factor = np.inf if gross_profit > 0 else np.nan
    else:
        profit_factor = gross_profit / gross_loss

    equity = values.cumsum()

    drawdown = equity - equity.cummax()

    return {
        "trades": int(len(values)),
        "total_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "win_rate": float(len(wins) / len(values)),
        "profit_factor": float(profit_factor),
        "max_drawdown_R": float(drawdown.min()),
    }


# =============================================================================
# WINDOW RESULTS
# =============================================================================


def build_window_results(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for window, group in df.groupby("window", sort=True):
        original = pd.to_numeric(
            group["_s2_R"],
            errors="coerce",
        )

        strategy = pd.to_numeric(
            group["_strategy_R"],
            errors="coerce",
        )

        rows.append(
            {
                "window": int(window),
                "trades": len(group),
                "original_total_R": float(original.sum()),
                "strategy_total_R": float(strategy.sum()),
                "delta_R": float(strategy.sum() - original.sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# AUDIT
# =============================================================================


def audit(
    result: pd.DataFrame,
    metrics: dict,
) -> None:

    print()
    print("=" * 100)
    print("S2R MODULAR INTEGRITY AUDIT")
    print("=" * 100)

    assert len(result) == 537
    assert result["_trade_key"].is_unique

    strategy = pd.to_numeric(
        result["_strategy_R"],
        errors="coerce",
    )

    assert strategy.notna().all()
    assert np.isfinite(strategy.to_numpy()).all()

    print("537 trades preserved              : PASS")

    print("Unique trade identity             : PASS")

    print("Finite strategy R                 : PASS")

    print()
    print(f"Recovery threshold MAE            : {MAE_THRESHOLD_R:.2f}R")

    print(f"Recovery target                   : +{RECOVERY_LEVEL_R:.2f}R")

    print(f"Recovery deadline                 : {RECOVERY_DEADLINE_BARS} bars")

    print()
    print(f"Final trades                      : {metrics['trades']}")

    print(f"Final Total R                     : {metrics['total_R']:.10f}")

    print(f"Benchmark Total R                 : {EXPECTED_TOTAL_R:.10f}")

    print(
        f"Difference                        : "
        f"{metrics['total_R'] - EXPECTED_TOTAL_R:+.10f}"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    print("=" * 100)
    print("S2R MODULAR — AUTHORITATIVE REPRODUCTION")
    print("=" * 100)

    print()
    print("FROZEN MODEL")
    print("-" * 100)
    print(f"MAE threshold     : >= {MAE_THRESHOLD_R:.2f}R")
    print(f"Recovery target   : >= +{RECOVERY_LEVEL_R:.2f}R")
    print(f"Deadline          : {RECOVERY_DEADLINE_BARS} bars")
    print()
    print("NO HMM RETRAINING.")
    print("NO ENTRY REDISCOVERY.")
    print("NO PARAMETER OPTIMIZATION.")

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    s2 = load_s2()

    s4 = load_s4()

    s26 = load_s26()

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = build_model_mapping(
        s4=s4,
        s26=s26,
    )

    # ---------------------------------------------------------
    # INTEGRATE
    # ---------------------------------------------------------

    result = integrate(
        s2=s2,
        model=model,
    )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    metrics = calculate_metrics(result)

    windows = build_window_results(result)

    # ---------------------------------------------------------
    # AUDIT
    # ---------------------------------------------------------

    audit(
        result=result,
        metrics=metrics,
    )

    # ---------------------------------------------------------
    # SAVE TRADES
    # ---------------------------------------------------------

    output = result.copy()

    output.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "strategy": "S2R_MODULAR_AUTHORITATIVE",
                **metrics,
                "mae_threshold_R": MAE_THRESHOLD_R,
                "recovery_level_R": RECOVERY_LEVEL_R,
                "recovery_deadline_bars": RECOVERY_DEADLINE_BARS,
                "benchmark_trades": EXPECTED_TRADES,
                "benchmark_total_R": EXPECTED_TOTAL_R,
                "trade_count_match": metrics["trades"] == EXPECTED_TRADES,
                "total_R_match": np.isclose(
                    metrics["total_R"],
                    EXPECTED_TOTAL_R,
                    atol=1e-10,
                ),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)

    print(f"Trades         : {metrics['trades']}")

    print(f"Total R        : {metrics['total_R']:.10f}")

    print(f"Mean R         : {metrics['mean_R']:.10f}")

    print(f"Win rate       : {metrics['win_rate']:.10f}")

    print(f"Profit factor  : {metrics['profit_factor']:.10f}")

    print(f"Max DD R       : {metrics['max_drawdown_R']:.10f}")

    print()
    print("=" * 100)
    print("BENCHMARK")
    print("=" * 100)

    print(f"Trades         : {EXPECTED_TRADES}")

    print(f"Total R        : {EXPECTED_TOTAL_R:.10f}")

    print()
    print(f"Trade count match : {metrics['trades'] == EXPECTED_TRADES}")

    print(
        f"Total R match     : "
        f"{np.isclose(metrics['total_R'], EXPECTED_TOTAL_R, atol=1e-10)}"
    )

    print()
    print("FILES")
    print("-" * 100)
    print(OUTPUT_TRADES)
    print(OUTPUT_WINDOWS)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
