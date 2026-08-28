from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


# =============================================================================
# S27 — FULL STRATEGY OOS
# =============================================================================
#
# COMPLETE S2 + FROZEN S26 MAE/RECOVERY MODEL
#
# S2:
#   537 complete benchmark trades
#
# S4:
#   195 trades with MAE/recovery enrichment
#
# S26:
#   audited frozen execution model
#
# Frozen rule:
#   MAE >= 0.70R
#   Recovery >= +0.20R
#   Deadline = 6 bars
#
# IMPORTANT:
#
# S26.original_index refers to the row index of S4.
#
# S4 is then linked to S2 using:
#
#   entry_timestamp
#   exit_timestamp
#   session_id
#
# No optimization is performed here.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"


S2_FILE = RESULTS_DIR / "s2_benchmark_trades_enriched.csv"

S4_FILE = RESULTS_DIR / "s4_adverse_recovery_enriched.csv"

S26_FILE = RESULTS_DIR / "s26_mae_recovery_integration_trades.csv"


SUMMARY_FILE = RESULTS_DIR / "s27_full_strategy_summary.csv"

TRADES_FILE = RESULTS_DIR / "s27_full_strategy_trades.csv"

WINDOWS_FILE = RESULTS_DIR / "s27_full_strategy_windows.csv"

ATTRIBUTION_FILE = RESULTS_DIR / "s27_full_strategy_attribution.csv"

OOS_FILE = RESULTS_DIR / "s27_full_strategy_oos_summary.csv"


MAE_THRESHOLD_R = 0.70
RECOVERY_LEVEL_R = 0.20
RECOVERY_DEADLINE = 6

DEVELOPMENT_WINDOWS = list(range(1, 12))
HOLDOUT_WINDOWS = list(range(12, 23))


# =============================================================================
# METRICS
# =============================================================================


def metrics(
    df: pd.DataFrame,
    r_col: str,
) -> dict:

    values = pd.to_numeric(
        df[r_col],
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "mean_R": np.nan,
            "total_R": 0.0,
            "gross_profit_R": 0.0,
            "gross_loss_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
        }

    wins = values[values > 0]
    losses = values[values <= 0]

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    if gross_loss == 0:
        profit_factor = math.inf if gross_profit > 0 else np.nan

    else:
        profit_factor = gross_profit / gross_loss

    equity = values.cumsum()

    drawdown = equity - equity.cummax()

    return {
        "trades": int(len(values)),
        "wins": int((values > 0).sum()),
        "losses": int((values <= 0).sum()),
        "win_rate": float((values > 0).mean()),
        "mean_R": float(values.mean()),
        "total_R": float(values.sum()),
        "gross_profit_R": gross_profit,
        "gross_loss_R": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_R": float(drawdown.min()),
    }


# =============================================================================
# TRADE KEY
# =============================================================================


def add_trade_key(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    required = [
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
    ]

    missing = [c for c in required if c not in df.columns]

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
        raise RuntimeError("Invalid entry timestamps detected.")

    if df["_exit_ts"].isna().any():
        raise RuntimeError("Invalid exit timestamps detected.")

    df["_trade_key"] = (
        df["_entry_ts"].astype(str)
        + "|"
        + df["_exit_ts"].astype(str)
        + "|"
        + df["session_id"].astype(str).str.strip()
    )

    if df["_trade_key"].duplicated().any():
        duplicates = df[df["_trade_key"].duplicated(keep=False)][
            [
                "entry_timestamp",
                "exit_timestamp",
                "session_id",
                "_trade_key",
            ]
        ]

        print(duplicates.head(20).to_string(index=False))

        raise RuntimeError("Trade identity key is not unique.")

    return df


# =============================================================================
# LOAD S2
# =============================================================================


def load_s2() -> pd.DataFrame:

    print()
    print("=" * 110)
    print("LOADING COMPLETE S2 BENCHMARK")
    print("=" * 110)

    df = pd.read_csv(S2_FILE)

    print(S2_FILE)
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    required = [
        "net_R",
        "window",
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"S2 missing columns: {missing}")

    df = add_trade_key(df)

    df["_s2_R"] = pd.to_numeric(
        df["net_R"],
        errors="coerce",
    )

    df["_window_numeric"] = pd.to_numeric(
        df["window"],
        errors="coerce",
    )

    if df["_s2_R"].isna().any():
        raise RuntimeError("S2 contains invalid net_R values.")

    print(f"S2 unique trade keys: {df['_trade_key'].is_unique}")

    return df


# =============================================================================
# LOAD S4
# =============================================================================


def load_s4() -> pd.DataFrame:

    print()
    print("=" * 110)
    print("LOADING S4 RECOVERY ENRICHMENT")
    print("=" * 110)

    df = pd.read_csv(S4_FILE)

    print(S4_FILE)
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    required = [
        "entry_timestamp",
        "exit_timestamp",
        "session_id",
        "net_R",
        "window",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"S4 missing columns: {missing}")

    df = add_trade_key(df)

    df["_s4_row_index"] = np.arange(len(df))

    return df


# =============================================================================
# LOAD S26
# =============================================================================


def load_s26() -> pd.DataFrame:

    print()
    print("=" * 110)
    print("LOADING AUDITED S26 RESULT")
    print("=" * 110)

    df = pd.read_csv(S26_FILE)

    print(S26_FILE)
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

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

    missing = [c for c in required if c not in df.columns]

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
# CONNECT S26 -> S4 -> S2
# =============================================================================


def build_model_mapping(
    s2: pd.DataFrame,
    s4: pd.DataFrame,
    s26: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 110)
    print("CONNECTING S26 -> S4 -> S2")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # S26 -> S4
    # -------------------------------------------------------------------------

    if (
        not s26["_s4_row_index"]
        .between(
            0,
            len(s4) - 1,
        )
        .all()
    ):
        raise RuntimeError("S26 contains invalid S4 row indices.")

    s26_s4 = s4.iloc[s26["_s4_row_index"].to_numpy()].copy()

    s26_s4 = s26_s4.reset_index(drop=True)

    print("S26 -> S4 index mapping:")

    print(f"  S26 rows: {len(s26)}")

    print(f"  Valid indices: {len(s26_s4)} / {len(s26)}")

    # -------------------------------------------------------------------------
    # Attach S26 execution fields.
    # -------------------------------------------------------------------------

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
            s26_s4.reset_index(drop=True),
            execution,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Verify identity keys are present in S2.
    # -------------------------------------------------------------------------

    s2_keys = set(s2["_trade_key"])

    matches = model["_trade_key"].isin(s2_keys)

    print("S4 -> S2 identity mapping:")

    print(f"  Matched: {int(matches.sum())} / {len(model)}")

    if not matches.all():
        bad = model.loc[
            ~matches,
            [
                "entry_timestamp",
                "exit_timestamp",
                "session_id",
                "_trade_key",
            ],
        ]

        print(bad.head(20).to_string(index=False))

        raise RuntimeError("Some S4 trades cannot be mapped to S2.")

    # -------------------------------------------------------------------------
    # Verify model uniqueness.
    # -------------------------------------------------------------------------

    if model["_trade_key"].duplicated().any():
        raise RuntimeError("S26/S4 model contains duplicate trade keys.")

    print("Model trade identity: PASS")

    return model


# =============================================================================
# INTEGRATE FULL S2
# =============================================================================


def integrate(
    s2: pd.DataFrame,
    model: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 110)
    print("INTEGRATING FROZEN MODEL INTO COMPLETE S2")
    print("=" * 110)

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

    print(f"Complete S2 trades : {len(result)}")

    print(f"S26/S4 matched     : {int(matched.sum())}")

    print(f"Unmatched S2        : {int(unmatched.sum())}")

    if int(matched.sum()) != len(model):
        raise RuntimeError("Not all S26 trades matched the complete S2 dataset.")

    # -------------------------------------------------------------------------
    # Strategy R.
    #
    # Matched trades:
    #     use S26 audited result.
    #
    # Unmatched trades:
    #     preserve original S2.
    # -------------------------------------------------------------------------

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
# AUDIT
# =============================================================================


def audit(
    df: pd.DataFrame,
):

    print()
    print("=" * 110)
    print("S27 INTEGRATION AUDIT")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # COMPLETE DATASET PRESERVED
    # -------------------------------------------------------------------------

    assert len(df) == 537

    print("537 trades preserved          : PASS")

    # -------------------------------------------------------------------------
    # UNIQUE TRADE IDENTITY
    # -------------------------------------------------------------------------

    assert df["_trade_key"].is_unique

    print("Unique trade identity         : PASS")

    # -------------------------------------------------------------------------
    # FINITE STRATEGY R
    # -------------------------------------------------------------------------

    strategy = pd.to_numeric(
        df["_strategy_R"],
        errors="coerce",
    )

    assert strategy.notna().all()

    assert np.isfinite(strategy.to_numpy()).all()

    print("Finite strategy R             : PASS")

    # -------------------------------------------------------------------------
    # MODEL COVERAGE
    # -------------------------------------------------------------------------

    matched = df[df["_model_match"] == "both"]

    unmatched = df[df["_model_match"] == "left_only"]

    assert len(matched) == 195
    assert len(unmatched) == 342

    print("195 model trades integrated   : PASS")

    print("342 original S2 trades kept   : PASS")

    # -------------------------------------------------------------------------
    # UNMATCHED TRADES MUST REMAIN EXACTLY S2
    # -------------------------------------------------------------------------

    assert np.allclose(
        unmatched["_strategy_R"].to_numpy(),
        unmatched["_s2_R"].to_numpy(),
        atol=1e-12,
    )

    print("Unmatched trades unchanged    : PASS")

    # -------------------------------------------------------------------------
    # MATCHED TRADES MUST EQUAL S26
    #
    # This is the critical audit.
    #
    # We do NOT assume that FAILED_TO_RECOVER == original S2 R.
    #
    # Instead, every enriched trade must use the audited S26 result.
    # -------------------------------------------------------------------------

    expected = pd.to_numeric(
        matched["_model_strategy_R"],
        errors="coerce",
    )

    actual = pd.to_numeric(
        matched["_strategy_R"],
        errors="coerce",
    )

    assert expected.notna().all()

    assert np.isfinite(expected.to_numpy()).all()

    assert np.allclose(
        actual.to_numpy(),
        expected.to_numpy(),
        atol=1e-12,
    )

    print("Matched trades equal S26     : PASS")

    # -------------------------------------------------------------------------
    # RECOVERED STATES
    # -------------------------------------------------------------------------

    recovered = df[df["_state"] == "RECOVERED"]

    if len(recovered) > 0:
        recovered_expected = pd.to_numeric(
            recovered["_model_strategy_R"],
            errors="coerce",
        )

        recovered_actual = pd.to_numeric(
            recovered["_strategy_R"],
            errors="coerce",
        )

        assert np.allclose(
            recovered_actual.to_numpy(),
            recovered_expected.to_numpy(),
            atol=1e-12,
        )

    print("Recovered states use S26     : PASS")

    # -------------------------------------------------------------------------
    # FAILED RECOVERY STATES
    #
    # IMPORTANT:
    #
    # Do NOT compare them to S2.
    # They are allowed to differ because S26 defines their execution result.
    # -------------------------------------------------------------------------

    failed = df[df["_state"] == "FAILED_TO_RECOVER"]

    if len(failed) > 0:
        failed_expected = pd.to_numeric(
            failed["_model_strategy_R"],
            errors="coerce",
        )

        failed_actual = pd.to_numeric(
            failed["_strategy_R"],
            errors="coerce",
        )

        assert np.allclose(
            failed_actual.to_numpy(),
            failed_expected.to_numpy(),
            atol=1e-12,
        )

    print("Failed recovery states use S26: PASS")

    # -------------------------------------------------------------------------
    # STATE ACCOUNTING
    # -------------------------------------------------------------------------

    recovered_count = int((df["_state"] == "RECOVERED").sum())

    failed_count = int((df["_state"] == "FAILED_TO_RECOVER").sum())

    no_model_count = int((df["_state"] == "NO_RECOVERY_ENRICHMENT").sum())

    assert recovered_count + failed_count + no_model_count == 537

    print("State accounting             : PASS")

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print()
    print("S27 AUDIT: PASS")


# =============================================================================
# SUMMARY
# =============================================================================


def build_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    samples = {
        "FULL_DATASET": df,
        "DEVELOPMENT": df[df["_window_numeric"].isin(DEVELOPMENT_WINDOWS)],
        "HOLDOUT_OOS": df[df["_window_numeric"].isin(HOLDOUT_WINDOWS)],
    }

    rows = []

    for name, subset in samples.items():
        original = metrics(
            subset,
            "_s2_R",
        )

        strategy = metrics(
            subset,
            "_strategy_R",
        )

        rows.append(
            {
                "sample": name,
                "trades": strategy["trades"],
                "original_total_R": original["total_R"],
                "strategy_total_R": strategy["total_R"],
                "delta_R": strategy["total_R"] - original["total_R"],
                "original_mean_R": original["mean_R"],
                "strategy_mean_R": strategy["mean_R"],
                "delta_mean_R": strategy["mean_R"] - original["mean_R"],
                "original_win_rate": original["win_rate"],
                "strategy_win_rate": strategy["win_rate"],
                "delta_win_rate": strategy["win_rate"] - original["win_rate"],
                "original_PF": original["profit_factor"],
                "strategy_PF": strategy["profit_factor"],
                "original_max_DD": original["max_drawdown_R"],
                "strategy_max_DD": strategy["max_drawdown_R"],
                "delta_max_DD": strategy["max_drawdown_R"] - original["max_drawdown_R"],
                "model_trades": int((subset["_model_match"] == "both").sum()),
                "recovered": int((subset["_state"] == "RECOVERED").sum()),
                "failed_recovery": int((subset["_state"] == "FAILED_TO_RECOVER").sum()),
                "changed_trades": int((subset["_delta_R"].abs() > 1e-12).sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# WINDOW REPORT
# =============================================================================


def build_windows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for window, group in df.groupby(
        "_window_numeric",
        dropna=False,
    ):
        original = metrics(
            group,
            "_s2_R",
        )

        strategy = metrics(
            group,
            "_strategy_R",
        )

        rows.append(
            {
                "window": window,
                "trades": len(group),
                "model_trades": int((group["_model_match"] == "both").sum()),
                "recovered": int((group["_state"] == "RECOVERED").sum()),
                "failed_recovery": int((group["_state"] == "FAILED_TO_RECOVER").sum()),
                "changed_trades": int((group["_delta_R"].abs() > 1e-12).sum()),
                "original_total_R": original["total_R"],
                "strategy_total_R": strategy["total_R"],
                "delta_R": strategy["total_R"] - original["total_R"],
                "original_mean_R": original["mean_R"],
                "strategy_mean_R": strategy["mean_R"],
                "original_win_rate": original["win_rate"],
                "strategy_win_rate": strategy["win_rate"],
                "original_PF": original["profit_factor"],
                "strategy_PF": strategy["profit_factor"],
                "original_max_DD": original["max_drawdown_R"],
                "strategy_max_DD": strategy["max_drawdown_R"],
            }
        )

    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


# =============================================================================
# ATTRIBUTION
# =============================================================================


def build_attribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    def bucket(row):

        delta = row["_delta_R"]
        original = row["_s2_R"]

        if np.isclose(
            delta,
            0.0,
            atol=1e-12,
        ):
            if original > 0:
                return "UNCHANGED_WIN"

            return "UNCHANGED_LOSS"

        if original <= 0 and delta > 0:
            return "LOSING_TRADE_IMPROVED"

        if original > 0 and delta > 0:
            return "WINNING_TRADE_IMPROVED"

        if original > 0 and delta < 0:
            return "WINNING_TRADE_WORSENED"

        if original <= 0 and delta < 0:
            return "LOSING_TRADE_WORSENED"

        return "OTHER"

    data["_attribution"] = data.apply(
        bucket,
        axis=1,
    )

    rows = []

    for name, group in data.groupby("_attribution"):
        rows.append(
            {
                "attribution": name,
                "trades": len(group),
                "original_total_R": group["_s2_R"].sum(),
                "strategy_total_R": group["_strategy_R"].sum(),
                "delta_R": group["_delta_R"].sum(),
                "mean_delta_R": group["_delta_R"].mean(),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# OOS SUMMARY
# =============================================================================


def build_oos_summary(
    windows: pd.DataFrame,
) -> pd.DataFrame:

    oos = windows[windows["window"].isin(HOLDOUT_WINDOWS)].copy()

    if oos.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "holdout_windows": len(oos),
                "positive_windows": int((oos["delta_R"] > 0).sum()),
                "negative_windows": int((oos["delta_R"] < 0).sum()),
                "flat_windows": int((oos["delta_R"] == 0).sum()),
                "positive_window_pct": float((oos["delta_R"] > 0).mean()),
                "total_original_R": float(oos["original_total_R"].sum()),
                "total_strategy_R": float(oos["strategy_total_R"].sum()),
                "total_delta_R": float(oos["delta_R"].sum()),
                "mean_window_delta_R": float(oos["delta_R"].mean()),
                "best_window_delta_R": float(oos["delta_R"].max()),
                "worst_window_delta_R": float(oos["delta_R"].min()),
            }
        ]
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 110)
    print("S27 FULL STRATEGY OOS")
    print("=" * 110)

    print()
    print("Complete S2 benchmark + audited S26 frozen recovery integration.")

    print()
    print("FROZEN MODEL")

    print(f"  MAE >= {MAE_THRESHOLD_R:.2f}R")

    print(f"  Recovery >= +{RECOVERY_LEVEL_R:.2f}R")

    print(f"  Deadline = {RECOVERY_DEADLINE} bars")

    print()
    print("NO OPTIMIZATION.")

    # =========================================================================
    # LOAD
    # =========================================================================

    s2 = load_s2()

    s4 = load_s4()

    s26 = load_s26()

    # =========================================================================
    # BUILD MODEL MAPPING
    # =========================================================================

    model = build_model_mapping(
        s2,
        s4,
        s26,
    )

    # =========================================================================
    # INTEGRATE
    # =========================================================================

    result = integrate(
        s2,
        model,
    )

    # =========================================================================
    # AUDIT
    # =========================================================================

    audit(result)

    # =========================================================================
    # REPORTS
    # =========================================================================

    summary = build_summary(result)

    windows = build_windows(result)

    attribution = build_attribution(result)

    oos = build_oos_summary(windows)

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================

    print()
    print("=" * 110)
    print("S2 ORIGINAL vs S2 + RECOVERY")
    print("=" * 110)

    print(summary.to_string(index=False))

    print()
    print("=" * 110)
    print("TRADE-LEVEL ATTRIBUTION")
    print("=" * 110)

    print(attribution.to_string(index=False))

    print()
    print("=" * 110)
    print("WINDOW-BY-WINDOW")
    print("=" * 110)

    print(windows.to_string(index=False))

    print()
    print("=" * 110)
    print("FINAL HOLDOUT OOS")
    print("=" * 110)

    print(oos.to_string(index=False))

    # =========================================================================
    # SAVE
    # =========================================================================

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    result.to_csv(
        TRADES_FILE,
        index=False,
    )

    windows.to_csv(
        WINDOWS_FILE,
        index=False,
    )

    attribution.to_csv(
        ATTRIBUTION_FILE,
        index=False,
    )

    oos.to_csv(
        OOS_FILE,
        index=False,
    )

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print(SUMMARY_FILE)
    print(TRADES_FILE)
    print(WINDOWS_FILE)
    print(ATTRIBUTION_FILE)
    print(OOS_FILE)

    print()
    print("=" * 110)
    print("S27 FULL STRATEGY OOS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
