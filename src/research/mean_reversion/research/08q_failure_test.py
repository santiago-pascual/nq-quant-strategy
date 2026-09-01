from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08Q
# FAILURE TEST — FROZEN CANDIDATES
# =============================================================================
#
# Purpose:
#   Study WHY the frozen candidates win and lose.
#
# Frozen candidates:
#
# MRS2 = SHORT | HMM 2 | VOL 80-100 | Z 2.0 | TP 5 | SL 2 | H 5
# MRL1 = LONG  | HMM 1 | VOL 20-40  | Z 2.5 | TP 5 | SL 2 | H 20
# MRL2 = LONG  | HMM 2 | VOL 60-80  | Z 3.5 | TP 5 | SL 2 | H 2
#
# NO:
#   - parameter optimization
#   - TP/SL optimization
#   - HMM retraining
#   - volatility optimization
#   - filtering
#   - strategy modification
#
# This research ONLY describes the behavior of the frozen trades.
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

TRADES_PATH = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

OUTPUT_TRADES = RESULTS_DIR / "research_08q_failure_trade_analysis.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08q_failure_summary.csv"

OUTPUT_HMM = RESULTS_DIR / "research_08q_failure_by_hmm.csv"

OUTPUT_VOL = RESULTS_DIR / "research_08q_failure_by_volatility.csv"

OUTPUT_WINDOW = RESULTS_DIR / "research_08q_failure_by_window.csv"

OUTPUT_Z = RESULTS_DIR / "research_08q_failure_by_zscore.csv"

OUTPUT_DURATION = RESULTS_DIR / "research_08q_failure_duration.csv"


# =============================================================================
# FROZEN CANDIDATES
# =============================================================================

CANDIDATES = {
    "MRS2": {
        "candidate_id": "C01",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 5,
    },
    "MRL1": {
        "candidate_id": "C02",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 20,
    },
    "MRL2": {
        "candidate_id": "C06",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 2,
    },
}


# =============================================================================
# HELPERS
# =============================================================================


def section(title):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_mean(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(series.mean()) if series.notna().any() else np.nan


def safe_median(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(series.median()) if series.notna().any() else np.nan


def safe_quantile(series, q):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return float(series.quantile(q)) if len(series) else np.nan


# =============================================================================
# LOAD TRADES
# =============================================================================


def load_trades():

    section("LOADING FROZEN FULL-SAMPLE TRADES")

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{TRADES_PATH}\n\nRun Research 08P first.")

    trades = pd.read_csv(TRADES_PATH)

    required = [
        "strategy_name",
        "candidate_id",
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "tp",
        "sl",
        "rr",
        "horizon",
        "event_id",
        "window",
        "timestamp",
        "entry",
        "result",
        "r",
        "bars_to_result",
    ]

    missing = [c for c in required if c not in trades.columns]

    if missing:
        raise RuntimeError(f"Missing trade columns: {missing}")

    trades["event_id"] = pd.to_numeric(
        trades["event_id"],
        errors="raise",
    ).astype(np.int64)

    trades["window"] = pd.to_numeric(
        trades["window"],
        errors="raise",
    ).astype(np.int16)

    trades["hmm_state"] = pd.to_numeric(
        trades["hmm_state"],
        errors="raise",
    ).astype(np.int8)

    trades["zscore"] = pd.to_numeric(
        trades["zscore"],
        errors="coerce",
    )

    trades["entry"] = pd.to_numeric(
        trades["entry"],
        errors="coerce",
    )

    trades["r"] = pd.to_numeric(
        trades["r"],
        errors="coerce",
    )

    trades["bars_to_result"] = pd.to_numeric(
        trades["bars_to_result"],
        errors="coerce",
    )

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
        errors="coerce",
    )

    print(f"Trades loaded: {len(trades):,}")

    print("\nStrategy counts:")

    print(trades["strategy_name"].value_counts())

    return trades


# =============================================================================
# LOAD METADATA
# =============================================================================


def load_metadata():

    section("LOADING EVENT METADATA")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)

    required = [
        "event_id",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in metadata.columns]

    if missing:
        raise RuntimeError(f"Missing metadata columns: {missing}")

    metadata["event_id"] = pd.to_numeric(
        metadata["event_id"],
        errors="raise",
    ).astype(np.int64)

    metadata["close"] = pd.to_numeric(
        metadata["close"],
        errors="coerce",
    )

    metadata["zscore_30"] = pd.to_numeric(
        metadata["zscore_30"],
        errors="coerce",
    )

    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"],
        utc=True,
        errors="coerce",
    )

    metadata = metadata[
        [
            "event_id",
            "timestamp",
            "close",
            "zscore_30",
        ]
    ]

    print(f"Metadata rows: {len(metadata):,}")

    return metadata


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_path_cache():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{PATH_CACHE_PATH}")

    cache = np.load(
        PATH_CACHE_PATH,
        allow_pickle=False,
    )

    required = [
        "future_close",
    ]

    for key in required:
        if key not in cache.files:
            raise RuntimeError(f"Missing cache key: {key}")

    future_close = cache["future_close"]

    print(f"future_close shape: {future_close.shape}")

    return future_close


# =============================================================================
# MAE / MFE RECONSTRUCTION
# =============================================================================


def calculate_excursions(
    trades,
    future_close,
):

    section("RECONSTRUCTING MAE / MFE")

    rows = []

    for row in trades.itertuples(index=False):
        event_id = int(row.event_id)

        entry = float(row.entry)

        horizon = int(row.horizon)

        path = future_close[
            event_id,
            :horizon,
        ]

        path = path[np.isfinite(path)]

        if len(path) == 0:
            mae = np.nan
            mfe = np.nan

        else:
            if row.side == "LONG":
                movement = path - entry

            else:
                movement = entry - path

            # Maximum favorable excursion
            mfe = float(np.max(movement))

            # Maximum adverse excursion
            mae = float(np.min(movement))

        rows.append(
            {
                "strategy_name": row.strategy_name,
                "candidate_id": row.candidate_id,
                "side": row.side,
                "hmm_state": row.hmm_state,
                "vol_bucket": row.vol_bucket,
                "zscore": row.zscore,
                "tp": row.tp,
                "sl": row.sl,
                "rr": row.rr,
                "horizon": row.horizon,
                "event_id": event_id,
                "window": row.window,
                "timestamp": row.timestamp,
                "entry": entry,
                "result": row.result,
                "r": row.r,
                "bars_to_result": row.bars_to_result,
                "mae": mae,
                "mfe": mfe,
            }
        )

    result = pd.DataFrame(rows)

    print(f"Excursion rows: {len(result):,}")

    return result


# =============================================================================
# MAE / MFE SUMMARY
# =============================================================================


def build_summary(trades):

    section("FAILURE / EXCURSION SUMMARY")

    rows = []

    for name in CANDIDATES:
        group = trades[trades["strategy_name"] == name]

        wins = group[group["result"] == "WIN"]

        losses = group[group["result"] == "LOSS"]

        rows.append(
            {
                "strategy_name": name,
                "observations": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "wr": (
                    len(wins) / (len(wins) + len(losses))
                    if (len(wins) + len(losses))
                    else np.nan
                ),
                "mean_mae_all": safe_mean(group["mae"]),
                "median_mae_all": safe_median(group["mae"]),
                "p10_mae_all": safe_quantile(
                    group["mae"],
                    0.10,
                ),
                "p90_mae_all": safe_quantile(
                    group["mae"],
                    0.90,
                ),
                "mean_mfe_all": safe_mean(group["mfe"]),
                "median_mfe_all": safe_median(group["mfe"]),
                "mean_mae_winners": safe_mean(wins["mae"]),
                "mean_mae_losers": safe_mean(losses["mae"]),
                "median_mae_winners": safe_median(wins["mae"]),
                "median_mae_losers": safe_median(losses["mae"]),
                "mean_mfe_winners": safe_mean(wins["mfe"]),
                "mean_mfe_losers": safe_mean(losses["mfe"]),
                "median_mfe_winners": safe_median(wins["mfe"]),
                "median_mfe_losers": safe_median(losses["mfe"]),
                "mean_bars_winners": safe_mean(wins["bars_to_result"]),
                "mean_bars_losers": safe_mean(losses["bars_to_result"]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# GROUP ANALYSIS
# =============================================================================


def group_analysis(
    trades,
    group_column,
):

    rows = []

    for (
        strategy_name,
        group_value,
    ), group in trades.groupby(
        [
            "strategy_name",
            group_column,
        ],
        dropna=False,
    ):
        wins = group[group["result"] == "WIN"]

        losses = group[group["result"] == "LOSS"]

        resolved = len(wins) + len(losses)

        rows.append(
            {
                "strategy_name": strategy_name,
                group_column: group_value,
                "observations": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "wr": (len(wins) / resolved if resolved else np.nan),
                "net_r": group["r"].sum(),
                "expectancy": (group["r"].mean() if len(group) else np.nan),
                "mean_mae": safe_mean(group["mae"]),
                "median_mae": safe_median(group["mae"]),
                "mean_mfe": safe_mean(group["mfe"]),
                "median_mfe": safe_median(group["mfe"]),
                "mean_bars": safe_mean(group["bars_to_result"]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Z-SCORE ANALYSIS
# =============================================================================


def build_zscore_analysis(trades):

    section("Z-SCORE FAILURE ANALYSIS")

    rows = []

    bins = [
        0.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        np.inf,
    ]

    labels = [
        "<1.5",
        "1.5-2.0",
        "2.0-2.5",
        "2.5-3.0",
        "3.0-3.5",
        "3.5-4.0",
        "4.0-5.0",
        "5.0+",
    ]

    work = trades.copy()

    work["abs_z"] = work["zscore"].abs()

    work["z_bucket"] = pd.cut(
        work["abs_z"],
        bins=bins,
        labels=labels,
        right=False,
    )

    grouped = group_analysis(
        work,
        "z_bucket",
    )

    return grouped


# =============================================================================
# OUTCOME COMPARISON
# =============================================================================


def print_outcome_comparison(
    trades,
):

    section("WINNER VS LOSER COMPARISON")

    for name in CANDIDATES:
        group = trades[trades["strategy_name"] == name]

        wins = group[group["result"] == "WIN"]

        losses = group[group["result"] == "LOSS"]

        print("\n" + "-" * 100)

        print(f"{name}")

        print(f"Winners: {len(wins):,}")

        print(f"Losers: {len(losses):,}")

        print("\nMAE:")

        print(f"  Winners mean  : {safe_mean(wins['mae']):.4f}")

        print(f"  Losers mean   : {safe_mean(losses['mae']):.4f}")

        print(f"  Winners median: {safe_median(wins['mae']):.4f}")

        print(f"  Losers median : {safe_median(losses['mae']):.4f}")

        print("\nMFE:")

        print(f"  Winners mean  : {safe_mean(wins['mfe']):.4f}")

        print(f"  Losers mean   : {safe_mean(losses['mfe']):.4f}")

        print(f"  Winners median: {safe_median(wins['mfe']):.4f}")

        print(f"  Losers median : {safe_median(losses['mfe']):.4f}")

        print("\nDuration:")

        print(f"  Winners mean  : {safe_mean(wins['bars_to_result']):.2f}")

        print(f"  Losers mean   : {safe_mean(losses['bars_to_result']):.2f}")


# =============================================================================
# FAILURE PATTERN FLAGS
# =============================================================================


def build_failure_flags(trades):

    section("IDENTIFYING DESCRIPTIVE FAILURE PATTERNS")

    result = trades.copy()

    # -------------------------------------------------------------------------
    # Important:
    #
    # These are DESCRIPTIVE flags only.
    # They are NOT filters.
    #
    # We deliberately do not optimize thresholds.
    # -------------------------------------------------------------------------

    result["mae_reached_1R"] = result["mae"] <= -result["sl"]

    result["mfe_reached_1R"] = result["mfe"] >= result["sl"]

    result["mfe_reached_TP"] = result["mfe"] >= result["tp"]

    result["deep_adverse"] = result["mae"] <= -1.5 * result["sl"]

    result["early_loss"] = (result["result"] == "LOSS") & (
        result["bars_to_result"] <= 2
    )

    result["late_loss"] = (result["result"] == "LOSS") & (result["bars_to_result"] > 2)

    return result


# =============================================================================
# PRINT FAILURE FLAGS
# =============================================================================


def print_failure_flags(
    trades,
):

    section("DESCRIPTIVE FAILURE FLAGS")

    for name in CANDIDATES:
        group = trades[trades["strategy_name"] == name]

        losses = group[group["result"] == "LOSS"]

        if losses.empty:
            continue

        print("\n" + "-" * 100)

        print(name)

        checks = [
            (
                "MAE reached 1R",
                "mae_reached_1R",
            ),
            (
                "MFE reached TP",
                "mfe_reached_TP",
            ),
            (
                "Deep adverse excursion",
                "deep_adverse",
            ),
            (
                "Early loss",
                "early_loss",
            ),
            (
                "Late loss",
                "late_loss",
            ),
        ]

        for label, column in checks:
            count = int(losses[column].sum())

            pct = count / len(losses)

            print(f"{label:30s}: {count:5d} ({pct:6.2%})")


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08Q")

    print("FAILURE TEST / MAE-MFE ANALYSIS")

    print("-" * 100)

    print("MRS2 / MRL1 / MRL2 ARE COMPLETELY FROZEN.")

    print("This script does NOT modify the strategy.")

    print("All findings are descriptive hypotheses.")

    trades = load_trades()

    metadata = load_metadata()

    future_close = load_path_cache()

    # -------------------------------------------------------------------------
    # Integrity check
    # -------------------------------------------------------------------------

    section("INTEGRITY CHECK")

    metadata_ids = set(metadata["event_id"].astype(np.int64))

    trade_ids = set(trades["event_id"].astype(np.int64))

    missing_metadata = trade_ids - metadata_ids

    print(f"Trade event IDs: {len(trade_ids):,}")

    print(f"Metadata event IDs: {len(metadata_ids):,}")

    print(f"Missing metadata IDs: {len(missing_metadata):,}")

    if missing_metadata:
        raise RuntimeError("Trade events missing from metadata.")

    # -------------------------------------------------------------------------
    # Reconstruct excursions
    # -------------------------------------------------------------------------

    analyzed = calculate_excursions(
        trades,
        future_close,
    )

    analyzed = build_failure_flags(analyzed)

    # -------------------------------------------------------------------------
    # Main summary
    # -------------------------------------------------------------------------

    summary = build_summary(analyzed)

    # -------------------------------------------------------------------------
    # Group analyses
    # -------------------------------------------------------------------------

    by_hmm = group_analysis(
        analyzed,
        "hmm_state",
    )

    by_vol = group_analysis(
        analyzed,
        "vol_bucket",
    )

    by_window = group_analysis(
        analyzed,
        "window",
    )

    by_z = build_zscore_analysis(analyzed)

    # -------------------------------------------------------------------------
    # Duration
    # -------------------------------------------------------------------------

    duration = (
        analyzed.groupby(
            [
                "strategy_name",
                "result",
                "bars_to_result",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="observations")
        .sort_values(
            [
                "strategy_name",
                "result",
                "bars_to_result",
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    print_outcome_comparison(analyzed)

    print_failure_flags(analyzed)

    section("BY HMM STATE")

    print(by_hmm.to_string(index=False))

    section("BY VOLATILITY BUCKET")

    print(by_vol.to_string(index=False))

    section("BY TEMPORAL WINDOW")

    print(by_window.to_string(index=False))

    section("BY Z-SCORE")

    print(by_z.to_string(index=False))

    section("DURATION DISTRIBUTION")

    print(duration.to_string(index=False))

    section("SUMMARY")

    print(summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyzed.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    by_hmm.to_csv(
        OUTPUT_HMM,
        index=False,
    )

    by_vol.to_csv(
        OUTPUT_VOL,
        index=False,
    )

    by_window.to_csv(
        OUTPUT_WINDOW,
        index=False,
    )

    by_z.to_csv(
        OUTPUT_Z,
        index=False,
    )

    duration.to_csv(
        OUTPUT_DURATION,
        index=False,
    )

    section("RESEARCH 08Q COMPLETE")

    print("Failure analysis completed.")

    print("No strategy parameters were changed.")

    print("No filters were selected.")

    print("\nFILES SAVED:")

    print(OUTPUT_TRADES)

    print(OUTPUT_SUMMARY)

    print(OUTPUT_HMM)

    print(OUTPUT_VOL)

    print(OUTPUT_WINDOW)

    print(OUTPUT_Z)

    print(OUTPUT_DURATION)

    print("\nNEXT STEP:")

    print("Review the failure patterns manually.")

    print(
        "Only after identifying a plausible "
        "causal condition should a new filter "
        "hypothesis be tested."
    )


if __name__ == "__main__":
    main()
