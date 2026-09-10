"""
Research 08X — Intrabar Reality Audit

Purpose
-------
Compare the frozen 08P close-to-close trade resolution against
the exact OHLC intrabar barrier-resolution methodology used by
Research 07.1.

08P:
    Entry = signal-bar close
    Future path = future_close
    TP/SL evaluated on future closes

08X:
    Entry = signal-bar close
    Future path = future_high / future_low
    TP/SL evaluated using intrabar OHLC extremes

Research 07 methodology:
    LONG:
        favorable = future_high - entry
        adverse   = entry - future_low

    SHORT:
        favorable = entry - future_low
        adverse   = future_high - entry

    TP before SL -> WIN
    SL before TP -> LOSS
    Same bar TP + SL -> LOSS
    Neither -> TIMEOUT

IMPORTANT
---------
This audit does NOT optimize anything.

The frozen candidates are not modified.

The OHLC paths are reconstructed directly from the same
raw/RTH dataset and the same event metadata used by
Research 07.1.

Frozen candidates:

    C01 = MRS2
    C02 = MRL1
    C06 = MRL2
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.data_loader import load_data


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRADES_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08p_full_confirmation_trades.csv"
)

EVENT_METADATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "cache"
    / "research_07_event_metadata.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

TRADE_AUDIT_PATH = OUTPUT_DIR / "research_08x_intrabar_trade_comparison.csv"

SUMMARY_PATH = OUTPUT_DIR / "research_08x_intrabar_summary.csv"

CROSSTAB_PATH = OUTPUT_DIR / "research_08x_intrabar_crosstab.csv"


# ============================================================
# FROZEN CANDIDATES
# ============================================================

CANDIDATES = {
    "C01": {
        "candidate_id": "C01",
        "strategy_name": "MRS2",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "VOL80-100",
        "z_threshold": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 5,
    },
    "C02": {
        "candidate_id": "C02",
        "strategy_name": "MRL1",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "VOL20-40",
        "z_threshold": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 20,
    },
    "C06": {
        "candidate_id": "C06",
        "strategy_name": "MRL2",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "VOL60-80",
        "z_threshold": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 2,
    },
}


# ============================================================
# PREPARE RTH
# ============================================================


def prepare_rth(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact RTH preparation used by Research 07.1.
    """

    required = {
        "timestamp ET",
        "market_period",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(data.columns)

    if missing:
        raise KeyError("Missing required columns: " + ", ".join(sorted(missing)))

    result = data.loc[data["market_period"].eq("RTH")].copy()

    result = result.sort_values("timestamp ET")

    result = result.reset_index(drop=True)

    return result


# ============================================================
# BUILD RAW OHLC FUTURE PATH
# ============================================================


def build_future_ohlc(
    data: pd.DataFrame,
    event_metadata: pd.DataFrame,
    max_horizon: int,
):
    """
    Reconstruct future_high and future_low exactly from
    Research 07.1 methodology.

    Research 07:
        event data_index points to the RTH dataframe.
        Future bars start at data_index + 1.
    """

    event_indices = event_metadata["data_index"].to_numpy(dtype=np.int64)

    entry_prices = event_metadata["close"].to_numpy(dtype=np.float32)

    n_events = len(event_indices)

    print(f"      Events: {n_events:,}")

    print(f"      Max horizon: {max_horizon}")

    # --------------------------------------------------------
    # RAW RTH ARRAYS
    # --------------------------------------------------------

    high = data["high"].to_numpy(dtype=np.float32)

    low = data["low"].to_numpy(dtype=np.float32)

    close = data["close"].to_numpy(dtype=np.float32)

    # --------------------------------------------------------
    # FUTURE INDEX MATRIX
    # --------------------------------------------------------

    offsets = np.arange(
        1,
        max_horizon + 1,
        dtype=np.int64,
    )

    index_matrix = event_indices[:, None] + offsets[None, :]

    valid = index_matrix < len(data)

    safe_indices = np.minimum(
        index_matrix,
        len(data) - 1,
    )

    # --------------------------------------------------------
    # FUTURE OHLC
    # --------------------------------------------------------

    future_high = high[safe_indices].astype(
        np.float32,
        copy=False,
    )

    future_low = low[safe_indices].astype(
        np.float32,
        copy=False,
    )

    future_close = close[safe_indices].astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # INVALID FUTURE BARS
    # --------------------------------------------------------

    future_high[~valid] = np.nan

    future_low[~valid] = np.nan

    future_close[~valid] = np.nan

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if not np.all(
        future_high[np.isfinite(future_high)] >= future_low[np.isfinite(future_low)]
    ):
        raise ValueError("Found future_high < future_low.")

    # --------------------------------------------------------
    # VERIFY ENTRY AGAINST EVENT METADATA
    # --------------------------------------------------------

    print("      Verifying event entry prices...")

    event_closes = data.iloc[event_indices]["close"].to_numpy(dtype=np.float32)

    entry_match = np.isclose(
        event_closes,
        entry_prices,
        atol=0.0,
        rtol=0.0,
    )

    print(f"      Entry matches: {entry_match.sum():,}/{len(entry_match):,}")

    if not entry_match.all():
        mismatch_count = (~entry_match).sum()

        raise ValueError(
            "Event metadata close does not "
            "match RTH data close for "
            f"{mismatch_count:,} events."
        )

    return (
        future_high,
        future_low,
        future_close,
    )


# ============================================================
# INTRABAR EVALUATION
# ============================================================


def evaluate_intrabar(
    entry: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    candidate: dict,
):
    """
    Exact barrier logic corresponding to Research 07.1.

    Returns:
        result
        R
        bars_to_result
        same_bar_conflict
    """

    horizon = candidate["horizon"]

    tp = candidate["tp"]

    sl = candidate["sl"]

    rr = candidate["rr"]

    side = candidate["side"]

    # --------------------------------------------------------
    # HORIZON
    # --------------------------------------------------------

    high_path = future_high[:horizon]

    low_path = future_low[:horizon]

    # --------------------------------------------------------
    # VALID FUTURE BARS
    # --------------------------------------------------------

    valid = np.isfinite(high_path) & np.isfinite(low_path)

    high_path = high_path[valid]

    low_path = low_path[valid]

    if len(high_path) == 0:
        return (
            "UNRESOLVED",
            0.0,
            0,
            False,
        )

    # --------------------------------------------------------
    # FAVORABLE / ADVERSE
    # --------------------------------------------------------

    if side == "LONG":
        favorable = high_path - entry

        adverse = entry - low_path

    elif side == "SHORT":
        favorable = entry - low_path

        adverse = high_path - entry

    else:
        raise ValueError(f"Unknown side: {side}")

    # --------------------------------------------------------
    # HIT ARRAYS
    # --------------------------------------------------------

    target_hit = favorable >= tp

    stop_hit = adverse >= sl

    has_target = bool(target_hit.any())

    has_stop = bool(stop_hit.any())

    # --------------------------------------------------------
    # FIRST TARGET / STOP
    # --------------------------------------------------------

    first_target = int(np.flatnonzero(target_hit)[0]) if has_target else horizon + 1

    first_stop = int(np.flatnonzero(stop_hit)[0]) if has_stop else horizon + 1

    # --------------------------------------------------------
    # SAME-BAR CONFLICT
    # --------------------------------------------------------

    same_bar_conflict = has_target and has_stop and first_target == first_stop

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if has_target and (not has_stop or first_target < first_stop):
        return (
            "WIN",
            float(rr),
            first_target + 1,
            False,
        )

    if has_stop and (not has_target or first_stop <= first_target):
        return (
            "LOSS",
            -1.0,
            first_stop + 1,
            bool(same_bar_conflict),
        )

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    return (
        "UNRESOLVED",
        0.0,
        len(high_path),
        False,
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    results: pd.Series,
    r_values: pd.Series,
):
    """
    Metrics in R-space.

    UNRESOLVED trades contribute 0R.
    """

    n = len(results)

    wins = int((results == "WIN").sum())

    losses = int((results == "LOSS").sum())

    unresolved = int((results == "UNRESOLVED").sum())

    resolved = wins + losses

    win_rate = wins / resolved if resolved > 0 else np.nan

    gross_profit = float(r_values[r_values > 0].sum())

    gross_loss = float(-r_values[r_values < 0].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    net_r = float(r_values.sum())

    expectancy = net_r / n if n > 0 else np.nan

    # --------------------------------------------------------
    # MAX DD
    # --------------------------------------------------------

    equity = r_values.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "resolved": resolved,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "net_r": net_r,
        "expectancy_r": expectancy,
        "max_dd_r": max_dd,
    }


# ============================================================
# RUN TRADE-LEVEL AUDIT
# ============================================================


def run_audit(
    trades: pd.DataFrame,
    future_high: np.ndarray,
    future_low: np.ndarray,
):
    """
    Reconstruct every 08P trade using OHLC intrabar data.
    """

    print("\n[4/4] Running intrabar reconstruction...")

    records = []

    for idx, row in trades.iterrows():
        candidate_id = str(row["candidate_id"])

        if candidate_id not in CANDIDATES:
            raise ValueError(f"Unknown candidate_id: {candidate_id}")

        candidate = CANDIDATES[candidate_id]

        event_id = int(row["event_id"])

        entry = float(row["entry"])

        # ----------------------------------------------------
        # ORIGINAL 08P RESULT
        # ----------------------------------------------------

        close_result = str(row["result"])

        close_r = float(row["r"])

        close_bars = int(row["bars_to_result"])

        # ----------------------------------------------------
        # INTRABAR RESULT
        # ----------------------------------------------------

        (
            intrabar_result,
            intrabar_r,
            intrabar_bars,
            same_bar_conflict,
        ) = evaluate_intrabar(
            entry=entry,
            future_high=future_high[event_id],
            future_low=future_low[event_id],
            candidate=candidate,
        )

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        result_same = close_result == intrabar_result

        r_same = np.isclose(
            close_r,
            intrabar_r,
            atol=1e-12,
            rtol=0.0,
        )

        bars_same = close_bars == intrabar_bars

        if result_same and r_same and bars_same:
            comparison = "IDENTICAL"

        elif result_same:
            comparison = "SAME_RESULT_DIFFERENT_PATH"

        else:
            comparison = "RESULT_CHANGE"

        records.append(
            {
                "trade_row": idx,
                "event_id": event_id,
                "candidate_id": candidate_id,
                "strategy_name": candidate["strategy_name"],
                "side": candidate["side"],
                "entry": entry,
                # 08P
                "close_result": close_result,
                "close_r": close_r,
                "close_bars": close_bars,
                # 08X
                "intrabar_result": intrabar_result,
                "intrabar_r": intrabar_r,
                "intrabar_bars": intrabar_bars,
                "same_bar_conflict": same_bar_conflict,
                # Comparison
                "comparison": comparison,
                "delta_r": intrabar_r - close_r,
                "delta_bars": intrabar_bars - close_bars,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    audit: pd.DataFrame,
):
    """
    Compare complete 08P and 08X performance
    for every frozen candidate.
    """

    rows = []

    for candidate_id, group in audit.groupby(
        "candidate_id",
        sort=True,
    ):
        strategy_name = group["strategy_name"].iloc[0]

        # ----------------------------------------------------
        # 08P
        # ----------------------------------------------------

        close_metrics = calculate_metrics(
            group["close_result"],
            group["close_r"],
        )

        # ----------------------------------------------------
        # 08X
        # ----------------------------------------------------

        intrabar_metrics = calculate_metrics(
            group["intrabar_result"],
            group["intrabar_r"],
        )

        # ----------------------------------------------------
        # PATH COMPARISON
        # ----------------------------------------------------

        identical = int((group["comparison"] == "IDENTICAL").sum())

        same_result = int((group["comparison"] == "SAME_RESULT_DIFFERENT_PATH").sum())

        result_changes = int((group["comparison"] == "RESULT_CHANGE").sum())

        conflicts = int(group["same_bar_conflict"].sum())

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        rows.append(
            {
                "candidate_id": candidate_id,
                "strategy_name": strategy_name,
                "trades": len(group),
                # 08P
                "close_wins": close_metrics["wins"],
                "close_losses": close_metrics["losses"],
                "close_unresolved": close_metrics["unresolved"],
                "close_win_rate": close_metrics["win_rate"],
                "close_net_r": close_metrics["net_r"],
                "close_expectancy_r": close_metrics["expectancy_r"],
                "close_profit_factor": close_metrics["profit_factor"],
                "close_max_dd_r": close_metrics["max_dd_r"],
                # 08X
                "intrabar_wins": intrabar_metrics["wins"],
                "intrabar_losses": intrabar_metrics["losses"],
                "intrabar_unresolved": intrabar_metrics["unresolved"],
                "intrabar_win_rate": intrabar_metrics["win_rate"],
                "intrabar_net_r": intrabar_metrics["net_r"],
                "intrabar_expectancy_r": intrabar_metrics["expectancy_r"],
                "intrabar_profit_factor": intrabar_metrics["profit_factor"],
                "intrabar_max_dd_r": intrabar_metrics["max_dd_r"],
                # Deltas
                "delta_net_r": (intrabar_metrics["net_r"] - close_metrics["net_r"]),
                "delta_expectancy_r": (
                    intrabar_metrics["expectancy_r"] - close_metrics["expectancy_r"]
                ),
                "delta_win_rate": (
                    intrabar_metrics["win_rate"] - close_metrics["win_rate"]
                ),
                "delta_profit_factor": (
                    intrabar_metrics["profit_factor"] - close_metrics["profit_factor"]
                ),
                "delta_max_dd_r": (
                    intrabar_metrics["max_dd_r"] - close_metrics["max_dd_r"]
                ),
                # Methodology
                "identical_trades": identical,
                "same_result_different_path": same_result,
                "result_changes": result_changes,
                "same_bar_conflicts": conflicts,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CROSSTAB
# ============================================================


def build_crosstab(
    audit: pd.DataFrame,
):
    return pd.crosstab(
        audit["close_result"],
        audit["intrabar_result"],
        rownames=["08P_CLOSE"],
        colnames=["08X_INTRABAR"],
    )


# ============================================================
# PRINT RESULTS
# ============================================================


def print_results(
    audit,
    summary,
    crosstab,
):

    print("\n" + "=" * 72)

    print("08X — INTRABAR REALITY RESULTS")

    print("=" * 72)

    # --------------------------------------------------------
    # CROSSTAB
    # --------------------------------------------------------

    print("\nGlobal result crosstab:")

    print(crosstab.to_string())

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nPer-strategy summary:")

    columns = [
        "candidate_id",
        "strategy_name",
        "trades",
        "close_net_r",
        "intrabar_net_r",
        "delta_net_r",
        "close_expectancy_r",
        "intrabar_expectancy_r",
        "close_win_rate",
        "intrabar_win_rate",
        "close_profit_factor",
        "intrabar_profit_factor",
        "close_max_dd_r",
        "intrabar_max_dd_r",
        "identical_trades",
        "same_result_different_path",
        "result_changes",
        "same_bar_conflicts",
    ]

    print(summary[columns].to_string(index=False))

    # --------------------------------------------------------
    # GLOBAL COMPARISON
    # --------------------------------------------------------

    total = len(audit)

    identical = int((audit["comparison"] == "IDENTICAL").sum())

    same_result = int((audit["comparison"] == "SAME_RESULT_DIFFERENT_PATH").sum())

    result_changes = int((audit["comparison"] == "RESULT_CHANGE").sum())

    conflicts = int(audit["same_bar_conflict"].sum())

    print("\nGlobal path comparison:")

    print(f"  Identical: {identical:,}/{total:,} ({identical / total:.2%})")

    print(
        f"  Same result / different path: "
        f"{same_result:,}/{total:,} "
        f"({same_result / total:.2%})"
    )

    print(
        f"  Result changes: {result_changes:,}/{total:,} ({result_changes / total:.2%})"
    )

    print(
        f"  Same-bar TP+SL conflicts: {conflicts:,}/{total:,} ({conflicts / total:.2%})"
    )

    # --------------------------------------------------------
    # RESULT CHANGE BREAKDOWN
    # --------------------------------------------------------

    changed = audit.loc[audit["comparison"] == "RESULT_CHANGE"]

    if len(changed) > 0:
        print("\nResult-change breakdown:")

        change_table = pd.crosstab(
            changed["close_result"],
            changed["intrabar_result"],
            rownames=["08P"],
            colnames=["08X"],
        )

        print(change_table.to_string())

    else:
        print("\nNo result classification changes.")


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    audit,
    summary,
    crosstab,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        TRADE_AUDIT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    crosstab.to_csv(
        CROSSTAB_PATH,
    )

    print("\nOutputs saved:")

    print(f"  Trade comparison:\n  {TRADE_AUDIT_PATH}")

    print(f"\n  Summary:\n  {SUMMARY_PATH}")

    print(f"\n  Crosstab:\n  {CROSSTAB_PATH}")


# ============================================================
# FINAL VERDICT
# ============================================================


def final_verdict(
    audit,
):

    total = len(audit)

    result_changes = int((audit["comparison"] == "RESULT_CHANGE").sum())

    same_result = int((audit["comparison"] == "SAME_RESULT_DIFFERENT_PATH").sum())

    identical = int((audit["comparison"] == "IDENTICAL").sum())

    print("\n" + "=" * 72)

    print("FINAL VERDICT")

    print("=" * 72)

    print("\n08X COMPLETE.")

    print("\nThis audit is methodological.")

    print("It does not optimize or modify the frozen strategies.")

    print("\nClassification changes:")

    print(f"  {result_changes:,}/{total:,} ({result_changes / total:.2%})")

    print("\nIdentical trades:")

    print(f"  {identical:,}/{total:,} ({identical / total:.2%})")

    print("\nSame result but different exit path:")

    print(f"  {same_result:,}/{total:,} ({same_result / total:.2%})")

    if result_changes == 0:
        print("\nConclusion:")

        print(
            "The final trade classification is invariant "
            "between close-to-close and OHLC intrabar "
            "resolution."
        )

    else:
        print("\nConclusion:")

        print("The frozen strategies are sensitive to the path-resolution methodology.")

        print(
            "The magnitude and direction of that sensitivity "
            "must be evaluated from the saved summary."
        )

    print("\nNext step after reviewing these results:")

    print(
        "Freeze the execution/path methodology, then "
        "perform cost + slippage validation."
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08X — INTRABAR REALITY AUDIT")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD 08P
    # --------------------------------------------------------

    print("\n[1/4] Loading 08P trades...")

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"08P trade file not found:\n{TRADES_PATH}")

    trades = pd.read_csv(TRADES_PATH)

    print(f"      Trades: {len(trades):,}")

    # --------------------------------------------------------
    # LOAD EVENT METADATA
    # --------------------------------------------------------

    print("\n[2/4] Loading event metadata...")

    if not EVENT_METADATA_PATH.exists():
        raise FileNotFoundError(f"Event metadata not found:\n{EVENT_METADATA_PATH}")

    event_metadata = pd.read_csv(EVENT_METADATA_PATH)

    print(f"      Events: {len(event_metadata):,}")

    # --------------------------------------------------------
    # LOAD SAME DATA AS RESEARCH 07
    # --------------------------------------------------------

    print("\n[3/4] Loading and preparing RTH data...")

    data = load_data()

    print(f"      Raw rows: {len(data):,}")

    data = prepare_rth(data)

    print(f"      RTH rows: {len(data):,}")

    # --------------------------------------------------------
    # RECONSTRUCT OHLC PATH
    # --------------------------------------------------------

    max_horizon = max(candidate["horizon"] for candidate in CANDIDATES.values())

    (
        future_high,
        future_low,
        future_close,
    ) = build_future_ohlc(
        data=data,
        event_metadata=event_metadata,
        max_horizon=max_horizon,
    )

    # --------------------------------------------------------
    # RUN AUDIT
    # --------------------------------------------------------

    audit = run_audit(
        trades=trades,
        future_high=future_high,
        future_low=future_low,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = build_summary(audit)

    crosstab = build_crosstab(audit)

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_results(
        audit=audit,
        summary=summary,
        crosstab=crosstab,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_outputs(
        audit=audit,
        summary=summary,
        crosstab=crosstab,
    )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    final_verdict(audit)

    print("\n" + "=" * 72)

    print("08X COMPLETE")

    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
