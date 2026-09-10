"""
Research 08W.5 — Exact Reproduction Audit

Purpose
-------
Reproduce the exact trade-evaluation logic used by
08P_full_confirmation.py.

This audit verifies that every trade saved by 08P can be
reconstructed exactly from the Research-07 future_close cache.

IMPORTANT
---------
This audit intentionally reproduces 08P.

It does NOT use future_high/future_low.

Therefore this is NOT the intrabar/OHLC reality audit.

08P methodology:
    Entry = signal-bar close
    Path  = future_close[event_id][:horizon]

LONG:
    movement = future_close - entry

SHORT:
    movement = entry - future_close

TP:
    movement >= tp

SL:
    movement <= -sl

If TP occurs first:
    WIN, +RR

If SL occurs first:
    LOSS, -1R

If TP and SL occur on the same future-close bar:
    AMBIGUOUS, 0R

If neither occurs:
    UNRESOLVED, 0R
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


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

FUTURE_PATH_CACHE_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "cache"
    / "research_07_future_path_cache.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

AUDIT_OUTPUT_PATH = OUTPUT_DIR / "research_08w5_exact_reproduction_audit.csv"


# ============================================================
# EXACT 08P CANDIDATE MAPPING
# ============================================================
#
# IMPORTANT:
#
# The 08P CSV stores the INTERNAL candidate IDs:
#
#   C01 = MRS2
#   C02 = MRL1
#   C06 = MRL2
#
# Do NOT replace these IDs with strategy names.
#
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
# EXACT 08P TRADE EVALUATION
# ============================================================


def evaluate_trade_exact(
    entry: float,
    future_path: np.ndarray,
    candidate: dict,
) -> tuple[str, float, int]:

    horizon = candidate["horizon"]
    tp = candidate["tp"]
    sl = candidate["sl"]
    rr = candidate["rr"]
    side = candidate["side"]

    # --------------------------------------------------------
    # EXACT PATH USED BY 08P
    # --------------------------------------------------------

    path = future_path[:horizon]

    # --------------------------------------------------------
    # EXACT FINITE FILTER USED BY 08P
    # --------------------------------------------------------

    path = path[np.isfinite(path)]

    if len(path) == 0:
        return "UNRESOLVED", 0.0, 0

    # --------------------------------------------------------
    # PRICE MOVEMENT
    # --------------------------------------------------------

    if side == "LONG":
        movement = path - entry

    elif side == "SHORT":
        movement = entry - path

    else:
        raise ValueError(f"Unknown side: {side}")

    # --------------------------------------------------------
    # BARRIER DETECTION
    # --------------------------------------------------------

    tp_hits = np.flatnonzero(movement >= tp)

    sl_hits = np.flatnonzero(movement <= -sl)

    # --------------------------------------------------------
    # BOTH TP AND SL OCCUR
    # --------------------------------------------------------

    if len(tp_hits) > 0 and len(sl_hits) > 0:
        first_tp = int(tp_hits[0])
        first_sl = int(sl_hits[0])

        # TP occurs first
        if first_tp < first_sl:
            return (
                "WIN",
                float(rr),
                first_tp + 1,
            )

        # SL occurs first
        elif first_sl < first_tp:
            return (
                "LOSS",
                -1.0,
                first_sl + 1,
            )

        # Same future-close bar
        else:
            return (
                "AMBIGUOUS",
                0.0,
                first_tp + 1,
            )

    # --------------------------------------------------------
    # ONLY TP
    # --------------------------------------------------------

    if len(tp_hits) > 0:
        first_tp = int(tp_hits[0])

        return (
            "WIN",
            float(rr),
            first_tp + 1,
        )

    # --------------------------------------------------------
    # ONLY SL
    # --------------------------------------------------------

    if len(sl_hits) > 0:
        first_sl = int(sl_hits[0])

        return (
            "LOSS",
            -1.0,
            first_sl + 1,
        )

    # --------------------------------------------------------
    # NEITHER TP NOR SL
    # --------------------------------------------------------

    return (
        "UNRESOLVED",
        0.0,
        len(path),
    )


# ============================================================
# LOAD INPUTS
# ============================================================


def load_inputs():

    print("=" * 72)
    print("08W.5 — EXACT REPRODUCTION AUDIT")
    print("=" * 72)

    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    print("\n[1/5] Loading 08P trade file...")

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Trade file not found:\n{TRADES_PATH}")

    trades = pd.read_csv(TRADES_PATH)

    print(f"      Trades: {len(trades):,}")

    print(f"      Path:   {TRADES_PATH}")

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print("\n[2/5] Loading event metadata...")

    if not EVENT_METADATA_PATH.exists():
        raise FileNotFoundError(f"Event metadata not found:\n{EVENT_METADATA_PATH}")

    metadata = pd.read_csv(EVENT_METADATA_PATH)

    print(f"      Events: {len(metadata):,}")

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    print("\n[3/5] Loading future path cache...")

    if not FUTURE_PATH_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Future path cache not found:\n{FUTURE_PATH_CACHE_PATH}"
        )

    cache = np.load(FUTURE_PATH_CACHE_PATH)

    if "future_close" not in cache:
        raise KeyError("future_close not found in cache.")

    future_close = cache["future_close"]

    print(f"      future_close shape: {future_close.shape}")

    return (
        trades,
        metadata,
        future_close,
    )


# ============================================================
# CACHE IDENTITY VALIDATION
# ============================================================


def validate_cache_identity(
    trades,
    metadata,
    future_close,
):

    print("\n[4/5] Validating event/cache identity...")

    event_ids = trades["event_id"].astype(int).to_numpy()

    if len(event_ids) == 0:
        raise ValueError("No trades found.")

    if event_ids.min() < 0:
        raise ValueError("Negative event_id found.")

    if event_ids.max() >= len(future_close):
        raise ValueError("Trade event_id exceeds future_close cache.")

    metadata_event_ids = metadata["event_id"].astype(int).to_numpy()

    metadata_identity = np.array_equal(
        metadata_event_ids,
        np.arange(len(metadata)),
    )

    print(f"      Metadata event_id == row index: {metadata_identity}")

    print(f"      event_id range in trades: {event_ids.min()} -> {event_ids.max()}")

    print(f"      future_close rows: {len(future_close):,}")


# ============================================================
# REPRODUCE EVERY TRADE
# ============================================================


def reproduce_trades(
    trades,
    future_close,
):

    print("\n[5/5] Reproducing every trade...")

    records = []

    # --------------------------------------------------------
    # VALIDATE ALL CANDIDATE IDS BEFORE RUNNING
    # --------------------------------------------------------

    csv_candidate_ids = trades["candidate_id"].astype(str).unique().tolist()

    unknown_ids = sorted(set(csv_candidate_ids) - set(CANDIDATES.keys()))

    if unknown_ids:
        raise ValueError(
            "\nUnknown candidate_id(s) found "
            "in the 08P trade file:\n"
            f"{unknown_ids}\n\n"
            "Known candidate IDs in this audit:\n"
            f"{sorted(CANDIDATES.keys())}"
        )

    print(f"      Candidate IDs found: {sorted(csv_candidate_ids)}")

    # --------------------------------------------------------
    # TRADE LOOP
    # --------------------------------------------------------

    for idx, row in trades.iterrows():
        candidate_id = str(row["candidate_id"])

        candidate = CANDIDATES[candidate_id]

        event_id = int(row["event_id"])

        entry = float(row["entry"])

        # ----------------------------------------------------
        # EXACT CACHE ACCESS
        # ----------------------------------------------------

        future_path = future_close[event_id]

        # ----------------------------------------------------
        # EXACT 08P EVALUATION
        # ----------------------------------------------------

        (
            result_reproduced,
            r_reproduced,
            bars_reproduced,
        ) = evaluate_trade_exact(
            entry=entry,
            future_path=future_path,
            candidate=candidate,
        )

        records.append(
            {
                "trade_row": idx,
                "event_id": event_id,
                "candidate_id": candidate_id,
                "strategy_name": candidate["strategy_name"],
                "side": candidate["side"],
                "entry_original": entry,
                "result_original": str(row["result"]),
                "r_original": float(row["r"]),
                "bars_original": int(row["bars_to_result"]),
                "result_reproduced": result_reproduced,
                "r_reproduced": r_reproduced,
                "bars_reproduced": bars_reproduced,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# COMPARISON
# ============================================================


def compare_results(
    audit: pd.DataFrame,
):

    print("\n" + "=" * 72)

    print("REPRODUCTION RESULTS")

    print("=" * 72)

    total = len(audit)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_matches = audit["result_original"] == audit["result_reproduced"]

    # --------------------------------------------------------
    # R
    # --------------------------------------------------------

    r_matches = np.isclose(
        audit["r_original"].to_numpy(),
        audit["r_reproduced"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    # --------------------------------------------------------
    # BARS
    # --------------------------------------------------------

    bars_matches = audit["bars_original"] == audit["bars_reproduced"]

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    all_matches = result_matches & r_matches & bars_matches

    # --------------------------------------------------------
    # PRINT GLOBAL RESULTS
    # --------------------------------------------------------

    print(
        f"\nResult matches: "
        f"{result_matches.sum():,}/"
        f"{total:,} "
        f"({result_matches.mean():.2%})"
    )

    print(f"R matches:      {r_matches.sum():,}/{total:,} ({r_matches.mean():.2%})")

    print(
        f"Bars matches:   {bars_matches.sum():,}/{total:,} ({bars_matches.mean():.2%})"
    )

    print(f"ALL MATCH:      {all_matches.sum():,}/{total:,} ({all_matches.mean():.2%})")

    # --------------------------------------------------------
    # RESULT CROSSTAB
    # --------------------------------------------------------

    print("\nResult crosstab:")

    crosstab = pd.crosstab(
        audit["result_original"],
        audit["result_reproduced"],
        rownames=["08P"],
        colnames=["08W.5"],
    )

    print(crosstab.to_string())

    # --------------------------------------------------------
    # CANDIDATE SUMMARY
    # --------------------------------------------------------

    print("\nPer-candidate summary:")

    summary_rows = []

    for candidate_id, group in audit.groupby(
        "candidate_id",
        sort=True,
    ):
        strategy_name = group["strategy_name"].iloc[0]

        n = len(group)

        result_ok = group["result_original"] == group["result_reproduced"]

        r_ok = np.isclose(
            group["r_original"].to_numpy(),
            group["r_reproduced"].to_numpy(),
            atol=1e-12,
            rtol=0.0,
        )

        bars_ok = group["bars_original"] == group["bars_reproduced"]

        all_ok = result_ok & r_ok & bars_ok

        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "strategy_name": strategy_name,
                "trades": n,
                "result_matches": int(result_ok.sum()),
                "result_match_rate": result_ok.mean(),
                "r_matches": int(r_ok.sum()),
                "r_match_rate": r_ok.mean(),
                "bars_matches": int(bars_ok.sum()),
                "bars_match_rate": bars_ok.mean(),
                "all_matches": int(all_ok.sum()),
                "all_match_rate": all_ok.mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(summary.to_string(index=False))

    return (
        result_matches,
        r_matches,
        bars_matches,
        all_matches,
        summary,
    )


# ============================================================
# MISMATCH DETAILS
# ============================================================


def show_mismatches(
    audit,
    result_matches,
    r_matches,
    bars_matches,
):

    mismatch_mask = ~(result_matches & r_matches & bars_matches)

    mismatches = audit.loc[mismatch_mask].copy()

    print("\n" + "=" * 72)

    print("MISMATCH DETAILS")

    print("=" * 72)

    if len(mismatches) == 0:
        print("\nNo mismatches found.")

        return

    print(f"\nTotal mismatches: {len(mismatches):,}")

    columns = [
        "trade_row",
        "event_id",
        "candidate_id",
        "strategy_name",
        "entry_original",
        "result_original",
        "result_reproduced",
        "r_original",
        "r_reproduced",
        "bars_original",
        "bars_reproduced",
    ]

    print("\nFirst 20 mismatches:")

    print(mismatches[columns].head(20).to_string(index=False))

    # --------------------------------------------------------
    # MISMATCH TYPE
    # --------------------------------------------------------

    mismatch_types = []

    for _, row in mismatches.iterrows():
        differences = []

        if row["result_original"] != row["result_reproduced"]:
            differences.append("RESULT")

        if not np.isclose(
            row["r_original"],
            row["r_reproduced"],
            atol=1e-12,
            rtol=0.0,
        ):
            differences.append("R")

        if row["bars_original"] != row["bars_reproduced"]:
            differences.append("BARS")

        mismatch_types.append("+".join(differences))

    mismatch_counts = (
        pd.Series(mismatch_types)
        .value_counts()
        .rename_axis("mismatch_type")
        .reset_index(name="count")
    )

    print("\nMismatch types:")

    print(mismatch_counts.to_string(index=False))


# ============================================================
# SAVE
# ============================================================


def save_audit(
    audit: pd.DataFrame,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_OUTPUT_PATH,
        index=False,
    )

    print("\nAudit saved to:")

    print(AUDIT_OUTPUT_PATH)


# ============================================================
# FINAL VERDICT
# ============================================================


def final_verdict(
    all_matches,
):

    print("\n" + "=" * 72)

    print("FINAL VERDICT")

    print("=" * 72)

    total = len(all_matches)

    matched = int(all_matches.sum())

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    if matched == total:
        print("\nPASS")

        print("\n08P is exactly reproducible from the Research-07 future_close cache.")

        print("\nConfirmed:")

        print("  - event_id mapping")

        print("  - future_close cache access")

        print("  - entry convention")

        print("  - LONG/SHORT movement calculation")

        print("  - TP detection")

        print("  - SL detection")

        print("  - first-barrier logic")

        print("  - same-bar AMBIGUOUS logic")

        print("  - RR convention")

        print("  - bars_to_result")

        print("\nIMPORTANT:")

        print(
            "This does NOT establish that the "
            "close-to-close methodology is the "
            "best execution methodology."
        )

        print("It establishes only that 08P is internally reproducible.")

        print("\nNext step:")

        print("08X — Intrabar Reality Audit")

        return True

    # --------------------------------------------------------
    # FAIL
    # --------------------------------------------------------

    print("\nFAIL")

    print(f"\nOnly {matched:,}/{total:,} trades were reproduced.")

    print("\nDo NOT proceed to cost/slippage yet.")

    return False


# ============================================================
# MAIN
# ============================================================


def main():

    trades, metadata, future_close = load_inputs()

    validate_cache_identity(
        trades,
        metadata,
        future_close,
    )

    audit = reproduce_trades(
        trades,
        future_close,
    )

    (
        result_matches,
        r_matches,
        bars_matches,
        all_matches,
        summary,
    ) = compare_results(audit)

    show_mismatches(
        audit,
        result_matches,
        r_matches,
        bars_matches,
    )

    save_audit(audit)

    passed = final_verdict(all_matches)

    print("\n" + "=" * 72)

    if passed:
        print("08W.5 COMPLETE — PASS")

    else:
        print("08W.5 COMPLETE — FAIL")

    print("=" * 72)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
