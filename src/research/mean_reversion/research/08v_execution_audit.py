"""
MEAN REVERSION — RESEARCH 08V
================================

EXECUTION / INTRABAR AUDIT

Purpose
-------
Audit the frozen Mean Reversion trade-resolution logic using:

    1. 08P trade records
    2. Research 07 future-path cache
    3. Event IDs
    4. Recorded TP / SL / horizon
    5. Recorded result / bars_to_result

This audit checks:

    - signal / entry timestamp
    - entry price
    - TP / SL geometry
    - recorded result
    - recorded holding bars
    - intrabar TP detection
    - intrabar SL detection
    - same-bar TP + SL collisions
    - timeout handling
    - consistency between 08P and the future-path cache

IMPORTANT
---------
This script does NOT:

    - optimize parameters
    - modify frozen strategies
    - apply commissions
    - apply slippage
    - change historical results

It is an execution audit only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

TRADES_FILE = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

PATH_CACHE_FILE = CACHE_DIR / "research_07_future_path_cache.npz"

METADATA_FILE = CACHE_DIR / "research_07_event_metadata.csv"

AUDIT_OUTPUT = RESULTS_DIR / "research_08v_execution_audit.csv"

TRADE_AUDIT_OUTPUT = RESULTS_DIR / "research_08v_trade_level_execution_audit.csv"


# =============================================================================
# FROZEN STRATEGIES
# =============================================================================

CANDIDATES = {
    "MRL1": {
        "side": "LONG",
        "tp_points": 5.0,
        "sl_points": 2.0,
        "horizon": 20,
    },
    "MRS2": {
        "side": "SHORT",
        "tp_points": 5.0,
        "sl_points": 2.0,
        "horizon": 5,
    },
    "MRL2": {
        "side": "LONG",
        "tp_points": 5.0,
        "sl_points": 2.0,
        "horizon": 2,
    },
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================


def banner(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_check(
    label: str,
    value,
) -> None:

    print(f"  {label:<42}: {value}")


# =============================================================================
# LOAD 08P
# =============================================================================


def load_trades() -> pd.DataFrame:

    banner("LOADING 08P TRADE FILE")

    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"\nCould not find:\n{TRADES_FILE}")

    df = pd.read_csv(TRADES_FILE)

    print(f"File  : {TRADES_FILE}")

    print(f"Rows  : {len(df):,}")

    print()
    print("Columns:")

    for column in df.columns:
        print(f"  - {column}")

    return df


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_path_cache() -> dict[str, np.ndarray]:

    banner("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_FILE.exists():
        raise FileNotFoundError(f"\nCould not find:\n{PATH_CACHE_FILE}")

    cache = np.load(
        PATH_CACHE_FILE,
        allow_pickle=False,
    )

    print(f"File: {PATH_CACHE_FILE}")

    print()
    print("Arrays:")

    for key in cache.files:
        array = cache[key]

        print(f"  {key:<25} shape={array.shape} dtype={array.dtype}")

    required = [
        "future_close",
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    missing = [key for key in required if key not in cache.files]

    if missing:
        raise RuntimeError(
            "\nRequired path-cache arrays are missing:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    return {key: cache[key] for key in cache.files}


# =============================================================================
# LOAD EVENT METADATA
# =============================================================================


def load_metadata() -> pd.DataFrame | None:

    banner("LOADING EVENT METADATA")

    if not METADATA_FILE.exists():
        print("Metadata file not found.")

        print("Entry-price / signal-close audit will be skipped.")

        return None

    metadata = pd.read_csv(METADATA_FILE)

    print(f"File : {METADATA_FILE}")

    print(f"Rows : {len(metadata):,}")

    print()
    print("Metadata columns:")

    for column in metadata.columns:
        print(f"  - {column}")

    return metadata


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================


def validate_schema(
    df: pd.DataFrame,
) -> None:

    banner("VALIDATING 08P SCHEMA")

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

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(
            "\n08P schema is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    print("Schema: PASS")


# =============================================================================
# BASIC STRUCTURE
# =============================================================================


def audit_basic_structure(
    df: pd.DataFrame,
) -> pd.DataFrame:

    banner("BASIC STRUCTURE AUDIT")

    records = []

    for strategy_name, config in CANDIDATES.items():
        subset = df[
            df["strategy_name"].astype(str).str.upper().str.strip() == strategy_name
        ].copy()

        if subset.empty:
            print()
            print(f"{strategy_name}: NO TRADES")

            continue

        holding = pd.to_numeric(
            subset["bars_to_result"],
            errors="coerce",
        )

        horizon = pd.to_numeric(
            subset["horizon"],
            errors="coerce",
        )

        tp = pd.to_numeric(
            subset["tp"],
            errors="coerce",
        )

        sl = pd.to_numeric(
            subset["sl"],
            errors="coerce",
        )

        print()
        print(strategy_name)

        print_check(
            "Trades",
            f"{len(subset):,}",
        )

        print_check(
            "Frozen TP",
            config["tp_points"],
        )

        print_check(
            "Frozen SL",
            config["sl_points"],
        )

        print_check(
            "Frozen horizon",
            config["horizon"],
        )

        print_check(
            "Observed TP unique",
            sorted(tp.dropna().unique().tolist()),
        )

        print_check(
            "Observed SL unique",
            sorted(sl.dropna().unique().tolist()),
        )

        print_check(
            "Observed horizon unique",
            sorted(horizon.dropna().unique().tolist()),
        )

        tp_mismatch = (
            ~np.isclose(
                tp,
                config["tp_points"],
                equal_nan=False,
            )
        ).sum()

        sl_mismatch = (
            ~np.isclose(
                sl,
                config["sl_points"],
                equal_nan=False,
            )
        ).sum()

        horizon_mismatch = (horizon != config["horizon"]).sum()

        over_horizon = (holding > config["horizon"]).sum()

        negative_holding = (holding < 0).sum()

        print_check(
            "TP mismatches",
            int(tp_mismatch),
        )

        print_check(
            "SL mismatches",
            int(sl_mismatch),
        )

        print_check(
            "Horizon mismatches",
            int(horizon_mismatch),
        )

        print_check(
            "Trades > horizon",
            int(over_horizon),
        )

        print_check(
            "Negative holding bars",
            int(negative_holding),
        )

        # Result distribution

        print()
        print("Result distribution:")

        result_counts = subset["result"].astype(str).str.lower().value_counts()

        for result, count in result_counts.items():
            print(f"  {result:<20}: {count:,}")

        records.append(
            {
                "strategy": strategy_name,
                "trades": len(subset),
                "tp_mismatches": int(tp_mismatch),
                "sl_mismatches": int(sl_mismatch),
                "horizon_mismatches": int(horizon_mismatch),
                "over_horizon": int(over_horizon),
                "negative_holding": int(negative_holding),
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# RESULT NORMALIZATION
# =============================================================================


def normalize_result(value: object) -> str:

    text = str(value).strip().lower()

    if any(
        token in text
        for token in [
            "target",
            "tp",
            "win",
            "profit",
        ]
    ):
        return "TARGET"

    if any(
        token in text
        for token in [
            "stop",
            "sl",
            "loss",
        ]
    ):
        return "STOP"

    if any(
        token in text
        for token in [
            "timeout",
            "horizon",
            "time",
            "expire",
        ]
    ):
        return "TIMEOUT"

    return "UNKNOWN"


# =============================================================================
# PATH CACHE RESOLUTION
# =============================================================================


def resolve_trade_from_path(
    side: str,
    event_id: int,
    tp: float,
    sl: float,
    horizon: int,
    cache: dict[str, np.ndarray],
) -> dict:

    side = str(side).upper().strip()

    event_id = int(event_id)

    tp = float(tp)
    sl = float(sl)
    horizon = int(horizon)

    # -------------------------------------------------------------------------
    # Bounds
    # -------------------------------------------------------------------------

    n_events = cache["future_close"].shape[0]

    if event_id < 0 or event_id >= n_events:
        return {
            "status": "INVALID_EVENT_ID",
            "expected_result": "UNKNOWN",
            "expected_bar": np.nan,
            "same_bar_collision": False,
            "max_favorable": np.nan,
            "max_adverse": np.nan,
        }

    # -------------------------------------------------------------------------
    # Select directional arrays
    # -------------------------------------------------------------------------

    if side == "LONG":
        favorable = cache["long_favorable"][event_id]

        adverse = cache["long_adverse"][event_id]

    elif side == "SHORT":
        favorable = cache["short_favorable"][event_id]

        adverse = cache["short_adverse"][event_id]

    else:
        return {
            "status": "INVALID_SIDE",
            "expected_result": "UNKNOWN",
            "expected_bar": np.nan,
            "same_bar_collision": False,
            "max_favorable": np.nan,
            "max_adverse": np.nan,
        }

    # -------------------------------------------------------------------------
    # Limit to horizon
    # -------------------------------------------------------------------------

    horizon = min(
        horizon,
        len(favorable),
    )

    favorable = np.asarray(
        favorable[:horizon],
        dtype=float,
    )

    adverse = np.asarray(
        adverse[:horizon],
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Touch masks
    # -------------------------------------------------------------------------

    target_hit = favorable >= tp

    stop_hit = adverse >= sl

    # -------------------------------------------------------------------------
    # Same-bar collision
    # -------------------------------------------------------------------------

    collision_mask = target_hit & stop_hit

    collision_positions = np.flatnonzero(collision_mask)

    same_bar_collision = len(collision_positions) > 0

    # -------------------------------------------------------------------------
    # First target / stop
    # -------------------------------------------------------------------------

    target_positions = np.flatnonzero(target_hit)

    stop_positions = np.flatnonzero(stop_hit)

    first_target = int(target_positions[0]) if len(target_positions) else None

    first_stop = int(stop_positions[0]) if len(stop_positions) else None

    # -------------------------------------------------------------------------
    # Resolve
    #
    # IMPORTANT:
    #
    # Same-bar TP + SL -> STOP
    #
    # This is the conservative rule already used in 08E.
    # -------------------------------------------------------------------------

    if first_target is not None and first_stop is not None:
        if first_stop <= first_target:
            return {
                "status": "RESOLVED",
                "expected_result": "STOP",
                "expected_bar": first_stop + 1,
                "same_bar_collision": (first_stop == first_target),
                "first_target_bar": first_target + 1,
                "first_stop_bar": first_stop + 1,
                "max_favorable": float(np.nanmax(favorable)),
                "max_adverse": float(np.nanmax(adverse)),
            }

        return {
            "status": "RESOLVED",
            "expected_result": "TARGET",
            "expected_bar": first_target + 1,
            "same_bar_collision": False,
            "first_target_bar": first_target + 1,
            "first_stop_bar": first_stop + 1,
            "max_favorable": float(np.nanmax(favorable)),
            "max_adverse": float(np.nanmax(adverse)),
        }

    if first_target is not None:
        return {
            "status": "RESOLVED",
            "expected_result": "TARGET",
            "expected_bar": first_target + 1,
            "same_bar_collision": False,
            "first_target_bar": first_target + 1,
            "first_stop_bar": np.nan,
            "max_favorable": float(np.nanmax(favorable)),
            "max_adverse": float(np.nanmax(adverse)),
        }

    if first_stop is not None:
        return {
            "status": "RESOLVED",
            "expected_result": "STOP",
            "expected_bar": first_stop + 1,
            "same_bar_collision": False,
            "first_target_bar": np.nan,
            "first_stop_bar": first_stop + 1,
            "max_favorable": float(np.nanmax(favorable)),
            "max_adverse": float(np.nanmax(adverse)),
        }

    # -------------------------------------------------------------------------
    # Neither target nor stop
    #
    # Timeout.
    # -------------------------------------------------------------------------

    return {
        "status": "RESOLVED",
        "expected_result": "TIMEOUT",
        "expected_bar": horizon,
        "same_bar_collision": False,
        "first_target_bar": np.nan,
        "first_stop_bar": np.nan,
        "max_favorable": float(np.nanmax(favorable)),
        "max_adverse": float(np.nanmax(adverse)),
    }


# =============================================================================
# FULL TRADE-LEVEL AUDIT
# =============================================================================


def run_trade_level_audit(
    df: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> pd.DataFrame:

    banner("RUNNING TRADE-LEVEL INTRABAR AUDIT")

    records = []

    for index, row in df.iterrows():
        strategy = str(row["strategy_name"]).upper().strip()

        side = str(row["side"]).upper().strip()

        event_id = int(row["event_id"])

        tp = float(row["tp"])

        sl = float(row["sl"])

        horizon = int(row["horizon"])

        recorded_result = normalize_result(row["result"])

        recorded_bars = int(row["bars_to_result"])

        resolved = resolve_trade_from_path(
            side=side,
            event_id=event_id,
            tp=tp,
            sl=sl,
            horizon=horizon,
            cache=cache,
        )

        expected_result = resolved["expected_result"]

        result_match = recorded_result == expected_result

        bars_match = True

        if pd.notna(resolved["expected_bar"]):
            bars_match = recorded_bars == int(resolved["expected_bar"])

        records.append(
            {
                "row_index": index,
                "strategy_name": strategy,
                "candidate_id": row["candidate_id"],
                "side": side,
                "event_id": event_id,
                "timestamp": row["timestamp"],
                "entry": row["entry"],
                "tp": tp,
                "sl": sl,
                "horizon": horizon,
                "recorded_result": recorded_result,
                "expected_result": expected_result,
                "result_match": result_match,
                "recorded_bars": recorded_bars,
                "expected_bars": resolved["expected_bar"],
                "bars_match": bars_match,
                "same_bar_collision": resolved["same_bar_collision"],
                "first_target_bar": resolved.get(
                    "first_target_bar",
                    np.nan,
                ),
                "first_stop_bar": resolved.get(
                    "first_stop_bar",
                    np.nan,
                ),
                "max_favorable": resolved["max_favorable"],
                "max_adverse": resolved["max_adverse"],
                "recorded_r": row["r"],
                "window": row["window"],
            }
        )

    audit = pd.DataFrame(records)

    return audit


# =============================================================================
# ENTRY PRICE AUDIT
# =============================================================================


def audit_entry_price(
    trades: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> None:

    banner("ENTRY-PRICE / SIGNAL-CLOSE AUDIT")

    if metadata is None:
        print("SKIPPED — metadata unavailable.")

        return

    # -------------------------------------------------------------------------
    # Identify event ID
    # -------------------------------------------------------------------------

    metadata_event_column = None

    for candidate in [
        "event_id",
        "event",
        "index",
    ]:
        if candidate in metadata.columns:
            metadata_event_column = candidate

            break

    if metadata_event_column is None:
        print("Could not identify event_id in metadata.")

        print("Entry-price audit skipped.")

        return

    # -------------------------------------------------------------------------
    # Identify close
    # -------------------------------------------------------------------------

    close_column = None

    for candidate in [
        "close",
        "Close",
        "close_price",
        "entry_close",
    ]:
        if candidate in metadata.columns:
            close_column = candidate

            break

    if close_column is None:
        print("No close-price column found in metadata.")

        print("Cannot directly verify whether entry == signal-bar close.")

        print()
        print("Available metadata columns:")

        print(list(metadata.columns))

        return

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    left = trades[
        [
            "event_id",
            "entry",
        ]
    ].copy()

    right = metadata[
        [
            metadata_event_column,
            close_column,
        ]
    ].copy()

    right = right.rename(
        columns={
            metadata_event_column: "event_id",
            close_column: "_metadata_close",
        }
    )

    left["event_id"] = pd.to_numeric(
        left["event_id"],
        errors="coerce",
    )

    right["event_id"] = pd.to_numeric(
        right["event_id"],
        errors="coerce",
    )

    left["entry"] = pd.to_numeric(
        left["entry"],
        errors="coerce",
    )

    right["_metadata_close"] = pd.to_numeric(
        right["_metadata_close"],
        errors="coerce",
    )

    merged = left.merge(
        right,
        on="event_id",
        how="left",
    )

    valid = merged.dropna(
        subset=[
            "entry",
            "_metadata_close",
        ]
    )

    if valid.empty:
        print("No matching entry/close observations.")

        return

    difference = (valid["entry"] - valid["_metadata_close"]).abs()

    tolerance = 1e-9

    matches = difference <= tolerance

    print_check(
        "Matched observations",
        f"{len(valid):,}",
    )

    print_check(
        "Entry == metadata close",
        f"{matches.sum():,}",
    )

    print_check(
        "Entry != metadata close",
        f"{(~matches).sum():,}",
    )

    print_check(
        "Maximum absolute difference",
        f"{difference.max():.10f}",
    )

    print()

    if matches.all():
        print("ENTRY PRICE AUDIT: PASS")

        print("All audited entries equal the signal-bar close.")

    else:
        print("ENTRY PRICE AUDIT: FAIL / INVESTIGATE")

        print("Some entries do not equal the metadata close.")


# =============================================================================
# RESULT COMPARISON
# =============================================================================


def summarize_trade_audit(
    audit: pd.DataFrame,
) -> pd.DataFrame:

    banner("INTRABAR RESOLUTION RESULTS")

    summaries = []

    for strategy in CANDIDATES:
        subset = audit[audit["strategy_name"] == strategy]

        if subset.empty:
            continue

        result_matches = subset["result_match"].sum()

        bar_matches = subset["bars_match"].sum()

        collisions = subset["same_bar_collision"].sum()

        print()
        print(strategy)

        print_check(
            "Trades",
            f"{len(subset):,}",
        )

        print_check(
            "Result matches",
            f"{result_matches:,}",
        )

        print_check(
            "Result mismatches",
            f"{len(subset) - result_matches:,}",
        )

        print_check(
            "Holding-bar matches",
            f"{bar_matches:,}",
        )

        print_check(
            "Holding-bar mismatches",
            f"{len(subset) - bar_matches:,}",
        )

        print_check(
            "Same-bar TP + SL collisions",
            f"{collisions:,}",
        )

        summaries.append(
            {
                "strategy": strategy,
                "trades": len(subset),
                "result_matches": int(result_matches),
                "result_mismatches": int(len(subset) - result_matches),
                "bar_matches": int(bar_matches),
                "bar_mismatches": int(len(subset) - bar_matches),
                "same_bar_collisions": int(collisions),
            }
        )

    return pd.DataFrame(summaries)


# =============================================================================
# SHOW MISMATCHES
# =============================================================================


def show_mismatches(
    audit: pd.DataFrame,
) -> None:

    mismatches = audit[~audit["result_match"] | ~audit["bars_match"]].copy()

    banner("TRADE-LEVEL MISMATCHES")

    if mismatches.empty:
        print("NO MISMATCHES FOUND.")

        return

    print(f"Total mismatches: {len(mismatches):,}")

    print()

    columns = [
        "strategy_name",
        "event_id",
        "timestamp",
        "side",
        "tp",
        "sl",
        "horizon",
        "recorded_result",
        "expected_result",
        "recorded_bars",
        "expected_bars",
        "same_bar_collision",
        "first_target_bar",
        "first_stop_bar",
        "max_favorable",
        "max_adverse",
        "recorded_r",
    ]

    print(mismatches[columns].head(50).to_string(index=False))


# =============================================================================
# FINAL STATUS
# =============================================================================


def final_status(
    structure: pd.DataFrame,
    audit_summary: pd.DataFrame,
    trade_audit: pd.DataFrame,
) -> None:

    banner("FINAL EXECUTION AUDIT STATUS")

    structural_failures = 0

    if not structure.empty:
        structural_failures = int(
            structure[
                [
                    "tp_mismatches",
                    "sl_mismatches",
                    "horizon_mismatches",
                    "over_horizon",
                    "negative_holding",
                ]
            ]
            .sum()
            .sum()
        )

    result_mismatches = int((~trade_audit["result_match"]).sum())

    bar_mismatches = int((~trade_audit["bars_match"]).sum())

    print()
    print(f"Structural failures      : {structural_failures}")

    print(f"Result mismatches        : {result_mismatches}")

    print(f"Holding-bar mismatches   : {bar_mismatches}")

    print()

    if structural_failures == 0 and result_mismatches == 0 and bar_mismatches == 0:
        print("EXECUTION AUDIT: PASS")

        print()
        print(
            "08P results are internally consistent "
            "with the Research 07 future-path cache."
        )

        print()
        print(
            "This confirms that intrabar TP/SL touches "
            "represented by the path cache are being "
            "resolved consistently."
        )

    else:
        print("EXECUTION AUDIT: FAIL / INVESTIGATE")

        print()
        print("Do NOT proceed to Cost + Slippage yet.")

        print("Inspect the mismatched trades above.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("MEAN REVERSION — RESEARCH 08V")

    print("EXECUTION / INTRABAR AUDIT")

    print()
    print("No optimization.")

    print("No parameter changes.")

    print("No cost/slippage.")

    # =========================================================================
    # LOAD
    # =========================================================================

    trades = load_trades()

    validate_schema(trades)

    cache = load_path_cache()

    metadata = load_metadata()

    # =========================================================================
    # BASIC AUDIT
    # =========================================================================

    structure = audit_basic_structure(trades)

    # =========================================================================
    # ENTRY AUDIT
    # =========================================================================

    audit_entry_price(
        trades,
        metadata,
    )

    # =========================================================================
    # TRADE-LEVEL PATH AUDIT
    # =========================================================================

    trade_audit = run_trade_level_audit(
        trades,
        cache,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    audit_summary = summarize_trade_audit(trade_audit)

    # =========================================================================
    # MISMATCHES
    # =========================================================================

    show_mismatches(trade_audit)

    # =========================================================================
    # SAVE
    # =========================================================================

    structure.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    trade_audit.to_csv(
        TRADE_AUDIT_OUTPUT,
        index=False,
    )

    print()
    print(f"Summary saved to:")

    print(AUDIT_OUTPUT)

    print()
    print(f"Trade-level audit saved to:")

    print(TRADE_AUDIT_OUTPUT)

    # =========================================================================
    # FINAL
    # =========================================================================

    final_status(
        structure,
        audit_summary,
        trade_audit,
    )

    banner("08V EXECUTION AUDIT COMPLETE")


if __name__ == "__main__":
    main()
