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

METADATA_FILE = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE_FILE = CACHE_DIR / "research_07_future_path_cache.npz"

OUTPUT_FILE = RESULTS_DIR / "research_08w3_identity_mismatches.csv"


# =============================================================================
# DISPLAY
# =============================================================================


def section(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


# =============================================================================
# LOAD
# =============================================================================


def load_all():

    section("LOADING 08P / RESEARCH 07")

    trades = pd.read_csv(TRADES_FILE)

    metadata = pd.read_csv(METADATA_FILE)

    cache_file = np.load(
        PATH_CACHE_FILE,
        allow_pickle=False,
    )

    cache = {key: np.asarray(cache_file[key]) for key in cache_file.files}

    print(f"08P trades       : {len(trades):,}")

    print(f"Metadata rows    : {len(metadata):,}")

    print(f"Cache rows       : {len(cache['future_close']):,}")

    return trades, metadata, cache


# =============================================================================
# BASIC IDENTITY
# =============================================================================


def audit_event_id_identity(
    metadata: pd.DataFrame,
    cache: dict[str, np.ndarray],
):

    section("EVENT-ID / CACHE IDENTITY AUDIT")

    event_id = pd.to_numeric(
        metadata["event_id"],
        errors="coerce",
    ).to_numpy()

    data_index = pd.to_numeric(
        metadata["data_index"],
        errors="coerce",
    ).to_numpy()

    n = len(metadata)

    row_index = np.arange(
        n,
        dtype=np.int64,
    )

    # -------------------------------------------------------------------------
    # event_id == row position
    # -------------------------------------------------------------------------

    event_equals_row = event_id == row_index

    # -------------------------------------------------------------------------
    # data_index == row position
    # -------------------------------------------------------------------------

    data_equals_row = data_index == row_index

    # -------------------------------------------------------------------------
    # event_id == data_index
    # -------------------------------------------------------------------------

    event_equals_data = event_id == data_index

    print(f"event_id == row index : {event_equals_row.sum():,}/{n:,}")

    print(f"data_index == row index: {data_equals_row.sum():,}/{n:,}")

    print(f"event_id == data_index : {event_equals_data.sum():,}/{n:,}")

    # -------------------------------------------------------------------------
    # Ranges
    # -------------------------------------------------------------------------

    print()
    print("ID ranges:")

    print(f"event_id   : {np.nanmin(event_id):,.0f} → {np.nanmax(event_id):,.0f}")

    print(f"data_index : {np.nanmin(data_index):,.0f} → {np.nanmax(data_index):,.0f}")

    print(f"row index  : 0 → {n - 1:,}")

    # -------------------------------------------------------------------------
    # First mismatches
    # -------------------------------------------------------------------------

    mismatch_mask = ~(event_equals_row & data_equals_row & event_equals_data)

    mismatch_positions = np.flatnonzero(mismatch_mask)

    print()
    print(f"Identity mismatches: {len(mismatch_positions):,}")

    if len(mismatch_positions):
        print()
        print("First 20 mismatches:")

        for position in mismatch_positions[:20]:
            print(
                f"row={position:,} | "
                f"event_id={event_id[position]:,.0f} | "
                f"data_index={data_index[position]:,.0f}"
            )

    # -------------------------------------------------------------------------
    # Cache lengths
    # -------------------------------------------------------------------------

    print()

    for key, array in cache.items():
        if len(array) != n:
            print(f"WARNING: {key} length {len(array):,} != metadata {n:,}")

        else:
            print(f"{key:20s}: length PASS")

    return mismatch_mask


# =============================================================================
# EVENT-ID DIRECT INDEX TEST
# =============================================================================


def audit_direct_event_index(
    trades: pd.DataFrame,
    metadata: pd.DataFrame,
    cache: dict[str, np.ndarray],
):

    section("DIRECT EVENT-ID INDEX TEST")

    metadata_event_ids = (
        pd.to_numeric(
            metadata["event_id"],
            errors="coerce",
        )
        .astype(np.int64)
        .to_numpy()
    )

    metadata_position = {
        int(event_id): int(position)
        for position, event_id in enumerate(metadata_event_ids)
    }

    trade_event_ids = (
        pd.to_numeric(
            trades["event_id"],
            errors="coerce",
        )
        .astype(np.int64)
        .to_numpy()
    )

    print(f"Trades tested: {len(trade_event_ids):,}")

    rows = []

    # -------------------------------------------------------------------------
    # Sample every trade.
    # -------------------------------------------------------------------------

    for i, event_id in enumerate(trade_event_ids):
        if event_id < 0:
            continue

        if event_id >= len(cache["future_close"]):
            rows.append(
                {
                    "trade_row": i,
                    "event_id": event_id,
                    "metadata_position": metadata_position.get(
                        int(event_id),
                        -1,
                    ),
                    "direct_index_valid": False,
                    "issue": "EVENT_ID_OUT_OF_CACHE_RANGE",
                }
            )

            continue

        position = metadata_position.get(
            int(event_id),
            -1,
        )

        if position < 0:
            rows.append(
                {
                    "trade_row": i,
                    "event_id": event_id,
                    "metadata_position": -1,
                    "direct_index_valid": True,
                    "issue": "EVENT_ID_NOT_IN_METADATA",
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Compare the actual cache row selected by:
        #
        #   A = cache[event_id]
        #   B = cache[metadata_position]
        # ---------------------------------------------------------------------

        a = cache["future_close"][int(event_id)]

        b = cache["future_close"][int(position)]

        same = np.array_equal(
            a,
            b,
        )

        if not same:
            rows.append(
                {
                    "trade_row": i,
                    "event_id": event_id,
                    "metadata_position": position,
                    "direct_index_valid": True,
                    "issue": "DIRECT_INDEX_DIFFERS_FROM_METADATA_POSITION",
                }
            )

    mismatches = pd.DataFrame(rows)

    print()
    print(f"Identity/cache mismatches: {len(mismatches):,}")

    if len(mismatches):
        print()
        print(mismatches.head(20).to_string(index=False))

    return mismatches


# =============================================================================
# SAMPLE PATH COMPARISON
# =============================================================================


def compare_path_values(
    trades: pd.DataFrame,
    metadata: pd.DataFrame,
    cache: dict[str, np.ndarray],
):

    section("SAMPLE PATH VALUE COMPARISON")

    metadata_event_ids = metadata["event_id"].astype(np.int64).to_numpy()

    position_map = {
        int(event_id): int(position)
        for position, event_id in enumerate(metadata_event_ids)
    }

    sample = trades.head(min(25, len(trades)))

    rows = []

    for _, trade in sample.iterrows():
        event_id = int(trade["event_id"])

        position = position_map[event_id]

        side = str(trade["side"]).upper()

        if side == "LONG":
            favorable_key = "long_favorable"

            adverse_key = "long_adverse"

        else:
            favorable_key = "short_favorable"

            adverse_key = "short_adverse"

        direct_f = cache[favorable_key][event_id]

        mapped_f = cache[favorable_key][position]

        direct_a = cache[adverse_key][event_id]

        mapped_a = cache[adverse_key][position]

        rows.append(
            {
                "event_id": event_id,
                "metadata_position": position,
                "side": side,
                "direct_f_1": float(direct_f[0]),
                "mapped_f_1": float(mapped_f[0]),
                "direct_a_1": float(direct_a[0]),
                "mapped_a_1": float(mapped_a[0]),
                "f_equal": np.array_equal(
                    direct_f,
                    mapped_f,
                ),
                "a_equal": np.array_equal(
                    direct_a,
                    mapped_a,
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    print(comparison.to_string(index=False))

    return comparison


# =============================================================================
# EVENT TIMESTAMP / CLOSE ALIGNMENT
# =============================================================================


def audit_timestamp_alignment(
    trades: pd.DataFrame,
    metadata: pd.DataFrame,
):

    section("TIMESTAMP / CLOSE ALIGNMENT")

    metadata_copy = metadata.copy()

    metadata_copy["event_id"] = pd.to_numeric(
        metadata_copy["event_id"],
        errors="coerce",
    ).astype(np.int64)

    metadata_copy["timestamp"] = pd.to_datetime(
        metadata_copy["timestamp"],
        errors="coerce",
        utc=True,
    )

    metadata_copy["close"] = pd.to_numeric(
        metadata_copy["close"],
        errors="coerce",
    )

    merged = trades.merge(
        metadata_copy[
            [
                "event_id",
                "timestamp",
                "close",
            ]
        ],
        on="event_id",
        how="left",
        suffixes=(
            "_trade",
            "_metadata",
        ),
        validate="many_to_one",
    )

    timestamp_equal = merged["timestamp_trade"] == merged["timestamp_metadata"]

    close_equal = np.isclose(
        merged["entry"].to_numpy(dtype=float),
        merged["close"].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    )

    print(f"Timestamp matches : {timestamp_equal.sum():,}/{len(merged):,}")

    print(f"Entry == close    : {close_equal.sum():,}/{len(merged):,}")

    print()
    print(f"Timestamp mismatch: {(~timestamp_equal).sum():,}")

    print(f"Close mismatch    : {(~close_equal).sum():,}")


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08W.3")

    print("IDENTITY / INDEX / CACHE AUDIT")

    print()
    print("This script does NOT resolve trades.")

    print("This script does NOT change parameters.")

    print(
        "This script only determines whether "
        "08P and Research 07 reference the "
        "same event/cache rows."
    )

    trades, metadata, cache = load_all()

    identity_mismatch = audit_event_id_identity(
        metadata,
        cache,
    )

    cache_mismatches = audit_direct_event_index(
        trades,
        metadata,
        cache,
    )

    compare_path_values(
        trades,
        metadata,
        cache,
    )

    audit_timestamp_alignment(
        trades,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    if len(cache_mismatches):
        cache_mismatches.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(f"Detailed mismatches saved to:\n{OUTPUT_FILE}")

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    section("08W.3 FINAL RESULT")

    if len(identity_mismatch) == 0:
        print("Metadata identity: PASS")

    else:
        print("Metadata identity: FAIL")

    if len(cache_mismatches) == 0:
        print("Direct event_id cache mapping: PASS")

    else:
        print("Direct event_id cache mapping: FAIL")

    print()

    if len(identity_mismatch) == 0 and len(cache_mismatches) == 0:
        print("EVENT/CACHE IDENTITY IS NOT THE SOURCE OF THE 08W.2 MISMATCH.")

        print()
        print("Next investigation:")

        print(
            "  Research 07 path-cache construction "
            "and exact 08P trade-generation alignment."
        )

    else:
        print("EVENT/CACHE IDENTITY IS A REAL SOURCE OF THE 08W.2 MISMATCH.")

        print()
        print("Do NOT modify 08P yet.")

    print()
    print("08W.3 COMPLETE")


if __name__ == "__main__":
    main()
