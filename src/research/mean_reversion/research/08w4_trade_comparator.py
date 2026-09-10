from __future__ import annotations

from pathlib import Path

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

METADATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "cache"
    / "research_07_event_metadata.csv"
)

CACHE_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "cache"
    / "research_07_future_path_cache.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

OUTPUT_PATH = OUTPUT_DIR / "research_08w4_trade_comparison.csv"

MISMATCH_PATH = OUTPUT_DIR / "research_08w4_mismatches_only.csv"


# ============================================================
# HELPERS
# ============================================================


def normalize_result(value: str) -> str:
    """
    Normalize the result labels used by 08P to the terminology
    used by the path resolver.
    """

    value = str(value).strip().upper()

    mapping = {
        "WIN": "TARGET",
        "LOSS": "STOP",
        "UNRESOLVED": "TIMEOUT",
        "TIMEOUT": "TIMEOUT",
        "TARGET": "TARGET",
        "STOP": "STOP",
    }

    if value not in mapping:
        raise ValueError(f"Unknown result label: {value}")

    return mapping[value]


def resolve_path(
    favorable: np.ndarray,
    adverse: np.ndarray,
    target: float,
    stop: float,
    horizon: int,
):
    """
    Reproduce the barrier-resolution logic.

    Path starts at the first future bar.

    TARGET:
        favorable >= target

    STOP:
        adverse >= stop

    If both barriers are hit on the same bar:
        STOP wins.

    Returns:
        result
        R
        bars_to_result
        first_target_bar
        first_stop_bar
    """

    h = min(
        int(horizon),
        len(favorable),
        len(adverse),
    )

    if h <= 0:
        return (
            "TIMEOUT",
            0.0,
            0,
            None,
            None,
        )

    fav = np.asarray(
        favorable[:h],
        dtype=np.float64,
    )

    adv = np.asarray(
        adverse[:h],
        dtype=np.float64,
    )

    target_hits = fav >= target
    stop_hits = adv >= stop

    # First target hit
    if target_hits.any():
        first_target_bar = int(np.argmax(target_hits)) + 1
    else:
        first_target_bar = None

    # First stop hit
    if stop_hits.any():
        first_stop_bar = int(np.argmax(stop_hits)) + 1
    else:
        first_stop_bar = None

    # No barrier
    if first_target_bar is None and first_stop_bar is None:
        return (
            "TIMEOUT",
            0.0,
            h,
            None,
            None,
        )

    # Only stop
    if first_target_bar is None:
        return (
            "STOP",
            -float(stop),
            first_stop_bar,
            None,
            first_stop_bar,
        )

    # Only target
    if first_stop_bar is None:
        return (
            "TARGET",
            float(target),
            first_target_bar,
            first_target_bar,
            None,
        )

    # Both exist.
    #
    # IMPORTANT:
    # Same-bar conflict resolves as STOP.
    if first_stop_bar <= first_target_bar:
        return (
            "STOP",
            -float(stop),
            first_stop_bar,
            first_target_bar,
            first_stop_bar,
        )

    return (
        "TARGET",
        float(target),
        first_target_bar,
        first_target_bar,
        first_stop_bar,
    )


def safe_equal_float(
    a: float,
    b: float,
    atol: float = 1e-8,
) -> bool:
    """
    Exact-enough floating-point comparison.
    """

    if pd.isna(a) or pd.isna(b):
        return pd.isna(a) and pd.isna(b)

    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=0.0,
            atol=atol,
        )
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 80)
    print("08W.4 — TRADE-BY-TRADE COMPARATOR")
    print("=" * 80)

    # ========================================================
    # 1. LOAD 08P
    # ========================================================

    print("\n[1/6] Loading 08P trades...")

    trades = pd.read_csv(TRADES_PATH)

    print(f"08P trades: {len(trades):,}")

    required_trade_columns = [
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

    missing = [c for c in required_trade_columns if c not in trades.columns]

    if missing:
        raise RuntimeError("08P is missing required columns: " + ", ".join(missing))

    # ========================================================
    # 2. LOAD METADATA
    # ========================================================

    print("\n[2/6] Loading Research 07 metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Metadata rows: {len(metadata):,}")

    required_metadata_columns = [
        "event_id",
        "data_index",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required_metadata_columns if c not in metadata.columns]

    if missing:
        raise RuntimeError(
            "Research 07 metadata is missing columns: " + ", ".join(missing)
        )

    metadata = metadata.reset_index(drop=True)

    # ========================================================
    # 3. LOAD CACHE
    # ========================================================

    print("\n[3/6] Loading Research 07 path cache...")

    cache = np.load(CACHE_PATH)

    future_close = cache["future_close"]
    long_favorable = cache["long_favorable"]
    long_adverse = cache["long_adverse"]
    short_favorable = cache["short_favorable"]
    short_adverse = cache["short_adverse"]

    print(f"future_close:    {future_close.shape}")

    print(f"long_favorable:  {long_favorable.shape}")

    print(f"long_adverse:    {long_adverse.shape}")

    print(f"short_favorable: {short_favorable.shape}")

    print(f"short_adverse:   {short_adverse.shape}")

    # ========================================================
    # 4. GLOBAL CACHE VALIDATION
    # ========================================================

    print("\n[4/6] Validating cache identity...")

    n_metadata = len(metadata)
    n_cache = future_close.shape[0]

    if n_metadata != n_cache:
        raise RuntimeError(
            f"Metadata/cache row count mismatch: {n_metadata:,} vs {n_cache:,}"
        )

    expected_event_ids = np.arange(n_metadata)

    event_id_identity = np.array_equal(
        metadata["event_id"].to_numpy(),
        expected_event_ids,
    )

    print(f"event_id == cache row index: {event_id_identity}")

    if not event_id_identity:
        raise RuntimeError("event_id is NOT identical to cache row index.")

    # ========================================================
    # 5. TRADE-BY-TRADE COMPARISON
    # ========================================================

    print("\n[5/6] Comparing every trade...")

    rows = []

    for trade_index, trade in trades.iterrows():
        # ----------------------------------------------------
        # BASIC TRADE DATA
        # ----------------------------------------------------

        event_id = int(trade["event_id"])

        side = str(trade["side"]).strip().upper()

        target = float(trade["tp"])

        stop = float(trade["sl"])

        horizon = int(trade["horizon"])

        # ----------------------------------------------------
        # VALIDATE EVENT ID
        # ----------------------------------------------------

        if event_id < 0:
            raise RuntimeError(f"Negative event_id at trade {trade_index}: {event_id}")

        if event_id >= n_cache:
            raise RuntimeError(
                f"event_id out of cache range at trade {trade_index}: {event_id}"
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        meta = metadata.iloc[event_id]

        metadata_event_id = int(meta["event_id"])

        metadata_data_index = int(meta["data_index"])

        metadata_timestamp = meta["timestamp"]

        metadata_close = float(meta["close"])

        metadata_window = meta["window"]

        metadata_zscore = float(meta["zscore_30"])

        # ----------------------------------------------------
        # 08P VALUES
        # ----------------------------------------------------

        p_result_raw = str(trade["result"])

        p_result = normalize_result(p_result_raw)

        p_r = float(trade["r"])

        p_bars = int(trade["bars_to_result"])

        p_timestamp = trade["timestamp"]

        p_entry = float(trade["entry"])

        # ----------------------------------------------------
        # SIDE-SPECIFIC CACHE
        # ----------------------------------------------------

        if side == "LONG":
            favorable = long_favorable[event_id]

            adverse = long_adverse[event_id]

        elif side == "SHORT":
            favorable = short_favorable[event_id]

            adverse = short_adverse[event_id]

        else:
            raise RuntimeError(f"Unknown side '{side}' at trade {trade_index}")

        # ----------------------------------------------------
        # RESOLVE CACHE PATH
        # ----------------------------------------------------

        (
            cache_result,
            cache_r,
            cache_bars,
            first_target_bar,
            first_stop_bar,
        ) = resolve_path(
            favorable=favorable,
            adverse=adverse,
            target=target,
            stop=stop,
            horizon=horizon,
        )

        # ----------------------------------------------------
        # CACHE GEOMETRY
        # ----------------------------------------------------

        same_bar_conflict = (
            first_target_bar is not None
            and first_stop_bar is not None
            and (first_target_bar == first_stop_bar)
        )

        # ----------------------------------------------------
        # ENTRY CHECK
        # ----------------------------------------------------

        entry_matches_metadata = safe_equal_float(
            p_entry,
            metadata_close,
        )

        # ----------------------------------------------------
        # TIMESTAMP CHECK
        # ----------------------------------------------------

        timestamp_matches_metadata = str(p_timestamp) == str(metadata_timestamp)

        # ----------------------------------------------------
        # WINDOW CHECK
        # ----------------------------------------------------

        window_matches_metadata = str(trade["window"]) == str(metadata_window)

        # ----------------------------------------------------
        # EVENT ID CHECK
        # ----------------------------------------------------

        event_id_matches_metadata = event_id == metadata_event_id

        # ----------------------------------------------------
        # RESULT / R / BARS CHECKS
        # ----------------------------------------------------

        result_match = p_result == cache_result

        r_match = safe_equal_float(
            p_r,
            cache_r,
        )

        bars_match = p_bars == cache_bars

        # ----------------------------------------------------
        # COMPLETE MATCH
        # ----------------------------------------------------

        all_match = result_match and r_match and bars_match

        # ----------------------------------------------------
        # MISMATCH CLASSIFICATION
        # ----------------------------------------------------

        if all_match:
            mismatch_type = "MATCH"

        elif not result_match:
            mismatch_type = "RESULT_MISMATCH"

        elif not r_match:
            mismatch_type = "R_MISMATCH"

        elif not bars_match:
            mismatch_type = "BARS_MISMATCH"

        else:
            mismatch_type = "OTHER"

        # ----------------------------------------------------
        # FIRST BARRIER DIFFERENCE
        # ----------------------------------------------------

        if first_target_bar is None and first_stop_bar is None:
            barrier_pattern = "NO_BARRIER"

        elif first_target_bar is None:
            barrier_pattern = "STOP_ONLY"

        elif first_stop_bar is None:
            barrier_pattern = "TARGET_ONLY"

        elif same_bar_conflict:
            barrier_pattern = "SAME_BAR"

        elif first_target_bar < first_stop_bar:
            barrier_pattern = "TARGET_FIRST"

        else:
            barrier_pattern = "STOP_FIRST"

        # ----------------------------------------------------
        # FUTURE CLOSE SNAPSHOT
        # ----------------------------------------------------

        h = min(
            horizon,
            future_close.shape[1],
        )

        fc = future_close[
            event_id,
            :h,
        ]

        future_close_values = {}

        for bar_number in [
            1,
            2,
            3,
            4,
            5,
            10,
            20,
        ]:
            idx = bar_number - 1

            if idx < len(fc):
                future_close_values[f"future_close_{bar_number}"] = float(fc[idx])

            else:
                future_close_values[f"future_close_{bar_number}"] = np.nan

        # ----------------------------------------------------
        # SAVE ROW
        # ----------------------------------------------------

        rows.append(
            {
                # Identity
                "trade_index": trade_index,
                "event_id": event_id,
                "metadata_event_id": metadata_event_id,
                "data_index": metadata_data_index,
                # Strategy
                "strategy_name": (trade["strategy_name"]),
                "candidate_id": (trade["candidate_id"]),
                "side": side,
                "hmm_state": (trade["hmm_state"]),
                "vol_bucket": (trade["vol_bucket"]),
                "zscore_08p": (trade["zscore"]),
                "zscore_metadata": (metadata_zscore),
                # Window / time
                "window_08p": (trade["window"]),
                "window_metadata": (metadata_window),
                "timestamp_08p": (p_timestamp),
                "timestamp_metadata": (metadata_timestamp),
                # Entry
                "entry_08p": p_entry,
                "entry_metadata": metadata_close,
                # Geometry
                "tp": target,
                "sl": stop,
                "rr": float(trade["rr"]),
                "horizon": horizon,
                # 08P
                "08p_result_raw": p_result_raw,
                "08p_result": p_result,
                "08p_r": p_r,
                "08p_bars": p_bars,
                # Cache
                "cache_result": cache_result,
                "cache_r": cache_r,
                "cache_bars": cache_bars,
                # Barrier diagnostics
                "first_target_bar": (first_target_bar),
                "first_stop_bar": (first_stop_bar),
                "barrier_pattern": (barrier_pattern),
                "same_bar_conflict": (same_bar_conflict),
                # Comparison
                "result_match": result_match,
                "r_match": r_match,
                "bars_match": bars_match,
                "all_match": all_match,
                # Identity diagnostics
                "event_id_matches_metadata": (event_id_matches_metadata),
                "entry_matches_metadata": (entry_matches_metadata),
                "timestamp_matches_metadata": (timestamp_matches_metadata),
                "window_matches_metadata": (window_matches_metadata),
                # Classification
                "mismatch_type": mismatch_type,
                # Future closes
                **future_close_values,
            }
        )

    comparison = pd.DataFrame(rows)

    # ========================================================
    # 6. SAVE RESULTS
    # ========================================================

    print("\n[6/6] Saving diagnostics...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    mismatches = comparison[~comparison["all_match"]].copy()

    mismatches.to_csv(
        MISMATCH_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    total = len(comparison)

    result_matches = int(comparison["result_match"].sum())

    r_matches = int(comparison["r_match"].sum())

    bars_matches = int(comparison["bars_match"].sum())

    all_matches = int(comparison["all_match"].sum())

    same_bar_conflicts = int(comparison["same_bar_conflict"].sum())

    print(f"Total trades:          {total:,}")

    print(
        f"Result matches:        "
        f"{result_matches:,}/{total:,} "
        f"({result_matches / total * 100:.2f}%)"
    )

    print(
        f"R matches:             "
        f"{r_matches:,}/{total:,} "
        f"({r_matches / total * 100:.2f}%)"
    )

    print(
        f"Bars matches:          "
        f"{bars_matches:,}/{total:,} "
        f"({bars_matches / total * 100:.2f}%)"
    )

    print(
        f"ALL fields match:      "
        f"{all_matches:,}/{total:,} "
        f"({all_matches / total * 100:.2f}%)"
    )

    print(
        f"Same-bar conflicts:    "
        f"{same_bar_conflicts:,}/{total:,} "
        f"({same_bar_conflicts / total * 100:.2f}%)"
    )

    # ========================================================
    # IDENTITY CHECKS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("IDENTITY CHECKS")
    print("=" * 80)

    for column, label in [
        (
            "event_id_matches_metadata",
            "event_id",
        ),
        (
            "entry_matches_metadata",
            "entry",
        ),
        (
            "timestamp_matches_metadata",
            "timestamp",
        ),
        (
            "window_matches_metadata",
            "window",
        ),
    ]:
        count = int(comparison[column].sum())

        print(f"{label:<12} {count:,}/{total:,} ({count / total * 100:.2f}%)")

    # ========================================================
    # MISMATCH TYPES
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MISMATCH TYPES")
    print("=" * 80)

    mismatch_counts = comparison["mismatch_type"].value_counts().sort_index()

    for name, count in mismatch_counts.items():
        pct = count / total * 100

        print(f"{name:<22}{count:>8,} ({pct:6.2f}%)")

    # ========================================================
    # RESULT CROSS-TAB
    # ========================================================

    print("\n")
    print("=" * 80)
    print("08P RESULT vs CACHE RESULT")
    print("=" * 80)

    result_table = pd.crosstab(
        comparison["08p_result"],
        comparison["cache_result"],
        margins=True,
    )

    print(result_table.to_string())

    # ========================================================
    # BY STRATEGY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("BY STRATEGY")
    print("=" * 80)

    for strategy, group in comparison.groupby(
        "strategy_name",
        sort=True,
    ):
        n = len(group)

        result_match_count = int(group["result_match"].sum())

        r_match_count = int(group["r_match"].sum())

        bars_match_count = int(group["bars_match"].sum())

        all_match_count = int(group["all_match"].sum())

        ambiguous_count = int(group["same_bar_conflict"].sum())

        print(f"\n{strategy}")

        print(f"  Trades:          {n:,}")

        print(
            f"  Result matches:  "
            f"{result_match_count:,}/{n:,} "
            f"({result_match_count / n * 100:.2f}%)"
        )

        print(
            f"  R matches:       "
            f"{r_match_count:,}/{n:,} "
            f"({r_match_count / n * 100:.2f}%)"
        )

        print(
            f"  Bars matches:    "
            f"{bars_match_count:,}/{n:,} "
            f"({bars_match_count / n * 100:.2f}%)"
        )

        print(
            f"  ALL match:       "
            f"{all_match_count:,}/{n:,} "
            f"({all_match_count / n * 100:.2f}%)"
        )

        print(
            f"  Same-bar:        "
            f"{ambiguous_count:,}/{n:,} "
            f"({ambiguous_count / n * 100:.2f}%)"
        )

        print("  Mismatch types:")

        strategy_mismatches = group["mismatch_type"].value_counts().sort_index()

        for name, count in strategy_mismatches.items():
            pct = count / n * 100

            print(f"    {name:<20}{count:>7,} ({pct:6.2f}%)")

    # ========================================================
    # BARRIER PATTERNS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("BARRIER PATTERNS")
    print("=" * 80)

    barrier_counts = comparison["barrier_pattern"].value_counts().sort_index()

    for name, count in barrier_counts.items():
        print(f"{name:<20}{count:>8,} ({count / total * 100:6.2f}%)")

    # ========================================================
    # FIRST MISMATCHES
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FIRST 30 MISMATCHES")
    print("=" * 80)

    if mismatches.empty:
        print("\nNO MISMATCHES FOUND.")

    else:
        display_columns = [
            "trade_index",
            "event_id",
            "strategy_name",
            "candidate_id",
            "side",
            "tp",
            "sl",
            "horizon",
            "08p_result",
            "08p_r",
            "08p_bars",
            "cache_result",
            "cache_r",
            "cache_bars",
            "first_target_bar",
            "first_stop_bar",
            "barrier_pattern",
            "same_bar_conflict",
            "mismatch_type",
        ]

        print(mismatches[display_columns].head(30).to_string(index=False))

    # ========================================================
    # SPECIFIC RESULT MISMATCHES
    # ========================================================

    result_mismatches = comparison[~comparison["result_match"]].copy()

    print("\n")
    print("=" * 80)
    print("FIRST 20 RESULT MISMATCHES")
    print("=" * 80)

    if result_mismatches.empty:
        print("\nNONE.")

    else:
        result_columns = [
            "trade_index",
            "event_id",
            "strategy_name",
            "side",
            "tp",
            "sl",
            "horizon",
            "08p_result",
            "cache_result",
            "08p_r",
            "cache_r",
            "08p_bars",
            "cache_bars",
            "first_target_bar",
            "first_stop_bar",
            "barrier_pattern",
        ]

        print(result_mismatches[result_columns].head(20).to_string(index=False))

    # ========================================================
    # FILE LOCATIONS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print("\nFull comparison:")

    print(OUTPUT_PATH)

    print("\nMismatches only:")

    print(MISMATCH_PATH)

    print("\n")
    print("=" * 80)
    print("08W.4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
