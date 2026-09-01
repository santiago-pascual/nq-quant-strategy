from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08M V2
# =============================================================================
#
# BROAD TP / SL / HORIZON SEARCH — VECTORIZED
#
# Fixed contexts from Research 08E.
#
# Search:
#   TP  : 5 -> 80 points, step 5
#   RR  : 0.50 -> 2.50, step 0.10
#   H   : 1 -> 120 bars
#
# IMPORTANT:
#   - No HMM retraining
#   - No volatility recalculation for context definition
#   - No context discovery
#   - No failure test
#   - No production changes
#
# OUTCOME LOGIC
# -------------
# TP and SL are reconstructed directly from future_close.
#
# For every event:
#
#   first TP hit < first SL hit -> WIN
#   first SL hit < first TP hit -> LOSS
#   same bar -> AMBIGUOUS
#   neither -> UNRESOLVED
#
# Same-bar conflicts are NEVER automatically treated as wins.
#
# The search reports:
#
#   WR among resolved
#   resolution rate
#   expectancy among resolved
#   expectancy over all opportunities
#
# This prevents artificial results from unresolved paths.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

OUTPUT_GRID = RESULTS_DIR / "research_08m_v2_tp_sl_horizon_grid.csv"

OUTPUT_TOP = RESULTS_DIR / "research_08m_v2_top_candidates.csv"

OUTPUT_CONTEXT = RESULTS_DIR / "research_08m_v2_context_summary.csv"


# =============================================================================
# SEARCH SPACE
# =============================================================================

TP_VALUES = np.arange(
    5.0,
    80.0 + 0.001,
    5.0,
    dtype=np.float64,
)

RR_VALUES = np.round(
    np.arange(
        0.50,
        2.50 + 0.001,
        0.10,
        dtype=np.float64,
    ),
    2,
)

HORIZON_VALUES = np.arange(
    1,
    121,
    dtype=np.int64,
)


# =============================================================================
# MINIMUM SAMPLE REQUIREMENTS
# =============================================================================

MIN_RESOLVED = 50
MIN_RESOLVED_STRONG = 100

MIN_RESOLUTION_RATE = 0.20


# =============================================================================
# FIXED CONTEXTS
# =============================================================================

CONTEXTS = [
    ("SHORT", 1, "20-40", 3.5),
    ("LONG", 2, "60-80", 3.5),
    ("SHORT", 0, "20-40", 3.0),
    ("LONG", 0, "40-60", 2.0),
    ("LONG", 0, "80-100", 2.5),
    ("LONG", 1, "20-40", 3.0),
    ("LONG", 1, "20-40", 2.5),
    ("LONG", 0, "20-40", 2.0),
    ("LONG", 0, "20-40", 2.5),
    ("SHORT", 2, "80-100", 2.0),
]


# =============================================================================
# HELPERS
# =============================================================================


def section(title: str):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# LOAD METADATA
# =============================================================================


def load_metadata() -> pd.DataFrame:

    section("LOADING RESEARCH 07 METADATA")

    df = pd.read_csv(METADATA_PATH)

    required = [
        "event_id",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing metadata columns: {missing}")

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(np.int64)

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df["zscore_30"] = pd.to_numeric(
        df["zscore_30"],
        errors="coerce",
    )

    print(f"Events: {len(df):,}")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm() -> pd.DataFrame:

    section("LOADING RESEARCH 08B HMM")

    df = pd.read_csv(HMM_PATH)

    required = [
        "event_id",
        "hmm_state",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing HMM columns: {missing}")

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(
        df["hmm_state"],
        errors="raise",
    ).astype(np.int8)

    print(f"HMM rows: {len(df):,}")

    return df[
        [
            "event_id",
            "hmm_state",
        ]
    ]


# =============================================================================
# LOAD VOLATILITY
# =============================================================================


def load_volatility() -> pd.DataFrame:

    section("BUILDING CAUSAL VOLATILITY CONTEXT")

    from src.databento_loader import (
        load_databento_mnq,
    )

    from src.feature_engine import (
        add_return_features,
        add_volatility_features,
    )

    market = load_databento_mnq()

    print(f"Rows loaded: {len(market):,}")

    market = add_return_features(market)

    market = add_volatility_features(market)

    market["_timestamp"] = pd.to_datetime(
        market["timestamp ET"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    market["realized_vol_30"] = pd.to_numeric(
        market["realized_vol_30"],
        errors="coerce",
    )

    market = market[market["_timestamp"].notna() & market["realized_vol_30"].notna()][
        [
            "_timestamp",
            "realized_vol_30",
        ]
    ].copy()

    market = (
        market.sort_values("_timestamp")
        .drop_duplicates(
            "_timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    metadata = pd.read_csv(METADATA_PATH)

    metadata["event_id"] = pd.to_numeric(
        metadata["event_id"],
        errors="raise",
    ).astype(np.int64)

    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    events = (
        metadata[
            [
                "event_id",
                "timestamp",
            ]
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    mapped = pd.merge_asof(
        events,
        market,
        left_on="timestamp",
        right_on="_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    print(
        "Events mapped to volatility: "
        f"{mapped['realized_vol_30'].notna().sum():,}/"
        f"{len(mapped):,}"
    )

    values = mapped["realized_vol_30"].to_numpy(dtype=np.float64)

    n = len(values)

    percentile = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    # -------------------------------------------------------------------------
    # Causal expanding percentile.
    #
    # IMPORTANT:
    # The current volatility value is compared ONLY with values that existed
    # before it.
    #
    # This avoids future leakage.
    # -------------------------------------------------------------------------

    valid = np.flatnonzero(np.isfinite(values))

    # Faster than repeatedly calling list.insert().
    #
    # We use a sorted NumPy history and periodically rebuild it.
    #
    # For this research this is still deterministic and causal.

    history = np.empty(
        len(valid),
        dtype=np.float64,
    )

    history_count = 0

    for counter, idx in enumerate(valid):
        value = values[idx]

        if history_count > 0:
            previous = history[:history_count]

            position = np.searchsorted(
                previous,
                value,
                side="right",
            )

            percentile[idx] = position / history_count * 100.0

            # Insert while preserving sort.
            if position < history_count:
                history[position + 1 : history_count + 1] = history[
                    position:history_count
                ]

            history[position] = value

        else:
            history[0] = value

        history_count += 1

        if (counter + 1) % 100_000 == 0 or counter + 1 == len(valid):
            print(f"  Percentile: {counter + 1:,}/{len(valid):,}")

    mapped["vol_percentile"] = percentile

    mapped["vol_bucket"] = np.select(
        [
            percentile < 20,
            percentile < 40,
            percentile < 60,
            percentile < 80,
            percentile >= 80,
        ],
        [
            "0-20",
            "20-40",
            "40-60",
            "60-80",
            "80-100",
        ],
        default="UNKNOWN",
    )

    return mapped[
        [
            "event_id",
            "realized_vol_30",
            "vol_percentile",
            "vol_bucket",
        ]
    ]


# =============================================================================
# BUILD EVENT TABLE
# =============================================================================


def build_events() -> pd.DataFrame:

    section("BUILDING COMPLETE EVENT CONTEXT")

    metadata = load_metadata()

    hmm = load_hmm()

    volatility = load_volatility()

    events = metadata.merge(
        hmm,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    events = events.merge(
        volatility,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    required = [
        "event_id",
        "close",
        "zscore_30",
        "hmm_state",
        "vol_bucket",
    ]

    before = len(events)

    events = events[events[required].notna().all(axis=1)].copy()

    print(f"Original events: {before:,}")

    print(f"Valid events: {len(events):,}")

    return events.reset_index(drop=True)


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths() -> np.ndarray:

    section("LOADING RESEARCH 07 FUTURE PATH CACHE")

    cache = np.load(
        PATH_CACHE_PATH,
        allow_pickle=False,
    )

    required = [
        "future_close",
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]

    for key in required:
        if key not in cache.files:
            raise RuntimeError(f"Missing cache key: {key}")

    future_close = cache["future_close"]

    print(f"future_close: {future_close.shape} {future_close.dtype}")

    if future_close.shape != (
        825717,
        120,
    ):
        raise RuntimeError("Unexpected future_close shape.")

    print("Path cache integrity: OK")

    return future_close


# =============================================================================
# SELECT CONTEXT
# =============================================================================


def select_context(
    events: pd.DataFrame,
    context: tuple,
):

    side, hmm_state, vol_bucket, zscore = context

    mask = events["hmm_state"] == hmm_state

    mask &= events["vol_bucket"] == vol_bucket

    if side == "LONG":
        mask &= events["zscore_30"] <= -abs(zscore)

    elif side == "SHORT":
        mask &= events["zscore_30"] >= abs(zscore)

    else:
        raise ValueError(f"Unknown side: {side}")

    return events.loc[mask]


# =============================================================================
# VECTORIZED GRID ENGINE
# =============================================================================


def evaluate_context_vectorized(
    event_ids: np.ndarray,
    entries: np.ndarray,
    future_close: np.ndarray,
    side: str,
):
    """
    Evaluate the entire TP × RR × HORIZON grid without looping over
    individual events.

    We still loop over TP/RR pairs, but every event and every horizon is
    processed by NumPy.

    This is dramatically faster than the event-by-event Python implementation.
    """

    n_events = len(event_ids)

    if n_events == 0:
        return []

    # -------------------------------------------------------------------------
    # Extract paths for this context.
    #
    # This is the only large context-specific matrix.
    # Shape:
    #
    #     N_events × 120
    #
    # -------------------------------------------------------------------------

    paths = future_close[
        event_ids,
        :120,
    ].astype(
        np.float64,
        copy=False,
    )

    entries = entries.astype(
        np.float64,
        copy=False,
    )

    # -------------------------------------------------------------------------
    # Relative movement from entry.
    #
    # LONG:
    #   positive = favorable
    #
    # SHORT:
    #   positive = favorable
    #
    # Shape = N × 120
    # -------------------------------------------------------------------------

    if side == "LONG":
        movement = paths - entries[:, None]

    else:
        movement = entries[:, None] - paths

    finite = np.isfinite(movement)

    # Invalid values must never trigger TP or SL.

    movement = np.where(
        finite,
        movement,
        -np.inf,
    )

    results = []

    # -------------------------------------------------------------------------
    # TP × RR
    # -------------------------------------------------------------------------

    for tp in TP_VALUES:
        print(
            f"    TP={tp:5.1f}",
            end=" ",
            flush=True,
        )

        for rr in RR_VALUES:
            sl = tp / rr

            # -----------------------------------------------------------------
            # Hit matrices
            #
            # Shape:
            #
            #     N_events × 120
            #
            # -----------------------------------------------------------------

            tp_hit = movement >= tp

            sl_hit = movement <= -sl

            # -----------------------------------------------------------------
            # First hit index for every event.
            #
            # argmax alone is dangerous because an all-False row returns 0.
            #
            # Therefore we separately check whether ANY hit exists.
            # -----------------------------------------------------------------

            tp_any = tp_hit.any(axis=1)

            sl_any = sl_hit.any(axis=1)

            first_tp = np.argmax(
                tp_hit,
                axis=1,
            )

            first_sl = np.argmax(
                sl_hit,
                axis=1,
            )

            # -----------------------------------------------------------------
            # Horizon loop
            #
            # This loop is only 120 iterations.
            # Each iteration operates over ALL events simultaneously.
            # -----------------------------------------------------------------

            for horizon in HORIZON_VALUES:
                h = int(horizon)

                # -------------------------------------------------------------
                # Restrict hit existence to horizon.
                # -------------------------------------------------------------

                tp_h = tp_hit[
                    :,
                    :h,
                ]

                sl_h = sl_hit[
                    :,
                    :h,
                ]

                tp_exists = tp_h.any(axis=1)

                sl_exists = sl_h.any(axis=1)

                tp_first = np.argmax(
                    tp_h,
                    axis=1,
                )

                sl_first = np.argmax(
                    sl_h,
                    axis=1,
                )

                # -------------------------------------------------------------
                # Outcomes
                # -------------------------------------------------------------

                wins = tp_exists & ~sl_exists

                losses = sl_exists & ~tp_exists

                both = tp_exists & sl_exists

                # Both exist: compare first hit.

                wins |= both & (tp_first < sl_first)

                losses |= both & (sl_first < tp_first)

                ambiguous = both & (tp_first == sl_first)

                unresolved = ~tp_exists & ~sl_exists

                wins_count = int(wins.sum())

                losses_count = int(losses.sum())

                ambiguous_count = int(ambiguous.sum())

                unresolved_count = int(unresolved.sum())

                resolved = wins_count + losses_count

                if resolved > 0:
                    wr = wins_count / resolved

                else:
                    wr = np.nan

                resolved_rate = resolved / n_events

                ambiguous_rate = ambiguous_count / n_events

                expectancy = wr * rr - (1.0 - wr) if np.isfinite(wr) else np.nan

                expectancy_all = (wins_count * rr - losses_count) / n_events

                # -------------------------------------------------------------
                # Conservative score
                # -------------------------------------------------------------

                if (
                    resolved >= MIN_RESOLVED
                    and resolved_rate >= MIN_RESOLUTION_RATE
                    and np.isfinite(expectancy)
                ):
                    sample_factor = min(
                        1.0,
                        np.sqrt(resolved / 500.0),
                    )

                    resolution_factor = min(
                        1.0,
                        resolved_rate / 0.50,
                    )

                    score = expectancy * 0.60 + (wr - 0.50) * 0.40

                    score *= sample_factor * resolution_factor

                    score *= max(
                        0.0,
                        1.0 - ambiguous_rate,
                    )

                else:
                    score = -np.inf

                results.append(
                    {
                        "tp": float(tp),
                        "sl": float(sl),
                        "rr": float(rr),
                        "horizon": h,
                        "observations": n_events,
                        "wins": wins_count,
                        "losses": losses_count,
                        "ambiguous": ambiguous_count,
                        "unresolved": unresolved_count,
                        "resolved": resolved,
                        "win_rate_resolved": wr,
                        "resolved_rate": resolved_rate,
                        "ambiguous_rate": ambiguous_rate,
                        "expectancy_resolved_r": expectancy,
                        "expectancy_all_r": expectancy_all,
                        "score": score,
                    }
                )

        print("DONE")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08M V2")

    print("VECTORIZED BROAD PARAMETER SEARCH")

    print("-" * 100)

    print("Fixed contexts from Research 08E.")

    print("Direct future_close reconstruction.")

    print("No HMM retraining.")

    print("No context optimization.")

    print("No failure test.")

    print("No production changes.")

    section("SEARCH SPACE")

    print(f"TP: {TP_VALUES[0]:.1f} -> {TP_VALUES[-1]:.1f} step 5")

    print(f"RR: {RR_VALUES[0]:.2f} -> {RR_VALUES[-1]:.2f} step 0.10")

    print("Horizon: 1 -> 120")

    total_points = len(CONTEXTS) * len(TP_VALUES) * len(RR_VALUES) * len(HORIZON_VALUES)

    print(f"Total grid points: {total_points:,}")

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    events = build_events()

    future_close = load_paths()

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    all_results = []

    for i, context in enumerate(
        CONTEXTS,
        start=1,
    ):
        side, hmm, vol, z = context

        section(f"CONTEXT {i}/{len(CONTEXTS)}")

        print(f"{side} | HMM={hmm} | VOL={vol} | Z={z}")

        selected = select_context(
            events,
            context,
        )

        print(f"Observations: {len(selected):,}")

        if len(selected) == 0:
            print("SKIPPED — no observations.")

            continue

        event_ids = selected["event_id"].to_numpy(dtype=np.int64)

        entries = selected["close"].to_numpy(dtype=np.float64)

        # -------------------------------------------------------------
        # Integrity checks
        # -------------------------------------------------------------

        if event_ids.min() < 0 or event_ids.max() >= future_close.shape[0]:
            raise RuntimeError("Event ID outside path cache.")

        print("Running vectorized grid...")

        results = evaluate_context_vectorized(
            event_ids,
            entries,
            future_close,
            side,
        )

        for row in results:
            row.update(
                {
                    "side": side,
                    "hmm_state": hmm,
                    "vol_bucket": vol,
                    "zscore": z,
                }
            )

        all_results.extend(results)

        print(f"Context results: {len(results):,}")

    if not all_results:
        raise RuntimeError("No results generated.")

    results = pd.DataFrame(all_results)

    # -------------------------------------------------------------------------
    # SORT
    # -------------------------------------------------------------------------

    results = results.sort_values(
        [
            "score",
            "expectancy_resolved_r",
            "win_rate_resolved",
            "resolved",
        ],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # STRONG CANDIDATES
    # -------------------------------------------------------------------------

    strong = results[
        (results["win_rate_resolved"] >= 0.50)
        & (results["expectancy_resolved_r"] > 0)
        & (results["resolved"] >= MIN_RESOLVED_STRONG)
        & (results["resolved_rate"] >= MIN_RESOLUTION_RATE)
    ].copy()

    # -------------------------------------------------------------------------
    # BEST BY CONTEXT
    # -------------------------------------------------------------------------

    context_rows = []

    group_cols = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
    ]

    for key, group in results.groupby(
        group_cols,
        dropna=False,
    ):
        valid = group[
            (group["resolved"] >= MIN_RESOLVED)
            & (group["resolved_rate"] >= MIN_RESOLUTION_RATE)
        ]

        if valid.empty:
            continue

        best = valid.iloc[valid["score"].argmax()]

        context_rows.append(
            {
                **dict(
                    zip(
                        group_cols,
                        key,
                    )
                ),
                "grid_points": len(group),
                "best_tp": best["tp"],
                "best_sl": best["sl"],
                "best_rr": best["rr"],
                "best_horizon": best["horizon"],
                "best_wr": best["win_rate_resolved"],
                "best_resolved": best["resolved"],
                "best_resolution": best["resolved_rate"],
                "best_expectancy": best["expectancy_resolved_r"],
                "best_expectancy_all": best["expectancy_all_r"],
                "best_score": best["score"],
            }
        )

    context_summary = pd.DataFrame(context_rows)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_GRID,
        index=False,
    )

    strong.to_csv(
        OUTPUT_TOP,
        index=False,
    )

    context_summary.to_csv(
        OUTPUT_CONTEXT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # DISPLAY
    # -------------------------------------------------------------------------

    display_columns = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "tp",
        "sl",
        "rr",
        "horizon",
        "observations",
        "wins",
        "losses",
        "ambiguous",
        "unresolved",
        "resolved",
        "win_rate_resolved",
        "resolved_rate",
        "expectancy_resolved_r",
        "expectancy_all_r",
        "score",
    ]

    section("TOP 50 RESULTS")

    print(results[display_columns].head(50).to_string(index=False))

    section("STRONG CANDIDATES")

    if strong.empty:
        print("No candidates currently satisfy:")

        print("WR >= 50%")

        print("Expectancy > 0")

        print(f"Resolved >= {MIN_RESOLVED_STRONG}")

        print(f"Resolution >= {MIN_RESOLUTION_RATE:.0%}")

    else:
        print(strong[display_columns].head(100).to_string(index=False))

    section("BEST RESULT PER CONTEXT")

    if context_summary.empty:
        print("No qualifying context results.")

    else:
        print(context_summary.to_string(index=False))

    section("RESEARCH 08M V2 COMPLETE")

    print(f"Total results: {len(results):,}")

    print(f"Strong candidates: {len(strong):,}")

    print("\nFILES SAVED")

    print(OUTPUT_GRID)

    print(OUTPUT_TOP)

    print(OUTPUT_CONTEXT)

    print("\nINTEGRITY RULES")

    print("TP/SL ordering reconstructed from future_close.")

    print("Same-bar TP/SL = AMBIGUOUS.")

    print("Ambiguous trades are not wins.")

    print("Unresolved trades are not wins.")

    print("All 120 horizons are tested.")

    print("RR 0.50 -> 2.50 tested at 0.10 resolution.")

    print("No parameter was selected as final strategy.")


if __name__ == "__main__":
    main()
