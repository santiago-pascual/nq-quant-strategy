from __future__ import annotations

from pathlib import Path
import bisect

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08L
# RAW FUTURE_CLOSE PATH RECONSTRUCTION AUDIT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT_TRADES = RESULTS_DIR / "research_08l_path_reconstruction_trades.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08l_path_reconstruction_summary.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08l_path_reconstruction_windows.csv"

OUTPUT_COMPARISON = RESULTS_DIR / "research_08l_comparison.csv"


# =============================================================================
# CANDIDATES
# =============================================================================

CANDIDATES = [
    {
        "candidate": "C1",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "40-60",
        "zscore": 2.0,
        "tp": 50.0,
        "sl": 12.5,
        "horizon": 3,
    },
    {
        "candidate": "A1",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "20-40",
        "zscore": 2.0,
        "tp": 50.0,
        "sl": 12.5,
        "horizon": 3,
    },
    {
        "candidate": "A2_H3",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 50.0,
        "sl": 12.5,
        "horizon": 3,
    },
    {
        "candidate": "A2_H4",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 50.0,
        "sl": 12.5,
        "horizon": 4,
    },
]


# =============================================================================
# TIMESTAMP NORMALIZATION
# =============================================================================


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Force timestamps to exactly datetime64[ns, UTC].

    pandas may preserve CSV timestamps as datetime64[us, UTC].
    merge_asof requires identical datetime dtypes.
    """

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")


# =============================================================================
# LOAD METADATA
# =============================================================================


def load_metadata() -> pd.DataFrame:

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 07 METADATA")
    print("=" * 100)

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

    df["timestamp"] = normalize_timestamp(df["timestamp"])

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm() -> pd.DataFrame:

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 08B HMM")
    print("=" * 100)

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
        errors="coerce",
    )

    if df["hmm_state"].isna().any():
        raise RuntimeError("HMM contains missing states.")

    df["hmm_state"] = df["hmm_state"].astype(int)

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

    print("\n" + "=" * 100)
    print("LOADING VOLATILITY CONTEXT")
    print("=" * 100)

    print("Rebuilding causal realized_vol_30 percentile.")

    print("Timestamp convention: timestamp ET")

    from src.databento_loader import (
        load_databento_mnq,
    )

    from src.feature_engine import (
        add_return_features,
        add_volatility_features,
    )

    # -------------------------------------------------------------------------
    # MARKET
    # -------------------------------------------------------------------------

    market = load_databento_mnq()

    print(f"Rows loaded: {len(market):,}")

    market = add_return_features(market)

    market = add_volatility_features(market)

    if "timestamp ET" not in market.columns:
        raise RuntimeError("Missing required market column: timestamp ET")

    print("Using market timestamp: timestamp ET")

    # -------------------------------------------------------------------------
    # FORCE EXACT SAME DTYPE
    # -------------------------------------------------------------------------

    market["_timestamp"] = normalize_timestamp(market["timestamp ET"])

    market["realized_vol_30"] = pd.to_numeric(
        market["realized_vol_30"],
        errors="coerce",
    )

    valid_market = market[
        market["_timestamp"].notna() & market["realized_vol_30"].notna()
    ][
        [
            "_timestamp",
            "realized_vol_30",
        ]
    ].copy()

    valid_market = (
        valid_market.sort_values("_timestamp")
        .drop_duplicates(
            "_timestamp",
            keep="last",
        )
        .reset_index(
            drop=True,
        )
    )

    print(f"Valid realized_vol_30: {len(valid_market):,}")

    print(
        "Market timestamp dtype:",
        valid_market["_timestamp"].dtype,
    )

    # -------------------------------------------------------------------------
    # EVENTS
    # -------------------------------------------------------------------------

    events = pd.read_csv(METADATA_PATH)

    events["event_id"] = pd.to_numeric(
        events["event_id"],
        errors="raise",
    ).astype(np.int64)

    events["timestamp"] = normalize_timestamp(events["timestamp"])

    events = (
        events[
            [
                "event_id",
                "timestamp",
            ]
        ]
        .sort_values("timestamp")
        .reset_index(
            drop=True,
        )
    )

    print(
        "Event timestamp dtype:",
        events["timestamp"].dtype,
    )

    # -------------------------------------------------------------------------
    # HARD DTYPE ASSERTION
    # -------------------------------------------------------------------------

    expected_dtype = "datetime64[ns, UTC]"

    if str(events["timestamp"].dtype) != expected_dtype:
        raise RuntimeError(
            f"Event timestamp dtype normalization failed: {events['timestamp'].dtype}"
        )

    if str(valid_market["_timestamp"].dtype) != expected_dtype:
        raise RuntimeError(
            "Market timestamp dtype normalization failed: "
            f"{valid_market['_timestamp'].dtype}"
        )

    # -------------------------------------------------------------------------
    # MERGE ASOF
    # -------------------------------------------------------------------------

    print("\nMapping events to realized_vol_30...")

    mapped = pd.merge_asof(
        events,
        valid_market,
        left_on="timestamp",
        right_on="_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    mapped_count = int(mapped["realized_vol_30"].notna().sum())

    print(f"Events mapped to realized_vol_30: {mapped_count:,}/{len(mapped):,}")

    if mapped_count == 0:
        raise RuntimeError("No Research 07 events could be mapped.")

    # -------------------------------------------------------------------------
    # CAUSAL PERCENTILE
    # -------------------------------------------------------------------------

    print("\nBuilding causal volatility percentile...")

    values = mapped["realized_vol_30"].to_numpy(dtype=np.float64)

    percentiles = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    history: list[float] = []

    progress_step = 100_000

    for i, value in enumerate(values):
        if np.isfinite(value):
            if history:
                rank = bisect.bisect_right(
                    history,
                    value,
                )

                percentiles[i] = rank / len(history) * 100.0

            bisect.insort(
                history,
                float(value),
            )

        if (i + 1) % progress_step == 0 or i + 1 == len(values):
            print(f"  Percentile progress: {i + 1:,}/{len(values):,}")

    mapped["vol_percentile"] = percentiles

    # -------------------------------------------------------------------------
    # BUCKETS
    # -------------------------------------------------------------------------

    mapped["vol_bucket"] = np.select(
        [
            mapped["vol_percentile"] < 20,
            mapped["vol_percentile"] < 40,
            mapped["vol_percentile"] < 60,
            mapped["vol_percentile"] < 80,
            mapped["vol_percentile"] >= 80,
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

    mapped.loc[
        mapped["vol_percentile"].isna(),
        "vol_bucket",
    ] = np.nan

    # -------------------------------------------------------------------------
    # DISTRIBUTION
    # -------------------------------------------------------------------------

    print("\nVOLATILITY BUCKET DISTRIBUTION")

    distribution = (
        mapped[mapped["vol_bucket"].notna()]
        .groupby(
            "vol_bucket",
            observed=True,
        )
        .agg(
            observations=(
                "event_id",
                "count",
            ),
            mean_realized_vol_30=(
                "realized_vol_30",
                "mean",
            ),
            median_realized_vol_30=(
                "realized_vol_30",
                "median",
            ),
            mean_percentile=(
                "vol_percentile",
                "mean",
            ),
        )
        .reset_index()
    )

    print(distribution.to_string(index=False))

    return mapped[
        [
            "event_id",
            "realized_vol_30",
            "vol_percentile",
            "vol_bucket",
        ]
    ]


# =============================================================================
# BUILD EVENT CONTEXT
# =============================================================================


def build_events() -> pd.DataFrame:

    metadata = load_metadata()

    hmm = load_hmm()

    volatility = load_volatility()

    print("\n" + "=" * 100)
    print("BUILDING COMPLETE EVENT CONTEXT")
    print("=" * 100)

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

    missing_hmm = int(events["hmm_state"].isna().sum())

    missing_vol = int(events["vol_bucket"].isna().sum())

    missing_z = int(events["zscore_30"].isna().sum())

    print(f"Events: {len(events):,}")

    print(f"Missing HMM: {missing_hmm:,}")

    print(f"Missing volatility: {missing_vol:,}")

    print(f"Missing z-score: {missing_z:,}")

    # -------------------------------------------------------------------------
    # HMM / Z-SCORE ARE REQUIRED
    # -------------------------------------------------------------------------

    if missing_hmm > 0:
        raise RuntimeError(f"Missing HMM states: {missing_hmm}")

    if missing_z > 0:
        raise RuntimeError(f"Missing z-score values: {missing_z}")

    # -------------------------------------------------------------------------
    # VOLATILITY
    #
    # A very small number of missing causal volatility observations is expected
    # at the beginning of the sample because percentile requires historical
    # observations.
    #
    # We DO NOT fabricate a bucket.
    # We simply remove those events from the audit.
    # -------------------------------------------------------------------------

    if missing_vol > 0:
        missing_pct = missing_vol / len(events) * 100.0

        print("\nWARNING:")

        print(
            f"{missing_vol:,} events "
            f"({missing_pct:.6f}%) "
            "have no causal volatility bucket."
        )

        print("These events will be EXCLUDED from the audit.")

        print("No volatility value or bucket will be fabricated.")

    # -------------------------------------------------------------------------
    # KEEP ONLY COMPLETE CONTEXT
    # -------------------------------------------------------------------------

    complete_mask = (
        events["hmm_state"].notna()
        & events["vol_bucket"].notna()
        & events["zscore_30"].notna()
    )

    events = events.loc[complete_mask].copy().reset_index(drop=True)

    print(f"\nComplete context events: {len(events):,}")

    print(f"Excluded events: {metadata.shape[0] - len(events):,}")

    # -------------------------------------------------------------------------
    # SANITY CHECK
    # -------------------------------------------------------------------------

    if events.empty:
        raise RuntimeError("No complete context events remain.")

    if events["vol_bucket"].isna().any():
        raise RuntimeError(
            "Volatility bucket still contains NaN after complete-context filtering."
        )

    # -------------------------------------------------------------------------
    # DISTRIBUTION
    # -------------------------------------------------------------------------

    print("\nFINAL VOLATILITY CONTEXT DISTRIBUTION")

    distribution = (
        events.groupby(
            "vol_bucket",
            observed=True,
        )
        .agg(
            observations=(
                "event_id",
                "count",
            ),
            mean_realized_vol_30=(
                "realized_vol_30",
                "mean",
            ),
            median_realized_vol_30=(
                "realized_vol_30",
                "median",
            ),
            mean_percentile=(
                "vol_percentile",
                "mean",
            ),
        )
        .reset_index()
    )

    print(distribution.to_string(index=False))

    print("\nContext coverage:")

    print(f"{len(events):,}/{len(metadata):,} usable events")

    return events


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths() -> np.ndarray:

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 07 PATH CACHE")
    print("=" * 100)

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

    missing = [key for key in required if key not in cache.files]

    if missing:
        raise RuntimeError(f"Missing cache arrays: {missing}")

    for key in required:
        print(f"{key}: {cache[key].shape} {cache[key].dtype}")

    future_close = cache["future_close"]

    if future_close.shape != (
        825717,
        120,
    ):
        raise RuntimeError(f"Unexpected future_close shape: {future_close.shape}")

    print("Research 07 path cache integrity: OK")

    return future_close


# =============================================================================
# RAW PATH RECONSTRUCTION
# =============================================================================


def reconstruct_trade(
    event_id: int,
    window: int,
    timestamp,
    entry: float,
    zscore: float,
    future_close: np.ndarray,
    tp: float,
    sl: float,
    horizon: int,
    side: str,
    candidate: str,
):

    path = future_close[
        event_id,
        :horizon,
    ]

    if side == "LONG":
        tp_level = entry + tp

        sl_level = entry - sl

        tp_hits = path >= tp_level

        sl_hits = path <= sl_level

        favorable = path - entry

        adverse = entry - path

    else:
        tp_level = entry - tp

        sl_level = entry + sl

        tp_hits = path <= tp_level

        sl_hits = path >= sl_level

        favorable = entry - path

        adverse = path - entry

    tp_indices = np.flatnonzero(tp_hits)

    sl_indices = np.flatnonzero(sl_hits)

    first_tp_bar = int(tp_indices[0]) + 1 if len(tp_indices) else np.nan

    first_sl_bar = int(sl_indices[0]) + 1 if len(sl_indices) else np.nan

    if len(tp_indices) and len(sl_indices):
        if first_tp_bar < first_sl_bar:
            outcome = "WIN"
            resolution = "TP_FIRST"

        elif first_sl_bar < first_tp_bar:
            outcome = "LOSS"
            resolution = "SL_FIRST"

        else:
            outcome = "CONFLICT"
            resolution = "SAME_BAR"

    elif len(tp_indices):
        outcome = "WIN"
        resolution = "TP_ONLY"

    elif len(sl_indices):
        outcome = "LOSS"
        resolution = "SL_ONLY"

    else:
        outcome = "UNRESOLVED"
        resolution = "NEITHER"

    return {
        "candidate": candidate,
        "event_id": int(event_id),
        "window": int(window),
        "timestamp": timestamp,
        "side": side,
        "entry": float(entry),
        "zscore": float(zscore),
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(tp / sl),
        "horizon": int(horizon),
        "tp_level": float(tp_level),
        "sl_level": float(sl_level),
        "first_tp_bar": first_tp_bar,
        "first_sl_bar": first_sl_bar,
        "tp_hit": bool(len(tp_indices)),
        "sl_hit": bool(len(sl_indices)),
        "outcome": outcome,
        "resolution": resolution,
        "max_favorable_raw": float(np.max(favorable)),
        "max_adverse_raw": float(np.max(adverse)),
    }


# =============================================================================
# AUDIT CANDIDATE
# =============================================================================


def audit_candidate(
    events: pd.DataFrame,
    future_close: np.ndarray,
    candidate: dict,
):

    print("\n" + "-" * 100)

    print(f"AUDITING {candidate['candidate']}")

    print(
        f"{candidate['side']} | "
        f"HMM={candidate['hmm_state']} | "
        f"VOL={candidate['vol_bucket']} | "
        f"Z={candidate['zscore']} | "
        f"TP={candidate['tp']} | "
        f"SL={candidate['sl']} | "
        f"H={candidate['horizon']}"
    )

    mask = events["hmm_state"] == candidate["hmm_state"]

    mask &= events["vol_bucket"] == candidate["vol_bucket"]

    #
    # Mean-reversion direction:
    #
    # LONG  = negative z-score
    # SHORT = positive z-score
    #

    if candidate["side"] == "LONG":
        mask &= events["zscore_30"] <= -abs(candidate["zscore"])

    else:
        mask &= events["zscore_30"] >= abs(candidate["zscore"])

    selected = events.loc[mask]

    print(f"Selected events: {len(selected):,}")

    rows = []

    for row in selected.itertuples(index=False):
        rows.append(
            reconstruct_trade(
                event_id=row.event_id,
                window=row.window,
                timestamp=row.timestamp,
                entry=row.close,
                zscore=row.zscore_30,
                future_close=future_close,
                tp=candidate["tp"],
                sl=candidate["sl"],
                horizon=candidate["horizon"],
                side=candidate["side"],
                candidate=candidate["candidate"],
            )
        )

    trades = pd.DataFrame(rows)

    if trades.empty:
        return trades, None

    total = len(trades)

    wins = int((trades["outcome"] == "WIN").sum())

    losses = int((trades["outcome"] == "LOSS").sum())

    conflicts = int((trades["outcome"] == "CONFLICT").sum())

    unresolved = int((trades["outcome"] == "UNRESOLVED").sum())

    resolved = wins + losses

    wr_resolved = wins / resolved if resolved else np.nan

    wr_all = wins / total if total else np.nan

    rr = candidate["tp"] / candidate["sl"]

    expectancy = (
        wr_resolved * rr - (1 - wr_resolved) if np.isfinite(wr_resolved) else np.nan
    )

    summary = {
        "candidate": candidate["candidate"],
        "side": candidate["side"],
        "hmm_state": candidate["hmm_state"],
        "vol_bucket": candidate["vol_bucket"],
        "zscore": candidate["zscore"],
        "tp": candidate["tp"],
        "sl": candidate["sl"],
        "rr": rr,
        "horizon": candidate["horizon"],
        "observations": total,
        "wins": wins,
        "losses": losses,
        "conflicts": conflicts,
        "unresolved": unresolved,
        "resolved": resolved,
        "win_rate_resolved": wr_resolved,
        "win_rate_all": wr_all,
        "expectancy_resolved_r": expectancy,
        "tp_hit_rate": trades["tp_hit"].mean(),
        "sl_hit_rate": trades["sl_hit"].mean(),
    }

    print("\nRAW FUTURE_CLOSE RESULTS")

    print(f"Observations: {total:,}")

    print(f"Wins:        {wins:,}")

    print(f"Losses:      {losses:,}")

    print(f"Conflicts:   {conflicts:,}")

    print(f"Unresolved:  {unresolved:,}")

    print(f"Resolved:    {resolved:,}")

    print(f"WR resolved: {wr_resolved:.4%}")

    print(f"WR all:      {wr_all:.4%}")

    print(f"RR:          {rr:.4f}")

    print(f"Expectancy:  {expectancy:.6f}R")

    return trades, summary


# =============================================================================
# WINDOW SUMMARY
# =============================================================================


def build_window_summary(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    if trades.empty:
        return pd.DataFrame()

    rows = []

    for (
        candidate,
        window,
    ), group in trades.groupby(
        [
            "candidate",
            "window",
        ]
    ):
        wins = int((group["outcome"] == "WIN").sum())

        losses = int((group["outcome"] == "LOSS").sum())

        conflicts = int((group["outcome"] == "CONFLICT").sum())

        unresolved = int((group["outcome"] == "UNRESOLVED").sum())

        resolved = wins + losses

        wr = wins / resolved if resolved else np.nan

        rows.append(
            {
                "candidate": candidate,
                "window": int(window),
                "observations": len(group),
                "wins": wins,
                "losses": losses,
                "conflicts": conflicts,
                "unresolved": unresolved,
                "resolved": resolved,
                "win_rate_resolved": wr,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("\n" + "=" * 100)
    print("MEAN REVERSION — RESEARCH 08L")
    print("=" * 100)

    print("RAW FUTURE_CLOSE PATH RECONSTRUCTION")

    print("-" * 100)

    print("Ground-truth audit of TP/SL ordering.")

    print("No optimization.")

    print("No HMM retraining.")

    print("No volatility optimization.")

    print("No failure test.")

    print("No production changes.")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    events = build_events()

    future_close = load_paths()

    # -------------------------------------------------------------------------
    # AUDIT
    # -------------------------------------------------------------------------

    all_trades = []

    summaries = []

    for candidate in CANDIDATES:
        trades, summary = audit_candidate(
            events,
            future_close,
            candidate,
        )

        if not trades.empty:
            all_trades.append(trades)

        if summary is not None:
            summaries.append(summary)

    if not all_trades:
        raise RuntimeError("No trade results generated.")

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    summary = pd.DataFrame(summaries)

    windows = build_window_summary(trades)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    # -------------------------------------------------------------------------
    # COMPARE 08K
    # -------------------------------------------------------------------------

    comparison_rows = []

    audit_08k = RESULTS_DIR / "research_08k_candidate_audit.csv"

    if audit_08k.exists():
        old = pd.read_csv(audit_08k)

        for row in summary.itertuples(index=False):
            old_rows = old[old["candidate"] == row.candidate]

            if old_rows.empty:
                continue

            old_row = old_rows.iloc[0]

            comparison_rows.append(
                {
                    "candidate": row.candidate,
                    "08K_observations": old_row["observations"],
                    "08K_wins": old_row["wins"],
                    "08K_losses": old_row["losses"],
                    "08K_wr_resolved": old_row["win_rate_resolved"],
                    "08L_observations": row.observations,
                    "08L_wins": row.wins,
                    "08L_losses": row.losses,
                    "08L_wr_resolved": row.win_rate_resolved,
                    "wr_difference": (
                        row.win_rate_resolved - old_row["win_rate_resolved"]
                    ),
                }
            )

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("08L FINAL SUMMARY")
    print("=" * 100)

    print(summary.to_string(index=False))

    if not comparison.empty:
        print("\n" + "=" * 100)
        print("08K vs 08L COMPARISON")
        print("=" * 100)

        print(comparison.to_string(index=False))

    print("\n" + "=" * 100)
    print("RESEARCH 08L COMPLETE")
    print("=" * 100)

    print("Raw future_close reconstruction completed.")

    print("\nSaved:")

    print(OUTPUT_TRADES)

    print(OUTPUT_SUMMARY)

    print(OUTPUT_WINDOWS)

    print(OUTPUT_COMPARISON)

    print("\nIMPORTANT:")

    print("This is an audit, not a strategy optimization.")


if __name__ == "__main__":
    main()
