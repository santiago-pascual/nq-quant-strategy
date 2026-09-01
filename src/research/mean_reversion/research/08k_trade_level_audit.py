from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08K — TRADE LEVEL AUDIT
# =============================================================================
#
# PURPOSE
#
# Independently audit the strongest Research 08J candidates at trade level.
#
# This research does NOT:
#
#   - optimize parameters
#   - retrain HMM
#   - redefine volatility
#   - search new contexts
#   - perform failure testing
#   - modify production code
#
# The purpose is to verify that the apparent 4:1 RR edge is mechanically real.
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"


METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

REFINEMENT_PATH = RESULTS_DIR / "research_08i_local_refinement.csv"

ROBUST_PATH = RESULTS_DIR / "research_08j_robust_candidates.csv"

OUTPUT_TRADES = RESULTS_DIR / "research_08k_trade_level_audit.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08k_candidate_audit.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08k_window_audit.csv"

OUTPUT_CONFLICTS = RESULTS_DIR / "research_08k_tp_sl_conflicts.csv"

OUTPUT_UNRESOLVED = RESULTS_DIR / "research_08k_unresolved_trades.csv"


# =============================================================================
# CANDIDATES
# =============================================================================
#
# These are the configurations that survived Research 08J.
#
# We explicitly audit all four.
#
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
# LOAD METADATA
# =============================================================================


def load_metadata():

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
        raise RuntimeError(f"Metadata missing columns: {missing}")

    df["event_id"] = pd.to_numeric(df["event_id"]).astype(np.int64)

    df["window"] = pd.to_numeric(df["window"]).astype(np.int64)

    df["zscore_30"] = pd.to_numeric(
        df["zscore_30"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm():

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
        raise RuntimeError(f"HMM missing columns: {missing}")

    df["event_id"] = pd.to_numeric(df["event_id"]).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(df["hmm_state"]).astype(int)

    return df[
        [
            "event_id",
            "hmm_state",
        ]
    ]


# =============================================================================
# LOAD VOLATILITY
# =============================================================================


def load_volatility():

    print("\n" + "=" * 100)
    print("LOADING VOLATILITY CONTEXT")
    print("=" * 100)

    #
    # Prefer an event-level file if 08H created one.
    #

    candidates = [
        RESULTS_DIR / "research_08h_event_context.csv",
        RESULTS_DIR / "research_08h_event_context.parquet",
        RESULTS_DIR / "research_08h_context_events.csv",
        CACHE_DIR / "research_08h_event_context.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        print(f"Found cached volatility context:\n{path}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)

        else:
            df = pd.read_csv(path)

        if "event_id" in df.columns and "vol_bucket" in df.columns:
            df["event_id"] = pd.to_numeric(df["event_id"]).astype(np.int64)

            return df[
                [
                    "event_id",
                    "vol_bucket",
                ]
            ]

    #
    # Fallback:
    #
    # Rebuild the SAME causal volatility percentile used previously.
    #
    # This is reconstruction, not optimization.
    #

    print("No cached event volatility context found.")

    print("Rebuilding causal realized_vol_30 percentile.")

    from src.databento_loader import (
        load_databento_mnq,
    )

    from src.feature_engine import (
        add_return_features,
        add_volatility_features,
    )

    market = load_databento_mnq()

    market = add_return_features(market)

    market = add_volatility_features(market)

    timestamp_column = None

    for column in [
        "timestamp",
        "timestamp ET",
        "ts_event",
    ]:
        if column in market.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise RuntimeError("No market timestamp column found.")

    market["_timestamp"] = pd.to_datetime(
        market[timestamp_column],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    market["realized_vol_30"] = pd.to_numeric(
        market["realized_vol_30"],
        errors="coerce",
    )

    market = market[
        [
            "_timestamp",
            "realized_vol_30",
        ]
    ].dropna()

    market = (
        market.sort_values("_timestamp")
        .drop_duplicates(
            "_timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    metadata = pd.read_csv(METADATA_PATH)

    metadata["event_id"] = pd.to_numeric(metadata["event_id"]).astype(np.int64)

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

    values = mapped["realized_vol_30"].to_numpy(dtype=np.float64)

    from bisect import (
        bisect_right,
        insort,
    )

    history = []

    percentile = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        position = bisect_right(
            history,
            value,
        )

        percentile[i] = position / (len(history) + 1) * 100.0

        insort(
            history,
            value,
        )

    mapped["vol_percentile"] = percentile

    mapped["vol_bucket"] = pd.cut(
        mapped["vol_percentile"],
        bins=[
            -np.inf,
            20,
            40,
            60,
            80,
            np.inf,
        ],
        labels=[
            "0-20",
            "20-40",
            "40-60",
            "60-80",
            "80-100",
        ],
        right=False,
    ).astype(str)

    return mapped[
        [
            "event_id",
            "vol_bucket",
        ]
    ]


# =============================================================================
# BUILD EVENT TABLE
# =============================================================================


def build_events():

    metadata = load_metadata()

    hmm = load_hmm()

    volatility = load_volatility()

    print("\n" + "=" * 100)
    print("BUILDING EVENT TABLE")
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

    print(f"Events: {len(events):,}")

    print(
        "Missing HMM:",
        events["hmm_state"].isna().sum(),
    )

    print(
        "Missing volatility:",
        events["vol_bucket"].isna().sum(),
    )

    if events["hmm_state"].isna().any():
        raise RuntimeError("Missing HMM states.")

    if events["vol_bucket"].isna().any():
        raise RuntimeError("Missing volatility buckets.")

    return events


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths():

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

    for key in required:
        if key not in cache.files:
            raise RuntimeError(f"Missing cache key: {key}")

        print(f"{key}: {cache[key].shape}")

    return {key: cache[key] for key in required}


# =============================================================================
# EXACT TRADE RESOLUTION
# =============================================================================
#
# This is the core of the audit.
#
# We record:
#
#   - first TP hit
#   - first SL hit
#   - whether both occur
#   - which happened first
#   - terminal excursion
#   - final classification
#
# IMPORTANT:
#
# A trade is NOT automatically a win merely because TP was reached.
#
# If SL occurred earlier, it is a loss.
#
# =============================================================================


def resolve_trades(
    event_ids,
    windows,
    timestamps,
    zscores,
    favorable,
    adverse,
    tp,
    sl,
    horizon,
    side,
    candidate_name,
):

    n = len(event_ids)

    rows = []

    for i in range(n):
        fav = favorable[
            i,
            :horizon,
        ]

        adv = adverse[
            i,
            :horizon,
        ]

        tp_hits = fav >= tp

        sl_hits = adv >= sl

        tp_indices = np.flatnonzero(tp_hits)

        sl_indices = np.flatnonzero(sl_hits)

        tp_hit = len(tp_indices) > 0

        sl_hit = len(sl_indices) > 0

        if tp_hit:
            first_tp = int(tp_indices[0]) + 1

        else:
            first_tp = np.nan

        if sl_hit:
            first_sl = int(sl_indices[0]) + 1

        else:
            first_sl = np.nan

        #
        # Exact classification.
        #

        if tp_hit and sl_hit:
            if first_tp < first_sl:
                outcome = "WIN"
                resolution = "TP_FIRST"

            elif first_sl < first_tp:
                outcome = "LOSS"
                resolution = "SL_FIRST"

            else:
                #
                # Same horizon/bar.
                #
                # We explicitly flag this.
                #
                outcome = "CONFLICT"
                resolution = "TP_SL_SAME_BAR"

        elif tp_hit:
            outcome = "WIN"
            resolution = "TP_ONLY"

        elif sl_hit:
            outcome = "LOSS"
            resolution = "SL_ONLY"

        else:
            #
            # Neither target was reached.
            #
            # We do NOT silently convert this into a win.
            #
            outcome = "UNRESOLVED"
            resolution = "NEITHER"

        terminal_fav = float(fav[horizon - 1])

        terminal_adv = float(adv[horizon - 1])

        rows.append(
            {
                "candidate": candidate_name,
                "event_id": int(event_ids[i]),
                "window": int(windows[i]),
                "timestamp": timestamps[i],
                "side": side,
                "zscore": float(zscores[i]),
                "tp": float(tp),
                "sl": float(sl),
                "rr": float(tp / sl),
                "horizon": int(horizon),
                "first_tp_bar": first_tp,
                "first_sl_bar": first_sl,
                "tp_hit": bool(tp_hit),
                "sl_hit": bool(sl_hit),
                "outcome": outcome,
                "resolution": resolution,
                "terminal_favorable": terminal_fav,
                "terminal_adverse": terminal_adv,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# AUDIT CANDIDATE
# =============================================================================


def audit_candidate(
    events,
    paths,
    candidate,
):

    print("\n" + "-" * 100)

    print(f"AUDITING {candidate['candidate']}")

    print(
        f"{candidate['side']} | "
        f"HMM={candidate['hmm_state']} | "
        f"VOL={candidate['vol_bucket']} | "
        f"Z>={candidate['zscore']} | "
        f"TP={candidate['tp']} | "
        f"SL={candidate['sl']} | "
        f"H={candidate['horizon']}"
    )

    mask = events["hmm_state"] == candidate["hmm_state"]

    mask &= events["vol_bucket"] == candidate["vol_bucket"]

    if candidate["side"] == "LONG":
        mask &= events["zscore_30"] <= -abs(candidate["zscore"])

        favorable_key = "long_favorable"

        adverse_key = "long_adverse"

    else:
        mask &= events["zscore_30"] >= abs(candidate["zscore"])

        favorable_key = "short_favorable"

        adverse_key = "short_adverse"

    selected = events.loc[mask].copy()

    print(f"Selected events: {len(selected):,}")

    if selected.empty:
        return (
            pd.DataFrame(),
            None,
        )

    ids = selected["event_id"].to_numpy(dtype=np.int64)

    windows = selected["window"].to_numpy(dtype=np.int64)

    timestamps = selected["timestamp"].to_numpy()

    zscores = selected["zscore_30"].to_numpy(dtype=np.float64)

    favorable = paths[favorable_key][ids]

    adverse = paths[adverse_key][ids]

    trades = resolve_trades(
        ids,
        windows,
        timestamps,
        zscores,
        favorable,
        adverse,
        candidate["tp"],
        candidate["sl"],
        candidate["horizon"],
        candidate["side"],
        candidate["candidate"],
    )

    #
    # Summary.
    #

    total = len(trades)

    wins = int((trades["outcome"] == "WIN").sum())

    losses = int((trades["outcome"] == "LOSS").sum())

    conflicts = int((trades["outcome"] == "CONFLICT").sum())

    unresolved = int((trades["outcome"] == "UNRESOLVED").sum())

    resolved = wins + losses

    wr_resolved = wins / resolved if resolved else np.nan

    wr_all = wins / total if total else np.nan

    rr = candidate["tp"] / candidate["sl"]

    expectancy_resolved = (
        wr_resolved * rr - (1.0 - wr_resolved) if np.isfinite(wr_resolved) else np.nan
    )

    expectancy_all = wr_all * rr - (1.0 - wr_all) if np.isfinite(wr_all) else np.nan

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
        "expectancy_resolved": expectancy_resolved,
        "expectancy_all": expectancy_all,
        "tp_hit_rate": (trades["tp_hit"].mean()),
        "sl_hit_rate": (trades["sl_hit"].mean()),
    }

    print("\nTRADE-LEVEL RESULTS")

    print(f"Observations: {total:,}")

    print(f"Wins:        {wins:,}")

    print(f"Losses:      {losses:,}")

    print(f"Conflicts:   {conflicts:,}")

    print(f"Unresolved:  {unresolved:,}")

    print(f"Resolved:    {resolved:,}")

    print(f"WR resolved: {wr_resolved:.4%}")

    print(f"WR all:      {wr_all:.4%}")

    print(f"RR:          {rr:.4f}")

    print(f"Expectancy resolved: {expectancy_resolved:.6f}R")

    print(f"Expectancy all:      {expectancy_all:.6f}R")

    return (
        trades,
        summary,
    )


# =============================================================================
# WINDOW SUMMARY
# =============================================================================


def build_window_summary(trades):

    rows = []

    if trades.empty:
        return pd.DataFrame()

    for (
        candidate,
        window,
    ), group in trades.groupby(
        [
            "candidate",
            "window",
        ],
        sort=True,
    ):
        total = len(group)

        wins = int((group["outcome"] == "WIN").sum())

        losses = int((group["outcome"] == "LOSS").sum())

        conflicts = int((group["outcome"] == "CONFLICT").sum())

        unresolved = int((group["outcome"] == "UNRESOLVED").sum())

        resolved = wins + losses

        wr = wins / resolved if resolved else np.nan

        rr = float(group["rr"].iloc[0])

        expectancy = wr * rr - (1.0 - wr) if np.isfinite(wr) else np.nan

        rows.append(
            {
                "candidate": candidate,
                "window": int(window),
                "observations": total,
                "wins": wins,
                "losses": losses,
                "conflicts": conflicts,
                "unresolved": unresolved,
                "resolved": resolved,
                "win_rate_resolved": wr,
                "expectancy_r": expectancy,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CONFLICT AUDIT
# =============================================================================


def build_conflict_report(trades):

    if trades.empty:
        return pd.DataFrame()

    conflicts = trades[trades["outcome"] == "CONFLICT"].copy()

    return conflicts[
        [
            "candidate",
            "event_id",
            "window",
            "timestamp",
            "side",
            "zscore",
            "tp",
            "sl",
            "horizon",
            "first_tp_bar",
            "first_sl_bar",
            "resolution",
        ]
    ]


# =============================================================================
# UNRESOLVED REPORT
# =============================================================================


def build_unresolved_report(trades):

    if trades.empty:
        return pd.DataFrame()

    unresolved = trades[trades["outcome"] == "UNRESOLVED"].copy()

    return unresolved[
        [
            "candidate",
            "event_id",
            "window",
            "timestamp",
            "side",
            "zscore",
            "tp",
            "sl",
            "horizon",
            "terminal_favorable",
            "terminal_adverse",
        ]
    ]


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("\n" + "=" * 100)

    print("MEAN REVERSION — RESEARCH 08K")

    print("=" * 100)

    print("TRADE-LEVEL EDGE AUDIT")

    print("-" * 100)

    print("Objective: verify the apparent 4:1 RR edge.")

    print("No optimization.")

    print("No HMM retraining.")

    print("No volatility optimization.")

    print("No failure test.")

    print("No production changes.")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    events = build_events()

    paths = load_paths()

    # -------------------------------------------------------------------------
    # AUDIT
    # -------------------------------------------------------------------------

    all_trades = []

    summaries = []

    for candidate in CANDIDATES:
        trades, summary = audit_candidate(
            events,
            paths,
            candidate,
        )

        if not trades.empty:
            all_trades.append(trades)

        if summary is not None:
            summaries.append(summary)

    if not all_trades:
        raise RuntimeError("No candidate trades generated.")

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    summary = pd.DataFrame(summaries)

    windows = build_window_summary(trades)

    conflicts = build_conflict_report(trades)

    unresolved = build_unresolved_report(trades)

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

    conflicts.to_csv(
        OUTPUT_CONFLICTS,
        index=False,
    )

    unresolved.to_csv(
        OUTPUT_UNRESOLVED,
        index=False,
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)

    print("FINAL AUDIT SUMMARY")

    print("=" * 100)

    print(summary.to_string(index=False))

    print("\n" + "=" * 100)

    print("CONFLICT CHECK")

    print("=" * 100)

    print(f"TP/SL same-bar conflicts: {len(conflicts):,}")

    if not conflicts.empty:
        print(conflicts.head(20).to_string(index=False))

    print("\n" + "=" * 100)

    print("UNRESOLVED CHECK")

    print("=" * 100)

    print(f"Unresolved trades: {len(unresolved):,}")

    print("\n" + "=" * 100)

    print("WINDOW AUDIT")

    print("=" * 100)

    print(windows.to_string(index=False))

    print("\n" + "=" * 100)

    print("RESEARCH 08K COMPLETE")

    print("=" * 100)

    print("Files saved:")

    print(OUTPUT_TRADES)

    print(OUTPUT_SUMMARY)

    print(OUTPUT_WINDOWS)

    print(OUTPUT_CONFLICTS)

    print(OUTPUT_UNRESOLVED)

    print("\nIMPORTANT:")

    print("This audit does NOT declare the strategy valid.")

    print(
        "The next decision depends on whether the "
        "trade-level mechanics reproduce the Research 08J edge."
    )


if __name__ == "__main__":
    main()
