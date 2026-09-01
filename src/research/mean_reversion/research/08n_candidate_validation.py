from __future__ import annotations

from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08N
# =============================================================================
# CANDIDATE VALIDATION / ANTI-OVERFITTING AUDIT
#
# Official strategy names:
#
#   MRS2 = Mean Reversion Short, HMM State 2
#   MRL1 = Mean Reversion Long,  HMM State 1
#   MRL2 = Mean Reversion Long,  HMM State 2
#
# Historical candidate IDs:
#   C01 -> MRS2
#   C02 -> MRL1
#   C06 -> MRL2
#
# The original C01/C02/C06 IDs are preserved internally for traceability.
#
# No HMM retraining.
# No volatility recalculation methodology change.
# No parameter optimization.
# No failure test.
# No production changes.
#
# This script validates the frozen candidates through:
#
#   1. Temporal window stability
#   2. Overall WR
#   3. Expectancy
#   4. Profit factor
#   5. Local TP / SL / horizon stability
#
# WR >= 50% is NOT a mandatory requirement.
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"


METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

PATH_CACHE_PATH = CACHE_DIR / "research_07_future_path_cache.npz"


OUTPUT_WINDOW = RESULTS_DIR / "research_08n_candidate_window_validation.csv"

OUTPUT_NEIGHBORHOOD = RESULTS_DIR / "research_08n_parameter_neighborhood.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08n_candidate_summary.csv"


# =============================================================================
# OFFICIAL CANDIDATE DEFINITIONS
# =============================================================================

CANDIDATES = [
    {
        "candidate_id": "C01",
        "strategy_name": "MRS2",
        "strategy_label": "Mean Reversion Short — HMM 2",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 5,
    },
    {
        "candidate_id": "C02",
        "strategy_name": "MRL1",
        "strategy_label": "Mean Reversion Long — HMM 1",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 20,
    },
    {
        "candidate_id": "C06",
        "strategy_name": "MRL2",
        "strategy_label": "Mean Reversion Long — HMM 2",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "rr": 2.5,
        "horizon": 2,
    },
]


# =============================================================================
# LOCAL PARAMETER NEIGHBOURHOOD
# =============================================================================
#
# These are NOT strategy optimization.
#
# They are used only to determine whether the candidate sits inside
# a robust positive region instead of being an isolated peak.
# =============================================================================

TP_DELTAS = [
    -1.0,
    0.0,
    1.0,
]

SL_DELTAS = [
    -0.5,
    0.0,
    0.5,
]

H_DELTAS = [
    -2,
    -1,
    0,
    1,
    2,
]


# =============================================================================
# PRINT HELPERS
# =============================================================================


def section(title: str):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# TIMESTAMP NORMALIZATION
# =============================================================================


def normalize_timestamp(
    series: pd.Series,
) -> pd.Series:

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")


# =============================================================================
# LOAD RESEARCH 07 METADATA
# =============================================================================


def load_metadata():

    section("LOADING RESEARCH 07 METADATA")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file:\n{METADATA_PATH}")

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
    ).astype(np.int16)

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df["zscore_30"] = pd.to_numeric(
        df["zscore_30"],
        errors="coerce",
    )

    df["timestamp"] = normalize_timestamp(df["timestamp"])

    df = df.sort_values("event_id").reset_index(drop=True)

    print(f"Events: {len(df):,}")

    print(
        "Timestamp dtype:",
        df["timestamp"].dtype,
    )

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm():

    section("LOADING RESEARCH 08B HMM")

    if not HMM_PATH.exists():
        raise FileNotFoundError(f"Missing HMM cache:\n{HMM_PATH}")

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

    print("\nHMM distribution:")

    print(df["hmm_state"].value_counts().sort_index().to_string())

    return df[
        [
            "event_id",
            "hmm_state",
        ]
    ]


# =============================================================================
# BUILD CAUSAL VOLATILITY CONTEXT
# =============================================================================


def build_volatility_context(
    metadata: pd.DataFrame,
):

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

    print("Building return features...")

    market = add_return_features(market)

    print("Building volatility features...")

    market = add_volatility_features(market)

    market["_timestamp"] = normalize_timestamp(market["timestamp ET"])

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

    event_times = metadata[
        [
            "event_id",
            "timestamp",
        ]
    ].copy()

    event_times["timestamp"] = normalize_timestamp(event_times["timestamp"])

    event_times = event_times.sort_values("timestamp").reset_index(drop=True)

    print("\nMerge timestamp dtypes:")

    print(
        "Events :",
        event_times["timestamp"].dtype,
    )

    print(
        "Market :",
        market["_timestamp"].dtype,
    )

    mapped = pd.merge_asof(
        event_times,
        market,
        left_on="timestamp",
        right_on="_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    mapped = mapped.sort_values("event_id").reset_index(drop=True)

    mapped_count = int(mapped["realized_vol_30"].notna().sum())

    print(f"\nEvents mapped: {mapped_count:,}/{len(mapped):,}")

    if mapped_count == 0:
        raise RuntimeError("No events mapped to realized_vol_30.")

    print("\nBuilding causal volatility percentile...")

    values = mapped["realized_vol_30"].to_numpy(dtype=np.float64)

    percentile = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    history = []

    valid_indices = np.flatnonzero(np.isfinite(values))

    total_valid = len(valid_indices)

    for counter, idx in enumerate(valid_indices):
        value = float(values[idx])

        if history:
            pos = bisect_right(
                history,
                value,
            )

            percentile[idx] = pos / len(history) * 100.0

            history.insert(
                pos,
                value,
            )

        else:
            history.append(value)

        if (counter + 1) % 100_000 == 0 or counter + 1 == total_valid:
            print(f"  Percentile: {counter + 1:,}/{total_valid:,}")

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

    print("\nVolatility distribution:")

    print(mapped["vol_bucket"].value_counts().sort_index().to_string())

    return mapped[
        [
            "event_id",
            "realized_vol_30",
            "vol_percentile",
            "vol_bucket",
        ]
    ]


# =============================================================================
# BUILD COMPLETE EVENT CONTEXT
# =============================================================================


def build_events():

    section("BUILDING COMPLETE EVENT CONTEXT")

    metadata = load_metadata()

    hmm = load_hmm()

    volatility = build_volatility_context(metadata)

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

    print(f"\nEvents: {len(events):,}")

    print(
        "Missing HMM:",
        int(events["hmm_state"].isna().sum()),
    )

    print(
        "Missing volatility:",
        int(events["vol_bucket"].isna().sum()),
    )

    print(
        "Missing z-score:",
        int(events["zscore_30"].isna().sum()),
    )

    if events["hmm_state"].isna().any():
        raise RuntimeError("Missing HMM states.")

    if events["zscore_30"].isna().any():
        raise RuntimeError("Missing z-score values.")

    unknown_mask = events["vol_bucket"] == "UNKNOWN"

    unknown_count = int(unknown_mask.sum())

    if unknown_count > 0:
        print("\nUndefined causal volatility percentile:")

        print(f"Removing {unknown_count:,} event(s).")

        print(
            "Reason: no prior volatility history existed "
            "for the first valid observation."
        )

        events = events.loc[~unknown_mask].copy()

        print(f"Events remaining: {len(events):,}")

    if events["vol_bucket"].isna().any():
        raise RuntimeError("Missing volatility buckets remain.")

    if (events["vol_bucket"] == "UNKNOWN").any():
        raise RuntimeError("UNKNOWN volatility bucket remains.")

    print("\nVolatility context integrity: OK")

    return events


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths():

    section("LOADING RESEARCH 07 PATH CACHE")

    if not PATH_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing path cache:\n{PATH_CACHE_PATH}")

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

    print("Cache keys:")

    print(cache.files)

    for key in required:
        if key not in cache.files:
            raise RuntimeError(f"Missing cache key: {key}")

        print(f"{key}: {cache[key].shape} {cache[key].dtype}")

    future_close = cache["future_close"]

    print("\nResearch 07 cache integrity: OK")

    print(f"Events: {future_close.shape[0]:,}")

    print(f"Maximum horizon: {future_close.shape[1]}")

    if future_close.shape[1] < 120:
        raise RuntimeError("Future path cache has fewer than 120 horizons.")

    return future_close


# =============================================================================
# SELECT CONTEXT
# =============================================================================


def select_context(
    events,
    candidate,
):

    mask = events["hmm_state"] == candidate["hmm_state"]

    mask &= events["vol_bucket"] == candidate["vol_bucket"]

    z = float(candidate["zscore"])

    if candidate["side"] == "LONG":
        mask &= events["zscore_30"] <= -z

    elif candidate["side"] == "SHORT":
        mask &= events["zscore_30"] >= z

    else:
        raise ValueError(f"Unknown side: {candidate['side']}")

    return events.loc[mask].copy()


# =============================================================================
# EVALUATE TP / SL / HORIZON
# =============================================================================


def evaluate_parameters(
    event_ids,
    entries,
    future_close,
    side,
    tp,
    sl,
    horizon,
):

    n = len(event_ids)

    if n == 0:
        return {
            "observations": 0,
            "wins": 0,
            "losses": 0,
            "ambiguous": 0,
            "unresolved": 0,
            "resolved": 0,
            "wr": np.nan,
            "resolution": np.nan,
            "expectancy": np.nan,
            "expectancy_all": np.nan,
            "profit_factor": np.nan,
            "net_r": np.nan,
        }

    h = int(
        max(
            1,
            min(
                120,
                horizon,
            ),
        )
    )

    paths = future_close[
        event_ids,
        :h,
    ].astype(
        np.float64,
        copy=False,
    )

    entries = entries.astype(
        np.float64,
        copy=False,
    )

    if side == "LONG":
        movement = paths - entries[:, None]

    else:
        movement = entries[:, None] - paths

    finite = np.isfinite(movement)

    tp_hit = (movement >= tp) & finite

    sl_hit = (movement <= -sl) & finite

    tp_exists = tp_hit.any(axis=1)

    sl_exists = sl_hit.any(axis=1)

    first_tp = np.argmax(
        tp_hit,
        axis=1,
    )

    first_sl = np.argmax(
        sl_hit,
        axis=1,
    )

    both = tp_exists & sl_exists

    wins = tp_exists & ~sl_exists

    losses = sl_exists & ~tp_exists

    wins |= both & (first_tp < first_sl)

    losses |= both & (first_sl < first_tp)

    ambiguous = both & (first_tp == first_sl)

    unresolved = ~tp_exists & ~sl_exists

    wins_count = int(wins.sum())

    losses_count = int(losses.sum())

    ambiguous_count = int(ambiguous.sum())

    unresolved_count = int(unresolved.sum())

    resolved = wins_count + losses_count

    if resolved > 0:
        wr = wins_count / resolved

        reward_r = tp / sl

        expectancy = wr * reward_r - (1.0 - wr)

        if losses_count > 0:
            profit_factor = wins_count * reward_r / losses_count

        else:
            profit_factor = np.inf

    else:
        wr = np.nan
        expectancy = np.nan
        profit_factor = np.nan

    net_r = wins_count * (tp / sl) - losses_count

    expectancy_all = net_r / n

    return {
        "observations": n,
        "wins": wins_count,
        "losses": losses_count,
        "ambiguous": ambiguous_count,
        "unresolved": unresolved_count,
        "resolved": resolved,
        "wr": wr,
        "resolution": (resolved / n),
        "expectancy": expectancy,
        "expectancy_all": expectancy_all,
        "profit_factor": profit_factor,
        "net_r": net_r,
    }


# =============================================================================
# TEMPORAL VALIDATION
# =============================================================================


def validate_candidate_windows(
    candidate,
    context_events,
    future_close,
):

    rows = []

    windows = sorted(context_events["window"].dropna().astype(int).unique())

    for window in windows:
        subset = context_events[context_events["window"] == window]

        event_ids = subset["event_id"].to_numpy(dtype=np.int64)

        entries = subset["close"].to_numpy(dtype=np.float64)

        metrics = evaluate_parameters(
            event_ids,
            entries,
            future_close,
            candidate["side"],
            candidate["tp"],
            candidate["sl"],
            candidate["horizon"],
        )

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "strategy_name": candidate["strategy_name"],
                "strategy_label": candidate["strategy_label"],
                "side": candidate["side"],
                "hmm_state": candidate["hmm_state"],
                "vol_bucket": candidate["vol_bucket"],
                "zscore": candidate["zscore"],
                "tp": candidate["tp"],
                "sl": candidate["sl"],
                "rr": candidate["rr"],
                "horizon": candidate["horizon"],
                "window": window,
                **metrics,
            }
        )

    return rows


# =============================================================================
# LOCAL PARAMETER STABILITY
# =============================================================================


def validate_neighbourhood(
    candidate,
    context_events,
    future_close,
):

    rows = []

    event_ids = context_events["event_id"].to_numpy(dtype=np.int64)

    entries = context_events["close"].to_numpy(dtype=np.float64)

    for tp_delta in TP_DELTAS:
        tp = candidate["tp"] + tp_delta

        if tp <= 0:
            continue

        for sl_delta in SL_DELTAS:
            sl = candidate["sl"] + sl_delta

            if sl <= 0:
                continue

            rr = tp / sl

            for h_delta in H_DELTAS:
                horizon = candidate["horizon"] + h_delta

                if horizon < 1 or horizon > 120:
                    continue

                metrics = evaluate_parameters(
                    event_ids,
                    entries,
                    future_close,
                    candidate["side"],
                    tp,
                    sl,
                    horizon,
                )

                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "strategy_name": candidate["strategy_name"],
                        "strategy_label": candidate["strategy_label"],
                        "side": candidate["side"],
                        "hmm_state": candidate["hmm_state"],
                        "vol_bucket": candidate["vol_bucket"],
                        "zscore": candidate["zscore"],
                        "base_tp": candidate["tp"],
                        "base_sl": candidate["sl"],
                        "base_rr": candidate["rr"],
                        "base_horizon": candidate["horizon"],
                        "tp": tp,
                        "sl": sl,
                        "rr": rr,
                        "horizon": horizon,
                        "tp_delta": tp_delta,
                        "sl_delta": sl_delta,
                        "h_delta": h_delta,
                        **metrics,
                    }
                )

    return rows


# =============================================================================
# SUMMARY
# =============================================================================


def summarize_candidate(
    candidate,
    window_df,
    neighbourhood_df,
):

    total_obs = int(window_df["observations"].sum())

    total_wins = int(window_df["wins"].sum())

    total_losses = int(window_df["losses"].sum())

    total_ambiguous = int(window_df["ambiguous"].sum())

    total_unresolved = int(window_df["unresolved"].sum())

    total_resolved = total_wins + total_losses

    if total_resolved > 0:
        overall_wr = total_wins / total_resolved

        overall_expectancy = ((total_wins * candidate["rr"]) - total_losses) / total_obs

        overall_pf = (
            (total_wins * candidate["rr"]) / total_losses
            if total_losses > 0
            else np.inf
        )

    else:
        overall_wr = np.nan
        overall_expectancy = np.nan
        overall_pf = np.nan

    valid_windows = window_df[window_df["resolved"] > 0]

    positive_windows = int((valid_windows["expectancy_all"] > 0).sum())

    negative_windows = int((valid_windows["expectancy_all"] < 0).sum())

    window_count = len(valid_windows)

    positive_window_ratio = positive_windows / window_count if window_count else np.nan

    worst_window_expectancy = (
        valid_windows["expectancy_all"].min() if window_count else np.nan
    )

    best_window_expectancy = (
        valid_windows["expectancy_all"].max() if window_count else np.nan
    )

    mean_window_wr = valid_windows["wr"].mean() if window_count else np.nan

    median_window_wr = valid_windows["wr"].median() if window_count else np.nan

    # -------------------------------------------------------------------------
    # Neighbourhood
    # -------------------------------------------------------------------------

    nb = neighbourhood_df[neighbourhood_df["candidate_id"] == candidate["candidate_id"]]

    positive_nb = int((nb["expectancy_all"] > 0).sum())

    nb_count = len(nb)

    true_nb = nb[
        ~(
            (nb["tp"] == candidate["tp"])
            & (nb["sl"] == candidate["sl"])
            & (nb["horizon"] == candidate["horizon"])
        )
    ]

    positive_true_nb = int((true_nb["expectancy_all"] > 0).sum())

    true_nb_count = len(true_nb)

    positive_true_nb_ratio = (
        positive_true_nb / true_nb_count if true_nb_count else np.nan
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "strategy_name": candidate["strategy_name"],
        "strategy_label": candidate["strategy_label"],
        "side": candidate["side"],
        "hmm_state": candidate["hmm_state"],
        "vol_bucket": candidate["vol_bucket"],
        "zscore": candidate["zscore"],
        "tp": candidate["tp"],
        "sl": candidate["sl"],
        "rr": candidate["rr"],
        "horizon": candidate["horizon"],
        "total_observations": total_obs,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_ambiguous": total_ambiguous,
        "total_unresolved": total_unresolved,
        "total_resolved": total_resolved,
        "overall_wr": overall_wr,
        "overall_expectancy_all": overall_expectancy,
        "overall_profit_factor": overall_pf,
        "positive_windows": positive_windows,
        "negative_windows": negative_windows,
        "window_count": window_count,
        "positive_window_ratio": positive_window_ratio,
        "mean_window_wr": mean_window_wr,
        "median_window_wr": median_window_wr,
        "worst_window_expectancy": worst_window_expectancy,
        "best_window_expectancy": best_window_expectancy,
        "neighbourhood_points": nb_count,
        "positive_neighbourhood_points": positive_nb,
        "positive_neighbourhood_ratio": (
            positive_nb / nb_count if nb_count else np.nan
        ),
        "true_neighbour_points": true_nb_count,
        "positive_true_neighbour_points": positive_true_nb,
        "positive_true_neighbour_ratio": positive_true_nb_ratio,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08N")

    print("CANDIDATE VALIDATION / ANTI-OVERFITTING AUDIT")

    print("-" * 100)

    print("OFFICIAL STRATEGY NAMES:")

    print("C01 -> MRS2")

    print("C02 -> MRL1")

    print("C06 -> MRL2")

    print("\nWR >= 50% is NOT a mandatory requirement.")

    print("Primary criteria:")

    print("positive expectancy + temporal stability + local parameter stability")

    print(f"\nCandidates: {len(CANDIDATES)}")

    # -------------------------------------------------------------------------
    # Build context
    # -------------------------------------------------------------------------

    events = build_events()

    # -------------------------------------------------------------------------
    # Load exact path cache
    # -------------------------------------------------------------------------

    future_close = load_paths()

    # -------------------------------------------------------------------------
    # Candidate validation
    # -------------------------------------------------------------------------

    all_window_rows = []

    all_neighbour_rows = []

    summary_rows = []

    for i, candidate in enumerate(
        CANDIDATES,
        start=1,
    ):
        section(
            f"{candidate['strategy_name']} "
            f"({candidate['candidate_id']}) "
            f"— CANDIDATE {i}/{len(CANDIDATES)}"
        )

        print(f"Strategy : {candidate['strategy_label']}")

        print(f"Side     : {candidate['side']}")

        print(f"HMM      : {candidate['hmm_state']}")

        print(f"Vol      : {candidate['vol_bucket']}")

        print(f"Z-score  : {candidate['zscore']}")

        print(f"TP       : {candidate['tp']}")

        print(f"SL       : {candidate['sl']}")

        print(f"RR       : {candidate['rr']}")

        print(f"Horizon  : {candidate['horizon']}")

        context_events = select_context(
            events,
            candidate,
        )

        print(f"\nContext observations: {len(context_events):,}")

        if context_events.empty:
            print("SKIPPED — no observations.")

            continue

        # ---------------------------------------------------------------------
        # Temporal windows
        # ---------------------------------------------------------------------

        print("\nValidating temporal windows...")

        window_rows = validate_candidate_windows(
            candidate,
            context_events,
            future_close,
        )

        all_window_rows.extend(window_rows)

        window_df = pd.DataFrame(window_rows)

        # ---------------------------------------------------------------------
        # Local parameter neighbourhood
        # ---------------------------------------------------------------------

        print("Testing local TP / SL / Horizon neighbourhood...")

        neighbourhood_rows = validate_neighbourhood(
            candidate,
            context_events,
            future_close,
        )

        all_neighbour_rows.extend(neighbourhood_rows)

        neighbourhood_df = pd.DataFrame(neighbourhood_rows)

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------

        summary = summarize_candidate(
            candidate,
            window_df,
            neighbourhood_df,
        )

        summary_rows.append(summary)

        print("\nRESULT:")

        print(f"Strategy: {summary['strategy_name']}")

        print(f"Overall WR: {summary['overall_wr']:.2%}")

        print(f"Overall expectancy: {summary['overall_expectancy_all']:.4f}R")

        print(f"Profit factor: {summary['overall_profit_factor']:.3f}")

        print(
            f"Positive windows: {summary['positive_windows']}/{summary['window_count']}"
        )

        print(
            f"Positive true neighbours: {summary['positive_true_neighbour_ratio']:.2%}"
        )

    # -------------------------------------------------------------------------
    # Dataframes
    # -------------------------------------------------------------------------

    window_df = pd.DataFrame(all_window_rows)

    neighbourhood_df = pd.DataFrame(all_neighbour_rows)

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        raise RuntimeError("No candidate validation results generated.")

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    statuses = []

    for _, row in summary_df.iterrows():
        expectancy = row["overall_expectancy_all"]

        positive_windows = row["positive_windows"]

        neighbour_ratio = row["positive_true_neighbour_ratio"]

        if (
            np.isfinite(expectancy)
            and expectancy > 0
            and positive_windows >= 3
            and np.isfinite(neighbour_ratio)
            and neighbour_ratio >= 0.40
        ):
            statuses.append("PROMISING")

        elif np.isfinite(expectancy) and expectancy > 0:
            statuses.append("POSITIVE_BUT_UNSTABLE")

        else:
            statuses.append("FAILED_VALIDATION")

    summary_df["validation_status"] = statuses

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_df.to_csv(
        OUTPUT_WINDOW,
        index=False,
    )

    neighbourhood_df.to_csv(
        OUTPUT_NEIGHBORHOOD,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    section("CANDIDATE VALIDATION SUMMARY")

    columns = [
        "candidate_id",
        "strategy_name",
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "tp",
        "sl",
        "rr",
        "horizon",
        "total_observations",
        "total_resolved",
        "overall_wr",
        "overall_expectancy_all",
        "overall_profit_factor",
        "positive_windows",
        "window_count",
        "positive_window_ratio",
        "worst_window_expectancy",
        "positive_true_neighbour_ratio",
        "validation_status",
    ]

    print(
        summary_df[columns]
        .sort_values(
            "overall_expectancy_all",
            ascending=False,
        )
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Window results
    # -------------------------------------------------------------------------

    section("WINDOW-LEVEL RESULTS")

    if window_df.empty:
        print("No window results.")

    else:
        print(
            window_df[
                [
                    "candidate_id",
                    "strategy_name",
                    "window",
                    "observations",
                    "wins",
                    "losses",
                    "ambiguous",
                    "unresolved",
                    "resolved",
                    "wr",
                    "resolution",
                    "expectancy_all",
                    "profit_factor",
                ]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Local parameter results
    # -------------------------------------------------------------------------

    section("LOCAL PARAMETER STABILITY")

    if neighbourhood_df.empty:
        print("No neighbourhood results.")

    else:
        display_columns = [
            "candidate_id",
            "strategy_name",
            "tp",
            "sl",
            "rr",
            "horizon",
            "observations",
            "wins",
            "losses",
            "wr",
            "expectancy_all",
        ]

        print(
            neighbourhood_df[display_columns]
            .sort_values(
                [
                    "candidate_id",
                    "expectancy_all",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(150)
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Official strategy names
    # -------------------------------------------------------------------------

    section("OFFICIAL STRATEGY IDENTIFIERS")

    print("MRS2 = Mean Reversion Short — HMM State 2")

    print("MRL1 = Mean Reversion Long  — HMM State 1")

    print("MRL2 = Mean Reversion Long  — HMM State 2")

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    section("RESEARCH 08N COMPLETE")

    print("Candidates validated:")

    print("MRS2")

    print("MRL1")

    print("MRL2")

    print("\nNo parameter optimization was performed.")

    print("Temporal stability evaluated.")

    print("Local TP / SL / Horizon stability evaluated.")

    print("No failure test was performed.")

    print("\nFILES SAVED:")

    print(OUTPUT_WINDOW)

    print(OUTPUT_NEIGHBORHOOD)

    print(OUTPUT_SUMMARY)

    print("\nNEXT STAGE:")

    print("Freeze MRS2 / MRL1 / MRL2.")

    print("Proceed to independent OOS / failure testing.")


if __name__ == "__main__":
    main()

