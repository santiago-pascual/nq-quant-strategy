from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08J
# ROBUST PARAMETER VALIDATION
# =============================================================================
#
# Purpose:
#
#   Validate the strongest Research 08I parameter configurations across
#   individual Research 07 windows.
#
# This is NOT:
#   - parameter optimization
#   - HMM retraining
#   - volatility optimization
#   - failure testing
#   - production strategy construction
#
# The objective is to determine whether the apparent edge is distributed
# across time or concentrated in a small number of windows.
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"


METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

VOL_CONTEXT_PATH = RESULTS_DIR / "research_08h_event_context.csv"

REFINEMENT_PATH = RESULTS_DIR / "research_08i_local_refinement.csv"

OUTPUT_RESULTS = RESULTS_DIR / "research_08j_robust_validation.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08j_robust_summary.csv"

OUTPUT_CANDIDATES = RESULTS_DIR / "research_08j_robust_candidates.csv"


# =============================================================================
# MINIMUM REQUIREMENTS
# =============================================================================

MIN_TOTAL_OBSERVATIONS = 100

MIN_WINDOW_OBSERVATIONS = 20

MIN_WINDOWS = 5

MIN_POSITIVE_WINDOW_RATIO = 0.50

MIN_MEAN_WR = 0.50

MIN_MEAN_EXPECTANCY = 0.0


# =============================================================================
# LOAD RESEARCH 08I
# =============================================================================


def load_refinement():

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 08I")
    print("=" * 100)

    df = pd.read_csv(REFINEMENT_PATH)

    required = [
        "context",
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "observations",
        "tp",
        "sl",
        "rr",
        "horizon",
        "win_rate",
        "expectancy_r",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Research 08I missing columns: {missing}")

    print(f"Parameter rows: {len(df):,}")

    return df


# =============================================================================
# SELECT STRONG PARAMETER CLUSTERS
# =============================================================================


def select_parameter_candidates(refinement):

    print("\n" + "=" * 100)
    print("SELECTING ROBUSTNESS CANDIDATES")
    print("=" * 100)

    #
    # We do NOT simply take the single best row.
    #
    # We take the strongest parameter configurations from each context.
    #
    # This prevents one potentially noisy combination from determining
    # the entire validation.
    #

    candidates = []

    for context, group in refinement.groupby(
        "context",
        sort=False,
    ):
        group = group.copy()

        #
        # Require positive expectancy.
        #

        group = group[group["expectancy_r"] > 0]

        if group.empty:
            continue

        #
        # Rank by a combination of:
        #
        #   expectancy
        #   WR
        #   observations
        #
        group["selection_score"] = (
            group["expectancy_r"]
            * (0.5 + group["win_rate"])
            * np.log1p(group["observations"])
        )

        group = group.sort_values(
            "selection_score",
            ascending=False,
        )

        #
        # Keep a small neighborhood rather than only one parameter set.
        #

        top = group.head(10)

        candidates.append(top)

    if not candidates:
        raise RuntimeError("No Research 08I candidates available.")

    candidates = pd.concat(
        candidates,
        ignore_index=True,
    )

    #
    # Remove duplicate parameter rows.
    #

    candidates = candidates.drop_duplicates(
        subset=[
            "context",
            "tp",
            "sl",
            "horizon",
        ]
    ).reset_index(drop=True)

    print(f"Robustness configurations selected: {len(candidates):,}")

    print(
        candidates[
            [
                "context",
                "side",
                "hmm_state",
                "vol_bucket",
                "zscore",
                "observations",
                "tp",
                "sl",
                "rr",
                "horizon",
                "win_rate",
                "expectancy_r",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

    return candidates


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
        "zscore_30",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Metadata missing: {missing}")

    df["event_id"] = pd.to_numeric(df["event_id"]).astype(np.int64)

    df["window"] = pd.to_numeric(df["window"]).astype(np.int64)

    df["zscore_30"] = pd.to_numeric(
        df["zscore_30"],
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


def load_volatility(metadata):

    print("\n" + "=" * 100)
    print("LOADING VOLATILITY CONTEXT")
    print("=" * 100)

    if VOL_CONTEXT_PATH.exists():
        df = pd.read_csv(VOL_CONTEXT_PATH)

        required = [
            "event_id",
            "vol_bucket",
        ]

        if all(c in df.columns for c in required):
            print("Using cached event-level volatility context.")

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
    # Reproduce the causal volatility context exactly as 08H/08I.
    #

    print("Cached event volatility context not found.")

    print("Rebuilding causal volatility context.")

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
        raise RuntimeError("Market timestamp unavailable.")

    market["_timestamp"] = pd.to_datetime(
        market[timestamp_column],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    market = market[
        [
            "_timestamp",
            "realized_vol_30",
        ]
    ].dropna()

    market = market.sort_values("_timestamp").drop_duplicates("_timestamp")

    events = metadata[
        [
            "event_id",
            "timestamp",
        ]
    ].copy()

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        utc=True,
        errors="coerce",
    ).astype("datetime64[ns, UTC]")

    events = events.sort_values("timestamp")

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

    percentiles = np.full(
        len(values),
        np.nan,
    )

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        position = bisect_right(
            history,
            value,
        )

        percentiles[i] = position / (len(history) + 1) * 100.0

        insort(
            history,
            value,
        )

    mapped["vol_percentile"] = percentiles

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

    volatility = load_volatility(metadata)

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

    print("\nEvents:", f"{len(events):,}")

    print(
        "Missing HMM:",
        events["hmm_state"].isna().sum(),
    )

    print(
        "Missing volatility:",
        events["vol_bucket"].isna().sum(),
    )

    return events


# =============================================================================
# LOAD PATHS
# =============================================================================


def load_paths():

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 07 PATH CACHE")
    print("=" * 100)

    cache = np.load(
        PATH_CACHE,
        allow_pickle=False,
    )

    for key in [
        "long_favorable",
        "long_adverse",
        "short_favorable",
        "short_adverse",
    ]:
        print(f"{key}: {cache[key].shape}")

    return {key: cache[key] for key in cache.files}


# =============================================================================
# EVALUATE ONE WINDOW
# =============================================================================


def evaluate_window(
    favorable,
    adverse,
    tp,
    sl,
    horizon,
):

    favorable = favorable[
        :,
        :horizon,
    ]

    adverse = adverse[
        :,
        :horizon,
    ]

    n = len(favorable)

    if n == 0:
        return None

    tp_hit = favorable >= tp

    sl_hit = adverse >= sl

    tp_any = tp_hit.any(axis=1)

    sl_any = sl_hit.any(axis=1)

    tp_first = np.full(
        n,
        horizon + 1,
    )

    sl_first = np.full(
        n,
        horizon + 1,
    )

    tp_rows = np.flatnonzero(tp_any)

    sl_rows = np.flatnonzero(sl_any)

    if len(tp_rows):
        tp_first[tp_rows] = (
            np.argmax(
                tp_hit[tp_rows],
                axis=1,
            )
            + 1
        )

    if len(sl_rows):
        sl_first[sl_rows] = (
            np.argmax(
                sl_hit[sl_rows],
                axis=1,
            )
            + 1
        )

    wins = tp_first < sl_first

    losses = sl_first < tp_first

    unresolved = ~(wins | losses)

    if unresolved.any():
        rows = np.flatnonzero(unresolved)

        fav = favorable[
            rows,
            horizon - 1,
        ]

        adv = adverse[
            rows,
            horizon - 1,
        ]

        wins[rows] = fav >= adv

        losses[rows] = fav < adv

    wins_n = int(wins.sum())

    wr = wins_n / n

    rr = tp / sl

    expectancy = wr * rr - (1.0 - wr)

    return {
        "observations": n,
        "wins": wins_n,
        "win_rate": wr,
        "rr": rr,
        "expectancy_r": expectancy,
    }


# =============================================================================
# CONTEXT MASK
# =============================================================================


def context_mask(
    events,
    candidate,
):

    mask = events["hmm_state"] == int(candidate["hmm_state"])

    mask &= events["vol_bucket"] == candidate["vol_bucket"]

    z = float(candidate["zscore"])

    if candidate["side"] == "LONG":
        mask &= events["zscore_30"] <= -abs(z)

    else:
        mask &= events["zscore_30"] >= abs(z)

    return mask


# =============================================================================
# VALIDATE
# =============================================================================


def validate_candidates(
    events,
    paths,
    candidates,
):

    print("\n" + "=" * 100)
    print("RUNNING WINDOW-BY-WINDOW ROBUSTNESS VALIDATION")
    print("=" * 100)

    rows = []

    for index, candidate in candidates.iterrows():
        context = candidate["context"]

        side = candidate["side"]

        mask = context_mask(
            events,
            candidate,
        )

        ids = events.loc[
            mask,
            "event_id",
        ].to_numpy(dtype=np.int64)

        if len(ids) < MIN_TOTAL_OBSERVATIONS:
            continue

        if side == "LONG":
            favorable = paths["long_favorable"][ids]

            adverse = paths["long_adverse"][ids]

        else:
            favorable = paths["short_favorable"][ids]

            adverse = paths["short_adverse"][ids]

        windows = events.loc[
            mask,
            "window",
        ].to_numpy(dtype=np.int64)

        tp = float(candidate["tp"])

        sl = float(candidate["sl"])

        horizon = int(candidate["horizon"])

        for window in np.unique(windows):
            window_mask = windows == window

            n = int(window_mask.sum())

            if n < MIN_WINDOW_OBSERVATIONS:
                continue

            result = evaluate_window(
                favorable[window_mask],
                adverse[window_mask],
                tp,
                sl,
                horizon,
            )

            if result is None:
                continue

            rows.append(
                {
                    "context": context,
                    "side": side,
                    "hmm_state": int(candidate["hmm_state"]),
                    "vol_bucket": candidate["vol_bucket"],
                    "zscore": float(candidate["zscore"]),
                    "tp": tp,
                    "sl": sl,
                    "rr": result["rr"],
                    "horizon": horizon,
                    "window": int(window),
                    "observations": result["observations"],
                    "wins": result["wins"],
                    "win_rate": result["win_rate"],
                    "expectancy_r": result["expectancy_r"],
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIZE
# =============================================================================


def summarize(results):

    print("\n" + "=" * 100)
    print("SUMMARIZING ROBUSTNESS")
    print("=" * 100)

    summaries = []

    group_columns = [
        "context",
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "tp",
        "sl",
        "rr",
        "horizon",
    ]

    for key, group in results.groupby(
        group_columns,
        sort=False,
    ):
        observations = int(group["observations"].sum())

        wins = int(group["wins"].sum())

        aggregate_wr = wins / observations

        window_wr = group["win_rate"]

        window_exp = group["expectancy_r"]

        qualifying_windows = int((group["win_rate"] >= 0.50).sum())

        positive_windows = int((group["expectancy_r"] > 0).sum())

        total_windows = len(group)

        summaries.append(
            {
                "context": key[0],
                "side": key[1],
                "hmm_state": key[2],
                "vol_bucket": key[3],
                "zscore": key[4],
                "tp": key[5],
                "sl": key[6],
                "rr": key[7],
                "horizon": key[8],
                "observations": observations,
                "wins": wins,
                "aggregate_wr": aggregate_wr,
                "mean_window_wr": window_wr.mean(),
                "median_window_wr": window_wr.median(),
                "std_window_wr": window_wr.std(),
                "min_window_wr": window_wr.min(),
                "max_window_wr": window_wr.max(),
                "mean_window_expectancy": window_exp.mean(),
                "median_window_expectancy": window_exp.median(),
                "min_window_expectancy": window_exp.min(),
                "positive_window_ratio": (positive_windows / total_windows),
                "wr_ge_50_window_ratio": (qualifying_windows / total_windows),
                "windows": total_windows,
            }
        )

    return pd.DataFrame(summaries)


# =============================================================================
# ROBUST CANDIDATE FILTER
# =============================================================================


def filter_robust_candidates(summary):

    print("\n" + "=" * 100)
    print("FILTERING ROBUST CANDIDATES")
    print("=" * 100)

    candidates = summary[
        (summary["observations"] >= MIN_TOTAL_OBSERVATIONS)
        & (summary["windows"] >= MIN_WINDOWS)
        & (summary["aggregate_wr"] >= MIN_MEAN_WR)
        & (summary["mean_window_expectancy"] > MIN_MEAN_EXPECTANCY)
        & (summary["positive_window_ratio"] >= MIN_POSITIVE_WINDOW_RATIO)
    ].copy()

    #
    # Robustness score.
    #
    # Reward:
    #   - aggregate WR
    #   - mean expectancy
    #   - positive-window ratio
    #   - number of observations
    #
    # Penalize:
    #   - window WR dispersion
    #

    candidates["robustness_score"] = (
        candidates["aggregate_wr"]
        * (1.0 + candidates["mean_window_expectancy"])
        * candidates["positive_window_ratio"]
        * np.log1p(candidates["observations"])
        / (1.0 + candidates["std_window_wr"])
    )

    candidates = candidates.sort_values(
        "robustness_score",
        ascending=False,
    ).reset_index(drop=True)

    print(f"Robust candidates: {len(candidates):,}")

    if not candidates.empty:
        print(
            candidates[
                [
                    "context",
                    "side",
                    "hmm_state",
                    "vol_bucket",
                    "zscore",
                    "tp",
                    "sl",
                    "rr",
                    "horizon",
                    "observations",
                    "aggregate_wr",
                    "mean_window_wr",
                    "median_window_wr",
                    "std_window_wr",
                    "min_window_wr",
                    "max_window_wr",
                    "mean_window_expectancy",
                    "positive_window_ratio",
                    "wr_ge_50_window_ratio",
                    "windows",
                    "robustness_score",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    else:
        print("No configuration satisfies all robustness requirements.")

    return candidates


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("\n" + "=" * 100)

    print("MEAN REVERSION — RESEARCH 08J")

    print("=" * 100)

    print("ROBUST PARAMETER VALIDATION")

    print("-" * 100)

    print("Research 08I candidates.")

    print("Window-by-window validation.")

    print("No parameter optimization.")

    print("No HMM retraining.")

    print("No volatility optimization.")

    print("No failure test.")

    print("No production changes.")

    refinement = load_refinement()

    candidates = select_parameter_candidates(refinement)

    events = build_events()

    paths = load_paths()

    results = validate_candidates(
        events,
        paths,
        candidates,
    )

    if results.empty:
        raise RuntimeError("No window validation results were generated.")

    summary = summarize(results)

    robust = filter_robust_candidates(summary)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    robust.to_csv(
        OUTPUT_CANDIDATES,
        index=False,
    )

    print("\n" + "=" * 100)
    print("RESEARCH 08J COMPLETE")
    print("=" * 100)

    print(f"Window results: {len(results):,}")

    print(f"Parameter configurations tested: {len(summary):,}")

    print(f"Robust candidates: {len(robust):,}")

    print("\nFILES SAVED")

    print(OUTPUT_RESULTS)

    print(OUTPUT_SUMMARY)

    print(OUTPUT_CANDIDATES)

    print("\nIMPORTANT")

    print("This is temporal robustness validation.")

    print("No final strategy has been frozen.")

    print("No failure test has been performed.")

    print("Failure testing comes only after a robust candidate survives this stage.")


if __name__ == "__main__":
    main()
