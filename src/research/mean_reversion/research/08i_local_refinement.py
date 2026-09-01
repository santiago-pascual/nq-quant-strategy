from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08I — LOCAL REFINEMENT
# =============================================================================
#
# Purpose:
#
#   Take the strongest parameter regions discovered in Research 08H
#   and perform a finer local search.
#
# FIXED:
#   - side
#   - HMM state
#   - volatility bucket
#   - Z-score threshold
#
# SEARCHED:
#   - TP
#   - SL
#   - horizon
#
# NOT PERFORMED:
#   - HMM retraining
#   - volatility recalculation
#   - context discovery
#   - failure test
#   - production changes
#
# IMPORTANT:
#
#   LONG  = zscore_30 <= -threshold
#   SHORT = zscore_30 >= +threshold
#
# Research 08H showed two major parameter regions:
#
# REGION A:
#   LONG
#   HMM 0
#   VOL 20-40
#   Z >= 2.0 / 2.5
#
# REGION B:
#   LONG
#   HMM 2
#   VOL 60-80
#   Z >= 3.5
#
# Additional promising regions:
#
#   LONG HMM 0 VOL 40-60 Z>=2.0
#   LONG HMM 1 VOL 20-40 Z>=2.5
#   LONG HMM 1 VOL 20-40 Z>=3.0
#
# We refine all of them.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_PATH = CACHE_DIR / "research_08b_causal_hmm_states.csv"

OUTPUT_ALL = RESULTS_DIR / "research_08i_local_refinement.csv"

OUTPUT_BEST_CONTEXT = RESULTS_DIR / "research_08i_best_per_context.csv"

OUTPUT_CANDIDATES = RESULTS_DIR / "research_08i_candidates.csv"

OUTPUT_STABILITY = RESULTS_DIR / "research_08i_parameter_stability.csv"


# =============================================================================
# CONTEXTS TO REFINE
# =============================================================================

CONTEXTS = [
    # -------------------------------------------------------------------------
    # REGION A — strongest large-sample region
    # -------------------------------------------------------------------------
    {
        "name": "A1",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "20-40",
        "zscore": 2.0,
    },
    {
        "name": "A2",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "20-40",
        "zscore": 2.5,
    },
    # -------------------------------------------------------------------------
    # REGION B — HMM 2 high-volatility tail
    # -------------------------------------------------------------------------
    {
        "name": "B1",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
    },
    # -------------------------------------------------------------------------
    # Additional promising regions
    # -------------------------------------------------------------------------
    {
        "name": "C1",
        "side": "LONG",
        "hmm_state": 0,
        "vol_bucket": "40-60",
        "zscore": 2.0,
    },
    {
        "name": "D1",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
    },
    {
        "name": "D2",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 3.0,
    },
    # -------------------------------------------------------------------------
    # SHORT regions that survived 08H
    # -------------------------------------------------------------------------
    {
        "name": "S1",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
    },
    {
        "name": "S2",
        "side": "SHORT",
        "hmm_state": 0,
        "vol_bucket": "20-40",
        "zscore": 3.0,
    },
    {
        "name": "S3",
        "side": "SHORT",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 3.5,
    },
]


# =============================================================================
# LOCAL PARAMETER GRID
# =============================================================================
#
# Region A:
# 08H found TP ~30-40 and SL ~15-20.
#
# We extend slightly beyond the observed optimum to see whether the
# expectancy/WR surface continues smoothly.
#
# =============================================================================

TP_VALUES = np.array(
    [
        25.0,
        27.5,
        30.0,
        32.5,
        35.0,
        37.5,
        40.0,
        42.5,
        45.0,
        47.5,
        50.0,
    ]
)

SL_VALUES = np.array(
    [
        10.0,
        12.5,
        15.0,
        17.5,
        20.0,
    ]
)

HORIZON_VALUES = np.array(
    [
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        15,
        20,
    ]
)


# Special grid for the HMM 2 / VOL 60-80 / Z3.5 region.

HMM2_TP_VALUES = np.array(
    [
        5.0,
        6.0,
        7.0,
        7.5,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        15.0,
    ]
)

HMM2_SL_VALUES = np.array(
    [
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.5,
    ]
)

HMM2_HORIZON_VALUES = np.array(
    [
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        15,
        20,
    ]
)


MIN_OBSERVATIONS = 100


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

    print(f"Events: {len(df):,}")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm():

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 08B HMM STATES")
    print("=" * 100)

    df = pd.read_csv(HMM_PATH)

    required = [
        "event_id",
        "hmm_state",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"HMM cache missing columns: {missing}")

    df["event_id"] = pd.to_numeric(df["event_id"]).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(
        df["hmm_state"],
        errors="coerce",
    )

    print(f"HMM rows: {len(df):,}")

    return df[
        [
            "event_id",
            "hmm_state",
        ]
    ]


# =============================================================================
# LOAD VOLATILITY BUCKETS
# =============================================================================


def load_volatility_buckets(metadata):

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 08H VOLATILITY CONTEXT")
    print("=" * 100)

    #
    # 08H already saved the causal volatility context.
    #
    # We use that exact output rather than recalculating volatility.
    #

    validation_path = RESULTS_DIR / "research_08h_context_validation.csv"

    #
    # The validation file only contains target context counts,
    # so it cannot provide event-level volatility.
    #
    # Therefore we use the event-level volatility mapping generated
    # by 08H if available.
    #

    event_candidates = [
        RESULTS_DIR / "research_08h_event_context.csv",
        RESULTS_DIR / "research_08h_event_context.parquet",
        RESULTS_DIR / "research_08h_context_events.csv",
        CACHE_DIR / "research_08h_event_context.csv",
    ]

    for path in event_candidates:
        if path.exists():
            print(f"Using existing event context:\n{path}")

            if path.suffix == ".parquet":
                df = pd.read_parquet(path)

            else:
                df = pd.read_csv(path)

            required = [
                "event_id",
                "vol_bucket",
            ]

            if all(c in df.columns for c in required):
                return df[
                    [
                        "event_id",
                        "vol_bucket",
                    ]
                ]

    #
    # If 08H did not save event-level volatility,
    # rebuild it using the same causal definition as 08H.
    #
    # This is NOT optimization.
    #

    print("Event-level volatility context not found.")

    print(
        "Rebuilding causal volatility buckets using the same Research 08H definition."
    )

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

    #
    # Find timestamp.
    #

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
        raise RuntimeError("Could not find market timestamp.")

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

    events = events.sort_values("timestamp").reset_index(drop=True)

    mapped = pd.merge_asof(
        events,
        market,
        left_on="timestamp",
        right_on="_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    values = mapped["realized_vol_30"].to_numpy(dtype=np.float64)

    percentile = np.full(
        len(mapped),
        np.nan,
    )

    #
    # Exact same causal concept:
    #
    # percentile = historical rank of current volatility.
    #
    history = []

    from bisect import (
        bisect_right,
        insort,
    )

    valid = np.flatnonzero(np.isfinite(values))

    for i in valid:
        value = values[i]

        position = bisect_right(
            history,
            value,
        )

        count = len(history) + 1

        percentile[i] = position / count * 100.0

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

    mapped = mapped.sort_values("event_id").reset_index(drop=True)

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

    volatility = load_volatility_buckets(metadata)

    print("\n" + "=" * 100)
    print("BUILDING EVENT CONTEXT")
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

    print(
        "Missing Z-score:",
        events["zscore_30"].isna().sum(),
    )

    if events["hmm_state"].isna().any():
        raise RuntimeError("Missing HMM states.")

    if events["vol_bucket"].isna().any():
        raise RuntimeError("Missing volatility buckets.")

    return events


# =============================================================================
# CONTEXT MASK
# =============================================================================


def get_context_mask(
    events,
    context,
):

    side = context["side"]

    state = context["hmm_state"]

    vol = context["vol_bucket"]

    z = context["zscore"]

    mask = events["hmm_state"].eq(state) & events["vol_bucket"].eq(vol)

    if side == "LONG":
        mask &= events["zscore_30"] <= -abs(z)

    elif side == "SHORT":
        mask &= events["zscore_30"] >= abs(z)

    else:
        raise ValueError(f"Invalid side: {side}")

    return mask


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_paths():

    print("\n" + "=" * 100)
    print("LOADING RESEARCH 07 PATH CACHE")
    print("=" * 100)

    cache = np.load(
        PATH_CACHE,
        allow_pickle=False,
    )

    required = [
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
# EVALUATE
# =============================================================================


def evaluate(
    favorable,
    adverse,
    tp,
    sl,
    horizon,
):

    horizon = int(horizon)

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
        dtype=np.int16,
    )

    sl_first = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
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

    #
    # Terminal excursion tie-break.
    #

    if unresolved.any():
        rows = np.flatnonzero(unresolved)

        terminal_fav = favorable[
            rows,
            horizon - 1,
        ]

        terminal_adv = adverse[
            rows,
            horizon - 1,
        ]

        wins[rows] = terminal_fav >= terminal_adv

        losses[rows] = terminal_fav < terminal_adv

    wins_n = int(wins.sum())

    losses_n = int(losses.sum())

    wr = wins_n / n

    rr = tp / sl

    expectancy = wr * rr - (1.0 - wr)

    breakeven = 1.0 / (1.0 + rr)

    return {
        "observations": n,
        "wins": wins_n,
        "losses": losses_n,
        "win_rate": wr,
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr),
        "horizon": horizon,
        "expectancy_r": expectancy,
        "breakeven_wr": breakeven,
        "edge_vs_breakeven": (wr - breakeven),
    }


# =============================================================================
# WINDOW STATS
# =============================================================================


def calculate_window_stability(
    favorable,
    adverse,
    event_windows,
    tp,
    sl,
    horizon,
):

    rows = []

    unique_windows = np.sort(np.unique(event_windows))

    for window in unique_windows:
        mask = event_windows == window

        if mask.sum() < 10:
            continue

        result = evaluate(
            favorable[mask],
            adverse[mask],
            tp,
            sl,
            horizon,
        )

        if result is None:
            continue

        rows.append(
            {
                "window": int(window),
                "observations": int(result["observations"]),
                "win_rate": result["win_rate"],
                "expectancy_r": result["expectancy_r"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SEARCH
# =============================================================================


def run_search(
    events,
    paths,
):

    print("\n" + "=" * 100)
    print("RUNNING RESEARCH 08I LOCAL REFINEMENT")
    print("=" * 100)

    all_results = []

    stability_rows = []

    total_contexts = len(CONTEXTS)

    for context_number, context in enumerate(
        CONTEXTS,
        start=1,
    ):
        print("\n" + "-" * 100)

        print(f"CONTEXT {context_number}/{total_contexts}")

        print(
            f"{context['name']} | "
            f"{context['side']} | "
            f"HMM={context['hmm_state']} | "
            f"VOL={context['vol_bucket']} | "
            f"|Z|>={context['zscore']}"
        )

        mask = get_context_mask(
            events,
            context,
        )

        event_ids = events.loc[
            mask,
            "event_id",
        ].to_numpy(dtype=np.int64)

        print(f"Observations: {len(event_ids):,}")

        if len(event_ids) < MIN_OBSERVATIONS:
            print("SKIPPED — insufficient observations.")

            continue

        if context["side"] == "LONG":
            favorable = paths["long_favorable"][event_ids]

            adverse = paths["long_adverse"][event_ids]

        else:
            favorable = paths["short_favorable"][event_ids]

            adverse = paths["short_adverse"][event_ids]

        event_windows = events.loc[
            mask,
            "window",
        ].to_numpy(dtype=np.int64)

        #
        # Special grid for HMM2.
        #

        if context["hmm_state"] == 2 and context["vol_bucket"] == "60-80":
            tp_values = HMM2_TP_VALUES

            sl_values = HMM2_SL_VALUES

            horizon_values = HMM2_HORIZON_VALUES

        else:
            tp_values = TP_VALUES

            sl_values = SL_VALUES

            horizon_values = HORIZON_VALUES

        combinations = itertools.product(
            tp_values,
            sl_values,
            horizon_values,
        )

        local_count = 0

        for tp, sl, horizon in combinations:
            result = evaluate(
                favorable,
                adverse,
                tp,
                sl,
                horizon,
            )

            if result is None:
                continue

            result.update(
                {
                    "context": context["name"],
                    "side": context["side"],
                    "hmm_state": context["hmm_state"],
                    "vol_bucket": context["vol_bucket"],
                    "zscore": context["zscore"],
                }
            )

            all_results.append(result)

            local_count += 1

        print(f"Parameter combinations evaluated: {local_count:,}")

        #
        # Preliminary best.
        #

        local = pd.DataFrame(
            [x for x in all_results if x["context"] == context["name"]]
        )

        if local.empty:
            continue

        candidates = local[(local["win_rate"] >= 0.50) & (local["expectancy_r"] > 0)]

        if not candidates.empty:
            best = candidates.sort_values(
                [
                    "expectancy_r",
                    "win_rate",
                ],
                ascending=False,
            ).iloc[0]

        else:
            best = local.sort_values(
                [
                    "expectancy_r",
                    "win_rate",
                ],
                ascending=False,
            ).iloc[0]

        print("\nBEST LOCAL RESULT:")

        print(
            f"TP={best.tp:.2f} | "
            f"SL={best.sl:.2f} | "
            f"RR={best.rr:.3f} | "
            f"H={int(best.horizon)} | "
            f"WR={best.win_rate:.2%} | "
            f"EXP={best.expectancy_r:.4f}R"
        )

        #
        # Window stability for the preliminary best.
        #

        stability = calculate_window_stability(
            favorable,
            adverse,
            event_windows,
            best.tp,
            best.sl,
            best.horizon,
        )

        if not stability.empty:
            stability["context"] = context["name"]

            stability["side"] = context["side"]

            stability["hmm_state"] = context["hmm_state"]

            stability["vol_bucket"] = context["vol_bucket"]

            stability["zscore"] = context["zscore"]

            stability["tp"] = best.tp

            stability["sl"] = best.sl

            stability["horizon"] = best.horizon

            stability_rows.append(stability)

    if not all_results:
        raise RuntimeError("No local parameter results were generated.")

    results = pd.DataFrame(all_results)

    if stability_rows:
        stability = pd.concat(
            stability_rows,
            ignore_index=True,
        )

    else:
        stability = pd.DataFrame()

    return results, stability


# =============================================================================
# RANK
# =============================================================================


def rank_results(results):

    print("\n" + "=" * 100)
    print("RANKING LOCAL RESULTS")
    print("=" * 100)

    results = results.copy()

    results["candidate"] = (results["win_rate"] >= 0.50) & (results["expectancy_r"] > 0)

    #
    # More conservative ranking:
    #
    # expectancy first,
    # then WR,
    # then observations.
    #

    results["research_score"] = (
        results["expectancy_r"]
        * (0.5 + results["win_rate"])
        * np.log1p(results["observations"])
    )

    results = results.sort_values(
        [
            "candidate",
            "research_score",
            "expectancy_r",
            "win_rate",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print(f"Total local evaluations: {len(results):,}")

    print(
        "WR >= 50% + positive expectancy:",
        int(results["candidate"].sum()),
    )

    return results


# =============================================================================
# BEST PER CONTEXT
# =============================================================================


def best_per_context(results):

    keys = [
        "context",
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
    ]

    rows = []

    for key, group in results.groupby(
        keys,
        sort=False,
    ):
        candidates = group[group["candidate"]]

        if candidates.empty:
            chosen = group.sort_values(
                [
                    "expectancy_r",
                    "win_rate",
                ],
                ascending=False,
            ).iloc[0]

        else:
            chosen = candidates.sort_values(
                [
                    "expectancy_r",
                    "win_rate",
                ],
                ascending=False,
            ).iloc[0]

        rows.append(chosen)

    return pd.DataFrame(rows)


# =============================================================================
# PARAMETER STABILITY
# =============================================================================


def calculate_parameter_stability(results):

    rows = []

    context_best = best_per_context(results)

    for _, best in context_best.iterrows():
        context = best["context"]

        group = results[results["context"] == context].copy()

        #
        # Parameter neighborhood:
        #
        # TP within 10 points
        # SL within 5 points
        # horizon within 5 bars
        #

        neighborhood = group[
            ((group["tp"] - best["tp"]).abs() <= 10.0)
            & ((group["sl"] - best["sl"]).abs() <= 5.0)
            & ((group["horizon"] - best["horizon"]).abs() <= 5)
        ]

        rows.append(
            {
                "context": context,
                "side": best["side"],
                "hmm_state": best["hmm_state"],
                "vol_bucket": best["vol_bucket"],
                "zscore": best["zscore"],
                "best_tp": best["tp"],
                "best_sl": best["sl"],
                "best_horizon": best["horizon"],
                "best_wr": best["win_rate"],
                "best_expectancy": best["expectancy_r"],
                "neighborhood_count": len(neighborhood),
                "neighborhood_mean_wr": (neighborhood["win_rate"].mean()),
                "neighborhood_mean_expectancy": (neighborhood["expectancy_r"].mean()),
                "neighborhood_median_expectancy": (
                    neighborhood["expectancy_r"].median()
                ),
                "neighborhood_positive_expectancy_ratio": (
                    (neighborhood["expectancy_r"] > 0).mean()
                ),
                "neighborhood_wr_ge_50_ratio": (
                    (neighborhood["win_rate"] >= 0.50).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================


def report(
    results,
    stability,
):

    banner_text = "RESEARCH 08I — RESULTS"

    print("\n" + "=" * 100)
    print(banner_text)
    print("=" * 100)

    columns = [
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
        "breakeven_wr",
        "edge_vs_breakeven",
    ]

    candidates = results[results["candidate"]]

    if not candidates.empty:
        print("\nTOP CANDIDATES:")

        print(candidates[columns].head(40).to_string(index=False))

    else:
        print("\nNo WR >= 50% + positive expectancy configuration found.")

    print("\n" + "=" * 100)

    print("BEST PER CONTEXT")

    best = best_per_context(results)

    print(best[columns + ["candidate"]].to_string(index=False))

    if not stability.empty:
        print("\n" + "=" * 100)

        print("WINDOW STABILITY OF BEST LOCAL PARAMETERS")

        print(
            stability[
                [
                    "context",
                    "window",
                    "observations",
                    "win_rate",
                    "expectancy_r",
                ]
            ].to_string(index=False)
        )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("\n" + "=" * 100)

    print("MEAN REVERSION — RESEARCH 08I")

    print("=" * 100)

    print("LOCAL PARAMETER REFINEMENT")

    print("-" * 100)

    print("Fixed contexts from Research 08E.")

    print("Local refinement of Research 08H parameter regions.")

    print("No HMM retraining.")

    print("No volatility optimization.")

    print("No context discovery.")

    print("No failure test.")

    print("No production changes.")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    events = build_events()

    paths = load_paths()

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    results, stability = run_search(
        events,
        paths,
    )

    # -------------------------------------------------------------------------
    # RANK
    # -------------------------------------------------------------------------

    results = rank_results(results)

    # -------------------------------------------------------------------------
    # PARAMETER STABILITY
    # -------------------------------------------------------------------------

    parameter_stability = calculate_parameter_stability(results)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    best = best_per_context(results)

    best.to_csv(
        OUTPUT_BEST_CONTEXT,
        index=False,
    )

    results[results["candidate"]].to_csv(
        OUTPUT_CANDIDATES,
        index=False,
    )

    parameter_stability.to_csv(
        OUTPUT_STABILITY,
        index=False,
    )

    if not stability.empty:
        stability.to_csv(
            RESULTS_DIR / "research_08i_window_stability.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    report(
        results,
        stability,
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print("\n" + "=" * 100)

    print("RESEARCH 08I COMPLETE")

    print("=" * 100)

    print(f"Total evaluations: {len(results):,}")

    print(
        "Candidate configurations:",
        int(results["candidate"].sum()),
    )

    print("\nFiles:")

    print(OUTPUT_ALL)

    print(OUTPUT_BEST_CONTEXT)

    print(OUTPUT_CANDIDATES)

    print(OUTPUT_STABILITY)

    print("\nIMPORTANT:")

    print("These are refined research candidates.")

    print("No strategy has been frozen.")

    print("No failure test has been performed.")

    print("Next step: robust temporal/OOS validation.")


if __name__ == "__main__":
    main()
