from __future__ import annotations

from bisect import bisect_right, insort
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from src.databento_loader import load_databento_mnq
from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)


# =============================================================================
# RESEARCH 08H
# =============================================================================
#
# LOCAL PARAMETER EXPANSION
#
# Research 08E has already selected the contexts.
#
# This research ONLY searches:
#
#   TP
#   SL
#   HORIZON
#
# Context is FIXED:
#
#   SIDE
#   HMM STATE
#   VOLATILITY BUCKET
#   Z-SCORE THRESHOLD
#
# IMPORTANT:
#
# LONG mean-reversion events:
#       zscore_30 <= -threshold
#
# SHORT mean-reversion events:
#       zscore_30 >= +threshold
#
# The positive threshold shown in Research 08E is therefore
# the absolute Z-score threshold.
#
# No:
#   HMM retraining
#   volatility optimization
#   context discovery
#   failure test
#   production changes
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

RESEARCH_07_METADATA = CACHE_DIR / "research_07_event_metadata.csv"

RESEARCH_07_PATH_CACHE = CACHE_DIR / "research_07_future_path_cache.npz"

HMM_CACHE = CACHE_DIR / "research_08b_causal_hmm_states.csv"

CONTEXT_RANKED = RESULTS_DIR / "research_08e_context_ranked.csv"

OUTPUT_ALL = RESULTS_DIR / "research_08h_local_parameter_results.csv"

OUTPUT_BEST_CONTEXT = RESULTS_DIR / "research_08h_best_per_context.csv"

OUTPUT_CANDIDATES = RESULTS_DIR / "research_08h_strategy_candidates.csv"

OUTPUT_BEST_SIDE = RESULTS_DIR / "research_08h_best_per_side.csv"


# =============================================================================
# FIXED CONTEXTS FROM RESEARCH 08E
# =============================================================================

TARGET_CONTEXTS = [
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
# PARAMETER GRID
# =============================================================================
#
# Research 07 path cache:
# maximum horizon = 120 bars.
#
# TP / SL are MNQ price points.
#
# =============================================================================

TP_VALUES = np.arange(
    2.5,
    42.5,
    2.5,
)

SL_VALUES = np.arange(
    2.5,
    22.5,
    2.5,
)

HORIZON_VALUES = [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    75,
    90,
    105,
    120,
]

MIN_OBSERVATIONS = 75


# =============================================================================
# UTILITY
# =============================================================================


def banner(text: str) -> None:

    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


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


def load_research_07_metadata() -> pd.DataFrame:

    banner("LOADING RESEARCH 07 METADATA")

    if not RESEARCH_07_METADATA.exists():
        raise FileNotFoundError(f"Missing:\n{RESEARCH_07_METADATA}")

    df = pd.read_csv(RESEARCH_07_METADATA)

    print(f"Events: {len(df):,}")

    required = [
        "event_id",
        "window",
        "timestamp",
        "close",
        "zscore_30",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Research 07 metadata missing:\n{missing}")

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["window"] = pd.to_numeric(
        df["window"],
        errors="raise",
    ).astype(np.int64)

    df["timestamp"] = normalize_timestamp(df["timestamp"])

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df["zscore_30"] = pd.to_numeric(
        df["zscore_30"],
        errors="coerce",
    )

    if df["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event_id in Research 07 metadata.")

    return df


# =============================================================================
# LOAD HMM
# =============================================================================


def load_hmm_states() -> pd.DataFrame:

    banner("LOADING RESEARCH 08B HMM STATES")

    if not HMM_CACHE.exists():
        raise FileNotFoundError(f"Missing:\n{HMM_CACHE}")

    df = pd.read_csv(HMM_CACHE)

    print(f"HMM rows: {len(df):,}")

    required = [
        "event_id",
        "window",
        "timestamp",
        "hmm_state",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"HMM cache missing:\n{missing}")

    df["event_id"] = pd.to_numeric(
        df["event_id"],
        errors="raise",
    ).astype(np.int64)

    df["hmm_state"] = pd.to_numeric(
        df["hmm_state"],
        errors="coerce",
    )

    if df["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event_id in HMM cache.")

    print("\nHMM state distribution:")

    print(df["hmm_state"].value_counts().sort_index().to_string())

    return df[
        [
            "event_id",
            "hmm_state",
        ]
    ]


# =============================================================================
# LOAD PATH CACHE
# =============================================================================


def load_path_cache():

    banner("LOADING RESEARCH 07 PATH CACHE")

    if not RESEARCH_07_PATH_CACHE.exists():
        raise FileNotFoundError(f"Missing:\n{RESEARCH_07_PATH_CACHE}")

    cache = np.load(
        RESEARCH_07_PATH_CACHE,
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

    missing = [x for x in required if x not in cache.files]

    if missing:
        raise RuntimeError(f"Missing path arrays:\n{missing}")

    arrays = {}

    for name in required:
        arrays[name] = cache[name]

        print(f"{name}: shape={arrays[name].shape} dtype={arrays[name].dtype}")

    n_events = arrays["future_close"].shape[0]

    for name in required:
        if arrays[name].shape[0] != n_events:
            raise RuntimeError("Path cache arrays have inconsistent event counts.")

    if n_events != 825717:
        raise RuntimeError(f"Unexpected Research 07 event count: {n_events:,}")

    print("\nResearch 07 cache integrity: OK")

    return arrays


# =============================================================================
# LOAD MARKET VOLATILITY
# =============================================================================


def load_market_volatility() -> pd.DataFrame:

    banner("LOADING MARKET DATA")

    print("Using project loader:")

    print("src.databento_loader.load_databento_mnq()")

    market = load_databento_mnq()

    print(f"Rows loaded: {len(market):,}")

    print("Building return features...")

    market = add_return_features(market)

    print("Building volatility features...")

    market = add_volatility_features(market)

    if "realized_vol_30" not in market.columns:
        raise RuntimeError("realized_vol_30 was not created.")

    #
    # Find timestamp.
    #

    timestamp_candidates = [
        "timestamp",
        "timestamp ET",
        "ts_event",
    ]

    timestamp_column = None

    for column in timestamp_candidates:
        if column in market.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise RuntimeError("Could not identify market timestamp.")

    print(f"Using market timestamp: {timestamp_column}")

    market["_timestamp_08h"] = normalize_timestamp(market[timestamp_column])

    market["realized_vol_30"] = pd.to_numeric(
        market["realized_vol_30"],
        errors="coerce",
    )

    print(f"Valid realized_vol_30: {market['realized_vol_30'].notna().sum():,}")

    print(f"Missing realized_vol_30: {market['realized_vol_30'].isna().sum():,}")

    market = market[
        [
            "_timestamp_08h",
            "realized_vol_30",
        ]
    ].dropna(
        subset=[
            "_timestamp_08h",
            "realized_vol_30",
        ]
    )

    market = market.sort_values("_timestamp_08h")

    market = market.drop_duplicates(
        subset="_timestamp_08h",
        keep="last",
    )

    return market.reset_index(drop=True)


# =============================================================================
# BUILD CAUSAL VOLATILITY CONTEXT
# =============================================================================


def build_volatility_context(
    metadata: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:

    banner("BUILDING CAUSAL VOLATILITY CONTEXT")

    #
    # Map realized_vol_30 to each Research 07 event.
    #
    # Backward as-of join:
    # use the most recent market observation at or before
    # the event timestamp.
    #

    events = metadata[
        [
            "event_id",
            "window",
            "timestamp",
        ]
    ].copy()

    events = events.sort_values("timestamp").reset_index(drop=True)

    market = market.sort_values("_timestamp_08h").reset_index(drop=True)

    events = pd.merge_asof(
        events,
        market,
        left_on="timestamp",
        right_on="_timestamp_08h",
        direction="backward",
        allow_exact_matches=True,
    )

    mapped = events["realized_vol_30"].notna()

    print(f"Events mapped to realized_vol_30: {mapped.sum():,}/{len(events):,}")

    if mapped.sum() == 0:
        raise RuntimeError("No events mapped to realized_vol_30.")

    #
    # =========================================================================
    # CAUSAL EXPANDING PERCENTILE
    # =========================================================================
    #
    # We intentionally calculate percentile sequentially.
    #
    # At event t:
    #
    #   percentile(t)
    #   = rank(volatility(t))
    #     among volatility observations available
    #     up to t.
    #
    # No future observations are used.
    #
    # =========================================================================

    values = events["realized_vol_30"].to_numpy(dtype=np.float64)

    percentile = np.full(
        len(events),
        np.nan,
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(np.isfinite(values))

    historical = []

    total = len(valid_indices)

    for counter, idx in enumerate(valid_indices):
        value = values[idx]

        position = bisect_right(
            historical,
            value,
        )

        count = len(historical) + 1

        percentile[idx] = position / count * 100.0

        insort(
            historical,
            value,
        )

        if (counter + 1) % 100000 == 0:
            print(f"  Percentile progress: {counter + 1:,}/{total:,}")

    events["vol_percentile"] = percentile

    #
    # Bucket.
    #

    events["vol_bucket"] = pd.cut(
        events["vol_percentile"],
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

    #
    # Restore event order.
    #

    events = events.sort_values("event_id").reset_index(drop=True)

    #
    # Diagnostic.
    #

    print("\nVOLATILITY BUCKET DISTRIBUTION:")

    distribution = (
        events.groupby(
            "vol_bucket",
            observed=False,
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

    return events[
        [
            "event_id",
            "vol_percentile",
            "vol_bucket",
            "realized_vol_30",
        ]
    ]


# =============================================================================
# BUILD EVENT CONTEXT
# =============================================================================


def build_event_context(
    metadata: pd.DataFrame,
    hmm: pd.DataFrame,
    volatility: pd.DataFrame,
) -> pd.DataFrame:

    banner("BUILDING COMPLETE EVENT CONTEXT")

    events = metadata[
        [
            "event_id",
            "window",
            "timestamp",
            "zscore_30",
        ]
    ].copy()

    events = events.merge(
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

    print(f"Missing HMM states: {events['hmm_state'].isna().sum():,}")

    print(f"Missing volatility buckets: {events['vol_bucket'].isna().sum():,}")

    print(f"Missing Z-scores: {events['zscore_30'].isna().sum():,}")

    if (
        events["hmm_state"].isna().any()
        or events["vol_bucket"].isna().any()
        or events["zscore_30"].isna().any()
    ):
        raise RuntimeError("Incomplete event context.")

    #
    # No side duplication here.
    #
    # Side is assigned only when the corresponding directional
    # Z-score condition is satisfied.
    #

    events["abs_zscore"] = events["zscore_30"].abs()

    return events


# =============================================================================
# CONTEXT MASK
# =============================================================================


def context_mask(
    events: pd.DataFrame,
    side: str,
    hmm_state: int,
    vol_bucket: str,
    zscore_threshold: float,
) -> pd.Series:

    base = events["hmm_state"].eq(hmm_state) & events["vol_bucket"].eq(vol_bucket)

    #
    # Mean-reversion direction:
    #
    # LONG  = negative Z-score
    # SHORT = positive Z-score
    #

    if side == "LONG":
        directional = events["zscore_30"] <= -abs(zscore_threshold)

    elif side == "SHORT":
        directional = events["zscore_30"] >= abs(zscore_threshold)

    else:
        raise ValueError(f"Unknown side: {side}")

    return base & directional


# =============================================================================
# GET CONTEXT EVENT IDS
# =============================================================================


def get_context_indices(
    events: pd.DataFrame,
    side: str,
    hmm_state: int,
    vol_bucket: str,
    zscore_threshold: float,
) -> np.ndarray:

    mask = context_mask(
        events,
        side,
        hmm_state,
        vol_bucket,
        zscore_threshold,
    )

    ids = events.loc[
        mask,
        "event_id",
    ].to_numpy(dtype=np.int64)

    return ids


# =============================================================================
# VALIDATE CONTEXTS
# =============================================================================


def validate_target_contexts(
    events: pd.DataFrame,
) -> pd.DataFrame:

    banner("VALIDATING TARGET CONTEXTS")

    rows = []

    for (
        side,
        hmm_state,
        vol_bucket,
        zscore,
    ) in TARGET_CONTEXTS:
        ids = get_context_indices(
            events,
            side,
            hmm_state,
            vol_bucket,
            zscore,
        )

        rows.append(
            {
                "side": side,
                "hmm_state": hmm_state,
                "vol_bucket": vol_bucket,
                "zscore": zscore,
                "observations": len(ids),
            }
        )

        print(
            f"{side:5s} | "
            f"HMM={hmm_state} | "
            f"VOL={vol_bucket:>5s} | "
            f"|Z|>={zscore:.1f} | "
            f"N={len(ids):,}"
        )

    validation = pd.DataFrame(rows)

    #
    # Critical sanity check.
    #

    if validation["observations"].sum() == 0:
        raise RuntimeError(
            "\nZERO observations found in ALL target contexts.\n"
            "\n"
            "This means the Research 08E context definition and "
            "the event-level mapping are inconsistent."
        )

    print("\nContext validation passed.")

    print(f"Total target-context observations: {validation['observations'].sum():,}")

    return validation


# =============================================================================
# PARAMETER EVALUATION
# =============================================================================


def evaluate_parameter(
    favorable: np.ndarray,
    adverse: np.ndarray,
    tp: float,
    sl: float,
    horizon: int,
):

    horizon = min(
        int(horizon),
        favorable.shape[1],
        adverse.shape[1],
    )

    n = favorable.shape[0]

    if n == 0:
        return None

    fav = favorable[
        :,
        :horizon,
    ]

    adv = adverse[
        :,
        :horizon,
    ]

    #
    # Barrier hits.
    #

    tp_hit = fav >= tp

    sl_hit = adv >= sl

    tp_any = tp_hit.any(axis=1)

    sl_any = sl_hit.any(axis=1)

    #
    # First TP.
    #

    tp_first = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
    )

    rows = np.flatnonzero(tp_any)

    if len(rows):
        tp_first[rows] = (
            np.argmax(
                tp_hit[rows],
                axis=1,
            )
            + 1
        )

    #
    # First SL.
    #

    sl_first = np.full(
        n,
        horizon + 1,
        dtype=np.int16,
    )

    rows = np.flatnonzero(sl_any)

    if len(rows):
        sl_first[rows] = (
            np.argmax(
                sl_hit[rows],
                axis=1,
            )
            + 1
        )

    #
    # Outcome.
    #

    wins = tp_first < sl_first

    losses = sl_first < tp_first

    unresolved = ~(wins | losses)

    #
    # If neither barrier was hit:
    #
    # classify using terminal excursion.
    #

    if unresolved.any():
        rows = np.flatnonzero(unresolved)

        terminal_fav = fav[
            rows,
            horizon - 1,
        ]

        terminal_adv = adv[
            rows,
            horizon - 1,
        ]

        terminal_win = terminal_fav >= terminal_adv

        wins[rows] = terminal_win

        losses[rows] = ~terminal_win

    #
    # Statistics.
    #

    wins_count = int(wins.sum())

    losses_count = int(losses.sum())

    unresolved_count = int(n - wins_count - losses_count)

    win_rate = wins_count / n

    loss_rate = losses_count / n

    rr = tp / sl

    breakeven_wr = 1.0 / (1.0 + rr)

    expectancy_r = win_rate * rr - loss_rate

    return {
        "observations": int(n),
        "wins": wins_count,
        "losses": losses_count,
        "unresolved": unresolved_count,
        "win_rate": float(win_rate),
        "loss_rate": float(loss_rate),
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr),
        "horizon": int(horizon),
        "breakeven_win_rate": float(breakeven_wr),
        "wr_edge_vs_breakeven": float(win_rate - breakeven_wr),
        "expectancy_r": float(expectancy_r),
    }


# =============================================================================
# RUN LOCAL SEARCH
# =============================================================================


def run_search(
    events: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:

    banner("RUNNING LOCAL PARAMETER SEARCH")

    combinations_per_context = len(TP_VALUES) * len(SL_VALUES) * len(HORIZON_VALUES)

    print(f"TP values: {len(TP_VALUES)}")

    print(f"SL values: {len(SL_VALUES)}")

    print(f"Horizon values: {len(HORIZON_VALUES)}")

    print(f"Combinations/context: {combinations_per_context:,}")

    all_results = []

    for context_number, (
        side,
        hmm_state,
        vol_bucket,
        zscore,
    ) in enumerate(
        TARGET_CONTEXTS,
        start=1,
    ):
        print("\n" + "-" * 100)

        print(f"CONTEXT {context_number}/{len(TARGET_CONTEXTS)}")

        print(f"{side} | HMM={hmm_state} | VOL={vol_bucket} | |Z|>={zscore:.1f}")

        event_ids = get_context_indices(
            events,
            side,
            hmm_state,
            vol_bucket,
            zscore,
        )

        n = len(event_ids)

        print(f"Observations: {n:,}")

        if n < MIN_OBSERVATIONS:
            print("SKIPPED — insufficient observations.")

            continue

        #
        # Cache integrity.
        #

        cache_n = arrays["future_close"].shape[0]

        if event_ids.min() < 0 or event_ids.max() >= cache_n:
            raise RuntimeError("Context event_id outside path cache.")

        #
        # Direction-specific path arrays.
        #

        if side == "LONG":
            favorable = arrays["long_favorable"][event_ids]

            adverse = arrays["long_adverse"][event_ids]

        else:
            favorable = arrays["short_favorable"][event_ids]

            adverse = arrays["short_adverse"][event_ids]

        local_results = []

        combinations = itertools.product(
            TP_VALUES,
            SL_VALUES,
            HORIZON_VALUES,
        )

        for tp, sl, horizon in combinations:
            result = evaluate_parameter(
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
                    "side": side,
                    "hmm_state": int(hmm_state),
                    "vol_bucket": vol_bucket,
                    "zscore": float(zscore),
                }
            )

            local_results.append(result)

        local = pd.DataFrame(local_results)

        if local.empty:
            continue

        all_results.append(local)

        #
        # Best WR.
        #

        best_wr = local.sort_values(
            [
                "win_rate",
                "expectancy_r",
            ],
            ascending=False,
        ).iloc[0]

        #
        # Best expectancy.
        #

        best_exp = local.sort_values(
            [
                "expectancy_r",
                "win_rate",
            ],
            ascending=False,
        ).iloc[0]

        #
        # Positive expectancy + WR >= 50%.
        #

        qualifying = local[(local["win_rate"] >= 0.50) & (local["expectancy_r"] > 0)]

        print("\nBEST WIN RATE:")

        print(
            f"TP={best_wr.tp:.1f} | "
            f"SL={best_wr.sl:.1f} | "
            f"RR={best_wr.rr:.2f} | "
            f"H={int(best_wr.horizon)} | "
            f"WR={best_wr.win_rate:.2%} | "
            f"EXP={best_wr.expectancy_r:.4f}R"
        )

        print("\nBEST EXPECTANCY:")

        print(
            f"TP={best_exp.tp:.1f} | "
            f"SL={best_exp.sl:.1f} | "
            f"RR={best_exp.rr:.2f} | "
            f"H={int(best_exp.horizon)} | "
            f"WR={best_exp.win_rate:.2%} | "
            f"EXP={best_exp.expectancy_r:.4f}R"
        )

        print("\nQUALIFYING:")

        print(f"WR >= 50% AND expectancy > 0: {len(qualifying):,}")

        if len(qualifying):
            candidate = qualifying.sort_values(
                [
                    "expectancy_r",
                    "win_rate",
                ],
                ascending=False,
            ).iloc[0]

            print("BEST QUALIFYING:")

            print(
                f"TP={candidate.tp:.1f} | "
                f"SL={candidate.sl:.1f} | "
                f"RR={candidate.rr:.2f} | "
                f"H={int(candidate.horizon)} | "
                f"WR={candidate.win_rate:.2%} | "
                f"EXP={candidate.expectancy_r:.4f}R"
            )

    if not all_results:
        raise RuntimeError("No parameter results were generated.")

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# =============================================================================
# RANK RESULTS
# =============================================================================


def rank_results(
    results: pd.DataFrame,
):

    banner("RANKING PARAMETER RESULTS")

    df = results.copy()

    df["wr_ge_50"] = df["win_rate"] >= 0.50

    df["positive_expectancy"] = df["expectancy_r"] > 0

    df["candidate"] = (
        df["wr_ge_50"]
        & df["positive_expectancy"]
        & (df["observations"] >= MIN_OBSERVATIONS)
    )

    #
    # Research ranking score.
    #

    df["research_score"] = (
        df["expectancy_r"] * (0.5 + df["win_rate"]) * np.log1p(df["observations"])
    )

    df = df.sort_values(
        [
            "candidate",
            "research_score",
            "expectancy_r",
            "win_rate",
        ],
        ascending=False,
    ).reset_index(drop=True)

    candidates = df[df["candidate"]].copy()

    print(f"Total evaluations: {len(df):,}")

    print(f"WR >= 50% + positive expectancy: {len(candidates):,}")

    return df, candidates


# =============================================================================
# BEST PER CONTEXT
# =============================================================================


def best_per_context(
    df: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
    ]

    ranked = df.sort_values(
        [
            "candidate",
            "expectancy_r",
            "win_rate",
        ],
        ascending=False,
    )

    return ranked.groupby(
        keys,
        as_index=False,
        sort=False,
    ).first()


# =============================================================================
# BEST PER SIDE
# =============================================================================


def best_per_side(
    df: pd.DataFrame,
) -> pd.DataFrame:

    ranked = df.sort_values(
        [
            "candidate",
            "expectancy_r",
            "win_rate",
        ],
        ascending=False,
    )

    return ranked.groupby(
        "side",
        as_index=False,
        sort=False,
    ).first()


# =============================================================================
# SAVE
# =============================================================================


def save_outputs(
    ranked: pd.DataFrame,
    candidates: pd.DataFrame,
):

    banner("SAVING RESEARCH 08H RESULTS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_best = best_per_context(ranked)

    side_best = best_per_side(ranked)

    ranked.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    context_best.to_csv(
        OUTPUT_BEST_CONTEXT,
        index=False,
    )

    candidates.to_csv(
        OUTPUT_CANDIDATES,
        index=False,
    )

    side_best.to_csv(
        OUTPUT_BEST_SIDE,
        index=False,
    )

    print("\nSaved:")

    print(OUTPUT_ALL)

    print(OUTPUT_BEST_CONTEXT)

    print(OUTPUT_CANDIDATES)

    print(OUTPUT_BEST_SIDE)


# =============================================================================
# REPORT
# =============================================================================


def report(
    ranked: pd.DataFrame,
    candidates: pd.DataFrame,
):

    banner("RESEARCH 08H — FINAL REPORT")

    columns = [
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
        "breakeven_win_rate",
        "wr_edge_vs_breakeven",
    ]

    if len(candidates):
        print("\n" + "=" * 100)

        print("CONFIGURATIONS WITH WR >= 50% + POSITIVE EXPECTANCY")

        print(candidates[columns].head(50).to_string(index=False))

    else:
        print("\nNO CONFIGURATION FOUND WITH:")

        print("WR >= 50% AND expectancy > 0")

        print("\nBEST POSITIVE-EXPECTANCY CONFIGURATIONS:")

        fallback = ranked[ranked["expectancy_r"] > 0].sort_values(
            [
                "win_rate",
                "expectancy_r",
            ],
            ascending=False,
        )

        if len(fallback):
            print(fallback[columns].head(50).to_string(index=False))

        else:
            print("No positive-expectancy configuration found.")

    print("\n" + "=" * 100)

    print("BEST CONFIGURATION PER CONTEXT")

    print(best_per_context(ranked)[columns + ["candidate"]].to_string(index=False))

    print("\n" + "=" * 100)

    print("BEST CONFIGURATION PER SIDE")

    print(best_per_side(ranked)[columns + ["candidate"]].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================


def main():

    banner("MEAN REVERSION — RESEARCH 08H")

    print("LOCAL PARAMETER EXPANSION")

    print("Fixed contexts from Research 08E.")

    print("LONG  = Z <= -threshold")

    print("SHORT = Z >= +threshold")

    print("Searching TP / SL / HORIZON only.")

    print("No HMM retraining.")

    print("No volatility optimization.")

    print("No failure test.")

    print("No production changes.")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    metadata = load_research_07_metadata()

    hmm = load_hmm_states()

    arrays = load_path_cache()

    # -------------------------------------------------------------------------
    # MARKET VOLATILITY
    # -------------------------------------------------------------------------

    market = load_market_volatility()

    volatility = build_volatility_context(
        metadata,
        market,
    )

    # -------------------------------------------------------------------------
    # EVENT CONTEXT
    # -------------------------------------------------------------------------

    events = build_event_context(
        metadata,
        hmm,
        volatility,
    )

    # -------------------------------------------------------------------------
    # CRITICAL VALIDATION
    # -------------------------------------------------------------------------

    validation = validate_target_contexts(events)

    # Save validation immediately.
    validation.to_csv(
        RESULTS_DIR / "research_08h_context_validation.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    results = run_search(
        events,
        arrays,
    )

    # -------------------------------------------------------------------------
    # RANK
    # -------------------------------------------------------------------------

    ranked, candidates = rank_results(results)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    save_outputs(
        ranked,
        candidates,
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    report(
        ranked,
        candidates,
    )

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    banner("RESEARCH 08H COMPLETE")

    print("Context fixed.")

    print("Only TP / SL / horizon searched.")

    print("Failure test NOT performed.")

    print("Next stage:")

    print("candidate robustness -> OOS validation -> strategy freeze -> failure test.")


if __name__ == "__main__":
    main()
